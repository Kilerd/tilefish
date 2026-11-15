//! Strict JSON in, Python's canonical JSON out.
//!
//! [`parse_strict`] reads what Python's `Manifest.load` reads and refuses what it refuses: an object with the same key
//! twice, the constants `NaN` and `Infinity`, a number that does not fit a float, and anything that is not JSON. It
//! also refuses what a Rust `String` cannot hold (a lone surrogate escape), a byte order mark and an integer beyond 64
//! bits, which a written manifest never has.
//!
//! [`canonical`] is `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)` of Python, byte for
//! byte: the digest of a manifest is a sha256 of that text, so a Rust host must produce it exactly. The subtle parts
//! are `ensure_ascii` (every character outside `0x20..=0x7e` is a lowercase `\uXXXX`, with surrogate pairs above the
//! basic plane, and `0x7f` is escaped too), the order of the keys (code points), and the spelling of a float, which is
//! `repr(float)` (`1e-05`, `1e+16`, `100000.0`, `-0.0`).
//!
//! The keys of every object [`parse_strict`] reads come out in code point order, whatever map serde_json was built with: a
//! `BTreeMap`, or an `IndexMap` in file order when a crate of the process turns on serde_json's `preserve_order` (cargo turns a
//! feature on for every crate of a build). Nothing this crate reads, refuses or says depends on that feature. Where the order of the
//! file means something (Python's dicts keep it: the buckets of a manifest entry make its stem in their order), the reader asks the
//! parser to record it for those objects, by their path, and takes it from there.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

const MAX_DEPTH: usize = 64;

/// Parse `text` as strict JSON. The error is the reason, worded like the Python one where Python has one. The keys of every object
/// are in code point order (see the module).
pub fn parse_strict(text: &str) -> Result<Value, String> {
    parse_strict_keeping(text, |_| false).map(|(v, _)| v)
}

/// One step of the path from the root of a document to one of its values: a key of an object, or an index of an array.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PathStep {
    Key(String),
    Index(usize),
}

/// The order the file gives the keys of the objects [`parse_strict_keeping`] was asked to keep, by the path of each.
#[derive(Debug, Default)]
pub(crate) struct KeyOrder(HashMap<Vec<PathStep>, Vec<String>>);

impl KeyOrder {
    /// The keys of the object at `path`, in the order of the file, when it was kept.
    pub(crate) fn keys(&self, path: &[PathStep]) -> Option<&[String]> {
        self.0.get(path).map(Vec::as_slice)
    }

    /// The members of `map`, the object at `path`, in the order of the file when it was kept, else in code point order.
    pub(crate) fn members<'m>(&self, path: &[PathStep], map: &'m Map<String, Value>) -> Vec<(&'m String, &'m Value)> {
        match self.keys(path) {
            Some(keys) => keys.iter().filter_map(|k| map.get_key_value(k)).collect(),
            None => map.iter().collect(),
        }
    }
}

/// [`parse_strict`], and the order of the keys of every object whose path `keep` accepts, as the file has them.
pub(crate) fn parse_strict_keeping(
    text: &str,
    keep: impl Fn(&[PathStep]) -> bool,
) -> Result<(Value, KeyOrder), String> {
    if text.starts_with('\u{feff}') {
        return Err("it starts with a byte order mark".to_owned());
    }
    let mut p = Parser { s: text.as_bytes(), text, i: 0, path: Vec::new(), keep: &keep, order: KeyOrder::default() };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("extra data at byte {}", p.i));
    }
    Ok((v, p.order))
}

/// The members of an object of a manifest entry whose order means something (its `buckets`, its `dims`): read in code point order, then
/// put in the order of the file by [`Members::follow`]. Looked up by key as a map is.
pub(crate) struct Members {
    map: Map<String, Value>,
    order: Vec<String>,
}

impl<'de> Deserialize<'de> for Members {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map = Map::deserialize(d)?;
        let order = map.keys().cloned().collect();
        Ok(Self { map, order })
    }
}

impl Members {
    /// Take the order of the file, which [`parse_strict_keeping`] kept (`None`: it was not kept, and the code point order stays).
    pub(crate) fn follow(&mut self, file_order: Option<&[String]>) {
        if let Some(keys) = file_order
            && keys.len() == self.map.len()
            && keys.iter().all(|k| self.map.contains_key(k))
        {
            self.order = keys.to_vec();
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    pub(crate) fn get<Q: ?Sized + Ord + Eq + Hash>(&self, key: &Q) -> Option<&Value>
    where
        String: Borrow<Q>,
    {
        self.map.get(key)
    }

    pub(crate) fn contains_key<Q: ?Sized + Ord + Eq + Hash>(&self, key: &Q) -> bool
    where
        String: Borrow<Q>,
    {
        self.map.contains_key(key)
    }

    /// The keys, in order.
    pub(crate) fn keys(&self) -> std::slice::Iter<'_, String> {
        self.order.iter()
    }

    /// The members, in order.
    pub(crate) fn iter(&self) -> MembersIter<'_> {
        MembersIter { map: &self.map, keys: self.order.iter() }
    }
}

impl<'a> IntoIterator for &'a Members {
    type Item = (&'a String, &'a Value);
    type IntoIter = MembersIter<'a>;

    fn into_iter(self) -> MembersIter<'a> {
        self.iter()
    }
}

/// The members of a [`Members`], in order.
pub(crate) struct MembersIter<'a> {
    map: &'a Map<String, Value>,
    keys: std::slice::Iter<'a, String>,
}

impl<'a> Iterator for MembersIter<'a> {
    type Item = (&'a String, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.keys.next().and_then(|k| self.map.get_key_value(k))
    }
}

/// As a `Map` writes itself (`{"M": Number(32)}`), in order.
impl fmt::Debug for Members {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

struct Parser<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
    /// The path of the value being read.
    path: Vec<PathStep>,
    keep: &'a dyn Fn(&[PathStep]) -> bool,
    order: KeyOrder,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.s.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err("it is nested too deeply".to_owned());
        }
        match self.s.get(self.i) {
            None => self.err("unexpected end of input"),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'N') => self.constant("NaN"),
            Some(b'I') => self.constant("Infinity"),
            Some(b'-') if self.s.get(self.i + 1) == Some(&b'I') => self.constant("-Infinity"),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.err("unexpected character"),
        }
    }

    fn constant(&self, name: &str) -> Result<Value, String> {
        if self.text[self.i..].starts_with(name) {
            Err(format!("it contains the constant {name}"))
        } else {
            self.err("unexpected character")
        }
    }

    fn literal(&mut self, word: &str, v: Value) -> Result<Value, String> {
        if self.text[self.i..].starts_with(word) {
            self.i += word.len();
            Ok(v)
        } else {
            self.err("unexpected character")
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, String> {
        self.i += 1;
        let mut map = Map::new();
        // the keys in the order of the file, when the reader wants it for this object
        let mut file_order = (self.keep)(&self.path).then(Vec::new);
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(self.finish_object(map, file_order));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return self.err("expected a string key");
            }
            let key = self.string()?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return self.err("expected ':'");
            }
            self.i += 1;
            self.ws();
            self.path.push(PathStep::Key(key));
            let v = self.value(depth + 1)?;
            let Some(PathStep::Key(key)) = self.path.pop() else { unreachable!("the key pushed above") };
            if map.contains_key(&key) {
                return Err(format!("an object has the key {key:?} twice"));
            }
            if let Some(keys) = &mut file_order {
                keys.push(key.clone());
            }
            map.insert(key, v);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(self.finish_object(map, file_order));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }

    /// The object in code point order whatever the map (an `IndexMap` keeps the order of insertion), its file order recorded when kept.
    fn finish_object(&mut self, mut map: Map<String, Value>, file_order: Option<Vec<String>>) -> Value {
        map.sort_keys();
        if let Some(keys) = file_order {
            self.order.0.insert(self.path.clone(), keys);
        }
        Value::Object(map)
    }

    fn array(&mut self, depth: usize) -> Result<Value, String> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.ws();
            self.path.push(PathStep::Index(items.len()));
            items.push(self.value(depth + 1)?);
            self.path.pop();
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        if self.s.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return self.err("expected a digit"),
        }
        let mut is_float = false;
        if self.s.get(self.i) == Some(&b'.') && matches!(self.s.get(self.i + 1), Some(b'0'..=b'9')) {
            is_float = true;
            self.i += 1;
            self.digits();
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            let mut j = self.i + 1;
            if matches!(self.s.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if matches!(self.s.get(j), Some(b'0'..=b'9')) {
                is_float = true;
                self.i = j;
                self.digits();
            }
        }
        let token = &self.text[start..self.i];
        if is_float {
            let x: f64 = token.parse().map_err(|_| format!("the number {token} is not a number"))?;
            match Number::from_f64(x) {
                Some(n) => Ok(Value::Number(n)),
                None => Err(format!("the number {token} does not fit a float")),
            }
        } else if let Ok(n) = token.parse::<i64>() {
            Ok(Value::Number(n.into())) // "-0" is the integer 0, as in Python
        } else if let Ok(n) = token.parse::<u64>() {
            Ok(Value::Number(n.into()))
        } else {
            Err(format!("the integer {token} does not fit 64 bits"))
        }
    }

    fn digits(&mut self) {
        while matches!(self.s.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let Some(h) = self.s.get(self.i..self.i + 4) else {
            return self.err("truncated \\u escape");
        };
        let mut v = 0u32;
        for b in h {
            let d = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return self.err("invalid \\u escape"),
            };
            v = v * 16 + u32::from(d);
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let run = self.i;
            while let Some(&b) = self.s.get(self.i) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(&self.text[run..self.i]);
            match self.s.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let Some(&e) = self.s.get(self.i) else {
                        return self.err("unterminated string");
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&hi) {
                                if self.s.get(self.i) != Some(&b'\\') || self.s.get(self.i + 1) != Some(&b'u') {
                                    return self.err("a lone surrogate escape");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return self.err("a lone surrogate escape");
                                }
                                0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                            } else if (0xdc00..0xe000).contains(&hi) {
                                return self.err("a lone surrogate escape");
                            } else {
                                hi
                            };
                            match char::from_u32(code) {
                                Some(c) => out.push(c),
                                None => return self.err("invalid \\u escape"),
                            }
                        }
                        _ => return self.err("invalid escape"),
                    }
                }
                Some(_) => return self.err("a control character in a string"),
            }
        }
    }
}

/// The shortest digits that read back as `x` (positive and finite) and where the decimal point goes: `x = 0.<digits> * 10^decpt`.
///
/// This must be CPython's digits, and not only some shortest digits: when two strings of the same length both round-trip, Python's
/// `repr` (David Gay's `dtoa` in shortest mode) takes the one closest to the exact value and, when the exact value is a tie between
/// them (it ends in a 5 one digit past the shortest), the even one. Rust's `{:e}` rounds such a tie up (`2^-25` is
/// `2.9802322387695312e-08` in Python and `...313e-08` there), so the digits come from Ryu, which rounds the tie to even.
fn shortest_digits(x: f64) -> (String, i32) {
    let mut buffer = ryu::Buffer::new();
    let text = buffer.format_finite(x);
    let (mantissa, exponent) = match text.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().expect("Ryu writes an integer exponent")),
        None => (text, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{int_part}{frac_part}");
    let mut decpt = int_part.len() as i32 + exponent;
    let leading = all.len() - all.trim_start_matches('0').len();
    decpt -= leading as i32;
    let digits = all.trim_matches('0').to_owned();
    (digits, decpt)
}

/// `repr(x)` of a Python float, for a finite `x`.
pub fn python_float_repr(x: f64) -> String {
    assert!(x.is_finite(), "a canonical document has only finite floats");
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    let (digits, decpt) = shortest_digits(x.abs()); // the value is 0.<digits> * 10^decpt
    let n = digits.len() as i32;
    let body = if decpt <= -4 || decpt > 16 {
        let e = decpt - 1;
        let m = if n == 1 { digits } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= n {
        format!("{digits}{}.0", "0".repeat((decpt - n) as usize))
    } else {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    };
    if x < 0.0 { format!("-{body}") } else { body }
}

fn push_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn push_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&python_float_repr(n.as_f64().expect("a JSON number is an integer or a float")));
            }
        }
        Value::String(s) => push_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0)); // by code point: the byte order of UTF-8 is the same
            out.push('{');
            for (i, (key, value)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_string(out, key);
                out.push(':');
                push_value(out, value);
            }
            out.push('}');
        }
    }
}

/// Python's `json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)`.
pub fn canonical(v: &Value) -> String {
    let mut out = String::new();
    push_value(&mut out, v);
    out
}

/// The name Python gives the type of what `json.loads` makes of `v` (`type(v).__name__`): `"list"` for an array, `"dict"` for an object.
pub(crate) fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_what_python_refuses() {
        for (text, why) in [
            (r#"{"a": 1, "a": 2}"#, "twice"),
            (r#"{"a": NaN}"#, "NaN"),
            (r#"[Infinity]"#, "Infinity"),
            (r#"[-Infinity]"#, "-Infinity"),
            (r#"[1e999]"#, "does not fit a float"),
            (r#"[1, 2,]"#, "unexpected"),
            (r#"{"a": 1,}"#, "string key"),
            ("[01]", "expected ','"),
            ("[1.]", "expected ','"),
            ("\"a\u{1}b\"", "control character"),
            (r#""\ud800""#, "lone surrogate"),
            (r#""\x""#, "invalid escape"),
            ("{}{}", "extra data"),
            ("", "unexpected end"),
            ("\u{feff}{}", "byte order mark"),
            ("[18446744073709551616]", "64 bits"),
        ] {
            let e = parse_strict(text).expect_err(text);
            assert!(e.contains(why), "{text:?}: {e}");
        }
    }

    #[test]
    fn nesting_is_bounded_exactly() {
        // a leaf at depth 64 is read, one at depth 65 is not: for arrays and for objects, so a hostile file cannot overflow the stack
        let arrays = |n: usize| format!("{}1{}", "[".repeat(n), "]".repeat(n));
        let objects = |n: usize| format!("{}1{}", "{\"a\":".repeat(n), "}".repeat(n));
        for make in [&arrays as &dyn Fn(usize) -> String, &objects] {
            assert!(parse_strict(&make(64)).is_ok());
            assert!(parse_strict(&make(65)).expect_err("65 levels").contains("nested too deeply"));
            assert!(parse_strict(&make(5000)).expect_err("5000 levels").contains("nested too deeply"));
        }
        // empty containers count too
        assert!(parse_strict(&format!("{}{}", "[".repeat(65), "]".repeat(65))).is_ok());
        assert!(parse_strict(&format!("{}{}", "[".repeat(66), "]".repeat(66))).is_err());
    }

    #[test]
    fn surrogate_escapes_are_read_at_the_edges_of_their_ranges() {
        let text = |escapes: &str| match parse_strict(&format!("\"{escapes}\"")) {
            Ok(Value::String(s)) => Ok(s),
            Ok(other) => panic!("{other:?}"),
            Err(e) => Err(e),
        };
        // the pairs at both ends, and the code points just outside the surrogate range on each side
        assert_eq!(text("\\ud800\\udc00").unwrap(), "\u{10000}");
        assert_eq!(text("\\udbff\\udfff").unwrap(), "\u{10ffff}");
        assert_eq!(text("\\ud83d\\ude00").unwrap(), "\u{1f600}");
        assert_eq!(text("\\ud7ff").unwrap(), "\u{d7ff}");
        assert_eq!(text("\\ue000").unwrap(), "\u{e000}");
        assert_eq!(text("\\uffff").unwrap(), "\u{ffff}");
        // a high surrogate needs a low one right after it, and a low one cannot stand alone
        for (escapes, what) in [
            ("\\ud800", "the first high surrogate, alone"),
            ("\\udbff", "the last high surrogate, alone"),
            ("\\udc00", "the first low surrogate, alone"),
            ("\\udfff", "the last low surrogate, alone"),
            ("\\ud800x", "a high surrogate and a character"),
            ("\\ud800\\n", "a high surrogate and another escape"),
            ("\\ud800\\u0041", "a high surrogate and a letter"),
            ("\\ud800\\ud800", "two high surrogates"),
            ("\\ud800\\udbff", "a high surrogate and one just below the low range"),
            ("\\ud800\\ue000", "a high surrogate and one just above the low range"),
            ("\\udc00\\ud800", "a low surrogate first"),
            ("\\udc00\\udc00", "two low surrogates"),
        ] {
            let e = text(escapes).expect_err(what);
            assert!(e.contains("lone surrogate"), "{what}: {e}");
        }
        // four hex digits, of either case, and nothing else
        assert_eq!(text("\\u00E9\\u00e9").unwrap(), "\u{e9}\u{e9}");
        for bad in ["\\u12", "\\u12g4", "\\u+123", "\\u 123"] {
            assert!(text(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn reads_what_python_reads() {
        let v = parse_strict(" {\"b\": [1, -0, 2.5, 1e2, true, null], \"a\": \"\\u00e9\\ud83d\\ude00\"} ").unwrap();
        assert_eq!(canonical(&v), "{\"a\":\"\\u00e9\\ud83d\\ude00\",\"b\":[1,0,2.5,100.0,true,null]}");
        assert_eq!(parse_strict("18446744073709551615").unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn keys_are_in_code_point_order_whatever_the_map_and_dumped_sorted() {
        // with or without serde_json's `preserve_order` (`cargo test --features serde_json/preserve_order`)
        let v = parse_strict(r#"{"z": 1, "a": 2, "m": {"y": 1, "b": 2}, "l": [{"q": 1, "p": 2}]}"#).unwrap();
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&v), ["a", "l", "m", "z"]);
        assert_eq!(keys(&v["m"]), ["b", "y"]);
        assert_eq!(keys(&v["l"][0]), ["p", "q"]);
        assert_eq!(v.to_string(), r#"{"a":2,"l":[{"p":2,"q":1}],"m":{"b":2,"y":1},"z":1}"#);
        assert_eq!(canonical(&v), r#"{"a":2,"l":[{"p":2,"q":1}],"m":{"b":2,"y":1},"z":1}"#);
    }

    #[test]
    fn the_file_order_of_the_objects_a_reader_keeps_is_recorded_by_their_path() {
        let text = r#"{"z": {"y": 1, "b": 2}, "list": [{"q": 1, "p": 2}, {"s": 1, "r": 2, "t": 3}], "e": {}}"#;
        let key = |k: &str| PathStep::Key(k.to_owned());
        let (v, order) = parse_strict_keeping(text, |path| {
            matches!(path, [PathStep::Key(k)] if k == "z" || k == "e")
                || matches!(path, [PathStep::Key(k), PathStep::Index(1)] if k == "list")
        })
        .unwrap();
        assert_eq!(v, parse_strict(text).unwrap(), "the same value, in code point order");
        assert_eq!(order.keys(&[key("z")]), Some(&["y".to_owned(), "b".to_owned()][..]));
        assert_eq!(
            order.keys(&[key("list"), PathStep::Index(1)]),
            Some(&["s".to_owned(), "r".to_owned(), "t".to_owned()][..])
        );
        assert_eq!(order.keys(&[key("e")]), Some(&[][..]));
        // what the reader did not ask for is not recorded: the root, the first object of the list
        assert_eq!(order.keys(&[]), None);
        assert_eq!(order.keys(&[key("list"), PathStep::Index(0)]), None);
        let members = |path: &[PathStep], at: &Value| -> Vec<String> {
            order.members(path, at.as_object().unwrap()).into_iter().map(|(k, _)| k.clone()).collect()
        };
        assert_eq!(members(&[key("z")], &v["z"]), ["y", "b"]);
        assert_eq!(
            members(&[key("list"), PathStep::Index(0)], &v["list"][0]),
            ["p", "q"],
            "not kept: code point order"
        );
        // a `Members` read from a kept object takes the file's order, and is looked up by key
        let mut m: Members = serde_json::from_value(v["list"][1].clone()).unwrap();
        assert_eq!(m.keys().cloned().collect::<Vec<_>>(), ["r", "s", "t"]);
        m.follow(order.keys(&[key("list"), PathStep::Index(1)]));
        assert_eq!(
            m.iter().map(|(k, v)| (k.as_str(), v.as_u64().unwrap())).collect::<Vec<_>>(),
            [("s", 1), ("r", 2), ("t", 3)]
        );
        assert_eq!((m.len(), m.get("r"), m.contains_key("q")), (3, Some(&Value::from(2)), false));
        assert_eq!(format!("{m:?}"), r#"{"s": Number(1), "r": Number(2), "t": Number(3)}"#);
        // an order that is not of these keys is not taken
        m.follow(Some(&["t".to_owned(), "s".to_owned()]));
        assert_eq!(m.keys().cloned().collect::<Vec<_>>(), ["s", "r", "t"]);
    }

    #[test]
    fn strings_are_escaped_like_ensure_ascii() {
        let v = Value::String("a\"b\\c\n\t\u{1}\u{7f} \u{e9}\u{4e2d}\u{1f600}".to_owned());
        assert_eq!(canonical(&v), "\"a\\\"b\\\\c\\n\\t\\u0001\\u007f \\u00e9\\u4e2d\\ud83d\\ude00\"");
    }

    #[test]
    fn floats_are_spelled_like_python() {
        for (x, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100000.0, "100000.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (0.5, "0.5"),
            (123.456, "123.456"),
            (-2.5e-7, "-2.5e-07"),
            (1e-6, "1e-06"),
            (0.1 + 0.2, "0.30000000000000004"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (9007199254740993.0, "9007199254740992.0"),
            // exact decimal ties between two shortest strings: Python takes the even one (Rust's `{:e}` rounds them up)
            (2.0f64.powi(-25), "2.9802322387695312e-08"),
            (5.0 * 2.0f64.powi(-23), "5.960464477539062e-07"),
            (9.0 * 2.0f64.powi(-23), "1.0728836059570312e-06"),
            (13.0 * 2.0f64.powi(-23), "1.5497207641601562e-06"),
            (2.0f64.powi(49) + 0.25, "562949953421312.2"),
            (2.0f64.powi(49) + 0.75, "562949953421312.8"),
        ] {
            assert_eq!(python_float_repr(x), want, "{x:e}");
        }
    }
}
