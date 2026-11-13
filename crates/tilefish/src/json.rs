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

use serde_json::{Map, Number, Value};

const MAX_DEPTH: usize = 64;

/// Parse `text` as strict JSON. The error is the reason, worded like the Python one where Python has one.
pub fn parse_strict(text: &str) -> Result<Value, String> {
    if text.starts_with('\u{feff}') {
        return Err("it starts with a byte order mark".to_owned());
    }
    let mut p = Parser { s: text.as_bytes(), text, i: 0 };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("extra data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
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
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Value::Object(map));
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
            let v = self.value(depth + 1)?;
            if map.contains_key(&key) {
                return Err(format!("an object has the key {key:?} twice"));
            }
            map.insert(key, v);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(map));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
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
            items.push(self.value(depth + 1)?);
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
    fn reads_what_python_reads() {
        let v = parse_strict(" {\"b\": [1, -0, 2.5, 1e2, true, null], \"a\": \"\\u00e9\\ud83d\\ude00\"} ").unwrap();
        assert_eq!(canonical(&v), "{\"a\":\"\\u00e9\\ud83d\\ude00\",\"b\":[1,0,2.5,100.0,true,null]}");
        assert_eq!(parse_strict("18446744073709551615").unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn keys_are_kept_in_file_order_and_dumped_sorted() {
        let v = parse_strict(r#"{"z": 1, "a": 2, "m": {"y": 1, "b": 2}}"#).unwrap();
        let keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, ["z", "a", "m"]);
        assert_eq!(canonical(&v), r#"{"a":2,"m":{"b":2,"y":1},"z":1}"#);
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
