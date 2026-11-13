//! The canonical JSON of a manifest digest is Python's `json.dumps(sort_keys=True, separators=(",", ":"))`, byte for byte.
//!
//! `data/python_canonical.in` is made by CPython: JSON texts (escapes, control characters, `0x7f`, characters above the
//! basic plane, keys whose code point order is not their UTF-16 order, the edges of the integers, floats, nesting, random
//! strings) with what `json.dumps(json.loads(text), ...)` gives for each. Reading with `parse_strict` and writing with
//! `canonical` must give the same text. (The spelling of floats alone is in `python_floats.rs`.)

use tilefish::json::{canonical, parse_strict};

include!("data/python_canonical.in");

#[test]
fn canonical_json_is_pythons() {
    assert!(CANONICAL.len() > 50);
    for (text, expected) in CANONICAL {
        let value = parse_strict(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(canonical(&value), *expected, "{text}");
    }
}
