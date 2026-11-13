//! The canonical JSON of a manifest digest is Python's `json.dumps(sort_keys=True, separators=(",", ":"))`, byte for byte.
//!
//! `data/python_canonical.in` is made by CPython: JSON texts (escapes, control characters, `0x7f`, characters above the
//! basic plane, keys whose code point order is not their UTF-16 order, the edges of the integers, floats, nesting, random
//! strings) with what `json.dumps(json.loads(text), ...)` gives for each. Reading with `parse_strict` and writing with
//! `canonical` must give the same text. (The spelling of floats alone is in `python_floats.rs`.)

use tilefish::json::{canonical, parse_strict};
use tilefish::manifest_digest;

include!("data/python_canonical.in");

mod manifests {
    include!("data/python_manifests.in");
}

#[test]
fn canonical_json_is_pythons() {
    assert!(CANONICAL.len() > 50);
    for (text, expected) in CANONICAL {
        let value = parse_strict(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(canonical(&value), *expected, "{text}");
    }
}

#[test]
fn the_digest_of_a_manifest_document_is_pythons() {
    let digests: Vec<&str> = manifests::MANIFESTS.iter().map(|(_, d)| *d).collect();
    for (text, expected) in manifests::MANIFESTS {
        let doc = parse_strict(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(manifest_digest(&doc).unwrap(), *expected, "{text}");
    }
    // the tool's version is not part of a digest, the order of the entries is not, the schema is
    assert_eq!(digests[1], digests[2]);
    assert_eq!(digests[3], digests[4]);
    assert_ne!(digests[1], digests[5]);
    assert_eq!(digests.len(), 8);
}
