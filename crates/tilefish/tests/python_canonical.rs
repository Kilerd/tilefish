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
    // schema 3: the empty one, the tool's version and the order of the entries (kernels and library) are not part of it
    let empty_v3 =
        digests.iter().position(|d| *d == "f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e");
    assert_eq!(empty_v3, Some(8), "the empty schema 3 manifest, the first of the schema 3 vectors");
    assert_eq!(digests[9], digests[10], "the version of the tool is not part of a schema 3 digest either");
    assert_eq!(digests[11], digests[12], "the order of the kernels is not");
    // a schema 3 digest is never a schema 2 one, and every vector is a different document
    let schema3: std::collections::HashSet<&&str> = digests[8..].iter().collect();
    assert!(digests[..8].iter().all(|d| !schema3.contains(d)));
    assert!(digests.len() >= 30);
}

/// What Python's `manifest_digest` refuses (a schema 3 document with no `library`, a list that is not one, an entry that is not an object, ...),
/// the Rust one refuses too, and a text that is not strict JSON never gets that far.
#[test]
fn a_document_python_cannot_digest_is_refused() {
    assert!(manifests::MANIFEST_ERRORS.len() >= 20);
    for text in manifests::MANIFEST_ERRORS {
        match parse_strict(text) {
            Err(_) => {} // NaN, Infinity and a number that does not fit a float are not strict JSON
            Ok(doc) => assert!(manifest_digest(&doc).is_err(), "{text}"),
        }
    }
}
