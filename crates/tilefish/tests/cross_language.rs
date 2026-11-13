//! The digests of Python and of Rust are the same, on real cubins.
//!
//! `tests/fixtures/sm_103` and `sm_120` are manifest directories that `tilefish.runtime.export` wrote on a B300 (the
//! second one cross-compiled): the five goldens of the lowering, a bucketed row kernel and three hand-written Triton
//! kernels, as 11 cubins each. `expected.json` has the `manifest_digest` of each manifest and the `sha256` and
//! `sha256_code` of each cubin *as Python computed them* (and `python/tests/test_rust_fixtures.py` checks that Python still
//! does). Here Rust computes them from the files, with its own JSON reader, canonical writer and ELF walk, and they must be
//! equal digit for digit. No GPU is involved.

mod common;

use common::{expected, fixtures};
use serde_json::Value;
use tilefish::elf::{Elf, sha256_hex};
use tilefish::json::{canonical, parse_strict};
use tilefish::{Manifest, Pin, manifest_digest};

const FIXTURES: [&str; 2] = ["sm_103", "sm_120"];

#[test]
fn the_manifest_digest_is_pythons() {
    for name in FIXTURES {
        let want = common::expected_digest(name);
        let text = std::fs::read_to_string(fixtures().join(name).join("manifest.json")).unwrap();
        let doc = parse_strict(&text).unwrap();
        assert_eq!(manifest_digest(&doc).unwrap(), want, "{name}: the digest of the document");
        // loading it with the pin Python computed accepts it, and says the same digest
        let manifest = Manifest::load(fixtures().join(name), Pin::Digest(&want)).unwrap();
        assert_eq!(manifest.digest(), want, "{name}");
        let kernels = common::expected()["fixtures"][name]["kernels"].as_array().unwrap().len();
        assert_eq!(manifest.entries().len(), kernels, "{name}: one entry for each binary expected.json lists");
    }
}

#[test]
fn the_digests_of_every_cubin_are_pythons() {
    let expected = expected();
    for name in FIXTURES {
        let manifest = Manifest::load(fixtures().join(name), Pin::Digest(&common::expected_digest(name))).unwrap();
        let kernels = expected["fixtures"][name]["kernels"].as_array().unwrap();
        assert_eq!(kernels.len(), manifest.entries().len(), "{name}");
        for want in kernels {
            let (stem, arch) = (want["stem"].as_str().unwrap(), want["arch"].as_str().unwrap());
            let entry = manifest.entry(stem, arch).unwrap_or_else(|| panic!("{name}: no entry for {stem} {arch}"));
            assert_eq!(entry.cubin_name, want["cubin"].as_str().unwrap());
            // from the bytes of the file, not from what the manifest says
            let bytes = std::fs::read(fixtures().join(name).join(&entry.cubin_name)).unwrap();
            assert_eq!(sha256_hex(&bytes), want["sha256"].as_str().unwrap(), "{name} {stem}: sha256");
            let elf = Elf::parse(&bytes).unwrap();
            assert_eq!(elf.code_digest(), want["sha256_code"].as_str().unwrap(), "{name} {stem}: sha256_code");
            assert_eq!(&*entry.cubin, &bytes[..]);
        }
    }
}

#[test]
fn the_two_architectures_are_other_binaries_of_the_same_kernels() {
    let a = Manifest::load(fixtures().join("sm_103"), Pin::TrustDirectory).unwrap();
    let b = Manifest::load(fixtures().join("sm_120"), Pin::TrustDirectory).unwrap();
    assert_ne!(a.digest(), b.digest());
    for entry in a.entries() {
        let other = b.entry(&entry.stem, "sm_120").unwrap_or_else(|| panic!("no sm_120 binary of {}", entry.stem));
        assert_eq!(entry.params, other.params);
        assert_eq!(entry.grid, other.grid);
        assert_eq!(entry.preconditions, other.preconditions);
        // (the declaration of a prebuilt binary holds the digest of the PTX or cubin it was made from, which is another file for each
        // architecture when it is a Triton build's PTX or a cubin)
        if entry.kind == tilefish::EntryKind::Binary && entry.source_digest != other.source_digest {
            assert_ne!(
                entry.declaration_digest, other.declaration_digest,
                "{}: another source is another declaration",
                entry.stem
            );
        } else {
            assert_eq!(entry.declaration_digest, other.declaration_digest, "{}", entry.stem);
        }
        assert_ne!(entry.sha256_code, other.sha256_code, "{}: another architecture is other machine code", entry.stem);
    }
}

#[test]
fn the_order_and_the_formatting_of_the_file_do_not_matter_to_the_digest() {
    let text = std::fs::read_to_string(fixtures().join("sm_103").join("manifest.json")).unwrap();
    let mut doc = parse_strict(&text).unwrap();
    let want = common::expected_digest("sm_103");
    // the same document, compact, with the entries reversed, and the top level written in another order
    doc["kernels"].as_array_mut().unwrap().reverse();
    let reordered = canonical(&doc);
    let again = parse_strict(&reordered).unwrap();
    assert_eq!(manifest_digest(&again).unwrap(), want);
    // the version of the tool that wrote it is not part of the digest
    let mut other = again.clone();
    other["tilefish"] = Value::String("9.9.9".to_owned());
    assert_eq!(manifest_digest(&other).unwrap(), want);
    // any field of any entry is
    let mut changed = again;
    changed["kernels"][0]["shared"] = Value::from(17);
    assert_ne!(manifest_digest(&changed).unwrap(), want);
}
