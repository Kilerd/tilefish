//! The digest of the machine code of a cubin is Python's `code_digest`, on the edges of the format as well as on real cubins.
//!
//! `data/python_elf.in` has small ELF files built by `python/tests/elf_builder.py` (the layout of a cubin: debug sections that
//! hold the build path, sections that name each other in `sh_link` and `sh_info`, `NOBITS` sections, files whose sections move,
//! other identity fields) with the `code_digest` Python computes for each. Two files that differ only in the build path or in
//! where their sections lie have the same digest; a file with other code or another identity has another.

use tilefish::elf::Elf;

include!("data/python_elf.in");

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex")).collect()
}

#[test]
fn the_code_digest_of_every_file_is_pythons() {
    assert!(ELFS.len() >= 13);
    for (name, hex, expected) in ELFS {
        let blob = unhex(hex);
        let elf = Elf::parse(&blob).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(elf.code_digest(), *expected, "{name}");
    }
}

#[test]
fn what_is_not_code_does_not_change_the_digest() {
    let digest = |name: &str| ELFS.iter().find(|(n, _, _)| *n == name).map(|(_, _, d)| *d).expect(name);
    assert_eq!(digest("default"), digest("another debug path"));
    assert_eq!(digest("default"), digest("offsets moved by a gap"));
    assert_eq!(digest("default"), digest("debug sections first")); // moving the debug sections moves the indices, not the references by name
    for other in [
        "no shared section",
        "empty shared section (sm_80)",
        "other code",
        "another OS ABI",
        "another machine",
        "other e_flags",
    ] {
        assert_ne!(digest("default"), digest(other), "{other}");
    }
}
