//! Nothing a file can contain makes the reader panic: a damaged cubin, a damaged manifest and a damaged expression are
//! errors. The mutations are deterministic (a small generator with a fixed seed), so a failure reproduces.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::fixtures;
use tilefish::elf::Elf;
use tilefish::json::parse_strict;
use tilefish::{Arch, Manifest, Pin};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn no_panic(what: &str, f: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(f)).is_ok(), "{what} panicked");
}

#[test]
fn a_damaged_cubin_is_an_error_never_a_panic() {
    let original = std::fs::read(fixtures().join("sm_103").join("softmax.sm_103.cubin")).unwrap();
    let section_table = u64::from_le_bytes(original[40..48].try_into().unwrap()) as usize;
    let mut rng = Lcg(1);
    let mut errors = 0;
    for round in 0..4000 {
        let mut bytes = original.clone();
        for _ in 0..1 + rng.below(4) {
            // the header and the section table are where the walk looks; the rest is where the data is
            let at = match rng.below(3) {
                0 => rng.below(64),
                1 => section_table + rng.below(bytes.len() - section_table),
                _ => rng.below(bytes.len()),
            };
            bytes[at] = rng.next() as u8;
        }
        if round % 7 == 0 {
            bytes.truncate(rng.below(bytes.len()));
        }
        no_panic("the ELF walk", || {
            if Elf::parse(&bytes).map(|e| e.code_digest()).is_err() {
                errors += 1;
            }
        });
    }
    assert!(errors > 100, "the mutations should break the file often ({errors})");
}

#[test]
fn damaged_json_is_an_error_never_a_panic() {
    let text = std::fs::read_to_string(fixtures().join("sm_103").join("manifest.json")).unwrap();
    let mut rng = Lcg(2);
    for _ in 0..3000 {
        let mut bytes = text.clone().into_bytes();
        for _ in 0..1 + rng.below(3) {
            let at = rng.below(bytes.len());
            bytes[at] = b"{}[]\",:\\ -.e0\xff\n"[rng.below(15)];
        }
        if rng.below(5) == 0 {
            bytes.truncate(rng.below(bytes.len()));
        }
        let damaged = String::from_utf8_lossy(&bytes).into_owned();
        no_panic("the strict JSON reader", || {
            let _ = parse_strict(&damaged);
        });
    }
}

#[test]
fn a_damaged_manifest_is_an_error_never_a_panic() {
    let dir = fixtures().join("sm_103");
    let text = std::fs::read(dir.join("manifest.json")).unwrap();
    let cubins: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "cubin"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap()))
        .collect();
    assert_eq!(cubins.len(), 12);
    let files: Vec<(&str, &[u8])> = cubins.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let mut rng = Lcg(3);
    let mut accepted = 0;
    for _ in 0..600 {
        let mut bytes = text.clone();
        for _ in 0..1 + rng.below(3) {
            let at = rng.below(bytes.len());
            // mostly digits and letters, so that the damage is a valid document more often than not
            bytes[at] = b"0123456789abcdefxyz_.\"{}[]"[rng.below(26)];
        }
        no_panic("the manifest reader", || {
            // the pin is the one thing a damaged file cannot match, so read it as a directory would be read without one
            if Manifest::from_embedded(&bytes, &files, Pin::TrustDirectory).is_ok() {
                accepted += 1;
            }
        });
    }
    println!("{accepted} damaged manifests were still valid ones");
}

#[test]
fn a_damaged_expression_is_an_error_never_a_panic() {
    // the same fixture, with a random expression where the manifest has a shape, a grid or the dimension a scalar carries
    let dir = fixtures().join("sm_103");
    let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    let cubins: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "cubin"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap()))
        .collect();
    let files: Vec<(&str, &[u8])> = cubins.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let alphabet: Vec<&str> =
        vec!["(", ")", ",", "+", "-", "*", "//", "cdiv", "n", "Count", "2", "1000000000000", "0", " ", "x_y", "#", "é"];
    let mut rng = Lcg(4);
    let mut accepted = 0;
    for round in 0..400 {
        let len = 1 + rng.below(12);
        let expression: String = (0..len).map(|_| alphabet[rng.below(alphabet.len())]).collect();
        let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        let entry = doc["kernels"].as_array_mut().unwrap().iter_mut().find(|k| k["stem"] == "gelu_erf").unwrap();
        match round % 3 {
            0 => entry["abi"]["params"][0]["shape"][0] = expression.clone().into(),
            1 => entry["grid"] = expression.clone().into(),
            _ => entry["abi"]["params"][2]["dim"] = expression.clone().into(),
        }
        let damaged = serde_json::to_vec(&doc).unwrap();
        no_panic("the expression parser", || {
            if let Ok(manifest) = Manifest::from_embedded(&damaged, &files, Pin::TrustDirectory) {
                accepted += 1;
                // an expression that reads must also plan and prepare without a panic
                if let Ok(plan) = manifest.plan("gelu_erf", &[], Arch::sm(103)) {
                    let shape = [4096usize];
                    // SAFETY: the address is never dereferenced.
                    let ptr = unsafe { tilefish::DevicePtr::from_raw(0x1000, 8192, 0) };
                    let a = tilefish::TensorArg::new(ptr, &shape, tilefish::DType::Bf16).unwrap();
                    let _ = plan.prepare(&[a, a], None);
                }
            }
        });
    }
    println!("{accepted} of 400 random expressions were valid ones");
}
