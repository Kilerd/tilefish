//! Helpers of the tests that read the fixtures (`tests/fixtures`, built by `python/tests/export_rust_fixtures.py`).
#![allow(dead_code)]

pub mod synthetic;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

/// The vectors Python made (`export_rust_vectors.py`).
pub fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data")
}

/// `expected.json`: what Python computed for each fixture.
pub fn expected() -> Value {
    let text = std::fs::read_to_string(fixtures().join("expected.json")).expect("expected.json");
    serde_json::from_str(&text).expect("expected.json is JSON")
}

pub fn expected_digest(fixture: &str) -> String {
    expected()["fixtures"][fixture]["manifest_digest"].as_str().expect("a digest").to_owned()
}

/// A copy of a fixture directory in a fresh temporary directory, removed when dropped.
pub struct Copy {
    pub dir: PathBuf,
}

impl Copy {
    pub fn of(fixture: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tilefish-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        for entry in std::fs::read_dir(fixtures().join(fixture)).expect("fixture dir") {
            let entry = entry.expect("entry");
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy");
        }
        Self { dir }
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.json")
    }

    pub fn manifest_text(&self) -> String {
        std::fs::read_to_string(self.manifest_path()).expect("manifest.json")
    }

    pub fn write_manifest(&self, text: &str) {
        std::fs::write(self.manifest_path(), text).expect("write manifest.json");
    }

    /// Change the parsed manifest and write it back.
    pub fn edit_json(&self, change: impl FnOnce(&mut Value)) {
        let mut doc: Value = serde_json::from_str(&self.manifest_text()).expect("a manifest");
        change(&mut doc);
        self.write_manifest(&serde_json::to_string_pretty(&doc).expect("json"));
    }

    /// The entry of `stem` in a parsed manifest.
    pub fn entry<'a>(doc: &'a mut Value, stem: &str) -> &'a mut Value {
        doc["kernels"].as_array_mut().expect("kernels").iter_mut().find(|k| k["stem"] == stem).expect("an entry")
    }

    /// Replace the first occurrence of `from` in the manifest text (which must be there).
    pub fn edit(&self, from: &str, to: &str) {
        let text = self.manifest_text();
        assert!(text.contains(from), "the manifest has no {from:?}");
        self.write_manifest(&text.replacen(from, to, 1));
    }
}

impl Drop for Copy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
