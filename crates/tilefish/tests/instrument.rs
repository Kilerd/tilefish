//! Instrumented exports read with no GPU (`docs/instrument.md`, manifest schema 3 section 4.8): the `instrument` of an entry is read only by
//! `Manifest::load_instrumented` and refused by `Manifest::load` and `Manifest::from_embedded`, malformed fields are refused with the rules of
//! Python's `_probe.check_field` (`python/tests/test_instrument.py` has the same cases), only a Triton entry has one, the field is digested like
//! every other, and the plans and plan sets of an instrumented export are refused unless loaded with `load_instrumented`.

mod common;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::synthetic::{Kernel, alignment, json, manifest_text, pointer, read, refusal};
use serde_json::Value;
use tilefish::{Error, Instrument, LaunchPlan, Manifest, Pin, PlanSet, Probe, ProbeSite, plan_digest, plan_set_digest};

/// A temporary directory, removed when dropped.
struct Dir(PathBuf);

impl Dir {
    fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tilefish-instrument-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The directory of a manifest of `kernels` (its manifest.json and cubins).
fn directory(kernels: &[Kernel]) -> Dir {
    let dir = Dir::new();
    std::fs::write(dir.0.join("manifest.json"), manifest_text(kernels, &[])).unwrap();
    for k in kernels {
        std::fs::write(dir.0.join(k.entry["cubin"].as_str().unwrap()), &k.cubin).unwrap();
    }
    dir
}

fn instrumented(dir: &Path) -> Result<Manifest, Error> {
    Manifest::load_instrumented(dir, Pin::TrustDirectory)
}

fn probe_table() -> Value {
    json!([
        {"id": 3, "name": "load.wait", "sites": [{"loc": "kernels/ws.py:33", "role": "worker 0", "function": "kernels.ws._load"}]},
        {"id": 4, "name": "compute.wait", "sites": [
            {"loc": "kernels/ws.py:46", "role": "default", "function": "kernels.ws._compute"},
            {"loc": "kernels/helpers.py:3", "role": "worker 0", "function": "kernels.ws._load"}
        ]}
    ])
}

/// `scale(x: Tensor[n, f32]) -> Tensor[n, f32]`, a Triton kernel, with `instrument` set to `field` (none: `Value::Null`).
fn scale(stem: &str, field: Value) -> Kernel {
    let mut k = Kernel::new(
        "triton",
        stem,
        103,
        vec![pointer(0, "x", "f32", "in", json!(["n"]), None), pointer(1, "out", "f32", "out", json!(["n"]), None)],
        vec![alignment(0, "x", 16), alignment(1, "out", 16)],
        json!({"n": {}}),
        "(cdiv(n, 1024),)",
    );
    if !field.is_null() {
        k.entry["instrument"] = field;
    }
    k
}

fn field(capacity: u64, probes: Value) -> Value {
    json!({"format": "tilefish.probes.v1", "capacity": capacity, "probes": probes})
}

#[test]
fn an_instrumented_export_is_read_only_when_asked_for() {
    let dir = directory(&[scale("scale", field(64, probe_table())), scale("other", Value::Null)]);
    let message = match Manifest::load(&dir.0, Pin::TrustDirectory) {
        Err(Error::Manifest(m)) => m,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(
        message.contains("is an instrumented export") && message.contains("Manifest::load_instrumented"),
        "{message}"
    );
    // an embedded manifest is a production host's: refused too
    assert!(refusal(&[scale("scale", field(64, probe_table()))], &[]).contains("is an instrumented export"));
    let m = instrumented(&dir.0).expect("load_instrumented reads it");
    assert!(m.is_instrumented());
    let e = m.entry("scale", "sm_103").unwrap();
    assert_eq!(
        e.instrument,
        Some(Instrument {
            capacity: 64,
            probes: vec![
                Probe {
                    id: 3,
                    name: "load.wait".into(),
                    sites: vec![ProbeSite {
                        loc: "kernels/ws.py:33".into(),
                        role: "worker 0".into(),
                        function: "kernels.ws._load".into()
                    }],
                },
                Probe {
                    id: 4,
                    name: "compute.wait".into(),
                    sites: vec![
                        ProbeSite {
                            loc: "kernels/ws.py:46".into(),
                            role: "default".into(),
                            function: "kernels.ws._compute".into()
                        },
                        ProbeSite {
                            loc: "kernels/helpers.py:3".into(),
                            role: "worker 0".into(),
                            function: "kernels.ws._load".into()
                        },
                    ],
                },
            ],
        })
    );
    assert_eq!(e.instrument.as_ref().unwrap().probes_json(), probe_table()); // the table written back as the manifest has it
    assert!(m.entry("other", "sm_103").unwrap().instrument.is_none()); // an entry of an operator the export did not instrument
    // a production manifest reads with either, and is not instrumented
    let prod = directory(&[scale("scale", Value::Null)]);
    let both = [Manifest::load(&prod.0, Pin::TrustDirectory).unwrap(), instrumented(&prod.0).unwrap()];
    assert!(both.iter().all(|m| !m.is_instrumented()));
    // the field is digested like every other: the instrumented variant has a digest of its own
    assert_ne!(m.digest(), read(&[scale("scale", Value::Null), scale("other", Value::Null)], &[]).unwrap().digest());
    let edited = directory(&[scale("scale", field(32, probe_table())), scale("other", Value::Null)]);
    assert_ne!(instrumented(&edited.0).unwrap().digest(), m.digest());
}

#[test]
fn a_malformed_instrument_is_refused_with_the_rules_python_has() {
    let site = json!({"loc": "k.py:1", "role": "", "function": "k"});
    let probe = json!({"id": 3, "name": "a", "sites": [site]});
    let cases: Vec<(Value, &str)> = vec![
        (json!("tilefish.probes.v1"), "instrument must be an object"),
        (
            json!({"format": "tilefish.probes.v0", "capacity": 8, "probes": []}),
            "instrument.format is \"tilefish.probes.v0\"",
        ),
        (field(0, json!([])), "instrument.capacity must be an integer from 1 to 65536"),
        (field(65537, json!([])), "instrument.capacity must be an integer from 1 to 65536"),
        (json!({"format": "tilefish.probes.v1", "capacity": true, "probes": []}), "invalid type: boolean"),
        (json!({"format": "tilefish.probes.v1", "capacity": 8.0, "probes": []}), "invalid type: floating point"),
        (json!({"format": "tilefish.probes.v1", "capacity": 8, "probes": [], "extra": 1}), "unknown field `extra`"),
        (json!({"format": "tilefish.probes.v1", "capacity": 8}), "missing field `probes`"),
        (
            field(8, json!([{"id": 2, "name": "a", "sites": []}])),
            "instrument.probes[0].id must be an integer from 3 to 65535",
        ),
        (
            field(8, json!([{"id": 65536, "name": "a", "sites": []}])),
            "instrument.probes[0].id must be an integer from 3 to 65535",
        ),
        (field(8, json!([probe, probe])), "instrument.probes[1].id must be an integer from 3 to 65535, each once"),
        (
            field(8, json!([probe, {"id": 4, "name": "a", "sites": []}])),
            "instrument.probes[1].name must be the name of a probe, each once",
        ),
        (
            field(8, json!([{"id": 3, "name": "not a name", "sites": []}])),
            "instrument.probes[0].name must be the name of a probe",
        ),
        (
            field(8, json!([{"id": 3, "name": "@begin", "sites": []}])),
            "instrument.probes[0].name must be the name of a probe",
        ),
        (
            field(8, json!([{"id": 3, "name": "a", "sites": [{"loc": 1, "role": "", "function": ""}]}])),
            "invalid type: integer",
        ),
        (field(8, json!([{"id": 3, "name": "a", "sites": [{"loc": "", "role": ""}]}])), "missing field `function`"),
        (field(8, json!([{"id": 3, "name": "a", "sites": [], "more": 1}])), "unknown field `more`"),
    ];
    for (bad, why) in cases {
        let dir = directory(&[scale("scale", bad.clone())]);
        let message = match instrumented(&dir.0) {
            Err(Error::Manifest(m)) => m,
            other => panic!("{bad}: expected a refusal, got {other:?}"),
        };
        assert!(message.contains(why), "{bad}: {why}: {message}");
    }
    let good = directory(&[scale("scale", field(65536, json!([{"id": 65535, "name": "x".repeat(64), "sites": []}])))]);
    assert!(instrumented(&good.0).is_ok());
}

#[test]
fn instrument_is_only_on_a_triton_entry() {
    let mut binary = Kernel::new(
        "binary",
        "axpy",
        103,
        vec![pointer(0, "x", "f32", "in", json!(["n"]), None)],
        vec![alignment(0, "x", 16)],
        json!({"n": {}}),
        "(1,)",
    );
    binary.entry["instrument"] = field(8, json!([]));
    let dir = directory(&[binary]);
    let message = match instrumented(&dir.0) {
        Err(Error::Manifest(m)) => m,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(message.contains("instrument is only on the entry of a Triton kernel"), "{message}");
}

// -- plans and plan sets ----------------------------------------------------------------------------------------------------------------------

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/tests/golden/plans")
}

/// A copy of the golden plan directory `name` whose manifest is made an instrumented export (its first Triton entry given `instrument`), with
/// every digest that names it rewritten (the plans' `manifest_digest` and `plan_digest`, a set's `manifest_digest`, plan digests and
/// `set_digest`): a consistent instrumented export, as Python would write one.
fn instrumented_copy(name: &str) -> (Dir, String) {
    let dir = Dir::new();
    for entry in std::fs::read_dir(golden().join(name)).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dir.0.join(entry.file_name())).unwrap();
    }
    let path = dir.0.join("manifest.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let old = Manifest::load(&dir.0, Pin::TrustDirectory).unwrap().digest().to_owned();
    assert!(text.contains("\"kind\": \"triton\""));
    let text = text.replacen(
        "\"kind\": \"triton\"",
        "\"instrument\": {\"format\": \"tilefish.probes.v1\", \"capacity\": 16, \"probes\": []}, \"kind\": \"triton\"",
        1,
    );
    std::fs::write(&path, text).unwrap();
    let new = instrumented(&dir.0).unwrap().digest().to_owned();
    let mut plans: Vec<(String, String, String)> = Vec::new(); // (file, old plan digest, new plan digest)
    for entry in std::fs::read_dir(&dir.0).unwrap() {
        let file = entry.unwrap().file_name().into_string().unwrap();
        if file.ends_with("plan.json") {
            let p = dir.0.join(&file);
            let text = std::fs::read_to_string(&p).unwrap().replace(&old, &new);
            let doc: Value = serde_json::from_str(&text).unwrap();
            let (was, now) = (doc["plan_digest"].as_str().unwrap().to_owned(), plan_digest(&doc));
            std::fs::write(&p, text.replace(&was, &now)).unwrap();
            plans.push((file, was, now));
        }
    }
    let index = dir.0.join("plans.json");
    let mut pin = plans.first().map(|p| p.2.clone()).unwrap_or_default();
    if index.is_file() {
        let mut text = std::fs::read_to_string(&index).unwrap().replace(&old, &new);
        for (_, was, now) in &plans {
            text = text.replace(was, now);
        }
        let doc: Value = serde_json::from_str(&text).unwrap();
        let (was, now) = (doc["set_digest"].as_str().unwrap().to_owned(), plan_set_digest(&doc));
        std::fs::write(&index, text.replace(&was, &now)).unwrap();
        pin = now;
    }
    (dir, pin)
}

#[test]
fn the_plan_of_an_instrumented_export_is_refused_unless_asked_for() {
    let (dir, pin) = instrumented_copy("block");
    let message = match LaunchPlan::load(&dir.0, Pin::Digest(&pin)) {
        Err(Error::Manifest(m)) => m,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(message.contains("is an instrumented export"), "{message}");
    let plan = LaunchPlan::load_instrumented(&dir.0, Pin::Digest(&pin)).expect("load_instrumented reads it");
    assert!(plan.manifest().is_instrumented() && plan.digest() == pin);
    // the golden production plan reads with either
    let e: Value = serde_json::from_str(&std::fs::read_to_string(golden().join("expected.json")).unwrap()).unwrap();
    let golden_pin = e["block"]["plan_digest"].as_str().unwrap();
    assert!(
        !LaunchPlan::load_instrumented(golden().join("block"), Pin::Digest(golden_pin))
            .unwrap()
            .manifest()
            .is_instrumented()
    );
}

#[test]
fn the_plan_set_of_an_instrumented_export_is_refused_unless_asked_for() {
    let (dir, pin) = instrumented_copy("set");
    // loaded with load_instrumented first: the verified manifest the process keeps for the set is not served to a production load
    let set = PlanSet::load_instrumented(&dir.0, Pin::Digest(&pin)).expect("load_instrumented reads it");
    assert!(set.manifest().is_instrumented());
    assert!(set.plan("tiles2").unwrap().manifest().is_instrumented());
    let message = match PlanSet::load(&dir.0, Pin::Digest(&pin)) {
        Err(Error::Manifest(m)) => m,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(
        message.contains("is an instrumented export") && message.contains("PlanSet::load_instrumented"),
        "{message}"
    );
    drop(set);
}
