//! Launch plans read with no GPU: the golden plan directories of `python/tests/golden/plans` (written by `export_plan_fixtures.py`: the
//! transformer block of `plan_model.py`, without and with reuse, with a real manifest of fake cubins) load with their pins and give every
//! number `expected.json` has; the digests are reproduced from the files; and every refusal of section 3 of `docs/plan-schema-1.md` (a wrong pin,
//! an edit, an edit that rewrites the digest, a plan of another manifest, a manifest or a cubin changed under the plan) is a refusal here.
//!
//! `python_plans.json` (`tests/data`, written by `export_rust_vectors.py`) holds documents made from the golden plans by every mutation of
//! Python's tests of the format, and many random ones, each with what Python's `read_plan` and `check_against_manifest` decided: Rust
//! decides the same for every one.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;
use tilefish::{BufferRole, Error, LaunchPlan, Manifest, Pin, PlanSet, StepKind, plan_digest};

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/tests/golden/plans")
}

fn expected() -> Value {
    serde_json::from_str(&std::fs::read_to_string(golden().join("expected.json")).unwrap()).unwrap()
}

/// A copy of a golden directory in a fresh temporary directory, removed when dropped.
struct Copy {
    dir: PathBuf,
}

impl Copy {
    fn of(name: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tilefish-plan-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(golden().join(name)).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
        }
        Self { dir }
    }

    fn doc(&self, file: &str) -> Value {
        serde_json::from_str(&std::fs::read_to_string(self.dir.join(file)).unwrap()).unwrap()
    }

    fn write(&self, file: &str, doc: &Value) {
        std::fs::write(self.dir.join(file), serde_json::to_string_pretty(doc).unwrap()).unwrap();
    }
}

impl Drop for Copy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn the_golden_plans_load_with_their_pins_and_give_every_number_python_has() {
    let expected = expected();
    for name in ["block", "block-reuse"] {
        let e = &expected[name];
        let pin = e["plan_digest"].as_str().unwrap();
        let plan = LaunchPlan::load(golden().join(name), Pin::Digest(pin)).unwrap();
        assert_eq!(plan.digest(), pin);
        assert_eq!(plan.manifest_digest(), e["manifest_digest"].as_str().unwrap());
        assert_eq!(plan.manifest().digest(), plan.manifest_digest());
        assert_eq!(plan.arch(), e["arch"].as_str().unwrap());
        assert_eq!(plan.graph(), e["graph"].as_str().unwrap());
        assert_eq!(plan.reuse(), e["reuse"].as_bool().unwrap());
        assert_eq!(plan.arena_bytes(), e["arena_bytes"].as_u64().unwrap());
        let offsets: serde_json::Map<String, Value> =
            plan.buffers().iter().filter_map(|b| b.offset.map(|o| (b.name.clone(), Value::from(o)))).collect();
        assert_eq!(Value::Object(offsets), e["offsets"]);
        let buckets: Vec<Value> = plan
            .buckets()
            .iter()
            .map(|b| Value::Object(b.values.iter().map(|(n, v)| (n.clone(), Value::from(*v))).collect()))
            .collect();
        assert_eq!(Value::Array(buckets), e["buckets"]);
        for bucket in plan.buckets() {
            assert_eq!(bucket.steps.len() as u64, e["steps_per_bucket"].as_u64().unwrap());
        }
        // what the steps are: 5 kernels and 4 GEMMs, q, k and v views of the fused output, the workspace in the arena
        let at64 = plan.bucket(&[("M", 64)]).unwrap();
        assert_eq!(at64.steps.iter().filter(|s| s.kind == StepKind::Library).count(), 4);
        assert_eq!(at64.steps[1].entry, "gemm_qkv.M64");
        assert_eq!(at64.steps[1].workspace.as_deref(), Some("workspace"));
        let attention = &at64.steps[2];
        assert_eq!(attention.entry, "attention");
        let views: Vec<usize> = attention.args[..3].iter().map(|a| a.view.as_ref().unwrap().offset).collect();
        assert_eq!(views, [0, 256, 512]);
        assert_eq!(attention.args[0].view.as_ref().unwrap().strides, [768, 64, 1]);
        assert_eq!(attention.args[4].binding, "n");
        assert_eq!(plan.buffer("n").unwrap().role, BufferRole::Devscalar);
        assert_eq!(plan.outputs(), ["out0"]);
        // the block declares no roles: the weights are labelled by their shapes, which a host does not bind once
        assert_eq!(plan.roles_declared(), e["roles_declared"].as_bool().unwrap());
        assert!(!plan.roles_declared());
        let weights: Vec<Value> = plan.weights().map(Value::from).collect();
        assert_eq!(Value::Array(weights), e["weights"]);
        // trusting the directory loads the same plan
        let trusted = LaunchPlan::load(golden().join(name).join("plan.json"), Pin::TrustDirectory).unwrap();
        assert_eq!(trusted.digest(), pin);
    }
}

/// The golden plan set `set` (`gpu_kernels_p6.tile_chain` for 1, 2 and 3 tiles: 2, 3 and 4 steps): one manifest with one entry per binary, the
/// index listing every plan's digest. `PlanSet::load` with the set's pin, every plan through it (sharing the manifest) and with its own pin; a
/// wrong pin, an edited index, an index of another manifest, a plan the index does not list and a plan file that is not the listed plan are refused.
#[test]
fn the_golden_plan_set_shares_one_manifest_and_one_pin_covers_every_plan() {
    let e = &expected()["set"];
    let pin = e["set_digest"].as_str().unwrap();
    let set = PlanSet::load(golden().join("set"), Pin::Digest(pin)).unwrap();
    assert_eq!(set.digest(), pin);
    assert_eq!(set.arch(), "sm_103");
    assert_eq!(set.manifest().digest(), e["manifest_digest"].as_str().unwrap());
    let stems: Vec<&str> = set.manifest().entries().iter().map(|x| x.stem.as_str()).collect();
    assert_eq!(stems, ["halve", "scale_add"], "one entry per binary, whatever the number of plans");
    assert_eq!(set.names().collect::<Vec<_>>(), ["tiles1", "tiles2", "tiles3"]);
    for name in ["tiles1", "tiles2", "tiles3"] {
        let plan = set.plan(name).unwrap();
        assert_eq!(plan.digest(), e["plans"][name].as_str().unwrap());
        assert_eq!(plan.buckets()[0].steps.len() as u64, e["steps"][name].as_u64().unwrap());
        assert_eq!(plan.manifest().digest(), set.manifest().digest());
        // the plan file alone, with its own pin, is the same plan
        let alone =
            LaunchPlan::load(golden().join("set").join(format!("{name}.plan.json")), Pin::Digest(plan.digest()))
                .unwrap();
        assert_eq!(alone.digest(), plan.digest());
        assert_eq!(tilefish::plan_file_name(name), format!("{name}.plan.json"));
    }
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(golden().join("set").join("plans.json")).unwrap()).unwrap();
    assert_eq!(tilefish::plan_set_digest(&doc), pin);
    // refusals
    let e1 = PlanSet::load(golden().join("set"), Pin::Digest(&"0".repeat(64))).unwrap_err();
    assert!(matches!(e1, Error::PlanPinMismatch { .. }), "{e1}");
    assert!(set.plan("tiles4").unwrap_err().to_string().contains("lists no plan of graph 'tiles4'"));
    let copy = Copy::of("set");
    let mut edited = copy.doc("plans.json");
    edited["arch"] = Value::from("sm_90");
    copy.write("plans.json", &edited);
    assert!(PlanSet::load(&copy.dir, Pin::TrustDirectory).unwrap_err().to_string().contains("was edited"));
    let mut other = copy.doc("plans.json");
    other["arch"] = Value::from("sm_103");
    other["manifest_digest"] = Value::from("0".repeat(64));
    other["set_digest"] = Value::from(tilefish::plan_set_digest(&other));
    copy.write("plans.json", &other);
    let e2 = PlanSet::load(&copy.dir, Pin::TrustDirectory).unwrap_err();
    assert!(e2.to_string().contains("built against the manifest"), "{e2}");
    let mut swapped = copy.doc("plans.json");
    swapped["manifest_digest"] = Value::from(e["manifest_digest"].as_str().unwrap());
    swapped["plans"]["tiles1"] = swapped["plans"]["tiles2"].clone();
    swapped["set_digest"] = Value::from(tilefish::plan_set_digest(&swapped));
    copy.write("plans.json", &swapped);
    let swapped_set = PlanSet::load(&copy.dir, Pin::TrustDirectory).unwrap();
    let e3 = swapped_set.plan("tiles1").unwrap_err();
    assert!(matches!(e3, Error::PlanPinMismatch { .. }), "{e3}");
    for (key, value) in [
        ("extra", Value::from(1)),
        ("plans", serde_json::json!({})),
        ("plans", serde_json::json!({"1x": "0".repeat(64)})),
    ] {
        let mut bad = copy.doc("plans.json");
        bad[key] = value;
        bad["set_digest"] = Value::from(tilefish::plan_set_digest(&bad));
        copy.write("plans.json", &bad);
        assert!(PlanSet::load(&copy.dir, Pin::TrustDirectory).is_err(), "{key}");
    }
}

/// The arena the plans of the golden set share (`docs/plan-schema-1.md`, section 15.1): the largest `arena_bytes` of the plans, read from their
/// pinned files, as Python's `share_arena` has it (`expected.json`), for every subset; the documents are unchanged, so the set's pin still covers
/// them; no plan, a plan given twice and a plan the set does not list are refused.
#[test]
fn the_plans_of_the_golden_set_share_an_arena_of_the_largest_of_theirs() {
    let e = &expected()["set"];
    let set = PlanSet::load(golden().join("set"), Pin::Digest(e["set_digest"].as_str().unwrap())).unwrap();
    let all = ["tiles1", "tiles2", "tiles3"];
    for name in all {
        assert_eq!(set.plan(name).unwrap().arena_bytes(), e["arena_bytes"][name].as_u64().unwrap());
    }
    assert_eq!(set.shared_arena_bytes(&all).unwrap(), e["shared_arena_bytes"].as_u64().unwrap());
    assert_eq!(set.shared_arena_bytes(&all).unwrap(), 2304);
    assert_eq!(set.shared_arena_bytes(&["tiles2", "tiles1"]).unwrap(), 1024);
    assert_eq!(set.shared_arena_bytes(&["tiles1"]).unwrap(), 256);
    let plans: Vec<LaunchPlan> = all.iter().map(|n| set.plan(n).unwrap()).collect();
    let refs: Vec<&LaunchPlan> = plans.iter().collect();
    assert_eq!(tilefish::SharedArena::bytes_for(&refs).unwrap(), 2304);
    let e1 = set.shared_arena_bytes(&[]).unwrap_err();
    assert!(e1.to_string().contains("name the plans of the set"), "{e1}");
    let e2 = set.shared_arena_bytes(&["tiles1", "tiles1"]).unwrap_err();
    assert!(e2.to_string().contains("given twice"), "{e2}");
    let e3 = set.shared_arena_bytes(&["tiles1", "tiles4"]).unwrap_err();
    assert!(e3.to_string().contains("lists no plan of graph 'tiles4'"), "{e3}");
    let e4 = tilefish::SharedArena::bytes_for(&[]).unwrap_err();
    assert!(e4.to_string().contains("one plan or more"), "{e4}");
    let e5 = tilefish::SharedArena::bytes_for(&[refs[0], refs[1], refs[0]]).unwrap_err();
    assert!(e5.to_string().contains("'tiles1' is given twice"), "{e5}");
}

/// The golden plan `given` (`plan_model.two_scales`): one prebuilt binary (one manifest entry) launched by two steps, each with the values its call
/// gives to the fixed arguments per call `k` (`i32`) and `c` (`i64`), the second `c` the bucket's `Rg`; the entry has them as `given` parameters.
#[test]
fn the_golden_plan_of_given_values_has_one_entry_and_the_values_of_each_step() {
    let e = &expected()["given"];
    let plan = LaunchPlan::load(golden().join("given"), Pin::Digest(e["plan_digest"].as_str().unwrap())).unwrap();
    let entry = plan.manifest().entry("scale_add", "sm_103").unwrap();
    let kinds: Vec<bool> = entry.params.iter().map(|p| matches!(p.kind, tilefish::ParamKind::Given { .. })).collect();
    assert_eq!(kinds, [false, false, false, true, true]);
    for rg in [8u64, 24] {
        let at = plan.bucket(&[("Rg", rg)]).unwrap();
        assert_eq!((at.steps[0].entry.as_str(), at.steps[1].entry.as_str()), ("scale_add", "scale_add"));
        assert_eq!(at.steps[0].given, [("c".to_owned(), -2), ("k".to_owned(), 3)]);
        assert_eq!(at.steps[1].given, [("c".to_owned(), rg as i64), ("k".to_owned(), -1)]);
    }
    let plan_of = tilefish::Plan::new(entry).unwrap();
    assert_eq!(plan_of.given_names().collect::<Vec<_>>(), ["k", "c"]);
}

/// The golden plan `steps` (`plan_model.steps`): a copy of a dense view into the arena, a step whose `r_out` is written over its `r` (one buffer for
/// both), a copy into the caller's `InOut` tensor, a result returned as all of its buffer in another shape, and outputs named by the graph.
#[test]
fn the_golden_plan_of_steps_has_copies_a_result_over_its_input_a_returned_reshape_and_named_outputs() {
    let e = &expected()["steps"];
    let pin = e["plan_digest"].as_str().unwrap();
    let plan = LaunchPlan::load(golden().join("steps"), Pin::Digest(pin)).unwrap();
    assert_eq!(plan.manifest_digest(), e["manifest_digest"].as_str().unwrap());
    assert_eq!(plan.arena_bytes(), e["arena_bytes"].as_u64().unwrap());
    let at = plan.bucket(&[("Rf", 24)]).unwrap();
    let kinds: Vec<StepKind> = at.steps.iter().map(|s| s.kind).collect();
    assert_eq!(kinds, [StepKind::Copy, StepKind::Kernel, StepKind::Copy, StepKind::Kernel]);
    assert_eq!(at.steps[0].entry, "", "a copy step launches no entry");
    let copy = &at.steps[0].args;
    assert_eq!((copy[0].name.as_str(), copy[0].binding.as_str(), copy[1].binding.as_str()), ("src", "x", "t0"));
    assert_eq!(copy[0].view.as_ref().unwrap().offset, 24 * 64, "x[1]: a dense view at an offset");
    let over = &at.steps[1].args;
    assert_eq!((over[1].binding.as_str(), over[3].binding.as_str()), ("t0", "t0"), "r and r_out: one buffer");
    assert_eq!((at.steps[2].args[0].binding.as_str(), at.steps[2].args[1].binding.as_str()), ("t0", "state"));
    let half = plan.buffer("half").unwrap();
    assert_eq!(half.shape.iter().map(ToString::to_string).collect::<Vec<_>>(), ["Rf", "64"]);
    assert_eq!(at.steps[3].args[1].view.as_ref().unwrap().shape, [24 * 64], "written flat into the [Rf, 64] output");
    // the outputs by the graph's names, and by out0, out1 (the names of before)
    assert_eq!(plan.outputs(), ["y", "half"]);
    // the graph declares its roles (`inputs=("x",)`): x an input, no weight
    assert!(plan.roles_declared() && e["roles_declared"].as_bool().unwrap());
    assert_eq!(plan.buffer("x").unwrap().role, BufferRole::Input);
    assert_eq!(plan.weights().count(), 0);
    for (name, found) in [("y", Some("y")), ("half", Some("half")), ("out0", Some("y")), ("out1", Some("half"))] {
        assert_eq!(plan.output_buffer(name), found, "{name}");
    }
    for name in ["out2", "out01", "out", "outx", "t0", "x", "state"] {
        assert_eq!(plan.output_buffer(name), None, "{name}");
    }
    // a plan whose output is called out0 has nothing else under that name
    let block = LaunchPlan::load(golden().join("block"), Pin::TrustDirectory).unwrap();
    assert_eq!(block.output_buffer("out0"), Some("out0"));
    assert_eq!(block.output_buffer("out1"), None);
}

#[test]
fn the_digests_are_recomputed_from_the_files() {
    let expected = expected();
    for name in ["block", "block-reuse"] {
        let copy = Copy::of(name);
        let doc = copy.doc("plan.json");
        assert_eq!(plan_digest(&doc), expected[name]["plan_digest"].as_str().unwrap());
        assert_eq!(doc["plan_digest"], expected[name]["plan_digest"]);
        let manifest = copy.doc("manifest.json");
        assert_eq!(tilefish::manifest_digest(&manifest).unwrap(), expected[name]["manifest_digest"].as_str().unwrap());
        // the version of the writer is not digested
        let mut other = doc.clone();
        other["tilefish"] = Value::from("9.9.9");
        assert_eq!(plan_digest(&other), plan_digest(&doc));
    }
}

fn refused(path: &Path, pin: Pin<'_>, fragment: &str) {
    match LaunchPlan::load(path, pin) {
        Ok(_) => panic!("{path:?} loaded, but it must be refused ({fragment})"),
        Err(e) => assert!(e.to_string().contains(fragment), "{fragment:?} is not in: {e}"),
    }
}

#[test]
fn a_wrong_pin_an_edit_and_a_plan_of_another_manifest_are_refused() {
    let e = expected();
    let pin = e["block"]["plan_digest"].as_str().unwrap().to_owned();
    let manifest_digest = e["block"]["manifest_digest"].as_str().unwrap().to_owned();
    let dir = golden().join("block");
    // a wrong pin, and the digest of the manifest (which is not the plan's)
    assert!(matches!(LaunchPlan::load(&dir, Pin::Digest(&"1".repeat(64))), Err(Error::PlanPinMismatch { .. })));
    refused(&dir, Pin::Digest(&manifest_digest), "but the pin is");
    for bad in ["", "abc", &"A".repeat(64), &"g".repeat(64)] {
        assert!(matches!(LaunchPlan::load(&dir, Pin::Digest(bad)), Err(Error::BadPin(_))), "{bad:?}");
    }
    // a byte of plan.json changed
    let copy = Copy::of("block");
    let text = std::fs::read_to_string(copy.dir.join("plan.json")).unwrap();
    std::fs::write(copy.dir.join("plan.json"), text.replacen("\"offset\": 256", "\"offset\": 512", 1)).unwrap();
    refused(&copy.dir, Pin::Digest(&pin), "but the pin is");
    refused(&copy.dir, Pin::TrustDirectory, "was edited: its plan_digest says");
    // an edit that rewrites the digest too: refused by the pin, and by the manifest when only the directory is trusted
    let copy = Copy::of("block");
    let mut doc = copy.doc("plan.json");
    doc["manifest_digest"] = Value::from("0".repeat(64));
    doc["plan_digest"] = Value::from(plan_digest(&doc));
    copy.write("plan.json", &doc);
    refused(&copy.dir, Pin::Digest(&pin), "but the pin is");
    refused(&copy.dir, Pin::Digest(doc["plan_digest"].as_str().unwrap()), "built against the manifest with digest");
    refused(&copy.dir, Pin::TrustDirectory, "built against the manifest with digest");
    // the manifest changed under the plan
    let copy = Copy::of("block");
    let mut manifest = copy.doc("manifest.json");
    manifest["kernels"][0]["shared"] = Value::from(manifest["kernels"][0]["shared"].as_u64().unwrap() + 16);
    copy.write("manifest.json", &manifest);
    refused(&copy.dir, Pin::Digest(&pin), "built against the manifest with digest");
    // a cubin changed under both digests
    let copy = Copy::of("block");
    let cubin = copy.dir.join("rmsnorm.sm_103.cubin");
    let mut bytes = std::fs::read(&cubin).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&cubin, bytes).unwrap();
    assert!(LaunchPlan::load(&copy.dir, Pin::Digest(&pin)).is_err());
    // a step that is not what the manifest's entry says, under a rewritten digest
    let copy = Copy::of("block");
    let mut doc = copy.doc("plan.json");
    for bucket in doc["buckets"].as_array_mut().unwrap() {
        bucket["steps"][0]["args"][0]["name"] = Value::from("y");
    }
    doc["plan_digest"] = Value::from(plan_digest(&doc));
    copy.write("plan.json", &doc);
    refused(&copy.dir, Pin::TrustDirectory, "is bound where the binary has 'x'");
}

#[test]
fn a_file_that_is_not_a_plan_or_is_read_through_a_link_is_refused() {
    let copy = Copy::of("block");
    let path = copy.dir.join("plan.json");
    for (text, fragment) in [
        ("[]", "is not a tilefish plan"),
        ("{\"schema\": \"tilefish.plan.v2\"}", "is not a tilefish plan"),
        ("{\"schema\": \"tilefish.plan.v1\", \"schema\": \"tilefish.plan.v1\"}", "twice"),
        ("{\"schema\": \"tilefish.plan.v1\", \"x\": NaN}", "NaN"),
        ("{\"schema\": \"tilefish.plan.v1\", \"x\": 9007199254740994}", "does not fit a double"),
        ("not json", "not valid JSON"),
    ] {
        std::fs::write(&path, text).unwrap();
        refused(&copy.dir, Pin::TrustDirectory, fragment);
    }
    std::fs::write(&path, [0xffu8, 0xfe, 0x00]).unwrap();
    refused(&copy.dir, Pin::TrustDirectory, "not UTF-8");
    #[cfg(unix)]
    {
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(golden().join("block").join("plan.json"), &path).unwrap();
        refused(&copy.dir, Pin::TrustDirectory, "symbolic link");
    }
}

#[test]
fn an_embedded_plan_is_checked_against_the_manifest_it_is_given() {
    let e = expected();
    let pin = e["block"]["plan_digest"].as_str().unwrap();
    let text = std::fs::read(golden().join("block").join("plan.json")).unwrap();
    let manifest = Manifest::load(golden().join("block"), Pin::TrustDirectory).unwrap();
    let plan = LaunchPlan::from_embedded(&text, manifest, Pin::Digest(pin)).unwrap();
    assert_eq!(plan.arena_bytes(), e["block"]["arena_bytes"].as_u64().unwrap());
    // another manifest (the reuse plan's is the same manifest: take the strict-reading fixture, which is another one)
    let other = Manifest::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("sm_103"),
        Pin::TrustDirectory,
    )
    .unwrap();
    let e = LaunchPlan::from_embedded(&text, other, Pin::Digest(pin)).unwrap_err();
    assert!(e.to_string().contains("built against the manifest with digest"), "{e}");
}

/// The node of `doc` at `path` (keys and indices).
fn node<'v>(doc: &'v mut Value, path: &[Value]) -> &'v mut Value {
    path.iter().fold(doc, |node, key| match key {
        Value::String(k) => &mut node[k.as_str()],
        Value::Number(i) => &mut node[i.as_u64().unwrap() as usize],
        other => panic!("a path has keys and indices, not {other}"),
    })
}

/// The `plan.json` text of a vector: its operations applied to the base document (and the digest computed again), or its edit of the base
/// file's text.
fn text_of(v: &Value) -> String {
    let base = std::fs::read_to_string(golden().join(v["base"].as_str().unwrap()).join("plan.json")).unwrap();
    if let Some(edit) = v.get("text") {
        let body = match edit["replace"].as_array() {
            Some(pair) => base.replacen(pair[0].as_str().unwrap(), pair[1].as_str().unwrap(), 1),
            None => base,
        };
        return format!("{}{body}{}", edit["prefix"].as_str().unwrap(), edit["suffix"].as_str().unwrap());
    }
    let mut doc: Value = serde_json::from_str(&base).unwrap();
    for op in v["ops"].as_array().unwrap() {
        let path = op[1].as_array().unwrap();
        match op[0].as_str().unwrap() {
            "set" => *node(&mut doc, path) = op[2].clone(),
            "delete" => {
                let (last, parent) = path.split_last().unwrap();
                node(&mut doc, parent).as_object_mut().unwrap().shift_remove(last.as_str().unwrap());
            }
            other => panic!("unknown operation {other}"),
        }
    }
    if v["rehash"].as_bool().unwrap() {
        doc["plan_digest"] = Value::from(plan_digest(&doc));
    }
    serde_json::to_string_pretty(&doc).unwrap()
}

/// The verdicts of Python on the documents of `python_plans.json`.
#[test]
fn every_document_python_accepts_or_refuses_is_accepted_or_refused_here() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("python_plans.json");
    let vectors: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (mut accepted, mut refused) = (0, 0);
    let mut disagreements = Vec::new();
    for v in vectors["documents"].as_array().unwrap() {
        let manifest = Manifest::load(golden().join(v["base"].as_str().unwrap()), Pin::TrustDirectory).unwrap();
        let text = text_of(v);
        let got = LaunchPlan::from_embedded(text.as_bytes(), manifest, Pin::TrustDirectory);
        let python_accepts = v["accepted"].as_bool().unwrap();
        match (&got, python_accepts) {
            (Ok(_), true) => accepted += 1,
            (Err(_), false) => refused += 1,
            (Ok(_), false) => {
                disagreements.push(format!("{}: Rust accepts what Python refuses ({})", v["name"], v["why"]))
            }
            (Err(e), true) => disagreements.push(format!("{}: Rust refuses what Python accepts: {e}", v["name"])),
        }
    }
    assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    assert!(accepted >= 5 && refused >= 100, "accepted {accepted}, refused {refused}");
    println!("{accepted} documents accepted and {refused} refused, as Python decides");
}
