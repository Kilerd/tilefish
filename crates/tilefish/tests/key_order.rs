//! Objects whose keys a file does not give in code point order are read in the order of the file wherever Python's reader (whose dicts keep
//! it) takes that order, and nothing else depends on it: the crate reads the same with and without serde_json's `preserve_order`, which another
//! crate of a host process may turn on (`cargo test -p tilefish` and `cargo test -p tilefish --features serde_json/preserve_order` both run
//! these). The buckets of a manifest entry (a kernel, a GEMM, a convolution) make its stem in their order and are its `buckets` in that order;
//! the `dims` of an entry and of a plan are in the order of the file, and so is the first of them a reader refuses; the first value of a
//! step's `given` that is not an int is the one refused.
//!
//! serde_json writes a map in code point order (in the order of insertion with `preserve_order`), so these tests write such objects as text,
//! in place of a placeholder string.

mod common;

use std::path::Path;

use common::Copy;
use serde_json::{Value, json};
use tilefish::{Error, LaunchPlan, Manifest, Pin, plan_digest};

/// `text` with the placeholder string `"@name"` (there once) replaced by the JSON text `object`.
fn put(text: &str, name: &str, object: &str) -> String {
    let placeholder = format!("\"@{name}\"");
    assert_eq!(text.matches(&placeholder).count(), 1, "the placeholder {placeholder}");
    text.replacen(&placeholder, object, 1)
}

/// The names of `items`, in order.
fn names<'a, T: 'a>(items: impl IntoIterator<Item = &'a T>, name: impl Fn(&T) -> &str) -> Vec<String> {
    items.into_iter().map(|i| name(i).to_owned()).collect()
}

// -- a kernel entry ----------------------------------------------------------------------------------------------------------------------

/// The `sm_103` fixture with the stem of `bucketed_rms.Db96` made `stem`, a second bucketed dimension `A` (its one bucket 7) after `Db` in its
/// `buckets` (`{"Db": 96, "A": 7}`, its bucket precondition last), and its `dims` the text `dims`.
fn kernel_with_two_buckets(stem: &str, dims: &str) -> Copy {
    let copy = Copy::of("sm_103");
    copy.edit_json(|doc| {
        let e = Copy::entry(doc, "bucketed_rms.Db96");
        e["stem"] = stem.into();
        e["buckets"] = "@buckets".into();
        e["dims"] = "@dims".into();
        e["preconditions"]
            .as_array_mut()
            .unwrap()
            .push(json!({"kind": "bucket", "dim": "A", "value": 7, "buckets": [7]}));
    });
    let text = put(&copy.manifest_text(), "buckets", r#"{"Db": 96, "A": 7}"#);
    copy.write_manifest(&put(&text, "dims", dims));
    copy
}

const KERNEL_DIMS: &str =
    r#"{"M": {}, "Db": {"buckets": [96, 512, 1000], "bucket": 96}, "A": {"buckets": [7], "bucket": 7}}"#;

fn manifest_refusal(copy: &Copy) -> String {
    match Manifest::load(&copy.dir, Pin::TrustDirectory) {
        Err(Error::Manifest(message)) => message,
        Err(other) => panic!("expected a manifest error, got {other:?}"),
        Ok(_) => panic!("the manifest was accepted"),
    }
}

#[test]
fn the_buckets_of_a_kernel_entry_make_its_stem_and_are_its_buckets_in_the_order_of_the_file() {
    let copy = kernel_with_two_buckets("bucketed_rms.Db96.A7", KERNEL_DIMS);
    let manifest = Manifest::load(&copy.dir, Pin::TrustDirectory).unwrap();
    let entry = manifest.entry("bucketed_rms.Db96.A7", "sm_103").unwrap();
    assert_eq!(entry.buckets, [("Db".to_owned(), 96), ("A".to_owned(), 7)]);
    assert_eq!(names(&entry.dims, |d| &d.name), ["M", "Db", "A"]);
    // the stem the buckets make in code point order is not theirs: Python's reader refuses it too
    let message = manifest_refusal(&kernel_with_two_buckets("bucketed_rms.A7.Db96", KERNEL_DIMS));
    assert!(
        message.ends_with(
            r#"stem "bucketed_rms.A7.Db96" does not match op "bucketed_rms" and buckets {"Db": Number(96), "A": Number(7)}"#
        ),
        "{message}"
    );
}

#[test]
fn the_first_dimension_of_an_entry_that_is_refused_is_the_first_of_the_file() {
    // two names of dims that are not identifiers: the first of the file is named, as Python's reader names it
    let dims = r#"{"M": {}, "Db": {"buckets": [96, 512, 1000], "bucket": 96}, "A": {"buckets": [7], "bucket": 7}, "9z": {}, "8y": {}}"#;
    let message = manifest_refusal(&kernel_with_two_buckets("bucketed_rms.Db96.A7", dims));
    assert!(message.ends_with(r#"dims name must be an identifier, got "9z""#), "{message}");
    // two buckets that have no entry in dims: the first of the buckets of the file (Python's reader of a library entry: `for name in buckets`)
    let message = manifest_refusal(&kernel_with_two_buckets("bucketed_rms.Db96.A7", r#"{"M": {}}"#));
    assert!(message.ends_with("every bucketed dimension needs its entry in dims: Db has none"), "{message}");
}

// -- library entries -------------------------------------------------------------------------------------------------------------------

/// The first entry Python accepts in `data/<vectors>` that has a bucket.
fn bucketed_vector(vectors: &str) -> Value {
    let path = common::data().join(vectors);
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    cases.into_iter().find(|c| c["accepted"] == true && c["entry"]["buckets"] != json!({})).unwrap()["entry"].clone()
}

/// A manifest of the library entry `entry` with a second bucketed dimension `B` (its one bucket 4) after its own bucket in `buckets` and in
/// `dims` (its bucket precondition last), and the stem `stem` (`{op}` is the entry's operator, `{own}` its own bucket as the stem has it).
fn library_with_two_buckets(entry: &Value, stem: &str) -> Result<Manifest, Error> {
    let (own, value) = entry["buckets"].as_object().unwrap().iter().next().unwrap();
    let mut entry = entry.clone();
    let own_dims = entry["dims"][own].to_string();
    entry["stem"] =
        stem.replace("{op}", entry["op"].as_str().unwrap()).replace("{own}", &format!("{own}{value}")).into();
    entry["buckets"] = "@buckets".into();
    entry["dims"] = "@dims".into();
    entry["preconditions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"kind": "bucket", "dim": "B", "value": 4, "buckets": [4]}));
    let doc = json!({"schema": 3, "tilefish": "0.0.3", "kernels": [], "library": [entry]});
    let text =
        put(&serde_json::to_string_pretty(&doc).unwrap(), "buckets", &format!(r#"{{"{own}": {value}, "B": 4}}"#));
    let text = put(&text, "dims", &format!(r#"{{"{own}": {own_dims}, "B": {{"bucket": 4, "buckets": [4]}}}}"#));
    Manifest::from_embedded(text.as_bytes(), &[], Pin::TrustDirectory)
}

#[test]
fn the_buckets_of_a_gemm_and_of_a_convolution_make_the_stem_and_are_the_buckets_in_the_order_of_the_file() {
    let gemm = bucketed_vector("python_library.json");
    let conv = bucketed_vector("python_conv.json");
    for (entry, what) in [(&gemm, "a GEMM"), (&conv, "a convolution")] {
        let manifest = library_with_two_buckets(entry, "{op}.{own}.B4").unwrap_or_else(|e| panic!("{what}: {e}"));
        let (buckets, dims) = match (manifest.library(), manifest.convolutions()) {
            ([gemm], []) => (gemm.buckets.clone(), names(&gemm.dims, |d| &d.name)),
            ([], [conv]) => (conv.buckets.clone(), names(&conv.dims, |d| &d.name)),
            _ => panic!("{what}: one entry"),
        };
        let (own, value) = entry["buckets"].as_object().unwrap().iter().next().unwrap();
        assert_eq!(buckets, [(own.clone(), value.as_u64().unwrap()), ("B".to_owned(), 4)], "{what}");
        assert_eq!(dims, [own.as_str(), "B"], "{what}");
        // the stem of the code point order (B first) is not theirs
        let message =
            library_with_two_buckets(entry, "{op}.B4.{own}").err().unwrap_or_else(|| panic!("{what}")).to_string();
        assert!(message.contains(".B4.") && message.contains("does not match op"), "{what}: {message}");
        assert!(
            message.ends_with(&format!(r#"buckets {{"{own}": Number({value}), "B": Number(4)}}"#)),
            "{what}: {message}"
        );
    }
}

// -- a launch plan ---------------------------------------------------------------------------------------------------------------------

fn golden(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/tests/golden/plans").join(name)
}

/// The golden plan `name` changed by `change` (its digest computed again), as text with each `(name, text)` of `objects` in place of its
/// placeholder, and read with the golden manifest.
fn plan_with(name: &str, change: impl FnOnce(&mut Value), objects: &[(&str, &str)]) -> Result<LaunchPlan, Error> {
    let mut doc: Value =
        serde_json::from_str(&std::fs::read_to_string(golden(name).join("plan.json")).unwrap()).unwrap();
    change(&mut doc);
    // the digest is of the content, whatever the order of the keys: the objects in place of their placeholders
    let mut digested = doc.clone();
    for (key, text) in objects {
        replace_placeholder(&mut digested, key, &serde_json::from_str(text).unwrap());
    }
    doc["plan_digest"] = plan_digest(&digested).into();
    let mut text = serde_json::to_string_pretty(&doc).unwrap();
    for (key, object) in objects {
        text = put(&text, key, object);
    }
    let manifest = Manifest::load(golden(name), Pin::TrustDirectory).unwrap();
    LaunchPlan::from_embedded(text.as_bytes(), manifest, Pin::TrustDirectory)
}

fn replace_placeholder(v: &mut Value, name: &str, with: &Value) {
    match v {
        Value::String(s) if *s == format!("@{name}") => *v = with.clone(),
        Value::Array(items) => items.iter_mut().for_each(|i| replace_placeholder(i, name, with)),
        Value::Object(map) => map.values_mut().for_each(|i| replace_placeholder(i, name, with)),
        _ => {}
    }
}

#[test]
fn the_dims_of_a_plan_are_in_the_order_of_the_file() {
    // the golden block with a second dimension Z before M: the buckets follow the combinations of the names sorted (M, then Z), as Python's do
    let plan = plan_with(
        "block",
        |doc| {
            doc["dims"] = "@dims".into();
            let buckets: Vec<Value> = doc["buckets"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|b| {
                    [1, 2].map(|z| {
                        let mut b = b.clone();
                        b["values"]["Z"] = z.into();
                        b
                    })
                })
                .collect();
            doc["buckets"] = buckets.into();
        },
        &[("dims", r#"{"Z": [1, 2], "M": [32, 64, 128]}"#)],
    )
    .unwrap();
    assert_eq!(plan.dims(), [("Z".to_owned(), vec![1, 2]), ("M".to_owned(), vec![32, 64, 128])]);
    let combinations: Vec<Vec<(String, u64)>> = plan.buckets().iter().map(|b| b.values.clone()).collect();
    assert_eq!(combinations[1], [("M".to_owned(), 32), ("Z".to_owned(), 2)]);
    assert_eq!(combinations.len(), 6);
}

#[test]
fn the_first_value_of_a_given_that_is_refused_is_the_first_of_the_file() {
    // Python writes the given values of a step in the order of the parameters (`{"k": 3, "c": -2}`): two that are not ints, k is named
    let e = plan_with(
        "given",
        |doc| doc["buckets"][0]["steps"][0]["given"] = "@given".into(),
        &[("given", r#"{"k": "three", "c": "minus two"}"#)],
    )
    .unwrap_err()
    .to_string();
    assert!(e.ends_with(r#"step #0: given maps argument names to ints, got "k": "three""#), "{e}");
    // and given values in that order are read
    let plan = plan_with(
        "given",
        |doc| doc["buckets"][0]["steps"][0]["given"] = "@given".into(),
        &[("given", r#"{"k": 3, "c": -2}"#)],
    )
    .unwrap();
    assert_eq!(plan.buckets()[0].steps[0].given, [("c".to_owned(), -2), ("k".to_owned(), 3)]);
}
