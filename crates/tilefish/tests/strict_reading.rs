//! A manifest directory is read as strictly as Python reads it: what it refuses, Rust refuses, with no GPU.
//!
//! Each test starts from a copy of the `sm_103` fixture and breaks one thing: tampered JSON, a wrong pin, a path that
//! leaves the directory, a symbolic link, a key twice, `NaN`, an unknown key, a field of the wrong type, a tampered
//! cubin (also one whose digest in the manifest was rewritten to match), an entry that two kernels share, a kernel that
//! uses shared memory the manifest does not account for.

mod common;

use common::{Copy, expected_digest};
use tilefish::elf::{Elf, sha256_hex};
use tilefish::{Arch, Error, Manifest, Pin, PreconditionKind};

fn trust(copy: &Copy) -> Result<Manifest, Error> {
    Manifest::load(&copy.dir, Pin::TrustDirectory)
}

fn refused(copy: &Copy) -> String {
    match trust(copy) {
        Err(Error::Manifest(message)) => message,
        Err(other) => panic!("expected a manifest error, got {other:?}"),
        Ok(_) => panic!("the manifest was accepted"),
    }
}

#[test]
fn an_untouched_copy_is_accepted_under_both_pins() {
    let copy = Copy::of("sm_103");
    let digest = expected_digest("sm_103");
    assert!(Manifest::load(&copy.dir, Pin::Digest(&digest)).is_ok());
    assert!(trust(&copy).is_ok());
    // the manifest file itself works as well as its directory
    assert!(Manifest::load(copy.manifest_path(), Pin::Digest(&digest)).is_ok());
}

#[test]
fn a_wrong_pin_is_refused_and_so_is_one_that_is_not_a_digest() {
    let copy = Copy::of("sm_103");
    let wrong = "0".repeat(64);
    match Manifest::load(&copy.dir, Pin::Digest(&wrong)) {
        Err(Error::PinMismatch { found, pinned, .. }) => {
            assert_eq!(found, expected_digest("sm_103"));
            assert_eq!(pinned, wrong);
        }
        other => panic!("{other:?}"),
    }
    // the digest of the other architecture's manifest is a wrong pin too
    assert!(matches!(
        Manifest::load(&copy.dir, Pin::Digest(&expected_digest("sm_120"))),
        Err(Error::PinMismatch { .. })
    ));
    for bad in ["", "abc", &"A".repeat(64), &"g".repeat(64), &"0".repeat(65)] {
        assert!(matches!(Manifest::load(&copy.dir, Pin::Digest(bad)), Err(Error::BadPin(_))), "{bad:?}");
    }
}

#[test]
fn tampered_json_changes_the_digest_so_a_pin_refuses_it() {
    let copy = Copy::of("sm_103");
    copy.edit("\"shared\": 16", "\"shared\": 17");
    let result = Manifest::load(&copy.dir, Pin::Digest(&expected_digest("sm_103")));
    assert!(matches!(result, Err(Error::PinMismatch { .. })), "{result:?}");
    // trusting the directory takes whatever it holds: that is what the pin is for
    let trusted = trust(&copy).expect("a consistent edit is accepted without a pin");
    assert_ne!(trusted.digest(), expected_digest("sm_103"));
    // and the pin of the edited manifest accepts it
    assert!(Manifest::load(&copy.dir, Pin::Digest(trusted.digest())).is_ok());
}

#[test]
fn a_pin_is_checked_before_a_cubin_is_read() {
    let copy = Copy::of("sm_103");
    std::fs::remove_file(copy.dir.join("softmax.sm_103.cubin")).unwrap();
    let result = Manifest::load(&copy.dir, Pin::Digest(&"1".repeat(64)));
    assert!(matches!(result, Err(Error::PinMismatch { .. })), "{result:?}");
}

#[test]
fn a_cubin_name_that_leaves_the_directory_is_refused() {
    for name in [
        "../sm_120/softmax.sm_120.cubin",
        "/etc/passwd",
        "sub/softmax.sm_103.cubin",
        "..",
        ".",
        ".softmax.cubin",
        "a\\\\b.cubin",
        "",
    ] {
        let copy = Copy::of("sm_103");
        copy.edit("\"cubin\": \"softmax.sm_103.cubin\"", &format!("\"cubin\": \"{name}\""));
        let message = refused(&copy);
        assert!(message.contains("plain file name"), "{name:?}: {message}");
    }
}

#[cfg(unix)]
#[test]
fn a_symbolic_link_is_refused_for_the_cubin_and_for_the_manifest() {
    let copy = Copy::of("sm_103");
    let cubin = copy.dir.join("softmax.sm_103.cubin");
    let real = copy.dir.join("elsewhere.bin");
    std::fs::rename(&cubin, &real).unwrap();
    std::os::unix::fs::symlink(&real, &cubin).unwrap();
    assert!(refused(&copy).contains("symbolic link"));

    let copy = Copy::of("sm_103");
    let real = copy.dir.join("real-manifest.json");
    std::fs::rename(copy.manifest_path(), &real).unwrap();
    std::os::unix::fs::symlink(&real, copy.manifest_path()).unwrap();
    assert!(refused(&copy).contains("symbolic link"));
}

#[test]
fn a_missing_cubin_is_refused() {
    let copy = Copy::of("sm_103");
    std::fs::remove_file(copy.dir.join("swiglu.sm_103.cubin")).unwrap();
    let message = refused(&copy);
    assert!(message.contains("swiglu") && message.contains("cannot read the cubin"), "{message}");
}

#[test]
fn what_is_not_strict_json_is_refused() {
    let cases: [(&str, &str, &str); 8] = [
        ("a key twice", "\"stem\": \"softmax\",", "\"stem\": \"softmax\", \"stem\": \"softmax\","),
        ("NaN", "\"shared\": 16", "\"shared\": NaN"),
        ("Infinity", "\"shared\": 16", "\"shared\": Infinity"),
        ("a number that does not fit a float", "\"num_stages\": 3", "\"num_stages\": 1e999"),
        ("a trailing comma", "\"schema\": 3,", "\"schema\": 3,,"),
        ("a leading zero", "\"num_stages\": 3", "\"num_stages\": 03"),
        ("a lone surrogate", "\"op\": \"softmax\"", "\"op\": \"soft\\ud800max\""),
        ("a single quote", "\"schema\": 3", "'schema': 3"),
    ];
    for (what, from, to) in cases {
        let copy = Copy::of("sm_103");
        copy.edit(from, to);
        let message = refused(&copy);
        assert!(message.contains("not strict JSON"), "{what}: {message}");
    }
    // every kind of repeated key, not only in an entry
    let copy = Copy::of("sm_103");
    copy.edit("\"schema\": 3,", "\"schema\": 3, \"schema\": 3,");
    assert!(refused(&copy).contains("twice"));
    let copy = Copy::of("sm_103");
    copy.edit("\"slot\": 0,", "\"slot\": 0, \"slot\": 0,");
    assert!(refused(&copy).contains("twice"));
    // not JSON at all, a byte order mark, text that is not UTF-8
    let copy = Copy::of("sm_103");
    copy.write_manifest("not json");
    assert!(refused(&copy).contains("not strict JSON"));
    let copy = Copy::of("sm_103");
    copy.write_manifest(&format!("\u{feff}{}", copy.manifest_text()));
    assert!(refused(&copy).contains("byte order mark"));
    let copy = Copy::of("sm_103");
    std::fs::write(copy.manifest_path(), [0xff, 0xfe, b'{', b'}']).unwrap();
    assert!(refused(&copy).contains("UTF-8"));
}

#[test]
fn an_unknown_key_a_missing_key_and_a_wrong_type_are_refused() {
    let cases: [(&str, &str, &str, &str); 16] = [
        (
            "an unknown key in an entry",
            "\"stem\": \"softmax\",",
            "\"stem\": \"softmax\", \"extra\": 1,",
            "unknown field",
        ),
        ("an unknown top-level key", "\"schema\": 3,", "\"schema\": 3, \"surprise\": true,", "unknown field"),
        ("an unknown key in the ABI", "\"slot\": 0,", "\"slot\": 0, \"align\": 8,", "unknown field"),
        ("a missing key", "\"symbol\": \"tf_softmax\",", "", "missing field"),
        ("shared as a string", "\"shared\": 16", "\"shared\": \"16\"", "invalid type"),
        ("num_warps as a float", "\"num_warps\": 4,", "\"num_warps\": 4.0,", "invalid type"),
        ("a bool where an int is", "\"num_stages\": 3", "\"num_stages\": true", "invalid type"),
        ("a negative size", "\"shared\": 16", "\"shared\": -16", "invalid value"),
        ("the schema as another number", "\"schema\": 3,", "\"schema\": 4,", "schema"),
        // 3.0 is 3 to Python's `==`: both hosts refuse it, only the integer 3 is a schema
        ("the schema as 3.0", "\"schema\": 3,", "\"schema\": 3.0,", "schema"),
        ("the schema as 3e0", "\"schema\": 3,", "\"schema\": 3e0,", "schema"),
        ("the schema as a string", "\"schema\": 3,", "\"schema\": \"3\",", "schema"),
        ("the schema as true", "\"schema\": 3,", "\"schema\": true,", "schema"),
        ("the schema as a list", "\"schema\": 3,", "\"schema\": [3],", "schema"),
        (
            "an unknown compile option",
            "\"num_stages\": 3\n      },\n      \"declaration_digest\"",
            "\"num_stages\": 3, \"maxnreg\": 64\n      },\n      \"declaration_digest\"",
            "unknown field",
        ),
        ("a digest that is not hex", "\"sha256\": \"", "\"sha256\": \"G", "sha256"),
    ];
    for (what, from, to, why) in cases {
        let copy = Copy::of("sm_103");
        copy.edit(from, to);
        let message = refused(&copy);
        assert!(message.contains(why), "{what}: {message}");
    }
    let copy = Copy::of("sm_103");
    copy.write_manifest("[1, 2]");
    assert!(refused(&copy).contains("not a JSON object") || refused(&copy).contains("not a tilefish manifest"));
}

/// Replace an object by the list of its values, in its order: what serde's derived reader of a struct also takes (its fields by position).
fn by_position(v: &mut serde_json::Value) {
    let values: Vec<serde_json::Value> = v.as_object().expect("an object").values().cloned().collect();
    *v = serde_json::Value::Array(values);
}

#[test]
fn an_object_given_as_the_list_of_its_values_is_refused() {
    // Python reads every one of these with `isinstance(x, dict)`; serde's derive would take the values by position
    type Pick = fn(&mut serde_json::Value) -> &mut serde_json::Value;
    let stride_of: Pick = |d| {
        let params = Copy::entry(d, "strided_copy")["abi"]["params"].as_array_mut().unwrap();
        params.iter_mut().find_map(|p| p.get_mut("stride_of")).expect("a scalar that carries a stride")
    };
    let cases: [(&str, Pick, &str); 14] = [
        ("a kernel entry", |d| Copy::entry(d, "softmax"), "a kernel entry must be an object"),
        ("abi", |d| &mut Copy::entry(d, "softmax")["abi"], "abi must be an object"),
        (
            "a parameter",
            |d| &mut Copy::entry(d, "softmax")["abi"]["params"][0],
            "a parameter of abi.params must be an object",
        ),
        ("stride_of", stride_of, "stride_of must be an object"),
        (
            "a scratch slot",
            |d| &mut Copy::entry(d, "softmax")["abi"]["scratch"][1],
            "a slot of abi.scratch must be an object",
        ),
        ("an empty dims entry", |d| &mut Copy::entry(d, "softmax")["dims"]["M"], "an entry of dims must be an object"),
        ("a dims entry", |d| &mut Copy::entry(d, "gelu_exact")["dims"]["T"], "an entry of dims must be an object"),
        (
            "a constexpr",
            |d| &mut Copy::entry(d, "softmax")["constexprs"][0],
            "an entry of constexprs must be an object",
        ),
        ("the compile options", |d| &mut Copy::entry(d, "softmax")["options"], "options must be an object"),
        ("the toolchain", |d| &mut Copy::entry(d, "softmax")["toolchain"], "toolchain must be an object"),
        ("an empty toolchain", |d| &mut Copy::entry(d, "axpy_cubin")["toolchain"], "toolchain must be an object"),
        ("the launch attributes", |d| &mut Copy::entry(d, "softmax")["launch"], "launch must be an object"),
        ("a library entry", |d| &mut d["library"][0], "a library entry must be an object"),
        ("the abi of a library entry", |d| &mut d["library"][0]["abi"], "abi must be an object"),
    ];
    for (what, pick, why) in cases {
        let copy = Copy::of("sm_103");
        copy.edit_json(|doc| by_position(pick(doc)));
        let message = refused(&copy);
        assert!(message.contains(why) && message.contains("got list"), "{what}: {message}");
    }
    let copy = Copy::of("sm_103");
    copy.edit_json(|doc| by_position(&mut doc["library"][0]["abi"]["params"][0]));
    let message = refused(&copy);
    assert!(
        message.contains("library entry #0") && message.contains("a parameter of abi.params must be an object"),
        "{message}"
    );
    // the whole document
    let copy = Copy::of("sm_103");
    copy.edit_json(by_position);
    assert!(refused(&copy).contains("it is not a JSON object"));
    // and the launch attributes read by themselves
    let e = serde_json::from_str::<tilefish::LaunchAttributes>("[0, 0, false, [1, 1, 1], 1, false]").unwrap_err();
    assert!(e.to_string().contains("launch must be an object (its fields by name), got list"), "{e}");
}

#[test]
fn a_tampered_cubin_is_refused_by_its_file_digest() {
    let copy = Copy::of("sm_103");
    let path = copy.dir.join("gelu_exact.sm_103.cubin");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();
    let message = refused(&copy);
    assert!(
        message.contains("gelu_exact") && message.contains("sha256") && message.contains("not the one that was built"),
        "{message}"
    );
}

#[test]
fn a_tampered_cubin_whose_file_digest_was_rewritten_is_refused_by_its_code_digest() {
    let copy = Copy::of("sm_103");
    let path = copy.dir.join("gelu_exact.sm_103.cubin");
    let mut bytes = std::fs::read(&path).unwrap();
    let old_file = sha256_hex(&bytes);
    // flip one bit of the machine code itself
    let section = {
        let elf = Elf::parse(&bytes).unwrap();
        let s = elf.section(".text.tf_gelu_exact").expect("the kernel's code section").clone();
        assert!(s.size > 16);
        s
    };
    bytes[section.offset as usize + 8] ^= 0x10;
    std::fs::write(&path, &bytes).unwrap();
    copy.edit(&old_file, &sha256_hex(&bytes)); // the attacker who can write the directory rewrites this one too
    let message = refused(&copy);
    assert!(message.contains("sha256_code") && message.contains("changed after it was built"), "{message}");
    // and a pin refuses the rewritten manifest before any of that
    assert!(matches!(
        Manifest::load(&copy.dir, Pin::Digest(&expected_digest("sm_103"))),
        Err(Error::PinMismatch { .. })
    ));
}

#[test]
fn a_cubin_that_is_not_an_elf_file_is_refused() {
    let copy = Copy::of("sm_103");
    let junk = vec![7u8; 4096];
    std::fs::write(copy.dir.join("gelu_exact.sm_103.cubin"), &junk).unwrap();
    let old = common::expected()["fixtures"]["sm_103"]["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["stem"] == "gelu_exact")
        .unwrap()["sha256"]
        .as_str()
        .unwrap()
        .to_owned();
    copy.edit(&old, &sha256_hex(&junk));
    let message = refused(&copy);
    assert!(message.contains("is not a valid cubin") && message.contains("bad magic"), "{message}");
}

#[test]
fn an_entry_that_two_kernels_share_is_refused() {
    let copy = Copy::of("sm_103");
    let mut doc: serde_json::Value = serde_json::from_str(&copy.manifest_text()).unwrap();
    let first = doc["kernels"][0].clone();
    doc["kernels"].as_array_mut().unwrap().push(first);
    copy.write_manifest(&serde_json::to_string_pretty(&doc).unwrap());
    assert!(refused(&copy).contains("twice"));
}

#[test]
fn shared_memory_the_manifest_does_not_account_for_is_refused() {
    let copy = Copy::of("sm_103");
    // softmax reduces through shared memory: a launch with `shared = 0` would fault and poison the context
    let mut doc: serde_json::Value = serde_json::from_str(&copy.manifest_text()).unwrap();
    let softmax = doc["kernels"].as_array_mut().unwrap().iter_mut().find(|k| k["stem"] == "softmax").unwrap();
    softmax["shared"] = serde_json::Value::from(0);
    copy.write_manifest(&serde_json::to_string_pretty(&doc).unwrap());
    let message = refused(&copy);
    assert!(message.contains("shared = 0") && message.contains("poison"), "{message}");
}

#[test]
fn entries_that_contradict_themselves_are_refused() {
    type Change = fn(&mut serde_json::Value);
    let cases: [(&str, Change, &str); 8] = [
        ("a block that is not 32 threads a warp", |e| e["block"][0] = 64.into(), "block must be"),
        ("an op that does not match the stem", |e| e["op"] = "softmaxx".into(), "does not match op"),
        ("an alignment other than 16", |e| e["preconditions"][0]["bytes"] = 8.into(), "pointer_alignment must be 16"),
        (
            "a pointer type that is not its dtype",
            |e| e["abi"]["params"][0]["type"] = "*bf16".into(),
            "does not match dtype",
        ),
        (
            "a max_elements that is not 2^31-1",
            |e| e["preconditions"][2]["value"] = 2147483646u64.into(),
            "max_elements",
        ),
        ("a slot out of place", |e| e["abi"]["params"][1]["slot"] = 5.into(), "slot"),
        (
            "scratch slots that are not the two null pointers",
            |e| e["abi"]["scratch"][0]["value"] = 1.into(),
            "scratch slots",
        ),
        ("a grid with no trailing comma", |e| e["grid"] = "(M)".into(), "trailing comma"),
    ];
    for (what, change, why) in cases {
        let copy = Copy::of("sm_103");
        copy.edit_json(|doc| change(Copy::entry(doc, "softmax")));
        let message = refused(&copy);
        assert!(message.contains(why), "{what}: {message}");
    }
}

#[test]
fn a_manifest_that_drops_a_pointers_alignment_check_is_refused() {
    let copy = Copy::of("sm_103");
    let mut doc: serde_json::Value = serde_json::from_str(&copy.manifest_text()).unwrap();
    let gelu = doc["kernels"].as_array_mut().unwrap().iter_mut().find(|k| k["stem"] == "gelu_exact").unwrap();
    let pre = gelu["preconditions"].as_array_mut().unwrap();
    let at = pre.iter().position(|p| p["kind"] == "pointer_alignment").unwrap();
    pre.remove(at);
    copy.write_manifest(&serde_json::to_string_pretty(&doc).unwrap());
    let message = refused(&copy);
    assert!(message.contains("no pointer_alignment precondition"), "{message}");
}

#[test]
fn a_binary_that_needs_launch_attributes_a_plain_launch_does_not_set_up_is_refused_when_it_is_planned() {
    let copy = Copy::of("sm_103");
    let mut doc: serde_json::Value = serde_json::from_str(&copy.manifest_text()).unwrap();
    doc["kernels"][0]["launch"]["launch_pdl"] = serde_json::Value::Bool(true);
    let stem = doc["kernels"][0]["stem"].as_str().unwrap().to_owned();
    copy.write_manifest(&serde_json::to_string_pretty(&doc).unwrap());
    let manifest = trust(&copy).expect("it reads: it is the launch that cannot be made");
    match manifest.plan(&stem, &[], Arch::sm(103)) {
        Err(Error::Unsupported(message)) => assert!(message.contains("launch_pdl=True"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_kernel_or_an_architecture_the_manifest_does_not_have_is_named() {
    let manifest = Manifest::load(common::fixtures().join("sm_120"), Pin::TrustDirectory).unwrap();
    match manifest.plan("softmax", &[], Arch::sm(103)) {
        Err(Error::NoBinary(message)) => {
            assert!(message.contains("no binary for 'softmax' on sm_103") && message.contains("sm_120"), "{message}")
        }
        other => panic!("{other:?}"),
    }
    assert!(manifest.plan("softmax", &[], Arch::sm(120)).is_ok());
    match manifest.plan("no_such_kernel", &[], Arch::sm(120)) {
        Err(Error::NoBinary(message)) => assert!(message.contains("no kernel 'no_such_kernel'"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn buckets_are_exact() {
    let manifest = Manifest::load(common::fixtures().join("sm_103"), Pin::TrustDirectory).unwrap();
    let arch = Arch::sm(103);
    // an operator and its bucket values, or the stem that says them
    assert_eq!(manifest.plan("bucketed_rms", &[("Db", 96)], arch).unwrap().stem(), "bucketed_rms.Db96");
    assert_eq!(manifest.plan("bucketed_rms.Db512", &[], arch).unwrap().stem(), "bucketed_rms.Db512");
    assert_eq!(manifest.plan("bucketed_rms.Db512", &[("Db", 512)], arch).unwrap().stem(), "bucketed_rms.Db512");
    for wrong in [&[("Db", 100)][..], &[][..], &[("Db", 96), ("X", 1)][..], &[("Dc", 96)][..]] {
        match manifest.plan("bucketed_rms", wrong, arch) {
            Err(e) => assert_eq!(e.precondition_kind(), Some(PreconditionKind::Bucket), "{wrong:?}: {e}"),
            Ok(_) => panic!("{wrong:?} was accepted"),
        }
    }
    let message = manifest.plan("bucketed_rms", &[("Db", 100)], arch).unwrap_err().to_string();
    assert!(message.contains("no padding") && message.contains("[96, 512, 1000]"), "{message}");
    // a stem that is not bucketed does not take a bucket
    assert!(manifest.plan("softmax", &[("M", 4)], arch).is_err());
}

#[test]
fn a_manifest_of_an_older_schema_is_refused_with_a_request_to_export_it_again() {
    for older in [1, 2] {
        let copy = Copy::of("sm_103");
        copy.edit("\"schema\": 3,", &format!("\"schema\": {older},"));
        let message = refused(&copy);
        assert!(
            message.contains(&format!("schema {older}"))
                && message.contains("older tilefish")
                && message.contains("Export it again"),
            "{message}"
        );
    }
    // a real schema 2 manifest has no `library` and no `kind`: it is told to export again, not that a field is missing
    let copy = Copy::of("sm_103");
    copy.edit_json(|doc| {
        doc["schema"] = 2.into();
        doc.as_object_mut().unwrap().remove("library");
        for k in doc["kernels"].as_array_mut().unwrap() {
            k.as_object_mut().unwrap().remove("kind");
            k["abi"].as_object_mut().unwrap().remove("scratch_slots");
        }
    });
    let message = refused(&copy);
    assert!(message.contains("schema 2") && message.contains("Export it again"), "{message}");
    // an integer that is not an older schema (0, negative, 2.0, true) is not a manifest of schema 3, and is not told to export again
    for not_older in ["0", "-1", "2.0", "true", "\"2\"", "null"] {
        let copy = Copy::of("sm_103");
        copy.edit("\"schema\": 3,", &format!("\"schema\": {not_older},"));
        let message = refused(&copy);
        assert!(
            !message.contains("older tilefish") && message.contains("not a tilefish manifest of schema 3"),
            "{not_older}: {message}"
        );
    }
    // and an entry without its shapes or dims is not read either
    for what in ["dims", "shape"] {
        let copy = Copy::of("sm_103");
        copy.edit_json(|doc| {
            let entry = Copy::entry(doc, "softmax");
            match what {
                "dims" => {
                    entry.as_object_mut().unwrap().remove("dims");
                }
                _ => {
                    entry["abi"]["params"][0].as_object_mut().unwrap().remove("shape");
                }
            }
        });
        let message = refused(&copy);
        assert!(message.contains("missing field `dims`") || message.contains("has no shape"), "{what}: {message}");
    }
}

#[test]
fn shapes_and_dimensions_that_contradict_themselves_are_refused() {
    type Change = fn(&mut serde_json::Value);
    let cases: [(&str, Change, &str); 13] = [
        (
            "a shape entry that is neither a size nor an expression",
            |e| e["abi"]["params"][0]["shape"][1] = 1.5.into(),
            "shape[1]",
        ),
        ("a size of 0", |e| e["abi"]["params"][0]["shape"][1] = 0.into(), "shape[1]"),
        (
            "an expression that is not in the grammar",
            |e| e["abi"]["params"][0]["shape"][0] = "M ** 2".into(),
            "shape[0]",
        ),
        (
            "a dimension no entry of dims names",
            |e| e["abi"]["params"][0]["shape"][0] = "Rows".into(),
            "dims does not name",
        ),
        (
            "a dims entry removed",
            |e| {
                e["dims"].as_object_mut().unwrap().remove("M");
            },
            "dims does not name",
        ),
        (
            "a static size that is not the static_size precondition",
            |e| e["abi"]["params"][0]["shape"][1] = 999.into(),
            "static_size",
        ),
        ("a dims entry with an unknown key", |e| e["dims"]["M"]["extra"] = 1.into(), "unknown field"),
        ("a divisor of 0", |e| e["dims"]["M"]["divisible_by"] = 0.into(), "divisible_by"),
        ("a bucket that the entry does not have", |e| e["dims"]["M"]["bucket"] = 4.into(), "bucket"),
        ("buckets out of order", |e| e["dims"]["M"]["buckets"] = serde_json::json!([4, 2]), "sorted"),
        ("a dims name that is not an identifier", |e| e["dims"]["a b"] = serde_json::json!({}), "identifier"),
        ("a dims value that is not an object", |e| e["dims"]["M"] = 3.into(), "dims.M"),
        ("a shape that is not a list", |e| e["abi"]["params"][0]["shape"] = "M".into(), "expected a sequence"),
    ];
    for (what, change, why) in cases {
        let copy = Copy::of("sm_103");
        copy.edit_json(|doc| change(Copy::entry(doc, "softmax")));
        let message = refused(&copy);
        assert!(message.contains(why), "{what}: {message}");
    }
}

#[test]
fn the_bucket_of_a_bucketed_entry_is_in_dims_and_agrees_with_the_precondition() {
    let ok = Manifest::load(common::fixtures().join("sm_103"), Pin::TrustDirectory).unwrap();
    let entry = ok.entry("bucketed_rms.Db96", "sm_103").unwrap();
    let db = entry.dims.iter().find(|d| d.name == "Db").unwrap();
    assert_eq!((db.bucket, db.buckets.clone()), (Some(96), Some(vec![96, 512, 1000])));
    assert!(entry.dims.iter().any(|d| d.name == "M" && d.bucket.is_none() && d.divisible_by.is_none()));
    // the entry's bucket, its dims and its precondition say one thing; any two that differ are refused
    for (what, change) in [
        (
            "dims says another bucket",
            (|e: &mut serde_json::Value| e["dims"]["Db"]["bucket"] = 512.into()) as fn(&mut serde_json::Value),
        ),
        ("dims has no bucket", |e| {
            e["dims"]["Db"].as_object_mut().unwrap().remove("bucket");
        }),
        ("the bucket is not among the buckets", |e| e["dims"]["Db"]["buckets"] = serde_json::json!([512, 1000])),
    ] {
        let copy = Copy::of("sm_103");
        copy.edit_json(|doc| change(Copy::entry(doc, "bucketed_rms.Db96")));
        let message = refused(&copy);
        assert!(message.contains("bucket"), "{what}: {message}");
    }
}

/// Make the runtime scalar `name` of an entry carry `dim` (and its divisible precondition, if it has one, say so).
fn carry(entry: &mut serde_json::Value, name: &str, dim: &str) {
    for p in entry["abi"]["params"].as_array_mut().unwrap().iter_mut().filter(|p| p["name"] == name) {
        p["dim"] = dim.into();
    }
    for p in entry["preconditions"].as_array_mut().unwrap().iter_mut() {
        if p["kind"] == "divisible" && p["arg"] == name {
            p["dim"] = dim.into();
        }
    }
}

/// A runtime scalar `n` that counts the elements of `x`, added after the last parameter of `bucketed_rms.Db96` (`x` is `[M, Db]`, `Db = 96`).
fn count_x(entry: &mut serde_json::Value, dim: &str) {
    let next_index = entry["constexprs"].as_array().unwrap().len() + entry["abi"]["params"].as_array().unwrap().len();
    let params = entry["abi"]["params"].as_array_mut().unwrap();
    let slot = params.len();
    params.push(serde_json::json!({"slot": slot, "index": next_index, "name": "n", "kind": "scalar", "type": "i32", "ptx_type": "u32",
                                   "size": 4, "dim": dim, "count_of": "x"}));
    for (k, s) in entry["abi"]["scratch"].as_array_mut().unwrap().iter_mut().enumerate() {
        s["slot"] = (slot + 1 + k).into();
    }
}

#[test]
fn a_scalar_that_counts_the_elements_of_a_pointer_carries_the_product_of_its_shape() {
    // Python's `_check_count_of`: when the product of the shape (with the bucket values in) is affine, the scalar carries that expression, as
    // `DimExpr` compares two; when it is not (two symbolic sizes), the scalar carries a fresh dimension name
    type Change = fn(&mut serde_json::Value);
    let cases: [(&str, &str, Change, Option<&str>); 11] = [
        // swiglu's `n` counts `gate` [M, 3072]
        ("3072*M spelled M*3072", "swiglu", |e| carry(e, "n", "M*3072"), None),
        ("half the product", "swiglu", |e| carry(e, "n", "1536*M"), Some("it must carry that product")),
        ("the product and more", "swiglu", |e| carry(e, "n", "3072*M + 16"), Some("it must carry that product")),
        ("one axis of it", "swiglu", |e| carry(e, "n", "M"), Some("it must carry that product")),
        // gelu_exact's `n` counts `x` [T]: a fresh name is not the count of a shape whose product is affine
        ("twice the count", "gelu_exact", |e| carry(e, "n", "2*T"), Some("it must carry that product")),
        ("a fresh name", "gelu_exact", |e| carry(e, "n", "Count"), Some("it must carry that product")),
        // bucketed_rms at Db = 96: x [M, Db] has 96*M elements, which is what the compile spec's shape [M, 96] says
        ("the product with the bucket in", "bucketed_rms.Db96", |e| count_x(e, "96*M"), None),
        (
            "the product with the bucket's name",
            "bucketed_rms.Db96",
            |e| count_x(e, "M*Db"),
            Some("it must carry that product"),
        ),
        (
            "a fresh name for an affine product",
            "bucketed_rms.Db96",
            |e| count_x(e, "n"),
            Some("it must carry that product"),
        ),
        // mm's `m` made the count of `c_ptr` [MmM, MmN], whose product is not affine
        (
            "a fresh name for a product that is not affine",
            "mm",
            |e| {
                carry(e, "m", "Mc");
                e["abi"]["params"][3]["count_of"] = "c_ptr".into();
                e["dims"]["Mc"] = serde_json::json!({});
            },
            None,
        ),
        (
            "the product that is not affine",
            "mm",
            |e| {
                carry(e, "m", "MmM*MmN");
                e["abi"]["params"][3]["count_of"] = "c_ptr".into();
            },
            Some("must carry a fresh dimension name"),
        ),
    ];
    for (what, stem, change, refusal) in cases {
        let copy = Copy::of("sm_103");
        copy.edit_json(|doc| change(Copy::entry(doc, stem)));
        match refusal {
            None => assert!(trust(&copy).is_ok(), "{what}: {:?}", trust(&copy).err()),
            Some(why) => {
                let message = refused(&copy);
                assert!(message.contains(why) && message.contains("count_of"), "{what}: {message}");
            }
        }
    }
}

#[test]
fn a_max_elements_precondition_is_on_the_shape_of_its_argument_with_the_bucket_values_in() {
    // Python's `p["shape"] == a["shape"]`: the precondition says the declared shape of its argument, axis by axis, with each bucket's value
    type Change = fn(&mut serde_json::Value);
    let cases: [(&str, &str, Change, bool); 8] = [
        ("another static size", "softmax", |e| shape_of_limit(e, "x", serde_json::json!(["M", "999"])), false),
        ("an axis fewer", "softmax", |e| shape_of_limit(e, "x", serde_json::json!(["M"])), false),
        ("an axis more", "softmax", |e| shape_of_limit(e, "x", serde_json::json!(["M", "1000", "1"])), false),
        ("the axes swapped", "mm", |e| shape_of_limit(e, "a_ptr", serde_json::json!(["MmK", "MmM"])), false),
        (
            "the name of the bucket",
            "bucketed_rms.Db96",
            |e| shape_of_limit(e, "x", serde_json::json!(["M", "Db"])),
            false,
        ),
        (
            "another bucket's value",
            "bucketed_rms.Db96",
            |e| shape_of_limit(e, "x", serde_json::json!(["M", "512"])),
            false,
        ),
        ("the bucket's value", "bucketed_rms.Db96", |e| shape_of_limit(e, "out", serde_json::json!(["M", "96"])), true),
        ("the same expression", "swiglu", |e| shape_of_limit(e, "gate", serde_json::json!(["M", "3072"])), true),
    ];
    for (what, stem, change, accepted) in cases {
        let copy = Copy::of("sm_103");
        copy.edit_json(|doc| change(Copy::entry(doc, stem)));
        if accepted {
            assert!(trust(&copy).is_ok(), "{what}: {:?}", trust(&copy).err());
        } else {
            let message = refused(&copy);
            assert!(message.contains("its argument's own shape"), "{what}: {message}");
        }
    }
}

/// Set the shape of the max_elements precondition of `arg`.
fn shape_of_limit(entry: &mut serde_json::Value, arg: &str, shape: serde_json::Value) {
    let pres = entry["preconditions"].as_array_mut().unwrap();
    pres.iter_mut().find(|p| p["kind"] == "max_elements" && p["arg"] == arg).expect("a max_elements")["shape"] = shape;
}

#[test]
fn a_scalar_has_a_dim_and_no_shape() {
    let copy = Copy::of("sm_103");
    copy.edit_json(|doc| Copy::entry(doc, "swiglu")["abi"]["params"][3]["shape"] = serde_json::json!([1]));
    assert!(refused(&copy).contains("a scalar"));
}

#[test]
fn the_launch_attributes_are_read_strictly_and_each_one_a_plain_launch_cannot_set_up_is_named() {
    let read = |text: &str| serde_json::from_str::<tilefish::LaunchAttributes>(text);
    let plain = r#"{"global_scratch_size": 0, "profile_scratch_size": 0, "launch_pdl": false,
                    "cluster_dims": [1, 1, 1], "num_ctas": 1, "cooperative": false}"#;
    let attrs = read(plain).unwrap();
    assert!(attrs.blockers().is_empty());
    // a field that is not one of the six is refused, and so is a missing one, wherever it is
    let with = |extra: &str| plain.replacen('{', &format!("{{{extra},"), 1);
    let e = read(&with(r#""launch_cluster": true"#)).unwrap_err().to_string();
    assert!(e.contains("unknown field") && e.contains("launch_cluster"), "{e}");
    assert!(read(&plain.replace("\"num_ctas\": 1,", "")).unwrap_err().to_string().contains("missing field"));
    // the same inside a manifest entry
    let copy = Copy::of("sm_103");
    copy.edit("\"cooperative\": false", "\"cooperative\": false, \"preferred_cluster\": [2, 1, 1]");
    assert!(refused(&copy).contains("unknown field"));
    // each attribute is a blocker by itself, and says so
    for (from, to, want) in [
        ("\"global_scratch_size\": 0", "\"global_scratch_size\": 128", "global_scratch_size=128"),
        ("\"profile_scratch_size\": 0", "\"profile_scratch_size\": 8", "profile_scratch_size=8"),
        ("\"launch_pdl\": false", "\"launch_pdl\": true", "launch_pdl=True"),
        ("\"cluster_dims\": [1, 1, 1]", "\"cluster_dims\": [2, 1, 1]", "cluster_dims=[2, 1, 1], num_ctas=1"),
        ("\"cluster_dims\": [1, 1, 1]", "\"cluster_dims\": [1, 1, 2]", "cluster_dims=[1, 1, 2], num_ctas=1"),
        ("\"num_ctas\": 1", "\"num_ctas\": 2", "cluster_dims=[1, 1, 1], num_ctas=2"),
        ("\"cooperative\": false", "\"cooperative\": true", "launch_cooperative_grid=True"),
    ] {
        let attrs = read(&plain.replace(from, to)).unwrap();
        assert_eq!(attrs.blockers(), [want], "{to}");
    }
    // several at once are all listed, in the order of the fields
    let many = plain
        .replace("\"launch_pdl\": false", "\"launch_pdl\": true")
        .replace("\"cooperative\": false", "\"cooperative\": true")
        .replace("\"global_scratch_size\": 0", "\"global_scratch_size\": 1");
    assert_eq!(
        read(&many).unwrap().blockers(),
        ["global_scratch_size=1", "launch_pdl=True", "launch_cooperative_grid=True"]
    );
}

#[test]
fn a_loaded_manifest_can_be_checked_against_a_digest_afterwards() {
    // what a generated launcher does with the digest it was generated from: the manifest it is given must be that one
    let copy = Copy::of("sm_103");
    let manifest = trust(&copy).unwrap();
    let digest = expected_digest("sm_103");
    assert_eq!(manifest.digest(), digest);
    assert!(manifest.require_digest(&digest).is_ok());
    // another manifest's digest, and one that only differs in its last character
    match manifest.require_digest(&expected_digest("sm_120")) {
        Err(Error::PinMismatch { found, pinned, directory }) => {
            assert_eq!((found.as_str(), pinned.as_str()), (digest.as_str(), expected_digest("sm_120").as_str()));
            assert_eq!(directory, manifest.label());
        }
        other => panic!("{other:?}"),
    }
    let last = if digest.ends_with('0') { '1' } else { '0' };
    let near = format!("{}{last}", &digest[..63]);
    assert!(matches!(manifest.require_digest(&near), Err(Error::PinMismatch { .. })));
    // a pin that is not a digest is another error, and is refused even when it is what the manifest would say
    for bad in ["", "abc", &digest.to_uppercase(), &format!("{digest}0"), &digest[..63]] {
        assert!(matches!(manifest.require_digest(bad), Err(Error::BadPin(_))), "{bad:?}");
    }
}
