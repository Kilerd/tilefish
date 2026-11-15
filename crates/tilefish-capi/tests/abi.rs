//! The C ABI without a GPU: the layout of the argument structs, the status of each kind of failure and the thread's last error, NULL and
//! invalid arguments, loading and verifying the committed manifests and plans (with their pins, a wrong pin, a bad pin, trusting the directory,
//! from memory), what the handles are made of (`Send + Sync`), and the refusal of a context on a machine without a driver.

use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};

use serde_json::Value;
use tilefish_capi::*;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_dir() -> PathBuf {
    repo().join("crates/tilefish/tests/fixtures/sm_103")
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The digest of the committed sm_103 manifest.
fn fixture_pin() -> String {
    json(&repo().join("crates/tilefish/tests/fixtures/expected.json"))["fixtures"]["sm_103"]["manifest_digest"]
        .as_str()
        .unwrap()
        .to_owned()
}

fn golden(name: &str) -> (PathBuf, Value) {
    let dir = repo().join("python/tests/golden/plans");
    (dir.join(name), json(&dir.join("expected.json"))[name].clone())
}

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn text(p: *const c_char) -> String {
    assert!(!p.is_null());
    unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned()
}

fn last() -> (tilefish_status, String) {
    let mut info = std::mem::MaybeUninit::<tilefish_error_info>::uninit();
    unsafe { tilefish_last_error(info.as_mut_ptr()) };
    let info = unsafe { info.assume_init() };
    (info.status, text(info.message))
}

/// The sizes the header promises (`tests/c/abi.c` checks the same numbers on the C side).
#[test]
fn the_argument_structs_have_the_layout_of_the_header() {
    use std::mem::{offset_of, size_of};
    assert_eq!(size_of::<tilefish_tensor>(), 56);
    assert_eq!(offset_of!(tilefish_tensor, device), 16);
    assert_eq!(offset_of!(tilefish_tensor, rank), 28);
    assert_eq!(offset_of!(tilefish_tensor, shape), 32);
    assert_eq!(offset_of!(tilefish_tensor, offset), 48);
    assert_eq!(size_of::<tilefish_named_tensor>(), 64);
    assert_eq!(size_of::<tilefish_tap>(), 72);
    assert_eq!(size_of::<tilefish_devscalar>(), 32);
    assert_eq!(offset_of!(tilefish_devscalar, int_value), 16);
    assert_eq!(size_of::<tilefish_bucket>(), 16);
    assert_eq!(size_of::<tilefish_given>(), 16);
    assert_eq!(size_of::<tilefish_file>(), 24);
    assert_eq!(size_of::<tilefish_error_info>(), 40);
    assert_eq!(size_of::<tilefish_buffer_info>(), 32);
}

#[test]
fn versions_and_names() {
    assert_eq!(tilefish_abi_version() >> 16, TILEFISH_ABI_MAJOR);
    let crate_toml = std::fs::read_to_string(repo().join("crates/tilefish/Cargo.toml")).unwrap();
    assert!(crate_toml.contains(&format!("version = \"{}\"", text(tilefish_version()))));
    assert_eq!(tilefish_manifest_schema(), 3);
    assert_eq!(text(tilefish_plan_schema()), "tilefish.plan.v1");
    assert_eq!(text(tilefish_status_name(TILEFISH_ERROR_PRECONDITION)), "TILEFISH_ERROR_PRECONDITION");
    assert_eq!(text(tilefish_status_name(TILEFISH_OK)), "TILEFISH_OK");
    assert_eq!(text(tilefish_status_name(1234)), "TILEFISH_ERROR_UNKNOWN");
}

#[test]
fn a_manifest_loads_with_its_pin_and_every_wrong_pin_has_its_status() {
    let dir = c(fixture_dir().to_str().unwrap());
    let pin = fixture_pin();
    let mut m = null_mut();
    assert_eq!(unsafe { tilefish_manifest_load(dir.as_ptr(), c(&pin).as_ptr(), &mut m) }, TILEFISH_OK);
    assert_eq!(text(unsafe { tilefish_manifest_digest(m) }), pin);
    assert!(text(unsafe { tilefish_manifest_label(m) }).contains("sm_103"));
    // a copy for a partition is a manifest of the same digest
    let mut part = null_mut();
    assert_eq!(unsafe { tilefish_manifest_with_sm_partition(m, 16, &mut part) }, TILEFISH_OK);
    assert_eq!(text(unsafe { tilefish_manifest_digest(part) }), pin);
    unsafe { tilefish_manifest_destroy(part) };
    unsafe { tilefish_manifest_destroy(m) };

    let mut m = std::ptr::dangling_mut::<tilefish_manifest>(); // a failure leaves NULL in the out-parameter
    let wrong = "0".repeat(64);
    assert_eq!(
        unsafe { tilefish_manifest_load(dir.as_ptr(), c(&wrong).as_ptr(), &mut m) },
        TILEFISH_ERROR_PIN_MISMATCH
    );
    assert!(m.is_null());
    let (status, message) = last();
    assert_eq!(status, TILEFISH_ERROR_PIN_MISMATCH);
    assert!(message.contains(&pin) && message.contains(&wrong), "{message}");
    assert_eq!(
        unsafe { tilefish_manifest_load(dir.as_ptr(), c("not a digest").as_ptr(), &mut m) },
        TILEFISH_ERROR_BAD_PIN
    );
    let missing = c(repo().join("no such directory").to_str().unwrap());
    assert_eq!(unsafe { tilefish_manifest_load(missing.as_ptr(), c(&pin).as_ptr(), &mut m) }, TILEFISH_ERROR_IO);
    assert_eq!(unsafe { tilefish_manifest_load_trusting_directory(dir.as_ptr(), &mut m) }, TILEFISH_OK);
    unsafe { tilefish_manifest_destroy(m) };
}

#[test]
fn a_tampered_manifest_is_refused_with_the_status_of_the_crate() {
    let copy = std::env::temp_dir().join(format!("tilefish-capi-tampered-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&copy);
    std::fs::create_dir_all(&copy).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), copy.join(entry.file_name())).unwrap();
    }
    let cubin = copy.join("gelu_erf.sm_103.cubin");
    let mut bytes = std::fs::read(&cubin).unwrap();
    let last_byte = bytes.len() - 1;
    bytes[last_byte] ^= 1;
    std::fs::write(&cubin, bytes).unwrap();
    let mut m = null_mut();
    let status =
        unsafe { tilefish_manifest_load(c(copy.to_str().unwrap()).as_ptr(), c(&fixture_pin()).as_ptr(), &mut m) };
    assert_eq!(status, TILEFISH_ERROR_MANIFEST);
    assert!(last().1.contains("gelu_erf"), "{}", last().1);
    std::fs::remove_dir_all(&copy).unwrap();
}

#[test]
fn a_manifest_in_memory_is_verified_like_a_directory() {
    let dir = fixture_dir();
    let manifest = std::fs::read(dir.join("manifest.json")).unwrap();
    let names: Vec<CString> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".cubin"))
        .map(|n| c(&n))
        .collect();
    let datas: Vec<Vec<u8>> = names.iter().map(|n| std::fs::read(dir.join(n.to_str().unwrap())).unwrap()).collect();
    let files: Vec<tilefish_file> = names
        .iter()
        .zip(&datas)
        .map(|(n, d)| tilefish_file { name: n.as_ptr(), bytes: d.as_ptr(), len: d.len() })
        .collect();
    let pin = c(&fixture_pin());
    let mut m = null_mut();
    let status = unsafe {
        tilefish_manifest_from_embedded(
            manifest.as_ptr(),
            manifest.len(),
            files.as_ptr(),
            files.len(),
            pin.as_ptr(),
            &mut m,
        )
    };
    assert_eq!(status, TILEFISH_OK, "{}", last().1);
    assert_eq!(text(unsafe { tilefish_manifest_label(m) }), "<embedded>");
    unsafe { tilefish_manifest_destroy(m) };
    // without its cubins
    let status =
        unsafe { tilefish_manifest_from_embedded(manifest.as_ptr(), manifest.len(), null(), 0, pin.as_ptr(), &mut m) };
    assert_ne!(status, TILEFISH_OK);
    assert!(m.is_null());
}

#[test]
fn null_and_invalid_arguments_are_refused() {
    let mut m = null_mut();
    assert_eq!(unsafe { tilefish_manifest_load(null(), null(), &mut m) }, TILEFISH_ERROR_INVALID_ARGUMENT);
    assert!(last().1.contains("NULL"));
    assert_eq!(
        unsafe { tilefish_manifest_load(c("x").as_ptr(), c("y").as_ptr(), null_mut()) },
        TILEFISH_ERROR_INVALID_ARGUMENT
    );
    let bad = [0xffu8, 0];
    assert_eq!(
        unsafe { tilefish_manifest_load(bad.as_ptr().cast(), c("y").as_ptr(), &mut m) },
        TILEFISH_ERROR_INVALID_ARGUMENT
    );
    assert!(last().1.contains("UTF-8"));
    // a count without an array
    assert_eq!(
        unsafe { tilefish_manifest_from_embedded(b"{}".as_ptr(), 2, null(), 3, c("y").as_ptr(), &mut m) },
        TILEFISH_ERROR_INVALID_ARGUMENT
    );
    // accessors and destroys take NULL
    assert!(unsafe { tilefish_manifest_digest(null()) }.is_null());
    assert_eq!(unsafe { tilefish_plan_buffer_count(null()) }, 0);
    assert_eq!(unsafe { tilefish_kernel_tensor_count(null()) }, 0);
    unsafe { tilefish_manifest_destroy(null_mut()) };
    unsafe { tilefish_captured_plan_destroy(null_mut()) };
    unsafe { tilefish_last_error(null_mut()) };
}

#[test]
fn a_plan_loads_with_its_pin_and_describes_itself() {
    let (dir, expected) = golden("block");
    let pin = expected["plan_digest"].as_str().unwrap().to_owned();
    let mut p = null_mut();
    assert_eq!(unsafe { tilefish_plan_load(c(dir.to_str().unwrap()).as_ptr(), c(&pin).as_ptr(), &mut p) }, TILEFISH_OK);
    assert_eq!(text(unsafe { tilefish_plan_digest(p) }), pin);
    assert_eq!(text(unsafe { tilefish_plan_graph(p) }), "block");
    assert_eq!(text(unsafe { tilefish_plan_arch(p) }), "sm_103");
    assert_eq!(text(unsafe { tilefish_plan_manifest_digest(p) }), expected["manifest_digest"].as_str().unwrap());
    assert_eq!(unsafe { tilefish_plan_arena_bytes(p) }, expected["arena_bytes"].as_u64().unwrap());
    let mut names = Vec::new();
    for i in 0..unsafe { tilefish_plan_buffer_count(p) } {
        let mut info = std::mem::MaybeUninit::uninit();
        assert_eq!(unsafe { tilefish_plan_buffer(p, i, info.as_mut_ptr()) }, TILEFISH_OK);
        let info = unsafe { info.assume_init() };
        names.push((text(info.name), info.role));
    }
    assert!(names.contains(&("n".to_owned(), TILEFISH_ROLE_DEVSCALAR)), "{names:?}");
    assert!(names.contains(&("w_qkv".to_owned(), TILEFISH_ROLE_WEIGHT)), "{names:?}");
    let mut info = std::mem::MaybeUninit::uninit();
    assert_eq!(unsafe { tilefish_plan_buffer(p, 10_000, info.as_mut_ptr()) }, TILEFISH_ERROR_INVALID_ARGUMENT);
    assert_eq!(unsafe { tilefish_plan_output_count(p) }, 1);
    assert_eq!(text(unsafe { tilefish_plan_output(p, 0) }), "out0");
    assert!(unsafe { tilefish_plan_output(p, 1) }.is_null());
    assert_eq!(unsafe { tilefish_plan_dim_count(p) }, 1);
    let (mut name, mut values, mut n) = (null(), null(), 0usize);
    assert_eq!(unsafe { tilefish_plan_dim(p, 0, &mut name, &mut values, &mut n) }, TILEFISH_OK);
    assert_eq!(text(name), "M");
    assert_eq!(unsafe { std::slice::from_raw_parts(values, n) }, [32, 64, 128]);
    let mut m = null_mut();
    assert_eq!(unsafe { tilefish_plan_manifest(p, &mut m) }, TILEFISH_OK);
    assert_eq!(text(unsafe { tilefish_manifest_digest(m) }), expected["manifest_digest"].as_str().unwrap());
    // the plan again, over that manifest
    let mut q = null_mut();
    let status =
        unsafe { tilefish_plan_load_with_manifest(c(dir.to_str().unwrap()).as_ptr(), m, c(&pin).as_ptr(), &mut q) };
    assert_eq!(status, TILEFISH_OK);
    let bytes = std::fs::read(dir.join("plan.json")).unwrap();
    let mut r = null_mut();
    assert_eq!(
        unsafe { tilefish_plan_from_embedded(bytes.as_ptr(), bytes.len(), m, c(&pin).as_ptr(), &mut r) },
        TILEFISH_OK
    );
    for h in [p, q, r] {
        unsafe { tilefish_plan_destroy(h) };
    }
    unsafe { tilefish_manifest_destroy(m) };
    let wrong = c(&"1".repeat(64));
    assert_eq!(
        unsafe { tilefish_plan_load(c(dir.to_str().unwrap()).as_ptr(), wrong.as_ptr(), &mut p) },
        TILEFISH_ERROR_PLAN_PIN_MISMATCH
    );
}

#[test]
fn a_plan_set_loads_with_one_pin_and_gives_its_plans() {
    let (dir, expected) = golden("carried");
    let pin = expected["set_digest"].as_str().unwrap().to_owned();
    let mut set = null_mut();
    assert_eq!(
        unsafe { tilefish_plan_set_load(c(dir.to_str().unwrap()).as_ptr(), c(&pin).as_ptr(), &mut set) },
        TILEFISH_OK
    );
    assert_eq!(text(unsafe { tilefish_plan_set_digest(set) }), pin);
    let names: Vec<String> = (0..unsafe { tilefish_plan_set_name_count(set) })
        .map(|i| text(unsafe { tilefish_plan_set_name(set, i) }))
        .collect();
    assert!(names.contains(&"stage_a".to_owned()) && names.contains(&"stage_b".to_owned()), "{names:?}");
    let wanted: Vec<CString> = names.iter().map(|n| c(n)).collect();
    let ptrs: Vec<*const c_char> = wanted.iter().map(|n| n.as_ptr()).collect();
    let mut plans = vec![null_mut(); ptrs.len()];
    assert_eq!(unsafe { tilefish_plan_set_plans(set, ptrs.as_ptr(), ptrs.len(), plans.as_mut_ptr()) }, TILEFISH_OK);
    for (p, name) in plans.iter().zip(&names) {
        assert_eq!(&text(unsafe { tilefish_plan_graph(*p) }), name);
        unsafe { tilefish_plan_destroy(*p) };
    }
    let mut bytes = 0u64;
    assert_eq!(
        unsafe { tilefish_plan_set_shared_arena_bytes(set, ptrs.as_ptr(), ptrs.len(), &mut bytes) },
        TILEFISH_OK
    );
    assert!(bytes > 0);
    let mut one = null_mut();
    assert_eq!(unsafe { tilefish_plan_set_plan(set, c("no_such_graph").as_ptr(), &mut one) }, TILEFISH_ERROR_PLAN);
    assert!(one.is_null());
    unsafe { tilefish_plan_set_destroy(set) };
}

#[test]
fn without_a_driver_a_context_is_refused_and_with_one_it_is_made() {
    let mut ctx = null_mut();
    let status = unsafe { tilefish_context_primary(0, &mut ctx) };
    if tilefish_driver_available() {
        // a machine with a driver (and maybe no device): a context, or a refusal that is not a panic
        assert!(
            status == TILEFISH_OK || status == TILEFISH_ERROR_DRIVER || status == TILEFISH_ERROR_INVALID_ARGUMENT,
            "{status}"
        );
        unsafe { tilefish_context_destroy(ctx) };
    } else {
        assert_eq!(status, TILEFISH_ERROR_NO_DRIVER);
        assert!(ctx.is_null());
        assert_eq!(unsafe { tilefish_context_borrow(1usize as _, &mut ctx) }, TILEFISH_ERROR_NO_DRIVER);
    }
}

/// Every handle but a captured plan may be shared between threads.
#[test]
fn the_handles_are_send_and_sync() {
    fn both<T: Send + Sync>() {}
    both::<tilefish_context>();
    both::<tilefish_cudnn_handle>();
    both::<tilefish_manifest>();
    both::<tilefish_kernel>();
    both::<tilefish_gemm>();
    both::<tilefish_conv>();
    both::<tilefish_plan>();
    both::<tilefish_plan_set>();
    both::<tilefish_shared_arena>();
    both::<tilefish_set_run>();
    both::<tilefish_plan_run>();
    both::<tilefish_weighted_run>();
}
