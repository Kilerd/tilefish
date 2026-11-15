//! `include/tilefish.h` is what cbindgen generates from this crate with `cbindgen.toml`: the header a C or C++ program compiles against declares
//! exactly the functions, structs and constants the library exports. Run with `TILEFISH_BLESS=1` to write the header after a change of the ABI.

use std::path::Path;

#[test]
fn the_committed_header_is_what_cbindgen_generates() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("cbindgen.toml");
    let mut generated = Vec::new();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen generates the header")
        .write(&mut generated);
    let generated = String::from_utf8(generated).expect("the header is UTF-8");
    let path = crate_dir.join("include").join("tilefish.h");
    if std::env::var_os("TILEFISH_BLESS").is_some() {
        std::fs::write(&path, &generated).expect("write include/tilefish.h");
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != generated {
        let line = committed
            .lines()
            .zip(generated.lines())
            .position(|(a, b)| a != b)
            .map_or_else(|| committed.lines().count().min(generated.lines().count()) + 1, |i| i + 1);
        panic!(
            "include/tilefish.h is not what cbindgen generates from the crate (first difference at line {line}): run \
             `TILEFISH_BLESS=1 cargo test -p tilefish-capi --test header` and commit the header"
        );
    }
}

/// The ABI major version of the header is the one the build script gives the shared library's name.
#[test]
fn the_soname_has_the_abi_major_version() {
    let build = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs")).unwrap();
    assert!(
        build.contains(&format!("const ABI_MAJOR: u32 = {};", tilefish_capi::TILEFISH_ABI_MAJOR)),
        "build.rs and TILEFISH_ABI_MAJOR disagree"
    );
    assert_eq!(
        tilefish_capi::tilefish_abi_version(),
        (tilefish_capi::TILEFISH_ABI_MAJOR << 16) | tilefish_capi::TILEFISH_ABI_MINOR
    );
}
