//! The name the shared library is loaded by: `libtilefish_capi.so.<ABI major>` on Linux (its SONAME), `@rpath/libtilefish_capi.dylib` on
//! macOS. A program linked against it records that name, so a library of another ABI major is never loaded in its place. The CMake package
//! (`CMakeLists.txt`) puts the library under that name next to the unversioned one.

/// `TILEFISH_ABI_MAJOR` of `src/lib.rs` (a test checks that they agree).
const ABI_MAJOR: u32 = 1;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `tilefish_version()`: the version of the tilefish crate this library is built from (its path dependency)
    let manifest = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../tilefish/Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest.display());
    let text = std::fs::read_to_string(&manifest).expect("the tilefish crate's Cargo.toml");
    let version = text
        .lines()
        .find_map(|l| l.strip_prefix("version = \"").and_then(|v| v.strip_suffix('"')))
        .expect("the version of the tilefish crate");
    println!("cargo:rustc-env=TILEFISH_CRATE_VERSION={version}");
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libtilefish_capi.so.{ABI_MAJOR}"),
        Ok("macos") => println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libtilefish_capi.dylib"),
        _ => {}
    }
}
