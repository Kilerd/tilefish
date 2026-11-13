//! `repr(float)` of Python, from Rust: reads the bits of one `f64` per line on stdin (hex), writes its spelling per line.
//!
//! The canonical JSON of a manifest digest spells a float as CPython's `repr` does, so the two must agree on every double. This is
//! how `python/tests/test_rust_floats.py` holds Rust to CPython over hundreds of thousands of values (systematic ties among them),
//! more than are worth committing as vectors.

use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        let bits = u64::from_str_radix(line.trim(), 16).expect("hex bits");
        let x = f64::from_bits(bits);
        assert!(x.is_finite(), "{line} is not finite");
        writeln!(out, "{}", tilefish::json::python_float_repr(x)).expect("stdout");
    }
}
