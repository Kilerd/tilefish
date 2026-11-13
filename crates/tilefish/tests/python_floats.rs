//! The spelling of a float in the canonical JSON of a manifest digest is Python's `repr(float)`.
//!
//! `data/python_floats.in` is made by CPython (`repr(x)` of 500 floats: special values, powers of ten, random bit
//! patterns), as pairs of the bits of the float and its repr. A digest over a manifest with a float in it is only
//! Python's when all of them agree.

include!("data/python_floats.in");

#[test]
fn a_float_is_spelled_as_python_spells_it() {
    assert!(FLOATS.len() > 400);
    for (bits, repr) in FLOATS {
        let x = f64::from_bits(*bits);
        assert_eq!(tilefish::json::python_float_repr(x), *repr, "bits {bits:#018x}");
    }
}
