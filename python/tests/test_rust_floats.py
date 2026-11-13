"""The Rust host spells every float as CPython's ``repr`` does: 200,000 doubles, systematic exact ties among them.

The digest of a manifest is a sha256 over canonical JSON, and a float in it is spelled by ``repr``. Python's ``repr`` picks, among the
shortest strings that read back as the double, the one nearest its exact value, and on an exact tie the even one; Rust's own float
formatting rounds the tie up (``2**-25`` is ``2.9802322387695312e-08`` here and ``...313e-08`` there), so a digest of a manifest with
such a float would differ between the two hosts. ``crates/tilefish/examples/float_repr.rs`` reads doubles by their bits and writes
``tilefish::json::python_float_repr`` of each; this holds it to ``repr`` over dyadic rationals of every exponent (both signs,
subnormals, the edges of ``repr``'s formats) and random doubles. The committed vectors (``tests/python_floats.rs``) are the small
version of this, and run without Python.
"""

import math
import shutil
import struct
import subprocess

import pytest
from conftest import REPO_ROOT
from export_rust_vectors import large_float_set, systematic_floats

CRATE = REPO_ROOT / "crates" / "tilefish"


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


@pytest.mark.skipif(shutil.which("cargo") is None or not CRATE.exists(), reason="cargo or the Rust crate is not available")
def test_rust_spells_two_hundred_thousand_doubles_as_python_does():
    values = large_float_set()
    assert len(values) >= 200_000
    done = subprocess.run(
        ["cargo", "run", "--quiet", "--release", "--example", "float_repr", "--manifest-path", str(CRATE / "Cargo.toml")],
        input="".join(f"{bits(x):016x}\n" for x in values),
        capture_output=True,
        text=True,
        timeout=900,
        env={"CARGO_TARGET_DIR": str(REPO_ROOT / "target" / "float-repr"), **__import__("os").environ},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    spelled = done.stdout.splitlines()
    assert len(spelled) == len(values)
    wrong = [(x, repr(x), got) for x, got in zip(values, spelled) if repr(x) != got]
    assert not wrong, f"{len(wrong)} of {len(values)} doubles are spelled differently; the first: {wrong[:5]}"


def test_the_ties_the_set_is_built_around_are_ties():
    """A guard on the vectors themselves: Python's repr of these is the even one of two equally near candidates."""
    assert repr(2.0**-25) == "2.9802322387695312e-08"
    ties = [x for x in systematic_floats() if x > 0 and math.log2(x) > -70]
    assert len(ties) > 1000
