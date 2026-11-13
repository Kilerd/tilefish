"""Writes the test vectors that the Rust crate checks against CPython (``crates/tilefish/tests/data/python_*.in``). Not collected.

Run from ``python/tests`` (no GPU, no Triton)::

    python export_rust_vectors.py ../../crates/tilefish/tests/data

The digests of a manifest are computed by Python and by Rust, and they must be the same bytes. The fixtures (real manifests and
cubins) prove it on what ``tilefish`` writes; these vectors prove it on the edges of the formats, with CPython as the oracle:

* ``python_floats.in``: ``repr(x)`` of 500 floats (special values, powers of ten, random bit patterns), by their bits;
* ``python_canonical.in``: JSON texts and ``json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))`` of each:
  escapes, control characters, ``0x7f``, characters above the basic plane, keys whose code point order is not their UTF-16
  order, the edges of the integers, floats, nesting, random strings;
* ``python_elf.in``: small ELF files (``elf_builder.py``) and their ``code_digest``;
* ``python_launches.json``: calls on the binaries of ``host_ops.ops()`` (a valid call per size and every single change of one input:
  dtype, rank, an axis made larger, twice as large or empty, a misaligned address, a tensor too large for 32-bit offsets, a tensor with gaps given to a
  pointer that is dense, an alias; for manifest schema 3 also views with their strides, a size-1 axis, a stride that fits no ``i32`` or ``i64``, the
  limit of ``max_extent``, addresses of every alignment, device scalars of every shape and dtype, and prebuilt binaries with fixed arguments) and
  what ``runtime._prepare`` of Python decides for each (the grid and the ``kernelParams`` values, or the kind of the refusal), run on
  fake tensors; the Rust ``Plan::prepare`` must decide the same (``tests/python_launches.rs``). A tensor of a record is ``{"ptr", "shape", "dtype"}``, and
  ``"strides"`` (element strides, as torch reports them; ``ptr`` is the address of the first element) when it is a view; ``params`` are the values of the
  ``kernelParams``: pointers are addresses, ``i32`` and ``i64`` scalars their integer, a fixed argument its integer or the IEEE bits of an ``f32``, and a Triton
  kernel's two null scratch slots follow (a prebuilt binary has none);
* ``python_plans.json``: plan documents and what Python decides of each (``export_plan_vectors.py``);
* ``python_manifests.in``: ``MANIFESTS``, small manifest documents (schema 2 shapes and dimensions, other orders, other schemas, other
  alphabets; schema 3 with kernels and library entries in every order, every field it added, the spelling of floats, non-ASCII text escaped and as it is,
  and the schemas ``3.0``, ``true``, ``"3"`` and ``4``) and their ``manifest_digest``, which has its own tag per schema; and ``MANIFEST_ERRORS``, documents
  that ``manifest_digest`` refuses.

The generators are seeded, so the files only change when a format does.
"""

import json
import math
import pathlib
import random
import struct
import sys

from export_plan_vectors import plan_vectors_json
from elf_builder import SHF_ALLOC, SHF_EXECINSTR, SHT_NOBITS, SHT_PROGBITS, S, build_elf, cubin_like

from fake_torch import dense_strides
from tilefish._elf import code_digest


def systematic_floats(exponents=range(-60, 12), small_odd=range(1, 32, 2)):
    """Doubles where Python's ``repr`` has something to decide: exact decimal ties, the edges of the formats.

    The shortest string that reads back as a double is not always unique: when two of the same length both do, ``repr`` takes the one
    nearest the exact value and, if the exact value is halfway between them (it ends in a 5 one digit past the shortest), the even
    one. Rust's ``{:e}`` rounds that tie up, so ties are what a spelling has to be held to: ``m * 2**e`` for small odd ``m`` over a
    range of exponents (``2**-25`` is one: Python's ``2.9802322387695312e-08``), both signs; the values of ``[2**49, 2**50)`` with
    a ``.25`` or ``.75`` fraction (17 significant digits, a tie at the 16th); the subnormals and the edges between them and the
    normals; the largest and smallest doubles; and the neighbours of the points where ``repr`` changes its format (``1e16``, ``1e-4``).
    """
    out = []
    for e in exponents:
        for m in small_odd:
            out.append(math.ldexp(m, e))
    for k in range(48):
        base = 2.0**49 + k * 257
        out += [base + 0.25, base + 0.75, base + 0.5]
    for m in range(1, 65):
        out.append(math.ldexp(m, -1074))
    for edge in (math.ldexp(1, -1022), math.ldexp(1, -1023), 5e-324, 1.7976931348623157e308, 2.2250738585072014e-308):
        out.append(edge)
        for direction in (0.0, math.inf):
            x = edge
            for _ in range(4):
                x = math.nextafter(x, direction)
                if math.isfinite(x) and x != 0.0:
                    out.append(x)
    for point in (1e16, 1e15, 1e17, 1e-4, 1e-5, 1e-3, 1e22, 1e23, 9007199254740992.0, 1.0, 10.0, 0.1):
        out.append(point)
        for direction in (0.0, math.inf):
            x = point
            for _ in range(6):
                x = math.nextafter(x, direction)
                out.append(x)
    out += [1e16 - 2, 1e16 + 2, 9999999999999998.0, 12345678901234568.0, 0.00009999999999999999, 0.0001000000000000001]
    return out + [-x for x in out] + [0.0, -0.0]


def floats():
    random.seed(7)
    vals = [0.0, -0.0, 1.0, 0.5, 0.1, 0.1 + 0.2, 1e15, 1e16, 1e17, 9999999999999998.0, 123456789012345678.0, 1e-4, 1e-5, 9.999e-5]
    vals += [5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 1e22, 1e23, 1.5, 100.0, 1234.5678, 3.14159, 2.5e-7, 1e100, 1e-100]
    vals += [float(2**53), float(2**53 + 2), 0.3, 1 / 3, 2 / 3, 1e5, 1e6, 123456.789e3, 4.35, 0.000123, 12345678912345678.0]
    for e in range(-30, 30):
        vals.append(10.0**e)
        vals.append(1.2345 * 10.0**e)
    for _ in range(150):
        while True:
            x = struct.unpack("<d", struct.pack("<Q", random.getrandbits(64)))[0]
            if math.isfinite(x):
                vals.append(x)
                break
    for _ in range(100):
        vals.append(random.uniform(-1e6, 1e6))
        vals.append(random.uniform(0, 1) * 10.0 ** random.randint(-20, 20))
    vals += systematic_floats()
    seen, rows = set(), []
    for x in vals:
        bits = struct.unpack("<Q", struct.pack("<d", x))[0]
        if bits not in seen:
            seen.add(bits)
            rows.append((bits, repr(x)))
    out = ["// generated by python: (bits of an f64, repr(x))", "pub const FLOATS: &[(u64, &str)] = &["]
    out += [f'    (0x{bits:016x}, "{r}"),' for bits, r in rows]
    return "\n".join([*out, "];"]) + "\n"


def large_float_set(count=200_000, seed=2024):
    """A big set for ``test_rust_floats.py``: dyadic rationals over every exponent (both signs, subnormals included), and random doubles."""
    rng = random.Random(seed)
    out = list(systematic_floats(range(-1074, 972, 1), range(1, 40, 2)))
    while len(out) < count:
        kind = rng.randrange(4)
        if kind == 0:  # an odd mantissa of a few bits times a power of two: ties live here
            x = math.ldexp(rng.randrange(1, 1 << rng.randrange(1, 12), 2) | 1, rng.randrange(-1074, 960))
        elif kind == 1:  # a full 53-bit mantissa
            x = math.ldexp(rng.getrandbits(53) | (1 << 52), rng.randrange(-1074, 960))
        elif kind == 2:  # random bits
            x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        else:  # moderate numbers with short decimals: the formats the manifests have
            x = round(rng.uniform(-1e3, 1e3), rng.randrange(0, 9)) * 10.0 ** rng.randrange(-8, 9)
        if math.isfinite(x):
            out.append(x if rng.random() < 0.5 else -x)
    return out


def rust_string(s):
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif 0x20 <= o < 0x7F:
            out.append(ch)
        else:
            out.append(f"\\u{{{o:x}}}")
    out.append('"')
    return "".join(out)


def canonical():
    import json

    random.seed(11)
    docs = [
        '{"b": 1, "a": 2}',
        '{"\\u00e9": 1, "e": 2, "\\ud83d\\ude00": 3, "\\uffff": 4, "z": 5}',  # keys: code point order, not UTF-16 order
        '["\\u0000\\u0001\\u001f \\u007f\\u0080\\u00ff\\u0100\\u2028\\ud7ff\\ue000\\uffff"]',
        '["\\"\\\\\\/\\b\\f\\n\\r\\t"]',
        '["\\ud800\\udc00\\udbff\\udfff\\ud83d\\ude00"]',
        "[0, -0, 1, -1, 9223372036854775807, -9223372036854775808, 9223372036854775808, 18446744073709551615]",
        "[0.0, -0.0, 1.5, 1e5, 1E5, 1e-5, 1.0e+16, 123456789.123456789, 5e-324, 1.7976931348623157e308, 0.1, 100.0]",
        '{"a": {"b": [1, {"c": null}, true, false]}, "A": "x"}',
        '{"": 1, " ": 2, "a b": 3}',
        '"just a string"',
        "[]",
        "{}",
        # what schema 2 puts in an entry: a declared shape (names and sizes in one list) and the table of dimensions
        '{"shape": ["Rows", 1024], "dims": {"Rows": {}, "S": {"buckets": [32, 64], "bucket": 32}, "n": {"divisible_by": 16}}}',
        '{"params": [{"shape": ["M", "S"], "name": "x"}, {"shape": [3072, "3*M"], "name": "y"}], "grid": "(cdiv(M, 16), 2)"}',
    ]
    for _ in range(40):
        pick = [lambda: random.randint(0x20, 0x7E), lambda: random.randint(0, 0x1F), lambda: random.randint(0x80, 0x7FF)]
        pick += [lambda: random.randint(0x800, 0xD7FF), lambda: random.randint(0xE000, 0xFFFF), lambda: random.randint(0x10000, 0x10FFFF)]
        s = "".join(chr(random.choice(pick)()) for _ in range(random.randint(0, 12)))
        docs.append(json.dumps({s: [s, random.random() * 10 ** random.randint(-8, 8), random.randint(-(10**6), 10**6)], "k": s}))
    out = [
        "// generated by python: (a JSON text, json.dumps(json.loads(text), sort_keys=True, separators=(',', ':')))",
        "pub const CANONICAL: &[(&str, &str)] = &[",
    ]
    for text in docs:
        dumped = json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"), allow_nan=False)
        out.append(f"    ({rust_string(text)}, {rust_string(dumped)}),")
    return "\n".join([*out, "];"]) + "\n"


def elfs():
    variants = {
        "default": cubin_like(),
        "another debug path": cubin_like(debug="somewhere/else/kernel.py"),
        "offsets moved by a gap": cubin_like(gap=13),
        "debug sections first": cubin_like(debug_first=True),
        "no shared section": cubin_like(shared=None),
        "empty shared section (sm_80)": cubin_like(shared=0),
        "other code": cubin_like(code=b"\x09\x08\x07\x06" * 16),
        "another OS ABI": cubin_like(osabi=0x42),
        "another machine": cubin_like(machine=0x1234),
        "other e_flags": cubin_like(flags=0x0A00670A),
        "debug first and moved": cubin_like(debug_first=True, gap=3, debug="x.py"),
        "no sections but one": build_elf([S(".text.k", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\x01" * 32, align=16)]),
        "a nobits section with a size": build_elf([S(".nv.shared.k", SHT_NOBITS, flags=SHF_ALLOC, size=4096, align=16), S(".text.k", data=b"abc")]),
    }
    out = ["// generated by python: (a description, an ELF file in hex, python's code_digest of it)", "pub const ELFS: &[(&str, &str, &str)] = &["]
    out += [f'    ("{name}", "{blob.hex()}", "{code_digest(blob)}"),' for name, blob in variants.items()]
    return "\n".join([*out, "];"]) + "\n"


def _text_and_digest(doc, ensure_ascii=True, indent=1):
    """The JSON text of a document (as a file would hold it) and the digest Python gives it. The text is what is digested, not ``doc``: a
    float is what it reads back as (``1e-06``) and ``"schema": 3.0`` is the float 3.0."""
    import json

    from tilefish._manifest import manifest_digest

    text = json.dumps(doc, indent=indent, ensure_ascii=ensure_ascii)
    return text, manifest_digest(json.loads(text))


def manifest_documents():
    """``(a manifest document, ensure_ascii of its text, its indent)``: schema 2 (digested in the form the crate 0.0.1 uses), then schema 3 (the
    first few indented as a manifest file is, the rest on one line: the whitespace is not part of the digest) and the schemas that are neither."""

    def entry(stem, arch, **extra):
        return {"stem": stem, "arch": arch, "shared": 16, "shape": ["M", 1024], "dims": {"M": {}}, **extra}

    # an entry of schema 3: what a Triton kernel, a strided pointer, a device scalar, a prebuilt binary and a library entry write (the fields are
    # the ones of docs/manifest-schema-3.md, which the digest takes as they are)
    def triton(stem, arch="sm_103", **extra):
        return entry(
            stem,
            arch,
            op=stem.split(".")[0],
            kind="triton",
            abi={
                "params": [
                    {"slot": 0, "index": 0, "name": "x", "kind": "pointer", "type": "*bf16", "ptx_type": "u64", "size": 8, "dtype": "bf16", "access": "in",
                     "shape": ["S", 8, 64], "strides": ["x_s0", "x_s1", 1], "offset": "x_offset"},
                    {"slot": 1, "index": 1, "name": "x_s0", "kind": "scalar", "type": "i32", "ptx_type": "u32", "size": 4, "dim": "x_s0",
                     "stride_of": {"arg": "x", "axis": 0}},
                    {"slot": 2, "index": 2, "name": "n_tok", "kind": "pointer", "type": "*i32", "ptx_type": "u64", "size": 8, "dtype": "i32", "access": "in", "shape": []},
                ],
                "scratch": [{"name": "global_scratch", "type": "*i8", "value": 0}, {"name": "profile_scratch", "type": "*i8", "value": 0}],
                "scratch_slots": 2,
            },
            preconditions=[
                {"kind": "pointer_alignment", "arg": "x", "index": 0, "bytes": 2},
                {"kind": "max_extent", "arg": "x", "index": 0, "value": 2147483647},
                {"kind": "pointer_alignment", "arg": "n_tok", "index": 2, "bytes": 16},
            ],
            **extra,
        )

    def shared_stride(stem):
        """The entry of ``triton`` with its stride scalar carrying the stride of two views (a ``stride_of`` list, which M3.3 added)."""
        e = triton(stem)
        e["abi"]["params"][1]["stride_of"] = [{"arg": "x", "axis": 0}, {"arg": "y", "axis": 0}]
        return e

    def binary(stem, arch="sm_103", alpha=2.0):
        return entry(
            stem,
            arch,
            op=stem,
            kind="binary",
            abi={
                "params": [
                    {"slot": 0, "index": 0, "name": "x", "kind": "pointer", "type": "*fp32", "ptx_type": "u64", "size": 8, "dtype": "f32", "access": "in", "shape": ["n"]},
                    {"slot": 1, "index": 1, "name": "alpha", "kind": "fixed", "type": "f32", "ptx_type": "f32", "size": 4, "value": alpha},
                    {"slot": 2, "index": 2, "name": "zero", "kind": "fixed", "type": "i64", "ptx_type": "u64", "size": 8, "value": 0},
                    {"slot": 3, "index": 3, "name": "eps", "kind": "fixed", "type": "f32", "ptx_type": "f32", "size": 4, "value": {"float": "inf"}},
                ],
                "scratch": [],
                "scratch_slots": 0,
            },
            binary={"source": "ptx", "ptxas_flags": ["-lineinfo", "-v", "--gpu-name=sm_103a"]},
            source_digest="0123456789abcdef" * 4,
            source_digest_kind="tilefish.binary.v1",
            toolchain={"ptxas": "Cuda compilation tools, release 12.9, V12.9.86"},
            block=[128, 1, 1],
            max_elements=2**31 - 1,
        )

    def library(stem, arch="sm_103", algo="0123456789abcdef" * 8, version=120804):
        return {
            "stem": stem,
            "op": stem.split(".")[0],
            "arch": arch,
            "impl": "cublaslt.gemm",
            "abi": {"params": [{"index": 0, "name": "a", "kind": "pointer", "dtype": "bf16", "access": "in", "shape": ["M", 1024]}]},
            "buckets": {"M": 32},
            "dims": {"M": {"buckets": [32, 64], "bucket": 32}},
            "preconditions": [{"kind": "pointer_alignment", "arg": "a", "index": 0, "bytes": 16}],
            "dtypes": {"a": "bf16", "b": "bf16", "out": "bf16"},
            "compute_type": "CUBLAS_COMPUTE_32F",
            "scale_type": "CUDA_R_32F",
            "transa": "T",
            "transb": "N",
            "m": 32,
            "n": 2048,
            "k": 1024,
            "lda": 1024,
            "ldb": 1024,
            "ldc": 2048,
            "ldd": 2048,
            "epilogue": "none",
            "workspace_bytes": 8519680,
            "algo": algo,
            "cublaslt_version": version,
            "declaration_digest": "f" * 64,
        }

    legacy = [
        {"schema": 2, "kernels": []},
        {"schema": 2, "tilefish": "0.0.2", "kernels": [entry("a", "sm_90")]},
        {"schema": 2, "tilefish": "9.9.9", "kernels": [entry("a", "sm_90")]},  # the version of the tool is not part of it
        {"schema": 2, "kernels": [entry("b", "sm_90"), entry("a", "sm_100"), entry("a", "sm_90")]},
        {"schema": 2, "kernels": [entry("a", "sm_90"), entry("a", "sm_100"), entry("b", "sm_90")]},  # the same, in another order
        {"schema": 1, "kernels": [entry("a", "sm_90")]},  # the schema is part of it
        {"schema": 2, "kernels": [entry("caf\u00e9.\u4e2d", "sm_90", note="\U0001f600\x7f", pi=3.14159, big=2**63, tiny=1e-07)]},
        {"schema": 2, "kernels": [entry("a", "sm_90", dims={"S": {"buckets": [32, 64], "bucket": 32}, "M": {}})]},
    ]
    non_ascii = {
        "schema": 3,
        "kernels": [triton("café.中", note='\U0001f600\x7f  "quoted" \\', pi=3.14159, big=2**63, tiny=1e-07)],
        "library": [library("gé.M32")],
    }
    v3 = [
        {"schema": 3, "kernels": [], "library": []},  # f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e
        {"schema": 3, "tilefish": "0.0.3", "kernels": [triton("k")], "library": [library("g.M32")]},
        {"schema": 3, "tilefish": "9.9.9", "kernels": [triton("k")], "library": [library("g.M32")]},  # the version of the tool is not digested
        # kernels and library entries, in several orders: the digest does not depend on the order of either list
        {"schema": 3, "kernels": [triton("b"), triton("a", "sm_120"), triton("a")], "library": [library("y.M32"), library("x.M32", "sm_120"), library("x.M32")]},
        {"schema": 3, "kernels": [triton("a"), triton("a", "sm_120"), triton("b")], "library": [library("x.M32"), library("x.M32", "sm_120"), library("y.M32")]},
        {"schema": 3, "library": [library("x.M32"), library("x.M32", "sm_120"), library("y.M32")], "kernels": [triton("a"), triton("b"), triton("a", "sm_120")]},
        # the key order of a document is not the order of the digest, and the entries of the two lists sort apart
        {"schema": 3, "kernels": [triton("k", "sm_90"), triton("k", "sm_103")], "library": [library("k.M32", "sm_90"), library("k.M32", "sm_103")]},
        # every field a version 3 entry adds is in the digest: kind, scratch_slots, strides, offset, stride_of, a device scalar, fixed values, binary, max_extent
        {"schema": 3, "kernels": [triton("s")], "library": []},
        {"schema": 3, "kernels": [shared_stride("s")], "library": []},  # a stride that two views share: stride_of is a list
        {"schema": 3, "kernels": [binary("p")], "library": []},
        {"schema": 3, "kernels": [binary("p", alpha=2.5)], "library": []},
        # the spelling of a float: python's shortest repr (2.0, 1e-06, 1e+16, 5e-324) and what it reads back as
        {"schema": 3, "kernels": [binary("p", alpha=1e-06)], "library": []},
        {"schema": 3, "kernels": [binary("p", alpha=1e16)], "library": []},
        {"schema": 3, "kernels": [binary("p", alpha=5e-324)], "library": []},
        {"schema": 3, "kernels": [binary("p", alpha=0.1)], "library": []},
        {"schema": 3, "kernels": [binary("p", alpha=123456789.123456789)], "library": []},
        {"schema": 3, "kernels": [binary("p", alpha=-0.0)], "library": []},
        # the fields of a library entry: the algorithm and the version are in it
        {"schema": 3, "kernels": [], "library": [library("g.M32", algo="f" * 128)]},
        {"schema": 3, "kernels": [], "library": [library("g.M32", version=120805)]},
        # non-ASCII, escaped (and, at the end of the list, as it is)
        non_ascii,
        # a schema that is not the integer 3: the digest still has a form for each, and 3.0 is the same number as 3 in Python
        {"schema": 3.0, "kernels": [triton("k")], "library": [library("g.M32")]},
        {"schema": 3.0, "kernels": [], "library": []},
        {"schema": True, "kernels": [triton("k")]},
        {"schema": "3", "kernels": [triton("k")]},
        {"schema": 4, "kernels": [triton("k")], "library": [library("g.M32")]},
        {"schema": 0, "kernels": []},
    ]
    return (
        [(d, True, 1) for d in legacy]
        + [(d, True, 2 if i < 3 else None) for i, d in enumerate(v3)]
        + [(non_ascii, False, None)]  # the same document with its non-ASCII text as it is, not escaped: the same digest
    )


def manifest_digests():
    """Small manifests and their ``manifest_digest``: the tag, the sorting of the entries and the canonical form together."""
    out = ["// generated by python: (a manifest document, tilefish._manifest.manifest_digest of it)", "pub const MANIFESTS: &[(&str, &str)] = &["]
    for doc, ensure_ascii, indent in manifest_documents():
        text, digest = _text_and_digest(doc, ensure_ascii, indent)
        out.append(f"    ({rust_string(text)}, {rust_string(digest)}),")
    out += ["];", ""]
    out += [
        "// generated by python: documents that tilefish._manifest.manifest_digest refuses (it is not JSON data it can digest)",
        "pub const MANIFEST_ERRORS: &[&str] = &[",
    ]
    for text in manifest_error_documents():
        out.append(f"    {rust_string(text)},")
    return "\n".join([*out, "];"]) + "\n"


def manifest_error_documents():
    """JSON texts that Python's ``manifest_digest`` refuses (a schema 3 without ``library``, ``kernels`` that is not a list, a value JSON cannot hold ...)."""
    import json

    from tilefish._manifest import manifest_digest
    from tilefish.errors import TilefishManifestError

    texts = [
        '{"schema": 3, "kernels": []}',  # no library
        '{"schema": 3, "library": []}',  # no kernels
        '{"kernels": [], "library": []}',  # no schema
        '{"schema": 2}',  # no kernels
        '{"schema": 2, "kernels": 5}',
        '{"schema": 2, "kernels": "k"}',
        '{"schema": 3, "kernels": 5, "library": []}',
        '{"schema": 3, "kernels": [], "library": 5}',
        '{"schema": 3, "kernels": [], "library": "x"}',
        '{"schema": 3, "kernels": [5], "library": []}',  # an entry that is not an object
        '{"schema": 3, "kernels": [], "library": ["x"]}',
        '{"schema": 2, "kernels": [[]]}',
        '{"schema": 2, "kernels": [{"stem": "a", "arch": "sm_90", "v": NaN}]}',  # not JSON: a host that reads strict JSON refuses the file first
        '{"schema": 2, "kernels": [{"stem": "a", "arch": "sm_90", "v": Infinity}]}',
        '{"schema": 3, "kernels": [], "library": [{"stem": "g", "arch": "sm_90", "v": -Infinity}]}',
        '{"schema": 2, "kernels": [{"stem": "a", "arch": "sm_90", "v": 1e999}]}',  # a number that does not fit a double
        "[]",
        "null",
        '"a string"',
        "5",
        "true",
    ]
    for text in texts:  # each one must be refused: the file is regenerated only from what Python refuses
        try:
            manifest_digest(json.loads(text))
        except TilefishManifestError:
            continue
        raise AssertionError(f"Python digests {text!r}")
    return texts


# (the operator of host_ops.ops(), the inputs of a valid call for one size, the sizes): what a call is built from. An input is ``(dtype, shape)`` (a dense
# tensor) or ``(dtype, shape, strides)`` (a view: element strides as ``t.stride()`` reports them, and the address of its first element)
LAUNCH_CASES = [
    ("softmax", lambda r: [("f32", (r, 1000))], (1, 5)),
    ("layernorm", lambda r: [("bf16", (r, 768)), ("bf16", (768,)), ("bf16", (768,))], (1, 5)),
    ("residual_add_rmsnorm", lambda r: [("bf16", (r, 2560)), ("f32", (r, 2560)), ("bf16", (2560,))], (1, 5)),
    ("gelu_exact", lambda n: [("bf16", (n,))], (16, 1040)),
    ("swiglu", lambda r: [("bf16", (r, 3072)), ("bf16", (r, 3072))], (1, 5)),
    ("bucketed_rms", lambda rd: [("bf16", rd), ("bf16", rd[1:])], ((5, 96), (3, 512), (5, 1000))),
    ("rmsnorm", lambda r: [("bf16", (r, 2048)), ("bf16", (2048,))], (1, 5)),
    ("gelu_erf", lambda n: [("bf16", (n,))], (16, 1040)),
    ("axpy", lambda n: [("f32", (n,)), ("f32", (n,))], (16, 1040)),
    ("mm", lambda mnk: [("f16", (mnk[0], mnk[2])), ("f16", (mnk[2], mnk[1]))], ((128, 128, 64), (256, 384, 192))),
    # manifest schema 3: strided views (the q, k or v of a fused qkv: strides (1536, 64, 1)), device scalars, prebuilt binaries with fixed arguments
    ("head_copy", lambda r: [("bf16", (r, 8, 64), (1536, 64, 1))], (1, 5)),
    ("head_copy16", lambda r: [("bf16", (r, 8, 64), (1536, 64, 1))], (1, 5)),
    ("head_copy64", lambda r: [("bf16", (r, 8, 64), (1536, 64, 1))], (1, 5)),
    ("softmax3d", lambda ab: [("f32", (ab[0], ab[1], 128), (128, ab[0] * 128, 1))], ((5, 3), (2, 9))),
    ("masked_rows", lambda rs: [("f32", (rs[0], 128)), ("i32", rs[1])], ((5, ()), (3, (1,)))),
    ("scale_by", lambda r: [("f32", (r, 256)), ("f32", ())], (1, 5)),
    ("scale_flat", lambda n: [("f32", (n,)), ("f32", ()), ("f32", ())], (16, 1040)),
    ("axpy_ptx", lambda n: [("f32", (n,)), ("f32", (n,))], (16, 1040)),
    ("strided_copy", lambda r: [("bf16", (r, 64), (200, 1))], (1, 5)),
    # M3.3: an extern declared with enable_fp_fusion= (the option is in the manifest entry; a launch is as any other)
    ("fma_plain", lambda n: [("f32", (n,)), ("f32", (n,)), ("f32", (n,))], (16, 1040)),
    ("fma_fused", lambda n: [("f32", (n,)), ("f32", (n,)), ("f32", (n,))], (16, 1040)),
    # M3.3: three views read with one stride pair, the q, k and v of a fused [R, 3, 4, 64] tensor (strides (768, 64, 1) each)
    ("qkv_sum", lambda r: [("bf16", (r, 4, 64), (768, 64, 1))] * 3, (1, 5)),
    ("qkv_sum16", lambda r: [("bf16", (r, 4, 64), (768, 64, 1))] * 3, (1, 5)),
    # M3.3: a dimension only the Out tensor determines (a [B] mask from a device scalar): the inputs, then the Out tensors the call is given
    ("key_mask", lambda b: [("i32", ()), ("bf16", (b,))], (8, 24)),
    ("key_mask_any", lambda b: [("i32", (1,)), ("bf16", (b,))], (1, 1000)),
]
_OTHER_DTYPE = {"bf16": "f32", "f32": "f16", "f16": "bf16", "i32": "i64"}
I31 = (1 << 31) - 1


def _mutations(dtype, shape, ptr, strides):
    """Single changes to one input tensor: ``(what, (dtype, shape, ptr, strides))`` (``strides`` is ``None`` for a dense tensor)."""

    yield "dtype", (_OTHER_DTYPE[dtype], shape, ptr, strides)
    if len(shape) >= 2:  # the strides of a view go with its axes
        yield "drop an axis", (dtype, shape[:-1], ptr, None if strides is None else strides[:-1])
    yield "add an axis", (dtype, (*shape, 2), ptr, None if strides is None else (*strides, 1))
    for axis, size in enumerate(shape):
        yield f"axis {axis} + 1", (dtype, (*shape[:axis], size + 1, *shape[axis + 1 :]), ptr, strides)
        yield f"axis {axis} * 2", (dtype, (*shape[:axis], size * 2, *shape[axis + 1 :]), ptr, strides)
        yield f"axis {axis} = 0", (dtype, (*shape[:axis], 0, *shape[axis + 1 :]), ptr, strides)
    yield "address + 8", (dtype, shape, ptr + 8, strides)
    yield "address + 16", (dtype, shape, ptr + 16, strides)
    yield "more elements than 32-bit offsets reach", (dtype, (2_200_000, *shape[1:]), ptr, strides)
    yield "2^31 rows", (dtype, (1 << 31, *shape[1:]), ptr, strides)
    if len(shape) == 2 and strides is None:  # the row count at which the tensor has 2^31 - 1 elements or fewer, and the first that has more
        for rows in (((1 << 31) - 1) // shape[1], ((1 << 31) - 1) // shape[1] + 1):
            yield f"{rows} rows", (dtype, (rows, shape[1]), ptr, strides)
    if strides is None and any(size != 1 for size in shape):  # a dense pointer is given a tensor with gaps
        yield "not contiguous", (dtype, shape, ptr, tuple(2 * s for s in dense_strides(shape)))


def _views(key, ptr, high):
    """Calls of schema 3 that no single change of the valid call makes, for the first input of ``key``: ``(what, (dtype, shape, ptr, strides))``.

    A view with a huge extent starts at ``high``, above every other tensor of the call (the allocator of the fake torch puts them 64 GiB apart, and a view
    that reaches over the ``Out`` tensor is refused as an alias: that is not what these calls are about)."""
    qkv = "bf16", (40, 8, 64)
    if key in ("head_copy", "head_copy16", "head_copy64"):
        yield "transposed", (*qkv, ptr, (64, 2560, 1))
        yield "expanded (stride 0)", (*qkv, ptr, (0, 64, 1))
        yield "one row, torch's stride", ("bf16", (1, 8, 64), ptr, (1536, 64, 1))
        yield "one row, a stride that fits no i32", ("bf16", (1, 8, 64), ptr, (1 << 40, 64, 1))
        yield "last axis stride 2", (*qkv, ptr, (1024, 128, 2))
        yield "last axis stride 0", (*qkv, ptr, (1536, 64, 0))
        yield "a stride beyond i32", (*qkv, high, (1 << 31, 64, 1))
        yield "a stride of 2^31 - 1", (*qkv, high, (I31, 64, 1))
        yield "an extent of 2^31 - 1", ("bf16", (2, 8, 64), high, (I31 - 512, 64, 1))
        yield "an extent of 2^31", ("bf16", (2, 8, 64), high, (I31 - 511, 64, 1))
        yield "strides 2^41 and 2^40", ("bf16", (2, 8, 64), high, (1 << 41, 1 << 40, 1))
        yield "strides 2^63 and 2^62", ("bf16", (2, 8, 64), high, (1 << 63, 1 << 62, 1))
        yield "a stride of 2^63 - 1", ("bf16", (2, 8, 64), high, (2**63 - 1, 64, 1))
        for offset in (1, 2, 4, 6, 8, 10, 16):
            yield f"address + {offset}", (*qkv, ptr + offset, (1536, 64, 1))
        yield "a stride that is not a multiple of 8", (*qkv, ptr, (1537, 64, 1))
        yield "another stride that is not a multiple of 8", (*qkv, ptr, (1536, 60, 1))
    elif key == "softmax3d":
        yield "broadcast along the first axis", ("f32", (5, 3, 128), ptr, (0, 128, 1))
        yield "last axis stride 2", ("f32", (5, 3, 128), ptr, (256, 1280, 2))
        yield "one row", ("f32", (1, 3, 128), ptr, (1 << 40, 128, 1))
        yield "address + 4", ("f32", (5, 3, 128), ptr + 4, (128, 640, 1))
    elif key == "strided_copy":
        yield "a row stride of 2^33", ("bf16", (7, 64), high, (1 << 33, 1))
        yield "a row stride of 2^63 - 1", ("bf16", (7, 64), high, (2**63 - 1, 1))
        yield "a row stride of 2^63", ("bf16", (7, 64), high, (2**63, 1))
        yield "one row", ("bf16", (1, 64), ptr, (1 << 40, 1))
        yield "last axis stride 2", ("bf16", (7, 64), ptr, (200, 2))
        yield "address + 1", ("bf16", (7, 64), ptr + 1, (200, 1))
        yield "address + 2", ("bf16", (7, 64), ptr + 2, (200, 1))


def _shared_strides(key, base):
    """Calls of an op whose views share one stride pair (``qkv_sum``): another view (k, then v) of other strides than the first one's, which the first
    binds and the others must have, and all three of other strides alike: ``(input index, (dtype, shape, ptr, strides))``."""
    if key not in ("qkv_sum", "qkv_sum16"):
        return
    for j in (1, 2):
        dtype, shape, ptr, _ = base[j]
        for strides in ((769, 64, 1), (768, 65, 1), (256, 64, 1), (1536, 64, 1), (768, 128, 1), (0, 64, 1)):
            yield [j], (dtype, shape, ptr, strides)
    # the same strides on all three: separate dense tensors, a wider fused tensor, a broadcast first axis
    for strides in ((256, 64, 1), (1536, 64, 1), (0, 64, 1), (768, 64, 1)):
        yield [0, 1, 2], (base[0][0], base[0][1], None, strides)


def _output_bound(key, base):
    """Calls of an op whose ``Out`` tensor binds a dimension (``key_mask``): ``(input index, (dtype, shape, ptr, strides))`` for the ``Out`` tensor
    (index 1) of other numbers of keys, buckets and not, and of other ranks, and for the device scalar (index 0) of every shape."""
    if key not in ("key_mask", "key_mask_any"):
        return
    dtype, _, ptr, _ = base[1]
    for b in (1, 7, 8, 9, 16, 24, 25, 128, 129, 1 << 20):
        yield 1, (dtype, (b,), ptr, None)
    yield 1, (dtype, (24, 1), ptr, None)
    yield 1, (dtype, (), ptr, None)
    yield 1, ("f32", (24,), ptr, None)
    for shape in ((), (1,), (2,), (1, 1), (0,)):
        yield 0, ("i32", shape, base[0][2], None)


def _device_scalars(key, base):
    """Device scalars of every shape and alignment: ``(input index, (dtype, shape, ptr, strides))``."""
    if key not in ("masked_rows", "scale_by", "scale_flat"):
        return
    dtype = "i32" if key == "masked_rows" else "f32"
    for j in range(1, len(base)):  # every input after the first is a device scalar
        ptr = base[j][2]
        for shape in ((), (1,), (2,), (1, 1), (0,), (4,)):
            yield j, (dtype, shape, ptr, None)
        for offset in (4, 8, 16):
            yield j, (dtype, (), ptr + offset, None)


def launches():
    """Calls and what ``runtime._prepare`` of Python decides for each (the oracle for the Rust ``Plan::prepare``): ``(records, how_many_binaries)``.

    For each binary of ``host_ops.ops()`` a valid call per size, and every single change of one input (dtype, rank, an axis made
    larger, twice as large or empty, a misaligned address, a tensor too large for 32-bit offsets, a tensor with gaps given to a pointer that is
    dense, an in-place tensor aliased); and for what manifest schema 3 added: views (a transposed one, an expanded one, an axis of size 1 whose stride
    is not the one a kernel is given, a last stride that is not 1, a stride that fits no ``i32`` or no ``i64``, the extent at the limit of 32-bit offsets, addresses
    of every alignment, strides that are not multiples of what the binary assumes), device scalars of every shape, dtype and alignment, and
    prebuilt binaries (a fixed argument is in the ``kernelParams`` at its own slot, and no scratch slot follows).
    A record is the stem of a binary, the tensors of the call in the order of its ABI (an ``Out`` tensor is what Python would
    allocate; a view has its ``strides``) and what Python decided: the grid and the ``kernelParams`` values, or the kind of the refusal. A call that Python sends
    to another bucket is a valid call of that binary, and a ``bucket`` refusal of this one.
    """
    from fake_torch import FakeTensor, prepare, verdict
    from host_ops import abi_of, ops

    from tilefish.runtime import _target

    from tilefish._launch import bind_call, eval_dim
    from tilefish.errors import TilefishError

    first_slab, slab = 1 << 44, 1 << 36
    declared = ops()

    def fake(t):
        d, sh, ptr, st = t
        return FakeTensor(sh, d, ptr, st)

    def allocated_outs(op, call, after, fallback):
        """What Python would allocate for the ``Out`` tensors of ``call``: shaped by the dimensions its inputs bind (as ``_prepare`` does
        once they are bound), or ``fallback`` when they cannot be bound (the refusal is then about an input)."""
        target = _target(op)
        try:
            dims = bind_call(target.name, target.public, [fake(t) for t in call], {}).dims
        except TilefishError:
            return fallback
        outs = {}
        for i, p in enumerate(q for q in target.launch_extern().signature.params if q.kind == "tensor" and q.access == "out"):
            shape = [eval_dim(d, dims) for d in p.type.shape]
            if any(v is None for v in shape):
                return fallback
            outs[p.name] = FakeTensor(shape, p.type.dtype.name, after + i * slab)
        return outs
    records, seen = [], set()

    def add(stem, tensors, expect):
        record = {"stem": stem, "tensors": tensors, "expect": expect}
        key = json.dumps(record, sort_keys=True)
        if key not in seen:
            seen.add(key)
            records.append(record)

    def abi_tensors(spec, by_name):
        out = []
        for a in spec.args:
            if a.kind != "pointer":
                continue
            t = by_name[a.name]
            record = {"ptr": t.data_ptr(), "shape": list(t.shape), "dtype": t.dtype.short}
            if t.strides is not None:  # a view: its element strides, as torch reports them (the address is the one of its first element)
                record["strides"] = list(t.strides)
            out.append(record)
        return out

    stems = set()
    for key, build, sizes in LAUNCH_CASES:
        op = declared[key]
        _, inputs, outputs = abi_of(op)
        # an op with a dimension only its Out tensors determine is called with them (`outputs=`), after its inputs: every change of a call is a
        # change of one of them too
        given_outs = bool(_target(op).public.output_dims)
        names = inputs + outputs if given_outs else inputs
        n_in = len(inputs)

        def prepare_call(call, after):
            if not given_outs:
                return prepare(op, [fake(t) for t in call], next_slab=after)
            return prepare(op, [fake(t) for t in call[:n_in]], next_slab=after, outputs=[fake(t) for t in call[n_in:]])

        specs = {sp.stem: sp for sp in _target(op).compile_specs()}
        stems |= set(specs)
        for size in sizes:
            base = [(item[0], tuple(item[1]), first_slab + i * slab, tuple(item[2]) if len(item) > 2 else None) for i, item in enumerate(build(size))]
            after = first_slab + len(base) * slab
            prepared, error = prepare_call(base, after)
            assert error is None, (key, size, error)
            intended = prepared.spec
            by_abi_base = dict(prepared.tensors)  # the allocated Out tensors of the valid call
            inout = [i for i, name in enumerate(inputs) if any(a.name == name and a.access == "inout" for a in intended.args)]
            add(intended.stem, abi_tensors(intended, by_abi_base), {k: v for k, v in verdict(prepared, None).items() if k != "stem"})
            calls = []  # every call: the valid one with a single change
            for j, t in enumerate(base):
                for _, new in _mutations(*t):
                    calls.append([*base[:j], new, *base[j + 1 :]])
                for i in inout:
                    if i != j:
                        calls.append([*base[:j], (t[0], t[1], base[i][2], t[3]), *base[j + 1 :]])
            for _, new in _views(key, base[0][2], after + 8 * slab):
                calls.append([new, *base[1:]])
            for j, new in _device_scalars(key, base):
                calls.append([*base[:j], new, *base[j + 1 :]])
            for which, (dtype, shape, ptr, strides) in _shared_strides(key, base):
                call = list(base)
                for j in which:
                    call[j] = (dtype, shape, base[j][2] if ptr is None else ptr, strides)
                calls.append(call)
            for j, new in _output_bound(key, base):
                calls.append([*base[:j], new, *base[j + 1 :]])
            if key == "bucketed_rms":  # the bucketed dimension moved in every tensor that has it: another bucket, or none
                rows = size[0]
                for db in (95, 100, 512, 1000):
                    calls.append([(base[0][0], (rows, db), base[0][2], None), (base[1][0], (db,), base[1][2], None)])
            for call in calls:
                prepared, error = prepare_call(call, after)
                got = verdict(prepared, error)
                if prepared is not None:
                    # a binary Python selected: a valid call of that binary, and a bucket refusal of the one intended
                    selected = prepared.spec
                    add(selected.stem, abi_tensors(selected, dict(prepared.tensors)), {k: v for k, v in got.items() if k != "stem"})
                    if selected.stem != intended.stem:
                        by_name = {names[i]: fake(t) for i, t in enumerate(call)}
                        outs = allocated_outs(op, call, after, {n: t for n, t in by_abi_base.items() if n not in by_name})
                        add(intended.stem, abi_tensors(intended, {**outs, **by_name}), {"ok": False, "kind": "bucket"})
                else:
                    by_name = {names[i]: fake(t) for i, t in enumerate(call)}
                    outs = allocated_outs(op, call[:n_in], after, {n: t for n, t in by_abi_base.items() if n not in by_name})
                    add(intended.stem, abi_tensors(intended, {**outs, **by_name}), got)
    return records, len(stems)


def launches_json():
    records, _ = launches()
    return json.dumps(records, indent=None, separators=(",", ":")) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    directory = pathlib.Path(argv[0])
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("python_floats.in", floats()),
        ("python_canonical.in", canonical()),
        ("python_elf.in", elfs()),
        ("python_manifests.in", manifest_digests()),
        ("python_launches.json", launches_json()),
        ("python_plans.json", plan_vectors_json()),
    ):
        (directory / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
