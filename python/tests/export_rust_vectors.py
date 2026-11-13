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
  dtype, rank, an axis made larger, twice as large or empty, a misaligned address, a tensor too large for 32-bit offsets, an alias) and
  what ``runtime._prepare`` of Python decides for each (the grid and the ``kernelParams`` values, or the kind of the refusal), run on
  fake tensors; the Rust ``Plan::prepare`` must decide the same (``tests/python_launches.rs``);
* ``python_manifests.in``: small manifest documents (schema 2 shapes and dimensions, other orders, other schemas, other
  alphabets) and their ``manifest_digest``, which has its own tag.

The generators are seeded, so the files only change when a format does.
"""

import json
import math
import pathlib
import random
import struct
import sys

from elf_builder import SHF_ALLOC, SHF_EXECINSTR, SHT_NOBITS, SHT_PROGBITS, S, build_elf, cubin_like

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


def manifest_digests():
    """Small manifests and their ``manifest_digest``: the tag, the sorting of the entries and the canonical form together."""
    import json

    from tilefish._manifest import manifest_digest

    def entry(stem, arch, **extra):
        return {"stem": stem, "arch": arch, "shared": 16, "shape": ["M", 1024], "dims": {"M": {}}, **extra}

    docs = [
        {"schema": 2, "kernels": []},
        {"schema": 2, "tilefish": "0.0.2", "kernels": [entry("a", "sm_90")]},
        {"schema": 2, "tilefish": "9.9.9", "kernels": [entry("a", "sm_90")]},  # the version of the tool is not part of it
        {"schema": 2, "kernels": [entry("b", "sm_90"), entry("a", "sm_100"), entry("a", "sm_90")]},
        {"schema": 2, "kernels": [entry("a", "sm_90"), entry("a", "sm_100"), entry("b", "sm_90")]},  # the same, in another order
        {"schema": 1, "kernels": [entry("a", "sm_90")]},  # the schema is part of it
        {"schema": 2, "kernels": [entry("caf\u00e9.\u4e2d", "sm_90", note="\U0001f600\x7f", pi=3.14159, big=2**63, tiny=1e-07)]},
        {"schema": 2, "kernels": [entry("a", "sm_90", dims={"S": {"buckets": [32, 64], "bucket": 32}, "M": {}})]},
    ]
    out = ["// generated by python: (a manifest document, tilefish._manifest.manifest_digest of it)", "pub const MANIFESTS: &[(&str, &str)] = &["]
    for doc in docs:
        text = json.dumps(doc, indent=1)
        out.append(f"    ({rust_string(text)}, {rust_string(manifest_digest(json.loads(text)))}),")
    return "\n".join([*out, "];"]) + "\n"


# (the operator of host_ops.ops(), the inputs of a valid call for one size, the sizes): what a call is built from
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
]
_OTHER_DTYPE = {"bf16": "f32", "f32": "f16", "f16": "bf16"}


def _mutations(dtype, shape, ptr):
    """Single changes to one input tensor: ``(what, (dtype, shape, ptr))``."""
    yield "dtype", (_OTHER_DTYPE[dtype], shape, ptr)
    if len(shape) >= 2:
        yield "drop an axis", (dtype, shape[:-1], ptr)
    yield "add an axis", (dtype, (*shape, 2), ptr)
    for axis, size in enumerate(shape):
        yield f"axis {axis} + 1", (dtype, (*shape[:axis], size + 1, *shape[axis + 1 :]), ptr)
        yield f"axis {axis} * 2", (dtype, (*shape[:axis], size * 2, *shape[axis + 1 :]), ptr)
        yield f"axis {axis} = 0", (dtype, (*shape[:axis], 0, *shape[axis + 1 :]), ptr)
    yield "address + 8", (dtype, shape, ptr + 8)
    yield "address + 16", (dtype, shape, ptr + 16)
    yield "more elements than 32-bit offsets reach", (dtype, (2_200_000, *shape[1:]), ptr)
    yield "2^31 rows", (dtype, (1 << 31, *shape[1:]), ptr)
    if len(shape) == 2:  # the row count at which the tensor has 2^31 - 1 elements or fewer, and the first that has more
        for rows in (((1 << 31) - 1) // shape[1], ((1 << 31) - 1) // shape[1] + 1):
            yield f"{rows} rows", (dtype, (rows, shape[1]), ptr)


def launches():
    """Calls and what ``runtime._prepare`` of Python decides for each (the oracle for the Rust ``Plan::prepare``): ``(records, how_many_binaries)``.

    For each binary of ``host_ops.ops()`` a valid call per size, and every single change of one input (dtype, rank, an axis made
    larger, twice as large or empty, a misaligned address, a tensor too large for 32-bit offsets, an in-place tensor aliased).
    A record is the stem of a binary, the tensors of the call in the order of its ABI (an ``Out`` tensor is what Python would
    allocate) and what Python decided: the grid and the ``kernelParams`` values, or the kind of the refusal. A call that Python sends
    to another bucket is a valid call of that binary, and a ``bucket`` refusal of this one.
    """
    from fake_torch import FakeTensor, prepare, verdict
    from host_ops import abi_of, ops

    from tilefish.runtime import _target

    from tilefish._launch import bind_call, eval_dim
    from tilefish.errors import TilefishError

    first_slab, slab = 1 << 44, 1 << 36
    declared = ops()

    def allocated_outs(op, call, after, fallback):
        """What Python would allocate for the ``Out`` tensors of ``call``: shaped by the dimensions its inputs bind (as ``_prepare`` does
        once they are bound), or ``fallback`` when they cannot be bound (the refusal is then about an input)."""
        target = _target(op)
        try:
            dims = bind_call(target.name, target.public, [FakeTensor(sh, d, p) for d, sh, p in call], {}).dims
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
        return [
            {"ptr": by_name[a.name].data_ptr(), "shape": list(by_name[a.name].shape), "dtype": by_name[a.name].dtype.short}
            for a in spec.args
            if a.kind == "pointer"
        ]

    stems = set()
    for key, build, sizes in LAUNCH_CASES:
        op = declared[key]
        _, inputs, _ = abi_of(op)
        specs = {sp.stem: sp for sp in _target(op).compile_specs()}
        stems |= set(specs)
        for size in sizes:
            base = [(d, tuple(sh), first_slab + i * slab) for i, (d, sh) in enumerate(build(size))]
            after = first_slab + len(base) * slab
            prepared, error = prepare(op, [FakeTensor(sh, d, ptr) for d, sh, ptr in base], next_slab=after)
            assert error is None, (key, size, error)
            intended = prepared.spec
            by_abi_base = dict(prepared.tensors)  # the allocated Out tensors of the valid call
            inout = [i for i, name in enumerate(inputs) if any(a.name == name and a.access == "inout" for a in intended.args)]
            add(intended.stem, abi_tensors(intended, by_abi_base), {k: v for k, v in verdict(prepared, None).items() if k != "stem"})
            calls = []  # every call: the valid one with a single change
            for j, (d, sh, ptr) in enumerate(base):
                for _, new in _mutations(d, sh, ptr):
                    calls.append([*base[:j], new, *base[j + 1 :]])
                for i in inout:
                    if i != j:
                        calls.append([*base[:j], (d, sh, base[i][2]), *base[j + 1 :]])
            if key == "bucketed_rms":  # the bucketed dimension moved in every tensor that has it: another bucket, or none
                rows = size[0]
                for db in (95, 100, 512, 1000):
                    calls.append([(base[0][0], (rows, db), base[0][2]), (base[1][0], (db,), base[1][2])])
            for call in calls:
                prepared, error = prepare(op, [FakeTensor(sh_, d_, p_) for d_, sh_, p_ in call], next_slab=after)
                got = verdict(prepared, error)
                if prepared is not None:
                    # a binary Python selected: a valid call of that binary, and a bucket refusal of the one intended
                    selected = prepared.spec
                    add(selected.stem, abi_tensors(selected, dict(prepared.tensors)), {k: v for k, v in got.items() if k != "stem"})
                    if selected.stem != intended.stem:
                        by_name = {inputs[i]: FakeTensor(sh_, d_, p_) for i, (d_, sh_, p_) in enumerate(call)}
                        outs = allocated_outs(op, call, after, {n: t for n, t in by_abi_base.items() if n not in by_name})
                        add(intended.stem, abi_tensors(intended, {**outs, **by_name}), {"ok": False, "kind": "bucket"})
                else:
                    by_name = {inputs[i]: FakeTensor(sh_, d_, p_) for i, (d_, sh_, p_) in enumerate(call)}
                    outs = allocated_outs(op, call, after, {n: t for n, t in by_abi_base.items() if n not in by_name})
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
    ):
        (directory / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
