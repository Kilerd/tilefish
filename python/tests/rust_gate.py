"""The byte-identity gate between ``tilefish.run(..., mode="replay")`` and the Rust host, on a GPU. Not collected: the name
has no ``test_`` prefix.

The claim is that the Rust host runs the exact machine code Python's ``replay`` runs, so the same inputs give the same
bytes. The gate has three steps, on a machine with a GPU, the ``gpu`` extra and a Rust toolchain::

    python rust_gate.py prepare WORK                  # 1. Python: export (or take --manifest DIR), emit, run replay
    TILEFISH_GATE_GENERATED=WORK/generated.rs TILEFISH_GATE_DIR=WORK \\
        cargo test -p tilefish --features gpu-gate --test gpu_gate -- --test-threads=1 --nocapture
    python rust_gate.py compare WORK                  # 3. Python: compare the bytes, tensor by tensor

``prepare`` exports every operator of ``host_ops.ops()``, ``host_ops.gpu_ops()`` and ``host_ops.library_ops()`` (or reads ``--manifest``, for example the
committed fixture, with ``--pin``: a case whose stem the manifest does not have is skipped), writes the generated launchers to ``WORK/generated.rs``
(with the manifest and the cubins embedded), and for each case generates inputs, writes them as raw bytes, runs ``replay`` *with a pin* and
writes what it left in every tensor, outputs and arguments, so that a tensor that is updated in place is compared too. ``compare`` then
requires the files the Rust test wrote to be equal, byte for byte, to those. The cases:

* the five goldens, a bucketed row kernel over each of its buckets, the hand-written rmsnorm and exact gelu, an in-place axpy and a ``tl.dot``
  matrix product with more than 48 KiB of dynamic shared memory, over several sizes and seeds;
* **strided views** (``gpu_kernels_p1.py``, as ``test_gpu_strided.py``): the q, k or v of a fused qkv, a flat view that starts at any element
  (aligned and not), a transposed view, an expanded one (stride 0), a contiguous tensor, a row softmax over a permuted view, 64-bit strides;
* **device scalars**: a mask whose row count is a tensor of shape ``()`` or ``(1,)``, a lowered kernel that reads one and one that reads two;
* **prebuilt binaries**: the PTX of a Triton build declared as a prebuilt extern (checked here to give the bytes of the Triton path), hand-written
  PTX with a float fixed argument and with a 64-bit stride, and a cubin;
* **an extern's contraction** (``gpu_kernels_p4.py``): ``y = a * b + c`` declared with ``enable_fp_fusion=False`` (checked here to be torch's two
  roundings) and ``True`` (one rounding);
* **strides that several views share** (``gpu_kernels_p4.py``): ``o = q + k + v`` over the q, k and v of one fused tensor, read with one stride pair
  (``StrideOf(("q", "k", "v"), axis)``), and over three tensors of their own with the same strides; the plan ``fused_qkv`` binds the three views in one
  step;
* **an Out that may be written over an input** (``gpu_kernels_p5.py``): ``add_into`` launched with its outputs apart; the plan ``residual_chain``
  (without and with reuse) writes both of its calls in place, over the residual it made in the arena; the plan ``copy_steps`` has two copy steps
  (``tilefish.copy`` of a dense view into the arena, a store into an ``InOut`` parameter), launched and captured; the plan ``view_returns`` returns a
  reshape of a result (the caller's output is its buffer), a dense view of a parameter and the result again (copy steps);
* **a dimension only an Out tensor determines** (``gpu_kernels_p4.py``): a ``[B]`` mask made from a device scalar, run with ``outputs=`` (which is what
  binds ``B``), bucketed and not; the plan ``mask_rows`` makes it in a step whose ``B`` is the graph's;
* **pinned cuBLASLt GEMMs** (``gemm`` and ``gemm_bias``, ``M`` in 32, 64 and 320, five ``(N, K)`` shapes), run through ``tilefish.run`` of the library op;
* **launch plans** (:func:`prepare_plans`, ``WORK/plans.json``): the transformer block of ``plan_model`` (``M`` in 32, 64 and 128, two values of the device
  scalar each), without and with reuse, the four-layer unrolled forward (both), the graph of prebuilt binaries and the graph with an in-place tensor,
  exported and run through Python's plan replay; ``crates/tilefish/tests/gpu_plan_gate.rs`` launches and captures each case, and ``compare`` holds the
  Rust plan's launch and its CUDA graph replay to Python's bytes.

``WORK/cases.json`` is what ``crates/tilefish/tests/gpu_gate.rs`` reads::

    {"arch": "sm_103", "digest": ..., "manifest": "manifest", "cublaslt_version": 120804 (when there are library entries),
     "cases": [{"id", "launcher": <the Rust struct>, "stem", "kind": "kernel" | "gemm", "tensors": [...]}]}

``tensors`` are the pointers of the ABI, in order (for a GEMM: ``a``, ``b``, ``bias`` when it has one, ``out``: the workspace is the Rust test's to allocate).
Each is ``{"name", "role": "in" | "out" | "inout", "dtype", "shape", "nbytes", "expect", "input"?}``: the files hold the contiguous bytes of the tensor. A
**view** (a strided pointer given a tensor that is a view, or not contiguous, or does not start at its storage) has also ``"strides"`` (the element strides
as torch reports them) and ``"offset"`` (``storage_offset()``, in elements), and its files, and ``nbytes``, are the bytes of the **whole storage** the view is
over (all of ``qkv``, the whole flat buffer, the small storage of an expanded one), so that the Rust test uploads the storage and builds the view over it. A
device scalar has the shape ``[]`` (or ``[1]``) and the size of its element. ``expect`` of an ``out`` tensor is what ``replay`` returned; of an ``in`` tensor, the
storage after the run (unchanged). An ``out`` tensor that is legitimately all zeros has ``"may_be_zero": true``.
"""

import argparse
import dataclasses
import json
import pathlib
import shutil
import sys
from typing import Any, Callable

import torch
from gpu_util import randn, same_bytes
from host_ops import GEMM_SHAPES, abi_of, gemm_name, gpu_ops, library_ops, ops

import tilefish
from tilefish import runtime
from tilefish.aot import Manifest
from tilefish.hosts import rust

BF16, F16, F32, I32 = torch.bfloat16, torch.float16, torch.float32, torch.int32
SEEDS = (0, 1, 2)
ROW_SIZES = (1, 7, 64, 129)
FLAT_SIZES = (16, 1040, 65552, 1 << 20)
MM_SIZES = ((128, 128, 64), (256, 256, 256), (384, 128, 192), (128, 512, 128))  # (M, N, K): M and N multiples of 128, K of 64
PTX_SIZES = (1, 17, 1000, 4096, 100003)  # the hand-written axpy: any size
QKV_ROWS = (1, 7, 40)
FLAT_OFFSETS = (0, 1, 2, 3, 5, 8, 9)  # head_copy and head_copy64: starts at any element, 16-byte aligned or not
ALIGNED_OFFSETS = (0, 8, 16)  # head_copy16: a start on 16 bytes
SOFTMAX3D = ((5, 3), (16, 9), (2, 33))  # (a, b): a [b, a, 128] tensor permuted to [a, b, 128]
MASKED_N = (0, 1, 5, 17, 63, 64, 40)
SCALES = (0.5, -3.25, 1e-3)
GEMM_M = (32, 64, 320)

DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32", torch.int32: "i32"}


# -- the inputs of the cases ----------------------------------------------------------------------------------------------------------------------


def args_softmax(rows, seed):
    return (randn((rows, 1000), F32, seed, scale=3.0),)


def args_layernorm(rows, seed):
    return (randn((rows, 768), BF16, seed), randn((768,), BF16, seed + 100), randn((768,), BF16, seed + 200, scale=0.1))


def args_residual_add_rmsnorm(rows, seed):
    return (randn((rows, 2560), BF16, seed), randn((rows, 2560), F32, seed + 50), randn((2560,), BF16, seed + 100))


def args_gelu(n, seed):
    return (randn((n,), BF16, seed, scale=3.0),)


def args_swiglu(rows, seed):
    return (randn((rows, 3072), BF16, seed, scale=2.0), randn((rows, 3072), BF16, seed + 100))


def args_bucketed(d):
    return lambda rows, seed: (randn((rows, d), BF16, seed), randn((d,), BF16, seed + 100))


def args_rmsnorm(rows, seed):
    return (randn((rows, 2048), BF16, seed), randn((2048,), BF16, seed + 100))


def args_axpy(n, seed):
    return (randn((n,), F32, seed), randn((n,), F32, seed + 100))


def args_mm(size, seed):
    m, n, k = size
    return (randn((m, k), F16, seed, scale=0.5), randn((k, n), F16, seed + 100, scale=0.5))


def args_qkv(spec, seed):
    """``qkv[:, which]``: a view of a fused qkv, with strides ``(1536, 64, 1)`` that starts ``which * 512`` elements in."""
    rows, which = spec
    return (randn((rows, 3, 8, 64), BF16, seed * 100 + rows * 10 + which)[:, which],)


def args_flat_view(offset, seed):
    """A view that starts ``offset`` elements into a flat buffer: 16-byte aligned only when ``offset`` is a multiple of 8."""
    flat = randn((40 * 8 * 64 + 32,), BF16, seed * 100 + offset)
    return (flat[offset : offset + 40 * 8 * 64].view(40, 8, 64),)


def args_layout(kind, seed):
    if kind == "transposed":  # strides (64, 2560, 1)
        return (randn((8, 40, 64), BF16, seed + 3).permute(1, 0, 2),)
    if kind == "expanded":  # stride 0 on the first axis, over a storage of 1024 bytes
        return (randn((1, 8, 64), BF16, seed + 4).expand(40, 8, 64),)
    return (randn((40, 8, 64), BF16, seed + 5),)  # a contiguous tensor given to a pointer that takes a view


def args_softmax3d(spec, seed):
    a, b = spec
    return (randn((b, a, 128), F32, seed + a * 100 + b, scale=3.0).permute(1, 0, 2),)


def n_tok(value, shape=()):
    return torch.full(shape, value, dtype=I32, device="cuda")


def args_masked(spec, seed):
    value, shape = spec
    return (randn((64, 128), F32, seed + value), n_tok(value, shape))


def args_scale_by(spec, seed):
    rows, scale = spec
    return (randn((rows, 256), F32, seed + rows), torch.full((), scale, dtype=F32, device="cuda"))


def args_scale_flat(n, seed):
    return (randn((n,), F32, seed + n), torch.full((), 1.5, dtype=F32, device="cuda"), torch.full((), -0.25, dtype=F32, device="cuda"))


def args_strided_copy(spec, seed):
    rows, first = spec
    return (randn((rows, 200), BF16, seed + rows)[:, first : first + 64],)  # [rows, 64], strides (200, 1), a start of `first` elements


def operands(spec, seed):
    m, n, k, bias = spec
    a = randn((m, k), BF16, seed + m)
    b = randn((n, k), BF16, seed + m + 1, scale=1 / k**0.5)
    return (a, b, randn((n,), BF16, seed + m + 2)) if bias else (a, b)


# -- the cases -------------------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Case:
    """One call: the operator (a key of ``host_ops``), the stem of its binary, how to make its arguments and what else to hold true of the result."""

    op: str
    stem: str
    label: str
    make: Callable[[], tuple]
    check: "Callable[[tuple, Any], None] | None" = None  # ``(args, result)``: something Python knows about the answer, beyond what Rust must reproduce
    may_be_zero: "tuple[str, ...]" = ()  # the ``out`` tensors that are legitimately all zeros
    kind: str = "kernel"
    same_as: "str | None" = None  # another operator (a key of ``host_ops``) whose replay on the same inputs gives these very bytes
    outputs: "Callable[[tuple], tuple] | None" = None  # the tensors the call writes its results into (``outputs=``), made from its arguments

    @property
    def id(self) -> str:
        return f"{self.stem}-{self.label}"


def sized(size):
    return "x".join(map(str, size)) if isinstance(size, tuple) else str(size)


def plain(key, stem, make, sizes, seeds=SEEDS, **kw):
    return [Case(key, stem, f"{sized(size)}-seed{seed}", lambda size=size, seed=seed: make(size, seed), **kw) for size in sizes for seed in seeds]


def must_be_the_copy(args, got):
    assert same_bytes(got, args[0].contiguous())


def must_be_a_softmax(args, got):
    assert (got - torch.softmax(args[0], dim=-1)).abs().max() < 1e-6


def must_be_masked(args, got):
    x, n = args
    want = x * 2.0
    want[int(n.reshape(())) :] = 0.0
    assert same_bytes(got, want)


def must_scale(args, got):
    x, s = args
    assert same_bytes(got, x * s)


def must_round_twice(args, got):
    a, b, c = args
    assert same_bytes(got, torch.mul(a, b).add(c))


def must_round_once(args, got):
    a, b, c = args
    assert same_bytes(got, (a.double() * b.double() + c.double()).to(F32))


def args_fma(n, seed):
    return (randn((n,), F32, seed), randn((n,), F32, seed + 100), randn((n,), F32, seed + 200))


def args_qkv_fused(rows, seed):
    """The q, k and v of one fused ``[rows, 3, 4, 64]`` tensor: offsets 0, 256 and 512 elements, strides ``(768, 64, 1)`` each."""
    fused = randn((rows, 3, 4, 64), BF16, seed * 100 + rows)
    return fused[:, 0], fused[:, 1], fused[:, 2]


def args_qkv_separate(rows, seed):
    """Three tensors of their own: the same strides ``(256, 64, 1)``, three buffers."""
    return tuple(randn((rows, 4, 64), BF16, seed * 100 + rows + i) for i in range(3))


def must_sum(args, got):
    q, k, v = args
    assert same_bytes(got, ((q.float() + k.float()) + v.float()).to(BF16))


def must_mask(args, got):
    (n,) = args
    keys = torch.arange(got.shape[0], device=got.device)
    assert same_bytes(got, torch.where(keys < n.reshape(()), 0.0, float("-inf")).to(BF16))


def mask_of(b):
    """The ``Out`` tensor of a mask of ``b`` keys: what binds ``B``."""
    return lambda args: (torch.zeros((b,), dtype=BF16, device="cuda"),)


def args_add_into(rows, seed):
    return randn((rows, 64), F32, seed), randn((rows, 64), F32, seed + 50)


def must_add_into(args, got):
    x, r = args
    y = got  # the first result (the check of a case sees that one): y = 0.5 * (r + x), which needs r_out = r + x
    assert same_bytes(y, (r + x) * 0.5)


def build_cases():
    cases: "list[Case]" = [
        *plain("softmax", "softmax", args_softmax, ROW_SIZES),
        *plain("layernorm", "layernorm", args_layernorm, ROW_SIZES),
        *plain("residual_add_rmsnorm", "residual_add_rmsnorm", args_residual_add_rmsnorm, (1, 3, 33)),
        *plain("gelu_exact", "gelu_exact", args_gelu, FLAT_SIZES),
        *plain("swiglu", "swiglu", args_swiglu, (1, 3, 33, 130)),
        *[c for d in (96, 512, 1000) for c in plain("bucketed_rms", f"bucketed_rms.Db{d}", args_bucketed(d), (1, 5, 40))],
        *plain("rmsnorm", "rmsnorm", args_rmsnorm, (1, 5, 64)),
        *plain("gelu_erf", "gelu_erf", args_gelu, FLAT_SIZES),
        *plain("axpy", "axpy", args_axpy, FLAT_SIZES),
        *plain("mm", "mm", args_mm, MM_SIZES),  # a tl.dot kernel with more than 48 KiB of dynamic shared memory and a 2-D grid
    ]
    # -- strided views: the head split copies y[s, h, :] = x[s, h, :] for a view x of any strides
    qkv = [(rows, which) for rows in QKV_ROWS for which in (0, 1, 2)]
    for op in ("head_copy", "head_copy16", "head_copy64"):
        cases += plain(op, op, args_qkv, qkv, seeds=(0,), check=must_be_the_copy)
    for op, offsets in (("head_copy", FLAT_OFFSETS), ("head_copy64", FLAT_OFFSETS), ("head_copy16", ALIGNED_OFFSETS)):
        cases += plain(op, op, args_flat_view, offsets, seeds=(0,), check=must_be_the_copy)
    for op in ("head_copy", "head_copy16", "head_copy64"):
        cases += plain(op, op, args_layout, ("transposed", "expanded", "contiguous"), seeds=(0,), check=must_be_the_copy)
    cases += plain("softmax3d", "softmax3d", args_softmax3d, SOFTMAX3D, seeds=(0,), check=must_be_a_softmax)
    # -- device scalars
    cases += [
        Case("masked_rows", "masked_rows", f"n{value}-seed0", lambda value=value: args_masked((value, ()), 0), must_be_masked, ("y",) if value == 0 else ())
        for value in MASKED_N
    ]
    cases.append(Case("masked_rows", "masked_rows", "n7-shape1-seed0", lambda: args_masked((7, (1,)), 0), must_be_masked))
    cases += [
        Case("scale_by", "scale_by", f"{rows}x{scale}-seed0", lambda rows=rows, scale=scale: args_scale_by((rows, scale), 0), must_scale)
        for rows in (1, 20, 65)
        for scale in SCALES
    ]
    cases += plain("scale_flat", "scale_flat", args_scale_flat, (16, 4096, 65536), seeds=(0,))
    # -- prebuilt binaries
    cases += plain("gelu_ptx", "gelu_ptx", args_gelu, (16, 1024, 4096, 100000 - 100000 % 16), seeds=(0, 1), same_as="gelu_exact")
    cases += plain("softmax_ptx", "softmax_ptx", args_softmax, (1, 9, 130), seeds=(0, 1), same_as="softmax")
    cases += plain("axpy_ptx", "axpy_ptx", args_axpy, PTX_SIZES, seeds=(0, 1), check=lambda args, got: same_bytes(got, 2.0 * args[0] + args[1]))
    cases += plain("axpy_cubin", "axpy_cubin", args_axpy, PTX_SIZES, seeds=(0, 1), same_as="axpy_ptx")
    cases += plain("strided_copy", "strided_copy", args_strided_copy, [(rows, first) for rows in (1, 7, 33) for first in (0, 3, 8)], seeds=(0,), check=must_be_the_copy)
    # -- an extern's contraction
    cases += plain("fma_plain", "fma_plain", args_fma, (16, 1040, 65539), seeds=(0, 1), check=must_round_twice)
    cases += plain("fma_fused", "fma_fused", args_fma, (16, 1040, 65539), seeds=(0, 1), check=must_round_once)
    # -- strides that several views share
    for op in ("qkv_sum", "qkv_sum16"):
        cases += [Case(op, op, f"fused-{rows}-seed0", lambda rows=rows: args_qkv_fused(rows, 0), must_sum) for rows in (1, 7, 40)]
        cases += [Case(op, op, f"separate-{rows}-seed0", lambda rows=rows: args_qkv_separate(rows, 0), must_sum) for rows in (1, 7, 40)]
    # -- a dimension only the Out tensor determines: B is the size of the mask the call is given
    for op, sizes in (("key_mask", (8, 24)), ("key_mask_any", (1, 130, 1000))):
        cases += [
            Case(op, op, f"B{b}-n{n}", lambda n=n: (n_tok(n),), must_mask, may_be_zero=("mask",) if n >= b else (), outputs=mask_of(b))
            for b in sizes
            for n in (0, 5, b, b + 3)
        ]
    # -- an Out that may be written over an input (MayAlias): launched apart here; in place in the plan residual_chain
    cases += plain("add_into", "add_into", args_add_into, (8, 24), seeds=(0, 1), check=must_add_into)
    return cases


def gemm_cases():
    return [
        Case(
            gemm_name(n, k, bias),
            f"{gemm_name(n, k, bias)}.M{m}",
            f"seed{seed}",
            lambda spec=(m, n, k, bias), seed=seed: operands(spec, seed),
            kind="gemm",
        )
        for n, k in GEMM_SHAPES
        for bias in (False, True)
        for m in GEMM_M
        for seed in (0,)
    ]


# -- views and bytes -------------------------------------------------------------------------------------------------------------------------


def raw(t):
    """The bytes of a tensor as one contiguous tensor (a tensor of any rank, a scalar too)."""
    return t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def is_view(t):
    """Whether Rust is given the storage and builds the view over it (anything that is not a plain tensor of its own contiguous storage)."""
    return bool(t._is_view() or not t.is_contiguous() or t.storage_offset() != 0)


def storage_bytes(t):
    """The bytes of the whole storage that ``t`` is a view of."""
    storage = t.untyped_storage()
    everything = torch.empty(0, dtype=torch.uint8, device=t.device).set_(storage, 0, (storage.nbytes(),))
    return everything.cpu().numpy().tobytes()


def describe(t):
    """``(fields of its tensor entry, the bytes of its files)``: a plain tensor, or a view of its whole storage."""
    if is_view(t):
        data = storage_bytes(t)
        return {"shape": list(t.shape), "strides": [int(s) for s in t.stride()], "offset": int(t.storage_offset()), "nbytes": len(data)}, data
    data = raw(t)
    return {"shape": list(t.shape), "nbytes": len(data)}, data


def expected_pin(manifest_dir):
    """The digest ``expected.json`` (next to the fixture directories, which the Rust tests check as well) has for a fixture directory."""
    expected = pathlib.Path(manifest_dir).resolve().parent / "expected.json"
    try:
        return json.loads(expected.read_text())["fixtures"][pathlib.Path(manifest_dir).resolve().name]["manifest_digest"]
    except (OSError, KeyError, ValueError):
        sys.exit(f"--manifest needs --pin <the manifest's digest> ({expected} has none for {manifest_dir})")


def library_abi(op):
    """``(tensors, inputs, outputs)`` of a library op, like ``abi_of``: the pointers ``a``, ``b``, ``bias`` when it has one, and ``out``."""
    names = [p.name for p in op.signature.params]
    return [(n, "in") for n in names] + [("out", "out")], names, ["out"]


def prepare(work, manifest_dir=None, pin=None):
    work = pathlib.Path(work).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    major, minor = torch.cuda.get_device_capability()
    cap = major * 10 + minor
    arch = f"sm_{cap}"
    sources = work / "sources"
    if manifest_dir is not None:  # the prebuilt sources the manifest was made from (the PTX of a Triton build, a cubin): declared as they are
        committed = pathlib.Path(manifest_dir).resolve().parent / "sources" / pathlib.Path(manifest_dir).resolve().name
        if committed.is_dir():
            shutil.copytree(committed, sources)
    declared = {**ops(), **gpu_ops(sources, cap, reuse=manifest_dir is not None), **library_ops()}
    all_ops = list(declared.values())
    if manifest_dir is None:
        manifest = runtime.export(all_ops, work / "manifest", arch=cap)
        manifest_ref = "manifest"
    else:
        manifest = Manifest.load(manifest_dir, pin=pin or expected_pin(manifest_dir))  # the emitter and the replay both take a pinned manifest
        manifest_ref = str(pathlib.Path(manifest_dir).resolve())
    digest = manifest.digest
    directory = manifest_dir or work / "manifest"
    # embedded: the generated module also carries the manifest and the cubins (include_bytes!), which the Rust test runs too
    rust.emit(manifest, work / "generated.rs", ops=all_ops, embed=True)
    cases, skipped, tensors_total = [], [], 0
    for case in [*build_cases(), *gemm_cases()]:
        if manifest.get(case.stem, arch) is None and manifest.library_entry(case.stem, arch) is None:
            skipped.append(case.stem)
            continue
        op = declared[case.op]
        library = case.kind == "gemm"
        tensors, inputs, outputs = library_abi(op) if library else abi_of(op)
        (work / "cases" / case.id).mkdir(parents=True)
        args = case.make()
        # an argument that is a view is given as it is (a strided pointer only reads); a plain tensor is cloned, so an in-place one keeps its input
        given = [a if is_view(a) else a.clone() for a in args]
        extra = {} if case.outputs is None else {"outputs": case.outputs(args)}
        result = tilefish.run(op, *given, mode="replay", manifest=directory, pin=digest, **extra)
        results = [] if result is None else (list(result) if isinstance(result, tuple) else [result])
        assert len(results) == len(outputs), (case.id, len(results), outputs)
        if library:  # the pin made in memory runs the algorithm the manifest records: the same bytes
            assert same_bytes(results[0], tilefish.run(op, *given)), f"{case.id}: replay and jit differ"
        verify(case, declared, args, results, directory, digest)
        post = {**dict(zip(inputs, given)), **dict(zip(outputs, results))}
        entries = []
        for name, access in tensors:
            t = post[name]
            fields, expect = describe(t)
            entry = {"name": name, "role": access, "dtype": DTYPES[t.dtype], **fields, "expect": f"cases/{case.id}/{name}.expect.bin"}
            (work / entry["expect"]).write_bytes(expect)
            if access != "out":
                entry["input"] = f"cases/{case.id}/{name}.in.bin"
                (work / entry["input"]).write_bytes(describe(args[inputs.index(name)])[1])
            if name in case.may_be_zero:
                entry["may_be_zero"] = True
            entries.append(entry)
        tensors_total += len(entries)
        cases.append({"id": case.id, "launcher": rust._camel(case.stem), "stem": case.stem, "kind": case.kind, "tensors": entries})
    if not cases:
        sys.exit(f"no case has a binary in the manifest for {arch}")
    doc = {"arch": arch, "digest": digest, "manifest": manifest_ref}
    versions = sorted({e["cublaslt_version"] for e in manifest.library_entries if e["arch"] == arch})
    if versions:
        doc["cublaslt_version"] = versions[0] if len(versions) == 1 else versions
    doc["cases"] = cases
    (work / "cases.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    note = f" ({len(skipped)} cases skipped: the manifest has no binary of {sorted(set(skipped))[:6]}...)" if skipped else ""
    print(f"prepared {len(cases)} cases ({tensors_total} tensors) for {arch}, manifest digest {digest}{note}", file=sys.stderr)


def verify(case, declared, args, results, directory, digest):
    """What Python holds true of the result, beyond the bytes Rust must reproduce."""
    if case.check is not None:
        case.check(tuple(args), results[0])
    if case.same_as is not None:  # the PTX of a Triton build (or a cubin made from a PTX) gives the bytes of the other path
        other = tilefish.run(declared[case.same_as], *[a if is_view(a) else a.clone() for a in args], mode="replay", manifest=directory, pin=digest)
        assert same_bytes(results[0], other), f"{case.id}: not the bytes of {case.same_as}"


# -- launch plans ------------------------------------------------------------------------------------------------------------------------------


def plan_graphs():
    """``(name, graph, reuse, the arguments of each case)``: the plans the gate exports. A case is ``(label, {parameter: tensor})``, the
    parameters in the order of the graph's signature (a ``DevScalar`` as a CUDA tensor of one element)."""
    import plan_model as pm

    weights = pm.weights()
    n_values = {32: (32, 5), 64: (64, 17), 128: (128, 77)}  # all the keys, and some of them

    def block_cases():
        out = []
        for m in pm.BUCKETS:
            for n in n_values[m]:
                x = pm.activations(m, seed=m + n)
                out.append((f"M{m}-n{n}", {"x": x, "n": torch.tensor([n], dtype=I32, device="cuda"), **weights}))
        return out

    layers = 4
    deep = pm.deep_block(layers)
    _, stacked = pm.stacked_weights(layers)

    def deep_cases():
        out = []
        for m in pm.BUCKETS:
            for n in (m - 3, m // 2):
                x = pm.activations(m, seed=60 + m + n)
                out.append((f"M{m}-n{n}", {"x": x, "n": torch.tensor([n], dtype=I32, device="cuda"), **stacked}))
        return out

    def prebuilt_cases():
        out = []
        for b in (8, 24):
            for seed in (0, 1):
                args = {"x": randn((b, 64), F32, seed=b + seed), "y": randn((b, 64), F32, seed=b + seed + 1), "w": randn((b, 128), BF16, seed=b + seed + 2)}
                out.append((f"B{b}-seed{seed}", args))
        return out

    def accumulate_cases():
        out = []
        for q in (16, 32):
            for seed in (0, 1):
                out.append((f"Q{q}-seed{seed}", {"a": randn((q, 128), F32, seed=q + seed), "state": randn((q, 128), F32, seed=q + seed + 7)}))
        return out

    import gpu_kernels_p4 as p4

    def fused_qkv_cases():
        return [(f"Rb{rb}-seed{seed}", {"x": randn((rb, 768), BF16, seed=rb + seed)}) for rb in (8, 24) for seed in (0, 1)]

    def mask_rows_cases():
        return [
            (f"Rb{rb}-n{n}", {"x": randn((rb, 64), F32, seed=rb + n), "n": torch.tensor([n], dtype=I32, device="cuda")})
            for rb in (8, 24)
            for n in (0, 5, rb)
        ]

    import gpu_kernels_p5 as p5

    def copy_cases():
        return [
            (f"Rb{rb}-seed{seed}", {"x": randn((2, rb, 64), F32, seed=rb + seed), "state": randn((rb, 64), F32, seed=rb + seed + 9)})
            for rb in (8, 24)
            for seed in (0, 1)
        ]

    def view_cases():
        return [(f"Rb{rb}-seed{seed}", {"x": randn((2, rb, 64), F32, seed=rb + seed + 30)}) for rb in (8, 24) for seed in (0, 1)]

    def residual_cases():
        return [(f"Rb{rb}-seed{seed}", {"x": randn((rb, 64), F32, seed=rb + seed)}) for rb in (8, 24) for seed in (0, 1)]

    return [
        ("block", pm.block, False, block_cases),
        ("block-reuse", pm.block, True, block_cases),
        (f"deep{layers}", deep, False, deep_cases),
        (f"deep{layers}-reuse", deep, True, deep_cases),
        ("prebuilt", pm.prebuilt, False, prebuilt_cases),
        ("accumulate", pm.accumulate, False, accumulate_cases),
        ("fused_qkv", p4.fused_qkv, False, fused_qkv_cases),  # one step that reads three views with one stride pair
        ("mask_rows", p4.mask_rows, False, mask_rows_cases),  # a step whose B only its Out buffer determines
        ("residual_chain", p5.residual_chain, False, residual_cases),  # two steps whose Out is written over their input (MayAlias)
        ("residual_chain-reuse", p5.residual_chain, True, residual_cases),
        ("copy_steps", p5.copy_steps, False, copy_cases),  # a copy of a dense view into the arena and a store into an InOut parameter
        ("view_returns", p5.view_returns, False, view_cases),  # a reshape of a result returned (no copy), a dense view of a parameter and a value twice (copies)
    ]


def prepare_plans(work):
    """Export the plans of :func:`plan_graphs` to ``WORK/plans/<name>`` and run every case through ``tilefish.run(..., mode="replay", plan=...)`` with
    the pin; write ``WORK/plans.json`` for ``gpu_gate.rs``::

        {"plans": [{"name", "dir", "digest", "graph", "reuse", "arena_bytes", "cases": [{"id", "bucket": {dim: value}, "devscalars": {name: int},
          "externals": [{"name", "role", "dtype", "shape", "input"?, "expect"?}]}]}]}

    An external of role ``input``, ``weight`` or ``inout`` has ``input`` (its bytes before the call; a weight's file is shared by the cases of a
    plan); an ``output`` and an ``inout`` have ``expect``, the bytes Python's replay left in it (an ``inout`` is updated in place)."""
    work = pathlib.Path(work).resolve()
    root = work / "plans"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    plans = []
    total = 0
    for name, graph, reuse, make_cases in plan_graphs():
        directory = root / name
        exported = runtime.export(graph, directory, reuse=reuse)
        doc = exported.doc
        params = [b for b in doc["buffers"] if b["role"] in ("input", "weight", "inout", "devscalar")]
        shared = {}  # a weight's file, written once per plan
        cases = []
        for label, args in make_cases():
            case_id = f"{name}-{label}"
            case_dir = root / "cases" / case_id
            case_dir.mkdir(parents=True)
            given = [args[b["name"]].clone() for b in params]  # a fresh tensor per call: an inout one is updated in place
            before = {b["name"]: raw(args[b["name"]]) for b in params}
            result = tilefish.run(graph, *given, mode="replay", plan=directory, pin=exported.digest)
            results = list(result) if isinstance(result, tuple) else [result]
            after = {b["name"]: t for b, t in zip(params, given)}
            after.update(zip(doc["outputs"], results))
            values = {}
            dims = {}
            externals = []
            for b in doc["buffers"]:
                role = b["role"]
                if role == "devscalar":
                    values[b["name"]] = int(args[b["name"]].reshape(()).item())
                    continue
                if role not in ("input", "weight", "inout", "output"):
                    continue
                t = after[b["name"]]
                entry = {"name": b["name"], "role": role, "dtype": DTYPES[t.dtype], "shape": list(t.shape)}
                if role in ("input", "weight", "inout"):
                    if role == "weight":
                        if b["name"] not in shared:
                            shared[b["name"]] = f"plans/weights/{name}/{b['name']}.bin"
                            (work / shared[b["name"]]).parent.mkdir(parents=True, exist_ok=True)
                            (work / shared[b["name"]]).write_bytes(before[b["name"]])
                        entry["input"] = shared[b["name"]]
                    else:
                        entry["input"] = f"plans/cases/{case_id}/{b['name']}.in.bin"
                        (work / entry["input"]).write_bytes(before[b["name"]])
                if role in ("inout", "output"):
                    entry["expect"] = f"plans/cases/{case_id}/{b['name']}.expect.bin"
                    (work / entry["expect"]).write_bytes(raw(t))
                externals.append(entry)
            for b in doc["buffers"]:
                for d, e in zip(b["shape"], after[b["name"]].shape if b["name"] in after else ()):
                    if isinstance(d, str) and d in doc["dims"]:
                        dims[d] = int(e)
            bucket = next(bk["values"] for bk in doc["buckets"] if all(bk["values"][k] == v for k, v in dims.items()))
            cases.append({"id": case_id, "bucket": bucket, "devscalars": values, "externals": externals})
            total += 1
        plans.append(
            {"name": name, "dir": f"plans/{name}", "digest": exported.digest, "graph": doc["graph"], "reuse": reuse, "arena_bytes": doc["arena_bytes"], "cases": cases}
        )
    (work / "plans.json").write_text(json.dumps({"plans": plans}, indent=1) + "\n", encoding="utf-8")
    print(f"prepared {len(plans)} plans and {total} cases of plan replay", file=sys.stderr)


def compare_plans(work):
    """The outputs (and in-place tensors) the Rust plan wrote, for its launch (``rust_plans``) and its CUDA graph replay (``rust_plans_graph``), against
    what Python's plan replay left. Returns ``(passed, cases, failures)``."""
    path = work / "plans.json"
    if not path.exists():
        return 0, 0, []
    doc = json.loads(path.read_text(encoding="utf-8"))
    failed, passed, count = [], 0, 0
    for subdir in ("rust_plans", "rust_plans_graph"):
        for plan in doc["plans"]:
            for case in plan["cases"]:
                count += 1
                bad = []
                for t in case["externals"]:
                    if "expect" not in t:
                        continue
                    expect = (work / t["expect"]).read_bytes()
                    got_path = work / subdir / case["id"] / f"{t['name']}.bin"
                    got = got_path.read_bytes() if got_path.exists() else None
                    if got != expect:
                        bad.append(f"{t['name']} ({'missing' if got is None else sum(a != b for a, b in zip(got, expect))} bytes differ)")
                    elif not any(expect):
                        bad.append(f"{t['name']} is all zeros")
                if bad:
                    failed.append((f"{subdir}/{case['id']}", bad))
                else:
                    passed += 1
    return passed, count, failed


def compare(work):
    """Every file the Rust test wrote, from the manifest directory and from the embedded manifest, against what replay left."""
    work = pathlib.Path(work).resolve()
    doc = json.loads((work / "cases.json").read_text(encoding="utf-8"))
    failed, tensors, total = [], 0, 0
    passed = {}
    for subdir in ("rust", "rust_embedded"):
        passed[subdir] = 0
        for case in doc["cases"]:
            bad = []
            for t in case["tensors"]:
                expect = (work / t["expect"]).read_bytes()
                path = work / subdir / case["id"] / f"{t['name']}.bin"
                got = path.read_bytes() if path.exists() else None
                tensors += 1
                total += len(expect)
                if got != expect:
                    bad.append(f"{t['name']} ({'missing' if got is None else sum(a != b for a, b in zip(got, expect))} bytes differ)")
                elif t["role"] == "out" and not t.get("may_be_zero") and not any(expect):
                    bad.append(f"{t['name']} is all zeros: the kernel wrote nothing")
            if bad:
                failed.append((f"{subdir}/{case['id']}", bad))
            else:
                passed[subdir] += 1
    for case_id, bad in failed:
        print(f"FAIL {case_id}: {'; '.join(bad)}")
    n = len(doc["cases"])
    print(
        f"rust vs replay: {passed['rust']} of {n} cases byte-identical with the manifest directory, "
        f"{passed['rust_embedded']} of {n} with the embedded manifest ({tensors} tensors, {total} bytes compared)"
    )
    plan_passed, plan_cases, plan_failed = compare_plans(work)
    for case_id, bad in plan_failed:
        print(f"FAIL {case_id}: {'; '.join(bad)}")
    if plan_cases:
        print(f"rust plan vs python plan replay: {plan_passed} of {plan_cases} cases byte-identical (the launch and the CUDA graph replay of each case)")
    return 0 if not failed and not plan_failed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("work", type=pathlib.Path)
    p.add_argument("--manifest", type=pathlib.Path, help="use this manifest directory instead of exporting one")
    p.add_argument("--pin", help="the digest of --manifest (default: the one expected.json next to a fixture directory has)")
    c = sub.add_parser("compare")
    c.add_argument("work", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.work, args.manifest, args.pin)
        # The GEMM whose pinned algorithm needs a workspace (gpu_gate.rs reads WORK/workspace_case); after prepare, which empties WORK.
        import rust_workspace_case

        rust_workspace_case.main(args.work)
        prepare_plans(args.work)  # the launch plans (gpu_gate.rs reads WORK/plans.json)
        return 0
    return compare(args.work)


if __name__ == "__main__":
    sys.exit(main())
