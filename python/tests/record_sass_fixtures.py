"""Record the nvdisasm listings of ``tests/sass_fixtures`` (``test_sass.py``): a few kernels for sm_90, sm_100, sm_103 and sm_120, read by
CUDA 13.x's nvdisasm (and 12.8's, Triton's, where it decodes). From python/: PYTHONPATH=src python tests/record_sass_fixtures.py OUT NVDISASM_13."""
import json
import pathlib
import re
import subprocess
import sys
import tempfile

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tilefish import _sass

OUT = pathlib.Path(sys.argv[1]) if __name__ == "__main__" else pathlib.Path(".")  # (the kernels below stay on the lines the fixtures name)
TOOLS13 = sys.argv[2] if __name__ == "__main__" and len(sys.argv) > 2 else "nvdisasm"
TOOLS = {"13": TOOLS13, "12": str(triton.knobs.nvidia.nvdisasm.path)}


@triton.jit
def dot_bf16(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(a_ptr + m[:, None] * K + k[None, :])
    b = tl.load(b_ptr + k[:, None] * N + n[None, :])
    acc = tl.dot(a, b)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc)


@triton.jit
def dot_tf32(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(a_ptr + m[:, None] * K + k[None, :])
    b = tl.load(b_ptr + k[:, None] * N + n[None, :])
    acc = tl.dot(a, b, input_precision="tf32")
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc)


@triton.jit
def spill(x_ptr, out_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    x = tl.load(x_ptr + o[:, None] * B + o[None, :])
    y = tl.dot(x, x)
    tl.store(out_ptr + o[:, None] * B + o[None, :], tl.dot(y, x) + y)


@triton.jit
def atomics(x_ptr, acc_ptr, d_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + o, mask=o < n)
    tl.atomic_add(acc_ptr, tl.sum(x, 0))
    tl.store(d_ptr + o, x.to(tl.float64) * 3.0, mask=o < n)



CASES = [
    ("dot_bf16", dot_bf16, {"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32", "M": "constexpr", "N": "constexpr", "K": "constexpr"}, {"M": 128, "N": 64, "K": 32}, {}),
    ("dot_tf32", dot_tf32, {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "constexpr", "N": "constexpr", "K": "constexpr"}, {"M": 128, "N": 64, "K": 16}, {}),
    ("spill", spill, {"x_ptr": "*fp32", "out_ptr": "*fp32", "B": "constexpr"}, {"B": 64}, {"maxnreg": 32}),
    ("atomics", atomics, {"x_ptr": "*fp32", "acc_ptr": "*fp32", "d_ptr": "*fp64", "n": "i32", "BLOCK": "constexpr"}, {"BLOCK": 256}, {}),
]
KEEP = {"mma", "local", "tma", "atomic", "fp64"}


def trim(text: str, window: int = 3, head: int = 16, budget: int = 110) -> str:
    lines = text.splitlines()
    first = next((i for i, l in enumerate(lines) if l.strip().startswith(".section") and ".text." in l), len(lines))
    keep = set(range(min(first + 8, len(lines))))
    ins = [i for i, l in enumerate(lines) if _sass._INSTR.match(l)]
    for i in ins[:head]:
        keep.add(i)
    hits = []
    for i in ins:
        m = _sass._INSTR.match(lines[i])
        b = _sass._BODY.match(_sass._SCHED.sub("", m.group(2)).strip())
        if b and _sass.class_of_opcode(b.group(2)) in KEEP:
            hits.append(i)
    for i in hits:
        if len(keep) > budget:
            break
        keep.update(range(max(first, i - window), min(len(lines), i + window + 1)))
    for i in hits:  # the instruction that sets the descriptor register of a tcgen05 MMA (UMOV URn, 0x...), so that its kind can be read
        reg = _sass._IDESC.search(lines[i])
        sets = [j for j in ins if j < i and reg is not None and re.search(rf"\bUMOV {reg.group(1)}, 0x", lines[j])]
        if sets:
            keep.add(sets[-1])
    info = [i for i, line in enumerate(lines) if _sass._LINE_INFO.match(line)]
    for i in [k for k in keep if _sass._INSTR.match(lines[k])]:  # the line information an instruction is under, wherever it was printed
        before = [j for j in info if first < j < i]
        if before:
            keep.add(before[-1])
    out = []
    last = -2
    for i in sorted(keep):
        if i != last + 1 and out:
            out.append("        // [trimmed]")
        out.append(lines[i])
        last = i
    return "\n".join(out) + "\n"


_ABSOLUTE = re.compile(r'File "(/[^"]*)"')


def scrub(text: str) -> str:
    """The listing with each absolute source path named as a build of tilefish names it: the module-relative tail (``triton/language/standard.py``),
    or the file's own name (this script, run as ``__main__``)."""

    def name(m: "re.Match[str]") -> str:
        path = m.group(1)
        tail = path.split("/site-packages/", 1)[1] if "/site-packages/" in path else path.rsplit("/", 1)[-1]
        return f'File "{tail}"'

    return _ABSOLUTE.sub(name, text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for cap in (90, 100, 103, 120):
        for name, fn, sig, consts, opts in CASES:
            try:
                k = triton.compile(ASTSource(fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", cap, 32), options={"num_warps": 4, **opts})
            except Exception as e:  # noqa: BLE001
                print("skip", cap, name, type(e).__name__, str(e).splitlines()[0][:200])
                continue
            with tempfile.TemporaryDirectory() as d:
                path = pathlib.Path(d) / "k.cubin"
                path.write_bytes(k.asm["cubin"])
                for tag, tool in TOOLS.items():
                    if tag == "12" and cap not in (100, 120):  # 12.8 for the family base of sm_103 and for its scheduling annotations on sm_120
                        continue
                    g = subprocess.run([tool, *_sass.LISTING_FLAGS, str(path)], capture_output=True, text=True)
                    if g.returncode != 0 or "Cannot decode" in g.stderr:
                        print("cannot decode", cap, name, tag)
                        continue
                    gp = subprocess.run([tool, *_sass.PTX_LINE_FLAGS, str(path)], capture_output=True, text=True)
                    base = f"{name}.sm_{cap}.nvdisasm{tag}"
                    full = _sass.parse_listing(g.stdout, gp.stdout)
                    fn_name = next(iter(full.functions))
                    s = _sass.summary(full.instructions(fn_name))
                    small = scrub(trim(g.stdout))
                    (OUT / f"{base}.g.txt").write_text(small)
                    if name == "dot_bf16" and tag == "13":
                        (OUT / f"{base}.gp.txt").write_text(scrub(trim(gp.stdout)))
                    part = _sass.summary(_sass.parse_listing(small).instructions(fn_name))
                    meta[base] = {
                        "function": fn_name,
                        "target": full.target,
                        "full": {"instructions": s["instructions"], "mma": s["mma"], "tf32_mma": s["tf32_mma"], "spills": s["spills"]},
                        "trimmed": {"instructions": part["instructions"], "classes": part["classes"], "mma": part["mma"], "tf32_mma": part["tf32_mma"], "spills": part["spills"]},
                        "attributes": _sass.kernel_attributes(k.asm["cubin"], fn_name),
                        "ptx_mma": sorted({op for _, op in _sass.ptx_instructions(k.asm["ptx"])}),
                    }
                    print("ok", base, len(small.splitlines()), "lines", s["mma"][:1], s["spills"])
    (OUT / "expected.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
