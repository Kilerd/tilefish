"""A prebuilt PTX extern and its fake build, for the tests that need no GPU (not collected: the name has no ``test_`` prefix).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import hashlib
import pathlib
import stat

from elf_builder import arch_header, cubin_like

import tilefish
from tilefish import Dim, Out, Tensor, const, f32, i64
from tilefish._build import BuiltKernel, LaunchAttributes
from tilefish._elf import code_digest

n = Dim("n", divisible_by=16)
R = Dim("R")

PTX_TEXT = """\
// a stand-in: the tests of the manifest need an entry point and parameters, not code
.version 7.0
.target sm_90
.address_size 64

.visible .entry tf_k(
	.param .u64 x,
	.param .u64 y,
	.param .u32 n,
	.param .f32 alpha,
	.param .u64 scratch
)
{
	ret;
}
"""

# the cubin ptxas makes for sm_90 (EI_OSABI 0x33, the SM in the low byte of e_flags) and for sm_103a (EI_OSABI 0x41, the SM in the second byte)
SM90 = dict(osabi=0x33, abiversion=7, flags=0x0050055A)
SM103A = dict(osabi=0x41, abiversion=8, flags=0x0600670A)


def write_ptx(directory, name="k.ptx", text=PTX_TEXT):
    path = pathlib.Path(directory) / name
    path.write_text(text)
    return path


def make_ptx_op(path, *, name="axpy", **kw):
    """``name(x, y: Out, n_elem, alpha: const[f32], scratch: const[i64])``: the C ABI of PTX_TEXT."""

    @tilefish.extern(ptx=path, symbol="tf_k", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name=name, **kw)
    def op(x: Tensor[n, f32], y: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0, scratch: const[i64] = 0): ...

    return op


def fake_binary_build(spec, target, *, arch="sm_90", cubin=None, flags=("-lineinfo", "-v", "--gpu-name=sm_90"), **kw):
    """A ``BuiltKernel`` of a prebuilt-binary ``spec``, around a small ELF file whose entry point is ``tf_k`` (a declared cubin is used as it is, as
    ``build_binary`` does: its bytes are the build); ``kw`` overrides any field."""
    key = "ptx" if spec.binary["kind"] == "ptx" else arch
    if cubin is None and spec.binary["kind"] == "cubin" and key in target.sources:
        cubin = target.sources[key]
    cubin = cubin_like(shared=None, **arch_header(arch)) if cubin is None else cubin  # (a prebuilt binary may have static shared memory: the section is no sign of dynamic use)
    build = {"source": "ptx", "ptxas_flags": list(flags)} if spec.binary["kind"] == "ptx" else {"source": "cubin"}
    values = dict(
        spec=spec,
        arch=arch,
        cubin=cubin,
        ptx="",
        symbol=spec.binary["symbol"],
        shared=spec.binary["shared"],
        num_warps=0,
        sha256=hashlib.sha256(cubin).hexdigest(),
        sha256_code=code_digest(cubin),
        triton_version="",
        ptxas_version="Cuda compilation tools, release 12.9, V12.9.86" if spec.binary["kind"] == "ptx" else "",
        ptx_target="sm_90" if spec.binary["kind"] == "ptx" else "",
        options={},
        launch=LaunchAttributes(0, 0, False, (1, 1, 1), 1, False),
        ir_digest=None,
        binary_digest=target.digests[key],
        binary_build=build,
    )
    values.update(kw)
    return BuiltKernel(**values)


def fake_ptxas(directory, cubin_path):
    """An executable that answers ``--version`` and otherwise copies ``cubin_path`` to the file after ``-o``: ptxas for a machine that has none."""
    script = pathlib.Path(directory) / "ptxas"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "ptxas: NVIDIA (R) Ptx optimizing assembler"; echo "Cuda compilation tools, release 12.9, V12.9.86"; exit 0; fi\n'
        'echo "$@" > "$0.args"\n'
        'while [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then out="$2"; fi; shift; done\n'
        f'cp "{cubin_path}" "$out"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)
