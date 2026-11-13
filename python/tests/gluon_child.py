"""A process that declares the Gluon kernel of a project directory and exports it, or replays an export of it (not collected: no ``test_``).

``test_gpu_gluon.py`` writes ``g_kernel.py`` into a directory (:func:`write`), runs this file as a child process to export the kernel, edits the
file (its layout, or a constant of its body), and runs it again to replay the first export in a process that holds the edited kernel::

    python gluon_child.py DIRECTORY export EXPORT_DIR
    python gluon_child.py DIRECTORY replay EXPORT_DIR PIN

The last line it prints is ``RESULT <json>``: for ``export`` the manifest's ``pin``, the IR ``digest`` of the kernel and the sha256 of the bytes
``jit`` gave (``jit``); for ``replay`` the ``digest`` of the kernel this process holds and, for a replay with the pin (``pinned``) and one with
``trust_directory=True`` (``trusted``), the sha256 of its bytes (``replay``) or the message it was refused with (``refused``).
"""

import hashlib
import json
import pathlib
import sys

KERNEL = """from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def kernel(x, y, n_rows, C: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, {per_thread}], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    rows = gl.program_id(0) * 4 + gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    keep = rows[:, None] < n_rows
    v = gl.load(x + rows[:, None] * C + cols[None, :], mask=keep, other=0.0)
    gl.store(y + rows, gl.sum(v * {scale}, axis=1), mask=rows < n_rows)
"""


def write(directory, per_thread=4, scale="1.0"):
    """Write ``g_kernel.py`` into ``directory`` (a ``pathlib.Path``): the layout gives each thread ``per_thread`` contiguous columns."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "g_kernel.py").write_text(KERNEL.format(per_thread=per_thread, scale=scale))


def main(directory, role, export_dir, pin):
    sys.path.insert(0, directory)
    import torch

    import tilefish
    from tilefish import Dim, Out, Tensor, const, f32, runtime
    from tilefish.aot import compile_specs

    R = Dim("R")

    @tilefish.extern(gluon="g_kernel:kernel", num_warps=4, grid=lambda R: (tilefish.cdiv(R, 4),))
    def kernel(x: Tensor[R, 256, f32], y: Out[Tensor[R, f32]], n_rows: R, C: const = 256): ...

    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    x = torch.randn((37, 256), generator=gen, device="cuda")

    def sha(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    (spec,) = compile_specs(kernel)
    cap = torch.cuda.get_device_capability()
    out = {"digest": runtime._ir_digest_of(spec, runtime.resolve_triton(spec.triton), cap[0] * 10 + cap[1], {})}
    if role == "export":
        out["jit"] = sha(tilefish.run(kernel, x))
        out["pin"] = runtime.export(kernel, pathlib.Path(export_dir)).digest
    else:
        for how, trust in (("pinned", dict(pin=pin)), ("trusted", dict(trust_directory=True))):
            try:
                out[how] = {"replay": sha(tilefish.run(kernel, x, mode="replay", manifest=export_dir, **trust))}
            except tilefish.TilefishError as e:
                out[how] = {"refused": str(e)}
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else "")
