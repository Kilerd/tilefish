"""A process that declares an op with an implementation per architecture from a project directory and exports it, or replays an export of it
(not collected: no ``test_``).

``test_gpu_per_arch.py`` writes ``pa_edit.py`` into a directory (:func:`write`), runs this file as a child process to export the op on the device,
edits one kernel of the file, and runs it again to replay the first export in a process that holds the edited kernel::

    python per_arch_child.py DIRECTORY export EXPORT_DIR
    python per_arch_child.py DIRECTORY replay EXPORT_DIR PIN

The op runs ``pa_edit:tiled`` on sm_103 and ``pa_edit:whole`` on sm_120 (and as the fallback). The last line printed is ``RESULT <json>``: the
implementation the device runs, the IR ``digest`` of its kernel as this process holds it, for ``export`` the manifest's ``pin`` and the sha256 of the
bytes ``jit`` gave, and for ``replay`` (``pinned``, with the pin, and ``trusted``, with ``trust_directory=True``) the sha256 of its bytes (``replay``) or the
message it was refused with (``refused``).
"""

import hashlib
import json
import pathlib
import sys

KERNELS = """import triton
import triton.language as tl


@triton.jit
def tiled(x, y, n_rows, ROWS: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    keep = rows < n_rows
    acc = tl.zeros((ROWS,), dtype=tl.float32)
    for k in range(0, C, BLOCK):
        cols = k + tl.arange(0, BLOCK)
        acc += tl.sum(tl.load(x + rows[:, None] * C + cols[None, :], mask=keep[:, None], other=0.0) * {tiled}, axis=1)
    tl.store(y + rows, acc, mask=keep)


@triton.jit
def whole(x, y, n_rows, ROWS: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    keep = rows < n_rows
    cols = tl.arange(0, BLOCK)
    tl.store(y + rows, tl.sum(tl.load(x + rows[:, None] * C + cols[None, :], mask=keep[:, None], other=0.0) * {whole}, axis=1), mask=keep)
"""


def write(directory, tiled="1.0", whole="1.0"):
    """Write ``pa_edit.py`` into ``directory`` (a ``pathlib.Path``): each kernel scales what it loads by its constant."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pa_edit.py").write_text(KERNELS.format(tiled=tiled, whole=whole))


def main(directory, role, export_dir, pin):
    sys.path.insert(0, directory)
    import torch

    import tilefish
    from tilefish import Dim, Out, Tensor, const, f32, runtime
    from tilefish.aot import compile_specs

    R = Dim("R")

    @tilefish.extern(
        by_arch={
            "sm_103": tilefish.impl(triton="pa_edit:tiled", num_warps=4, consts={"ROWS": 2, "BLOCK": 128}),
            "sm_120": tilefish.impl(triton="pa_edit:whole", num_warps=8),
        },
        fallback=tilefish.impl(triton="pa_edit:whole", num_warps=4),
        grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),),
    )
    def row_sum(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 1024, BLOCK: const = 1024): ...

    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    x = torch.randn((37, 1024), generator=gen, device="cuda")

    def sha(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    major, minor = torch.cuda.get_device_capability()
    cap = major * 10 + minor
    (spec,) = compile_specs(row_sum, arch=cap)
    out = {"implementation": spec.implementation, "digest": runtime._ir_digest_of(spec, runtime.resolve_triton(spec.triton), cap, {})}
    if role == "export":
        out["jit"] = sha(tilefish.run(row_sum, x))
        out["pin"] = runtime.export(row_sum, pathlib.Path(export_dir)).digest
    else:
        for how, trust in (("pinned", dict(pin=pin)), ("trusted", dict(trust_directory=True))):
            try:
                out[how] = {"replay": sha(tilefish.run(row_sum, x, mode="replay", manifest=export_dir, **trust))}
            except tilefish.TilefishError as e:
                out[how] = {"refused": str(e)}
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else "")
