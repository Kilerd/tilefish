"""Every route a hand-written Triton kernel can read a constant by, as real files, and a process that builds, runs and replays one.

Not collected (the name has no ``test_`` prefix). ``test_gpu_identity.py`` writes one package per route and runs this file as a
child process (``python identity_routes.py ...``) so that "another process" is another process: its own imports, its own
environment (``R_VALUE`` is the constant), the same files, and, where asked, the same tilefish cache directory.

The routes are the ones that each earlier way of *finding* what Triton reads (a digest of the files, a walk over the namespaces, the
closures, a watch on the imports) missed in turn: an attribute of a module, a name of the kernel's own module, ``getattr`` with a literal
name and with one made at run time, an attribute of a class, a ``triton.constexpr_function`` that reads the environment when the kernel
is compiled, a ``tl.constexpr`` of an object whose ``repr`` hides its state, a user module with the name of a standard library module,
a closure variable, a helper two levels away. Triton's frontend folds each of them into the IR it builds, which is what a binary is
identified by now.
"""

import json
import sys

# the kernel; ``{imports}`` and ``{expr}`` are the route
KERNEL = """import triton
import triton.language as tl
{imports}

{prelude}
@triton.jit
def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, {expr}, mask=mask)
"""

ENV = 'float(os.environ["R_VALUE"])'
CFG_ATTRIBUTE = f"import os\n\nimport triton.language as tl\n\nSCALE = tl.constexpr({ENV})\n"
BODY = "x * {}"


def kernel(expr, *, imports="", prelude=""):
    return KERNEL.format(imports=imports, prelude=prelude, expr=expr)


# name -> (the files of the package, the modules the child must import afresh)
ROUTES = {
    "an attribute of a module": ({"r_kernel": kernel(BODY.format("r_cfg.SCALE"), imports="import r_cfg"), "r_cfg": CFG_ATTRIBUTE}, ("r_kernel", "r_cfg")),
    "a name of the kernel's own module": (
        {"r_kernel": kernel(BODY.format("SCALE"), imports="import os", prelude=f"SCALE = tl.constexpr({ENV})\n\n")},
        ("r_kernel",),
    ),
    "getattr with a literal name": ({"r_kernel": kernel(BODY.format('getattr(r_cfg, "SCALE")'), imports="import r_cfg"), "r_cfg": CFG_ATTRIBUTE}, ("r_kernel", "r_cfg")),
    "getattr with a name made at run time": (
        {"r_kernel": kernel(BODY.format('getattr(r_cfg, "SC" + "ALE")'), imports="import r_cfg"), "r_cfg": CFG_ATTRIBUTE},
        ("r_kernel", "r_cfg"),
    ),
    "an attribute of a class": (
        {
            "r_kernel": kernel(BODY.format("r_cfg.Inner.SCALE"), imports="import r_cfg"),
            "r_cfg": f"import os\n\nimport triton.language as tl\n\n\nclass Inner:\n    SCALE = tl.constexpr({ENV})\n",
        },
        ("r_kernel", "r_cfg"),
    ),
    "a constexpr_function that reads the environment": (
        {
            "r_kernel": kernel(BODY.format("r_cfg.scale()"), imports="import r_cfg"),
            "r_cfg": f"import os\n\nimport triton\n\n\n@triton.constexpr_function\ndef scale():\n    return {ENV}\n",
        },
        ("r_kernel", "r_cfg"),
    ),
    "a tl.constexpr of an object whose repr hides its state": (
        {
            "r_kernel": kernel(BODY.format("r_cfg.unwrap(r_cfg.HIDDEN)"), imports="import r_cfg"),
            "r_cfg": (
                "import os\n\nimport triton\nimport triton.language as tl\n\n\nclass Hidden:\n    def __init__(self, scale):\n        self.scale = scale\n\n"
                f"    def __repr__(self):\n        return 'Hidden()'\n\n\nHIDDEN = tl.constexpr(Hidden({ENV}))\n\n\n"
                "@triton.constexpr_function\ndef unwrap(hidden):\n    return hidden.scale\n"
            ),
        },
        ("r_kernel", "r_cfg"),
    ),
    "a user module named like a standard library module": (
        {"r_kernel": kernel(BODY.format("sched.SCALE"), imports="import sched"), "sched": CFG_ATTRIBUTE},
        ("r_kernel", "sched"),
    ),
    "a closure variable": (
        {
            "r_kernel": (
                "import os\n\nimport triton\nimport triton.language as tl\n\n\ndef make():\n"
                f"    S = tl.constexpr({ENV})\n\n    @triton.jit\n    def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):\n"
                "        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n        mask = offsets < n_elements\n"
                "        x = tl.load(x_ptr + offsets, mask=mask)\n        tl.store(y_ptr + offsets, x * S, mask=mask)\n\n    return kernel\n\n\nkernel = make()\n"
            )
        },
        ("r_kernel",),
    ),
    "a helper two levels away": (
        {
            "r_kernel": kernel("r_helper.helper(x)", imports="import r_helper"),
            "r_helper": "import triton\n\nimport r_cfg\n\n\n@triton.jit\ndef helper(x):\n    return x * r_cfg.SCALE\n",
            "r_cfg": CFG_ATTRIBUTE,
        },
        ("r_kernel", "r_helper", "r_cfg"),
    ),
}


def write(directory, route):
    """Write the package of ``route`` into ``directory`` (a ``pathlib.Path``)."""
    directory.mkdir(parents=True, exist_ok=True)
    files, modules = ROUTES[route]
    for name, text in files.items():
        (directory / f"{name}.py").write_text(text)
    return modules


def declare():
    """The ``extern`` of ``r_kernel:kernel``."""
    import tilefish
    from tilefish import Dim, Out, Tensor, f32

    N = Dim("N", divisible_by=16)

    @tilefish.extern(triton="r_kernel:kernel", num_warps=4, grid=lambda N: (tilefish.cdiv(N, 256),))
    def r_op(x_ptr: Tensor[N, f32], y_ptr: Out[Tensor[N, f32]], n_elements: N, *, BLOCK: tilefish.const = 256): ...

    return r_op


# -- the child ------------------------------------------------------------------------------------------------------------------------


def main(directory, route, role, export_dir, pin):
    """One process: ``first`` builds (``jit``) and exports; ``edit_then_export`` does that after editing ``r_cfg.py`` on disk without
    reloading it; ``other`` and ``again`` run ``jit`` and try to replay the export with its pin.

    Prints one JSON object: ``compiles`` (how many times ``triton.compile`` ran here), ``jit`` (the value of ``y``, the constant),
    ``replay`` (the same, or ``None``), ``refused`` (the message when the replay was refused as stale, else ``None``), ``pin``, ``digest``
    (the IR digest of the kernel as this process holds it, for the device).
    """
    sys.path.insert(0, directory)
    import torch
    import triton

    import pathlib

    import tilefish
    from tilefish import TilefishManifestError, runtime
    from tilefish._build import resolve_triton
    from tilefish.aot import compile_specs

    compiles = []
    real = triton.compile
    triton.compile = lambda *a, **k: (compiles.append(1), real(*a, **k))[1]
    op = declare()
    x = torch.ones(64, dtype=torch.float32, device="cuda")  # y is then the constant itself
    out = {"compiles": 0, "jit": None, "replay": None, "refused": None, "pin": None}
    y = tilefish.run(op, x, mode="jit")
    out["jit"] = float(y[0])
    major, minor = torch.cuda.get_device_capability()
    out["digest"] = runtime._ir_digest_of(compile_specs(op)[0], resolve_triton("r_kernel:kernel"), major * 10 + minor, {})
    if role == "edit_then_export":  # the file is edited after the process imported it, and the module is not reloaded
        path = pathlib.Path(directory) / "r_cfg.py"
        path.write_text(path.read_text().replace(ENV, ENV + " + 1.0"))
    if role in ("first", "edit_then_export"):
        manifest = runtime.export(op, export_dir)
        out["pin"] = manifest.digest
    else:
        try:
            r = tilefish.run(op, x, mode="replay", manifest=export_dir, pin=pin)
            out["replay"] = float(r[0])
        except TilefishManifestError as e:
            out["refused"] = str(e)
    out["compiles"] = len(compiles)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:6])
