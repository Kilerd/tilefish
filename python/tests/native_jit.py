"""Triton's own JIT launcher, as a test-only comparison (not collected: the name has no ``test_`` prefix).

``tilefish.run(..., mode="jit")`` builds the current source through its compile spec and launches the binary
through the CUDA driver, like ``replay``. Triton's native launcher is not a mode, because it specializes on the
values of its arguments (an ``i32`` equal to 1 becomes a constant, divisibility by 16 is assumed wherever it holds),
so its machine code can differ from the spec's binary. This helper runs it anyway, with the same preparation,
checks and launch configuration, so a test can compare. It needs a C compiler, which Triton uses to build its
launcher the first time.
"""

import torch

from tilefish import runtime


def native_jit(op, *args, **kwargs):
    """Run ``op`` through ``JITFunction.__getitem__(grid)(...)``; the results and in-place tensors are as in ``tilefish.run``."""
    target = runtime._target(op)
    prep = runtime._prepare(target, args, kwargs)
    fn = target.jit_function(prep.spec)
    values = [v for _, _, v in runtime._launch_values(prep)]
    options = dict(prep.spec.ast_source_inputs().options)  # num_warps, num_stages and, for a generated kernel, enable_fp_fusion
    with torch.cuda.device(runtime._device_index(torch, prep.device)):
        fn[prep.grid](*values, **options)
    return runtime._result(prep.results)
