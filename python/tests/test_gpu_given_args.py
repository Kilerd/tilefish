"""Fixed arguments given per call, on a real GPU.

``gpu_kernels_p6.scale_add`` is a prebuilt PTX (``y = x * k + c``) whose ``k: const[i32]`` and ``c: const[i64]`` every call gives. One binary serves
every value: a single launch in ``jit`` and ``replay`` (one manifest entry) gives the reference's bytes for values of either sign and at the edges of
their types; the plan of ``two_scales`` (two steps of that one binary, the second ``c`` the bucket's ``Rg``) gives them in replay, jit, reference
and a capture.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p6 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, TilefishTypeError, runtime  # noqa: E402

F32 = torch.float32
VALUES = [(3, -2), (-1, 7), (0, 0), (2**31 - 1, -(2**40)), (-(2**31), 2**24)]


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    directory = tmp_path_factory.mktemp("given")
    return directory, runtime.export(k.scale_add, directory)


@pytest.mark.parametrize("kc", VALUES)
def test_one_binary_gives_the_reference_for_every_value(manifest, kc):
    directory, m = manifest
    major, minor = torch.cuda.get_device_capability()
    entry = m.get("scale_add", f"sm_{major * 10 + minor}").entry
    assert [p["kind"] for p in entry["abi"]["params"]][-2:] == ["given", "given"]
    kk, c = kc
    x = randn((1000,), F32, seed=1)
    want = k.scale_add_reference(x, kk, c)
    for mode, extra in (("jit", {}), ("replay", {"manifest": directory, "pin": m.digest}), ("reference", {})):
        got = tilefish.run(k.scale_add, x, k=kk, c=c, mode=mode, **extra)
        assert same_bytes(got, want), (mode, kc)


def test_a_call_without_a_value_or_with_one_that_does_not_fit_is_refused(manifest):
    x = randn((16,), F32, seed=2)
    with pytest.raises(TilefishTypeError, match="missing a required argument: 'c'"):
        tilefish.run(k.scale_add, x, k=1)
    with pytest.raises(TilefishPreconditionError, match="does not fit the i32"):
        tilefish.run(k.scale_add, x, k=2**31, c=0)
    with pytest.raises(TilefishTypeError, match="takes an int"):
        tilefish.run(k.scale_add, x, k=1.5, c=0)


@pytest.mark.parametrize("rg", [8, 24])
def test_the_plan_of_two_calls_of_one_binary_gives_the_reference_in_every_mode(tmp_path, rg):
    plan = runtime.export(k.two_scales, tmp_path)
    assert [s["given"] for s in plan.steps(Rg=rg)] == [{"k": 3, "c": -2}, {"k": -1, "c": rg}]
    x = randn((rg, 64), F32, seed=rg)
    want = k.scale_add_reference(k.scale_add_reference(x.reshape(-1), 3, -2), -1, rg).reshape(rg, 64)
    for mode, extra in (("replay", {"plan": tmp_path, "pin": plan.digest}), ("jit", {}), ("reference", {})):
        assert same_bytes(tilefish.run(k.two_scales, x, mode=mode, **extra), want), mode
    cap = tilefish.capture(k.two_scales, x, mode="replay", plan=tmp_path, pin=plan.digest)
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(cap.outputs, want)
