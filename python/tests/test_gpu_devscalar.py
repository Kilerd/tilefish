"""Device scalars, on a real GPU: a request-dependent value that lives in device memory and that no binary is compiled for.

``n_tok: DevScalar[i32]`` is a pointer to one element (in the ABI, a pointer with the shape ``[]``): the kernel reads it with ``tl.load``, so the
same binary runs for every value, and a value changes by writing to memory, not by building. The test kernel zeroes the rows from ``n_tok``
on (``gpu_kernels_p1.masked_rows``); a lowered kernel reads one with ``tilefish.load(s)``. Each is run for several values with **one**
compile, in ``jit`` and in ``replay`` (which compiles nothing), and checked against torch.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p1 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, runtime  # noqa: E402

F32 = torch.float32


@pytest.fixture
def builds(monkeypatch):
    """The compiles (calls of ``tilefish._build.build`` through the runtime) made while the test runs."""
    runtime.clear_caches()
    calls = []
    real = runtime.build

    def counting(spec, **kw):
        calls.append(spec.stem)
        return real(spec, **kw)

    monkeypatch.setattr(runtime, "build", counting)
    return calls


def n_tok(value, shape=()):
    return torch.full(shape, value, dtype=torch.int32, device="cuda")


def expected_rows(x, n):
    y = x * 2.0
    y[n:] = 0.0
    return y


def test_the_masked_rows_are_right_for_every_n_with_one_compile(builds):
    x = randn((64, 128), F32, seed=1)
    for n in (0, 1, 5, 17, 63, 64, 40):
        got = tilefish.run(k.masked_rows, x, n_tok(n))
        assert same_bytes(got, expected_rows(x, n)), n
    assert builds == ["masked_rows"]  # one compile for seven values


def test_replay_runs_every_n_without_compiling(builds, tmp_path):
    m = runtime.export(k.masked_rows, tmp_path)
    runtime.clear_caches()
    builds.clear()
    x = randn((64, 128), F32, seed=2)
    for n in (3, 0, 64, 29):
        got = tilefish.run(k.masked_rows, x, n_tok(n), mode="replay", manifest=tmp_path, pin=m.digest)
        assert same_bytes(got, expected_rows(x, n)), n
    assert builds == []


def test_jit_and_replay_agree_byte_for_byte(tmp_path):
    m = runtime.export(k.masked_rows, tmp_path)
    x = randn((33, 128), F32, seed=3)
    for n in (1, 16, 33):
        jit = tilefish.run(k.masked_rows, x, n_tok(n))
        replay = tilefish.run(k.masked_rows, x, n_tok(n), mode="replay", manifest=tmp_path, pin=m.digest)
        assert same_bytes(jit, replay)


def test_a_value_written_after_the_launch_is_read_by_the_next_one():
    """The scalar is read when the kernel runs, not when it is launched: the same tensor, updated in place between two launches."""
    x = randn((16, 128), F32, seed=4)
    n = n_tok(4)
    first = tilefish.run(k.masked_rows, x, n)
    n.fill_(11)
    second = tilefish.run(k.masked_rows, x, n)
    assert same_bytes(first, expected_rows(x, 4)) and same_bytes(second, expected_rows(x, 11))


def test_a_scalar_of_shape_one_is_the_same_scalar():
    x = randn((16, 128), F32, seed=5)
    assert same_bytes(tilefish.run(k.masked_rows, x, n_tok(7, (1,))), tilefish.run(k.masked_rows, x, n_tok(7)))


@pytest.mark.parametrize("bad", [torch.zeros(2, dtype=torch.int32), torch.zeros((), dtype=torch.int64), torch.zeros((1, 1), dtype=torch.int32)])
def test_a_scalar_has_one_element_of_its_dtype(bad):
    x = randn((16, 128), F32, seed=6)
    with pytest.raises(TilefishPreconditionError):
        tilefish.run(k.masked_rows, x, bad.to("cuda"))


@pytest.mark.parametrize("scale", [0.5, -3.25, 1e-3])
def test_a_lowered_kernel_reads_a_device_scalar_with_tilefish_load(builds, scale):
    low = tilefish.lower(k.scale_by)
    assert "tl.load(s)" in low.source and "mask=" in low.source
    x = randn((20, 256), F32, seed=7)
    s = torch.full((), scale, dtype=F32, device="cuda")
    got = tilefish.run(k.scale_by, x, s)
    assert same_bytes(got, x * s)  # one f32 multiply: the same bytes as torch
    s.fill_(scale * 4)
    assert same_bytes(tilefish.run(k.scale_by, x, s), x * (scale * 4))
    assert builds == ["scale_by"]  # the second value did not compile


def test_a_lowered_kernel_with_two_scalars_agrees_with_replay_and_with_the_reference(tmp_path):
    x = randn((4096,), F32, seed=8)
    s = torch.full((), 1.5, dtype=F32, device="cuda")
    b = torch.full((), -0.25, dtype=F32, device="cuda")
    m = runtime.export(k.scale_flat, tmp_path)
    jit = tilefish.run(k.scale_flat, x, s, b)
    replay = tilefish.run(k.scale_flat, x, s, b, mode="replay", manifest=tmp_path, pin=m.digest)
    assert same_bytes(jit, replay)
    ref = tilefish.run(k.scale_flat, x, s, b, mode="reference")
    assert (jit - ref).abs().max() <= 2e-7 * (1.5 * x.abs().max() + 0.25)  # a * b + c may round once (FMA): within an ulp
    assert (jit - (x * 1.5 - 0.25)).abs().max() <= 2e-7 * 1.5 * x.abs().max() + 1e-6


def test_the_manifest_says_a_device_scalar_is_a_pointer_with_no_axes(tmp_path):
    import json

    runtime.export(k.masked_rows, tmp_path)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    n = next(p for p in entry["abi"]["params"] if p["name"] == "n_tok")
    assert (n["kind"], n["dtype"], n["shape"], n["access"]) == ("pointer", "i32", [], "in")
    assert {"kind": "pointer_alignment", "arg": "n_tok", "index": 2, "bytes": 16} in entry["preconditions"]
    assert not any(p.get("arg") == "n_tok" and p["kind"] in ("max_elements", "static_size") for p in entry["preconditions"])
