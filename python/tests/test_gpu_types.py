"""64-bit dimension scalars, null pointer arguments and the specialisation of a lowered kernel for views off 16 bytes, on a real GPU.

* ``add_one_wide`` takes its count as an ``i64[n]`` scalar: run on ``2**31 + 1000`` u8 elements in ``jit`` and ``replay``, every element is
  ``x + 1`` (an ``i32`` count could not carry the size: ``n_elem: n`` is refused at the launch);
* ``scale_null`` and ``scale_bias`` are one Triton kernel declared with a null bias and with a tensor, and ``scale_null_ptx`` / ``scale_bias_ptx`` a
  hand-written PTX that reads the bias only when it is not null (a pointer that is not 0 and not a tensor would fault or add what it points at) and takes a
  64-bit count: each gives ``2 * x`` (``+ bias``) byte for byte, in ``jit``, in ``replay`` and in a plan;
* ``plane_sum`` adds rows of a ``[3, R]`` int32 plane, two of which start off 16 bytes for an odd ``R``: its plan launches the specialisations
  ``add_rows__ua0`` and ``add_rows__ua1`` (no divisibility hint for those pointers) and gives the sum in every mode, and ``tilefish.run(add_rows, ...)``
  on a row that starts off 16 bytes runs the specialisation (``replay`` of a manifest without it says what to export).
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import types_kernels as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Out, Tensor, TilefishManifestError, TilefishPreconditionError, const, runtime, u8  # noqa: E402

F32 = torch.float32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("types")
    ops = [k.add_one_wide, k.scale_null, k.scale_bias, k.scale_null_ptx, k.scale_bias_ptx, k.add_rows, k.add_rows.unaligned("a")]
    manifest = runtime.export(ops, directory)
    return directory, manifest.digest


def run_modes(op, args, exported):
    directory, pin = exported
    return {
        "jit": tilefish.run(op, *args),
        "replay": tilefish.run(op, *args, mode="replay", manifest=directory, pin=pin),
    }


def test_an_i64_count_of_more_than_2_31_elements(exported):
    n = 2**31 + 1000
    x = torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda")
    want = x + 1
    for mode, y in run_modes(k.add_one_wide, (x,), exported).items():
        assert torch.equal(y, want), mode
        del y
    del want
    torch.cuda.empty_cache()

    @tilefish.extern(triton=k.add_one_wide_kernel, num_warps=4, grid=lambda n: (tilefish.cdiv(n, 1024),), index_bits=64, name="add_one_narrow")
    def add_one_narrow(x: Tensor[k.n, u8], y: Out[Tensor[k.n, u8]], n_elem: k.n, BLOCK: const = 1024): ...

    with pytest.raises(TilefishPreconditionError, match="does not fit the 32-bit signed integer"):
        tilefish.run(add_one_narrow, x)


@pytest.mark.parametrize("n", [1, 1000, 65539])
def test_an_i64_count_of_a_few_elements(exported, n):
    x = torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda")
    for mode, y in run_modes(k.add_one_wide, (x,), exported).items():
        assert torch.equal(y, x + 1), mode


@pytest.mark.parametrize("null_op, bias_op", [(k.scale_null, k.scale_bias), (k.scale_null_ptx, k.scale_bias_ptx)], ids=["triton", "ptx"])
@pytest.mark.parametrize("n", [1, 1000, 65539])
def test_a_null_pointer_is_null(exported, null_op, bias_op, n):
    x, b = randn((n,), F32, seed=n), randn((n,), F32, seed=n + 1)
    for mode, y in run_modes(null_op, (x,), exported).items():
        assert same_bytes(y, x * 2.0), (null_op.name, mode)
    for mode, y in run_modes(bias_op, (x, b), exported).items():
        assert same_bytes(y, x * 2.0 + b), (bias_op.name, mode)


def test_a_null_pointer_in_a_plan(tmp_path):
    nb = tilefish.Dim("nb", buckets=[1, 4099])

    @tilefish.graph
    def chain(x: Tensor[nb, tilefish.f32], b: Tensor[nb, tilefish.f32]):
        return k.scale_bias_ptx(k.scale_null(x), b)

    n = 4099
    x, b = randn((n,), F32, seed=3), randn((n,), F32, seed=4)
    want = x * 4.0 + b
    plan = runtime.export(chain, tmp_path)
    for mode, extra in (("replay", {"plan": tmp_path, "pin": plan.digest}), ("jit", {})):
        assert same_bytes(tilefish.run(chain, x, b, mode=mode, **extra), want), mode


def plane(r, seed=0):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + r)
    return torch.randint(-(2**20), 2**20, (3, r), dtype=torch.int32, device="cuda", generator=gen)


@pytest.mark.parametrize("r", [5, 8, 13])
def test_rows_of_a_plane_off_16_bytes_in_a_plan(tmp_path, r):
    plan = runtime.export(k.plane_sum, tmp_path)
    first, second = plan.steps(R=r)
    assert (first["entry"], second["entry"]) == ("add_rows__ua0", "add_rows__ua1")
    p = plane(r)
    want = p[1] + p[0] + p[2]
    for mode, extra in (("replay", {"plan": tmp_path, "pin": plan.digest}), ("jit", {}), ("reference", {})):
        assert torch.equal(tilefish.run(k.plane_sum, p, mode=mode, **extra), want), (mode, r)
    from tilefish import Manifest

    entry = Manifest.load(tmp_path, pin=plan.manifest_digest).get("add_rows__ua0", plan.arch)
    assert {"kind": "pointer_alignment", "arg": "a", "index": 0, "bytes": 4} in entry.entry["preconditions"]


def test_a_row_off_16_bytes_runs_the_specialisation(exported, tmp_path):
    directory, pin = exported
    p = plane(5, seed=9)
    a, b = p[1], p[0]  # a starts 20 bytes in
    assert a.data_ptr() % 16 and not b.data_ptr() % 16
    for mode, y in run_modes(k.add_rows, (a, b), exported).items():
        assert torch.equal(y, a + b), mode
    # a manifest that has the kernel's binaries and not the specialisation's says what to export
    only = runtime.export(k.add_rows, tmp_path)
    with pytest.raises(TilefishManifestError, match=r"runs the kernel's specialisation for pointers aligned to their element only, 'add_rows__ua0'"):
        tilefish.run(k.add_rows, a, b, mode="replay", manifest=tmp_path, pin=only.digest)
    # the kernel's own binaries still run an aligned call
    assert torch.equal(tilefish.run(k.add_rows, b, b, mode="replay", manifest=tmp_path, pin=only.digest), b + b)
