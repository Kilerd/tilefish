"""A plan set on a real GPU: the plans of ``gpu_kernels_p6.tile_chain`` for 1, 2 and 3 tiles (2, 3 and 4 steps) exported together with one manifest.

Each binary is built once (one entry each for ``scale_add`` and ``halve``); a replay of each graph through the set's pin and through its own pin gives
the bytes of jit and of the reference, and a capture replays.
"""

import json

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p6 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishManifestError, runtime  # noqa: E402

F32 = torch.float32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("set")
    return directory, runtime.export([k.TILES[t] for t in (1, 2, 3)], directory)


def test_one_manifest_holds_each_binary_once(exported):
    directory, s = exported
    manifest = json.loads((directory / "manifest.json").read_text())
    assert sorted(e["stem"] for e in manifest["kernels"]) == ["halve", "scale_add"]
    assert sorted(s.plans) == ["tiles1", "tiles2", "tiles3"] and all(p.manifest_digest == s.manifest_digest for p in s.plans.values())


@pytest.mark.parametrize("tiles", [1, 2, 3])
def test_each_plan_of_the_set_replays_through_either_pin(exported, tiles):
    directory, s = exported
    g = k.TILES[tiles]
    x = randn((tiles * 64,), F32, seed=tiles)
    want = k.tile_chain_reference(x, tiles)
    for pin in (s.digest, s[g.name].digest):
        assert same_bytes(tilefish.run(g, x, mode="replay", plan=directory, pin=pin), want), pin
    assert same_bytes(tilefish.run(g, x), want) and same_bytes(tilefish.run(g, x, mode="reference"), want)
    cap = tilefish.capture(g, x, mode="replay", plan=directory, pin=s.digest)
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(cap.outputs, want)
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        tilefish.run(g, x, mode="replay", plan=directory, pin="0" * 64)
