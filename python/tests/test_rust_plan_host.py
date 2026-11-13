"""``tilefish.hosts.rust.emit_plan``: a typed Rust runner of a launch plan, from a pinned plan only, and it compiles against the crate.

The plans are the golden ones (``golden/plans``) and plans of ``plan_model`` exported with the fake builds of ``plan_fakes`` (no GPU): the format, the
digests and the names are real, which is all the generated source depends on.
"""

import json
import os
import pathlib
import shutil
import subprocess

import plan_fakes
import plan_model as pm
import pytest

import tilefish
from tilefish import _plan
from tilefish.errors import TilefishManifestError
from tilefish.hosts import rust

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GOLDEN = HERE / "golden" / "plans"
CRATE = REPO_ROOT / "crates" / "tilefish"
needs_golden = pytest.mark.skipif(not (GOLDEN / "expected.json").exists(), reason="the golden plans are not available")
needs_cargo = pytest.mark.skipif(
    shutil.which("cargo") is None or not (CRATE / "src" / "plan_run.rs").exists(),
    reason="cargo is not installed, or the crate of this checkout does not run launch plans",
)


def golden_pin(name):
    return json.loads((GOLDEN / "expected.json").read_text())[name]["plan_digest"]


@needs_golden
def test_a_runner_is_generated_from_a_pinned_plan_and_is_deterministic():
    pin = golden_pin("block")
    source = rust.render_plan(GOLDEN / "block", pin=pin)
    assert source == rust.render_plan(GOLDEN / "block" / "plan.json", pin=pin)
    assert f'pub const PLAN_DIGEST: &str = "{pin}";' in source
    assert "pub struct Block {" in source and "pub struct BlockExternals<'a> {" in source and "pub struct BlockScalars {" in source
    assert "    pub x: Readable<'a, Bf16>,\n" in source and "    pub out0: TensorMut<'a, Bf16>,\n" in source and "    pub n: i32,\n" in source
    assert "LaunchPlan::load(directory, Pin::Digest(PLAN_DIGEST))" in source
    # the plan loaded back with its pin gives the same source
    loaded = _plan.load_plan(GOLDEN / "block", pm.block, pin=pin)
    assert rust.render_plan(loaded) == source


@needs_golden
def test_a_plan_nobody_vouched_for_is_refused():
    with pytest.raises(TilefishManifestError, match="without a trust anchor"):
        rust.render_plan(GOLDEN / "block")
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        rust.render_plan(GOLDEN / "block", pin=golden_pin("block-reuse"))
    with pytest.raises(TilefishManifestError, match="not pinned|arch must look like"):
        rust.render_plan(tilefish.plan(pm.block))  # made in memory: no architecture, no manifest, nothing pinned
    loaded = _plan.load_plan(GOLDEN / "block", pm.block, trust_directory=True)
    with pytest.raises(TilefishManifestError, match="not pinned"):
        rust.render_plan(loaded)
    assert rust.render_plan(loaded, pin=golden_pin("block")) == rust.render_plan(GOLDEN / "block", pin=golden_pin("block"))
    with pytest.raises(TilefishManifestError, match="takes a plan directory"):
        rust.render_plan(42)


def test_the_names_of_a_runner_follow_the_graph_and_its_buffers(tmp_path):
    exported = plan_fakes.export(tilefish.plan(pm.accumulate), tmp_path / "accumulate")
    source = rust.render_plan(exported)
    assert "pub struct Accumulate {" in source and "    pub state: TensorMut<'a, F32>,\n" in source and "    pub a: Readable<'a, F32>,\n" in source
    assert "pub struct AccumulateScalars;" in source  # no device scalar
    exported = plan_fakes.export(tilefish.plan(pm.two_outputs), tmp_path / "two")
    source = rust.render_plan(exported)
    assert "`out0`: a returned value (written), returned value 0" in source and "`out1`: a returned value (written), returned value 1" in source


@needs_golden
@needs_cargo
def test_the_runners_compile_against_the_crate(tmp_path):
    crate = tmp_path / "plan-runners-check"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "plan-runners-check"\nversion = "0.0.0"\nedition = "2024"\n\n'
        f"[dependencies]\ntilefish = {{ path = {json.dumps(str(CRATE))} }}\n\n[workspace]\n"
    )
    rust.emit_plan(GOLDEN / "block", crate / "src" / "block.rs", pin=golden_pin("block"))
    rust.emit_plan(GOLDEN / "block-reuse", crate / "src" / "block_reuse.rs", pin=golden_pin("block-reuse"))
    rust.emit_plan(plan_fakes.export(tilefish.plan(pm.accumulate), tmp_path / "accumulate"), crate / "src" / "accumulate.rs")
    rust.emit_plan(plan_fakes.export(tilefish.plan(pm.two_outputs), tmp_path / "two"), crate / "src" / "two_outputs.rs")
    (crate / "src" / "lib.rs").write_text("pub mod accumulate;\npub mod block;\npub mod block_reuse;\npub mod two_outputs;\n")
    env = {**os.environ, "RUSTFLAGS": "-D warnings", "CARGO_TARGET_DIR": str(REPO_ROOT / "target" / "plan-runners-check")}
    done = subprocess.run(
        ["cargo", "check", "--manifest-path", str(crate / "Cargo.toml"), "--color", "never"], capture_output=True, text=True, env=env, timeout=900
    )
    assert done.returncode == 0, done.stderr[-4000:]
