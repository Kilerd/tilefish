"""``TILEFISH_IMPORT_WATCH=0``: tilefish leaves the import system alone and falls back to size and modification time."""

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import importlib, importlib.machinery, os, sys, time
    sys.path.insert(0, sys.argv[1])
    import tilefish
    from tilefish import _source

    wrapped = hasattr(importlib.machinery.SourceFileLoader.exec_module, "_tilefish_watch")
    print("wrapped", wrapped, "lazy", _source._watch.lazy)

    path = os.path.join(sys.argv[1], "late_mod.py")
    with open(path, "w") as f:
        f.write("X = 1\\n")
    importlib.invalidate_caches()
    import late_mod

    def check():
        with open(path, "rb") as f:
            data = f.read()
            st = os.fstat(f.fileno())
        import hashlib
        _source._watch.check_file("late_mod", path, hashlib.sha256(data).hexdigest(), st)

    check()  # first read by a build: recorded (lazy) or compared with the import record (watched)
    with open(path, "w") as f:
        f.write("X = 22\\n")  # another size, so the fallback sees it even within one mtime tick
    try:
        check()
        print("edit", "passed")
    except _source.StaleSourceError:
        print("edit", "refused")
    """
)


def _run(tmp_path, env_value):
    env = dict(os.environ)
    env.pop("TILEFISH_IMPORT_WATCH", None)
    if env_value is not None:
        env["TILEFISH_IMPORT_WATCH"] = env_value
    script = tmp_path / "probe.py"
    script.write_text(SCRIPT)
    out = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], env=env, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return out[0], out[1]


def test_default_wraps_the_loader_and_refuses_an_edit(tmp_path):
    first, edit = _run(tmp_path, None)
    assert first == "wrapped True lazy False"
    assert edit == "edit refused"


def test_opt_out_leaves_the_loader_alone_and_still_refuses_an_edit_after_the_first_read(tmp_path):
    first, edit = _run(tmp_path, "0")
    assert first == "wrapped False lazy True"
    assert edit == "edit refused"


def test_any_other_value_keeps_the_watch(tmp_path):
    first, _ = _run(tmp_path, "1")
    assert first == "wrapped True lazy False"
