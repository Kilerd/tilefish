"""The plan documents the Rust reader is held to (``crates/tilefish/tests/data/python_plans.json``). Not collected. Needs no GPU.

    python export_plan_vectors.py ../../crates/tilefish/tests/data      # (export_rust_vectors.py writes it too)

Each document is a ``plan.json`` text made from one of the golden plans (``golden/plans/block`` and ``block-reuse``) by one change: every mutation of
``test_plan_schema.py`` (a field of the wrong type or value, a key missing or unknown, a view outside its buffer, overlapping regions, a step that is not
what the manifest's entry says, a buffer read before it is written, ...), shape expressions that do and do not fold (``2*M//2`` is ``M``, ``M*M//M`` is
not an affine expression), texts that are not strict JSON, and a few hundred random changes of one leaf (seeded). Unless the change is to the digest
itself, the ``plan_digest`` is computed again, so that what is held to is the format and not the digest. Each has ``accepted``: what Python's
``read_plan`` (with no pin) and ``check_against_manifest`` with the manifest of its directory decide. ``crates/tilefish/tests/launch_plans.rs``
requires the Rust reader to decide the same for every one, and ``test_plan_vectors.py`` that the committed file is what this makes.
"""

import copy
import json
import pathlib
import random
import sys
import tempfile

from tilefish import _plan
from tilefish._manifest import Manifest
from tilefish.errors import TilefishManifestError

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE / "golden" / "plans"
BASES = ("block", "block-reuse")
JUNK = [None, True, False, 0, -1, 1, 2**70, 1.5, "", "x", "M", "M*M", "3*M", "(((", [], {}, [1], {"a": 1}, "sm_103", "0" * 64]


def setter(path, value):
    def apply(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


def deleter(path):
    def apply(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return apply


def buffer_index(doc, name):
    return next(i for i, b in enumerate(doc["buffers"]) if b["name"] == name)


def mutations():
    """``(name, mutate)``: the rules of ``test_plan_schema.py``, the manifest checks, and shape expressions."""
    steps0 = ["buckets", 0, "steps"]
    out = [
        ("schema", setter(["schema"], "tilefish.plan.v2")),
        ("unknown key", setter(["extra"], 1)),
        ("missing key", deleter(["arena_bytes"])),
        ("version type", setter(["tilefish"], 1)),
        ("version", setter(["tilefish"], "9.9.9")),
        ("graph", setter(["graph"], "a b")),
        ("arch", setter(["arch"], "sm_9")),
        ("arch with a suffix", setter(["arch"], "sm_100a")),
        ("arch null", setter(["arch"], None)),
        ("manifest digest", setter(["manifest_digest"], "A" * 64)),
        ("manifest digest of another", setter(["manifest_digest"], "0" * 64)),
        ("manifest digest null", setter(["manifest_digest"], None)),
        ("reuse", setter(["reuse"], 1)),
        ("alignment", setter(["arena_alignment"], 128)),
        ("alignment float", setter(["arena_alignment"], 256.0)),
        ("arena bytes", setter(["arena_bytes"], 256)),
        ("arena bytes float", setter(["arena_bytes"], 9634048.0)),
        ("dims not sorted", setter(["dims", "M"], [64, 32, 128])),
        ("dims duplicate", setter(["dims", "M"], [32, 32, 64])),
        ("dims zero", setter(["dims", "M"], [0, 32])),
        ("dims bool", setter(["dims", "M"], [True, 32])),
        ("dims empty", setter(["dims", "M"], [])),
        ("dims name", setter(["dims"], {"1M": [32]})),
        ("dims not an object", setter(["dims"], [["M", [32]]])),
        ("buffers empty", setter(["buffers"], [])),
        ("buffer key", setter(["buffers", 0, "extra"], 1)),
        ("buffer positional", setter(["buffers", 0], ["x", "input", "bf16", ["M", 256], None, 65536])),
        ("buffer name", setter(["buffers", 1, "name"], "x")),
        ("buffer role", setter(["buffers", 0, "role"], "weights")),
        ("buffer role list", setter(["buffers", 0, "role"], ["input"])),
        ("buffer dtype", setter(["buffers", 0, "dtype"], "f64")),
        ("buffer dtype packed", setter(["buffers", 0, "dtype"], "e2m1")),
        ("buffer shape", setter(["buffers", 0, "shape"], [0, 256])),
        ("buffer shape bool", setter(["buffers", 0, "shape"], [True, 256])),
        ("buffer shape expr", setter(["buffers", 0, "shape"], ["M*M", 256])),
        ("buffer bytes", setter(["buffers", 0, "bytes"], 1)),
        ("external offset", setter(["buffers", 0, "offset"], 0)),
        ("arena offset null", setter(["buffers", 9, "offset"], None)),
        ("arena offset unaligned", setter(["buffers", 9, "offset"], 100)),
        ("arena offset negative", setter(["buffers", 9, "offset"], -256)),
        ("devscalar shape", setter(["buffers", 1, "shape"], [1])),
        ("workspace dtype", setter(["buffers", -1, "dtype"], "bf16")),
        ("outputs missing", setter(["outputs"], [])),
        ("outputs not a buffer", setter(["outputs"], ["x"])),
        ("outputs twice", setter(["outputs"], ["out0", "out0"])),
        ("buckets short", setter(["buckets"], [])),
        ("bucket values", setter(["buckets", 0, "values"], {"M": 33})),
        ("bucket values float", setter(["buckets", 0, "values"], {"M": 32.0})),
        ("bucket values extra", setter(["buckets", 0, "values"], {"M": 32, "N": 1})),
        ("bucket order", lambda d: d["buckets"].reverse()),
        ("bucket key", setter(["buckets", 0, "extra"], 1)),
        ("step key", setter([*steps0, 0, "extra"], 1)),
        ("step positional", setter([*steps0, 0], ["rmsnorm", "kernel", []])),
        ("step entry", setter([*steps0, 0, "entry"], "a b")),
        ("step kind", setter([*steps0, 0, "kind"], "launch")),
        ("step kind swapped", setter([*steps0, 0, "kind"], "library")),
        ("workspace on a kernel", setter([*steps0, 0, "workspace"], "workspace")),
        ("workspace missing", deleter([*steps0, 1, "workspace"])),
        ("workspace is not one", setter([*steps0, 1, "workspace"], "x")),
        ("workspace is a list", setter([*steps0, 1, "workspace"], ["workspace"])),
        ("args empty", setter([*steps0, 0, "args"], [])),
        ("arg key", setter([*steps0, 0, "args", 0, "extra"], 1)),
        ("arg positional", setter([*steps0, 0, "args", 0], ["x", "x"])),
        ("arg binding", setter([*steps0, 0, "args", 0, "binding"], "nothing")),
        ("arg names twice", setter([*steps0, 0, "args", 1, "name"], "x")),
        ("arg is the workspace", setter([*steps0, 0, "args", 0, "binding"], "workspace")),
        ("view key", setter([*steps0, 2, "args", 0, "view", "extra"], 1)),
        ("view positional", setter([*steps0, 2, "args", 0, "view"], [[32, 4, 64], [768, 64, 1], 0])),
        ("view shape", setter([*steps0, 2, "args", 0, "view", "shape"], [0, 4, 64])),
        ("view strides length", setter([*steps0, 2, "args", 0, "view", "strides"], [768, 64])),
        ("view negative stride", setter([*steps0, 2, "args", 0, "view", "strides"], [768, -64, 1])),
        ("view offset", setter([*steps0, 2, "args", 0, "view", "offset"], -1)),
        ("view past the buffer", setter([*steps0, 2, "args", 0, "view", "offset"], 10**6)),
        ("view to the last element", setter([*steps0, 2, "args", 2, "view", "offset"], 32 * 768 - (31 * 768 + 3 * 64 + 63) - 1)),
        ("view one past the last element", setter([*steps0, 2, "args", 2, "view", "offset"], 32 * 768 - (31 * 768 + 3 * 64 + 63))),
        ("view of a devscalar", setter([*steps0, 2, "args", 4, "view"], {"shape": [1], "strides": [1], "offset": 0})),
        ("view of a whole buffer", setter([*steps0, 0, "args", 0, "view"], {"shape": [32, 256], "strides": [256, 1], "offset": 0})),
        ("kinds differ across buckets", setter(["buckets", 1, "steps", 0, "kind"], "library")),
        ("bindings differ across buckets", setter(["buckets", 1, "steps", 0, "args", 0, "binding"], "w_ln1")),
        ("step counts differ", lambda d: d["buckets"][1]["steps"].pop()),
        ("overlap without reuse", setter(["buffers", 10, "offset"], 0)),
        # what the manifest's entries say (check_against_manifest)
        ("no such kernel", setter([*steps0, 0, "entry"], "nothing")),
        ("another bucket's gemm", setter([*steps0, 1, "entry"], "gemm_o.M32")),
        ("no such library entry", setter([*steps0, 1, "entry"], "nothing.M32")),
        ("another architecture", setter(["arch"], "sm_90")),
        ("argument name", setter([*steps0, 0, "args", 0, "name"], "y")),
        ("argument missing", lambda d: d["buckets"][0]["steps"][0]["args"].pop()),
        ("rank of a weight", setter([*steps0, 0, "args", 1, "binding"], "x")),
        ("rank of an input", setter([*steps0, 0, "args", 0, "binding"], "w_ln1")),
        ("dtype", lambda d: d["buffers"][0].update(dtype="f32", bytes=d["buffers"][0]["bytes"] * 2)),
        ("written weight", setter([*steps0, 4, "args", 3, "binding"], "w_o")),
        ("written input", setter([*steps0, 0, "args", 2, "binding"], "x")),
        ("devscalar as an input", lambda d: d["buffers"][1].update(role="input", offset=None)),
        ("devscalar to a tensor pointer", setter([*steps0, 0, "args", 0, "binding"], "n")),
        ("read before written", setter([*steps0, 1, "args", 0, "binding"], "t4")),
        ("written then read", lambda d: (d["buckets"][0]["steps"][0]["args"][2].update(binding="t5"), d["buckets"][0]["steps"][1]["args"][0].update(binding="t5"))),
        ("workspace too small", lambda d: d["buffers"][-1].update(bytes=1000, shape=[1000])),
        ("a later bucket", setter(["buckets", 2, "steps", 0, "entry"], "nothing")),
        ("an output no step writes", lambda d: [b["steps"][8]["args"][2].update(binding="t0") for b in d["buckets"]]),
    ]
    out += copy_mutations()
    # shape expressions: the same sizes, written so that Python's dimension algebra folds them (accepted) or cannot (refused)
    for text in ("2*M//2", "M + 0", "M*1", "(M+M)//2", "M//1", "cdiv(M, 1)", "cdiv(2*M, 2)", "M - M + M", "-(-M)", "cdiv(M, 2) * 2", "M*M//M",
                 "cdiv(M*M, M)", "M // 2 * 2", "M // 0", "cdiv(M, 0)", "M - 2*M + 2*M", "N", "M+N-N", "((M))", "M//2+M//2", "2*(M//2)", "M  "):
        out.append((f"shape {text!r}", setter(["buffers", 0, "shape"], [text, 256])))
    for text in ("256", "2*128", "0+256", "M - M + 256", "cdiv(512, 2)", "513//2", "cdiv(511, 2)", "256*1*1", "-(-256)", "128 + 128 "):
        out.append((f"static shape {text!r}", setter(["buffers", 2, "shape"], [text])))
    return out


def arg(name, binding, view=None):
    out = {"name": name, "binding": binding}
    if view is not None:
        out["view"] = view
    return out


def copy_step(src, dst, *extra):
    """A copy step made per bucket: ``src`` and ``dst`` are ``(binding, view(rows) or None)``; ``extra`` are more keys (refused)."""

    def make(rows):
        step = {"kind": "copy", "args": [arg("src", src[0], src[1] and src[1](rows)), arg("dst", dst[0], dst[1] and dst[1](rows))]}
        for k, v in extra:
            step[k] = v
        return step

    return make


def with_step(make, at=None, buckets=None):
    """Every bucket (or the ones of index in ``buckets``) gets the step ``make(M)``, appended or inserted at ``at``."""

    def apply(doc):
        for i, b in enumerate(doc["buckets"]):
            if buckets is not None and i not in buckets:
                continue
            step = make(b["values"]["M"])
            b["steps"].insert(len(b["steps"]) if at is None else at, step)

    return apply


def copy_mutations():
    """Copy steps (``kind: "copy"``) added to the block's plan: valid ones (a whole buffer, a dense view, into an output, two runs of one
    buffer apart), and every rule both readers hold one to: the fields, the two arguments in order, dense, one dtype and one count, no overlap
    in one buffer, a destination a step may write, a source written before, the same step in every bucket."""
    dense = lambda offset: lambda m: {"shape": [m, 256], "strides": [256, 1], "offset": offset(m)}  # noqa: E731
    strided = lambda m: {"shape": [m, 256], "strides": [768, 1], "offset": 0}  # noqa: E731
    whole = None
    return [
        ("copy of an input", with_step(copy_step(("x", whole), ("t0", whole)))),
        ("copy into an output", with_step(copy_step(("t0", whole), ("out0", whole)))),
        ("copy of a dense view", with_step(copy_step(("t1", dense(lambda m: 0)), ("t0", whole)))),
        ("copy of a dense view at an offset", with_step(copy_step(("t1", dense(lambda m: 2 * m * 256)), ("t0", whole)))),
        ("copy within a buffer, apart", with_step(copy_step(("t1", dense(lambda m: 0)), ("t1", dense(lambda m: m * 256))))),
        ("copy within a buffer, overlapping", with_step(copy_step(("t1", dense(lambda m: 0)), ("t1", dense(lambda m: 128))))),
        ("copy of a view that is not dense", with_step(copy_step(("t1", strided), ("t0", whole)))),
        ("copy of another count", with_step(copy_step(("t1", whole), ("t0", whole)))),
        ("copy of a device scalar", with_step(copy_step(("n", whole), ("t0", whole)))),
        ("copy into an input", with_step(copy_step(("t0", whole), ("x", whole)))),
        ("copy into a weight", with_step(copy_step(("t0", dense(lambda m: 0)), ("w_ln1", whole)))),
        ("copy into a device scalar", with_step(copy_step(("x", whole), ("n", whole)))),
        ("copy of the workspace", with_step(copy_step(("workspace", whole), ("t0", whole)))),
        ("copy with an entry", with_step(copy_step(("x", whole), ("t0", whole), ("entry", "copy")))),
        ("copy with a workspace", with_step(copy_step(("x", whole), ("t0", whole), ("workspace", "workspace")))),
        ("copy with its arguments swapped", with_step(lambda m: {"kind": "copy", "args": [arg("dst", "t0"), arg("src", "x")]})),
        ("copy with one argument", with_step(lambda m: {"kind": "copy", "args": [arg("src", "x")]})),
        ("copy with three arguments", with_step(lambda m: {"kind": "copy", "args": [arg("src", "x"), arg("dst", "t0"), arg("dst2", "t1")]})),
        ("copy reads before it is written", with_step(copy_step(("t8", whole), ("t0", whole)), at=0)),
        ("copy writes before it is read", with_step(copy_step(("x", whole), ("t0", whole)), at=0)),
        ("copy in one bucket only", with_step(copy_step(("x", whole), ("t0", whole)), buckets=[1])),
        ("copy kind is not a string", with_step(lambda m: {"kind": ["copy"], "args": [arg("src", "x"), arg("dst", "t0")]})),
    ]


def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (i,))
    else:
        yield path


def random_mutations(base, seed, count):
    rng = random.Random(seed)
    out = []
    for n in range(count):
        doc = copy.deepcopy(base)
        path = rng.choice(list(leaves(doc)))
        node = doc
        for key in path[:-1]:
            node = node[key]
        how = rng.choice(["set", "set", "delete", "dup"])
        if how == "set":
            node[path[-1]] = copy.deepcopy(rng.choice(JUNK))
        elif how == "delete":
            del node[path[-1]]
        elif isinstance(node, list):
            node.insert(rng.randrange(len(node) + 1), copy.deepcopy(node[path[-1]]))
        else:
            node[path[-1] + "_"] = node[path[-1]]
        out.append((f"random {seed}.{n}: {how} {list(path)}", doc, path == ("plan_digest",)))
    return out


def verdict(text, manifest):
    """``(accepted, why)``: Python's ``read_plan`` (no pin) and ``check_against_manifest`` of a ``plan.json`` text."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "plan.json"
        path.write_bytes(text.encode("utf-8"))
        try:
            doc = _plan.read_plan(path)
            _plan.check_against_manifest(doc, manifest)
        except TilefishManifestError as e:
            # where the files are (the temporary directory, the checkout) is not part of the reason: the same file in every checkout
            return False, str(e).replace(str(path), "plan.json").replace(tmp, ".").replace(str(GOLDEN), "golden/plans")[:300]
    return True, ""


def diff(a, b, path=()):
    """The operations that make ``b`` of ``a``: ``["set", path, value]`` and ``["delete", path]`` (a list whose length changed is set whole)."""
    if type(a) is type(b) and isinstance(a, dict):
        ops = []
        for k in a:
            ops += [["delete", [*path, k]]] if k not in b else diff(a[k], b[k], (*path, k))
        return ops + [["set", [*path, k], b[k]] for k in b if k not in a]
    if type(a) is type(b) and isinstance(a, list) and len(a) == len(b):
        return [op for i, (x, y) in enumerate(zip(a, b)) for op in diff(x, y, (*path, i))]
    if type(a) is type(b) and a == b:
        return []
    return [["set", list(path), b]]


# (name, the text replaced in the file and what replaces it, a prefix, a suffix): edits that do not rewrite the digest, texts that are not strict JSON
TEXT_EDITS = [
    ("the file as it is", None, "", ""),
    ("an edit under the digest", ('"offset": 256', '"offset": 512'), "", ""),
    ("a duplicate key", ('"reuse":', '"reuse": false, "reuse":'), "", ""),
    ("NaN", ('"arena_alignment": 256', '"arena_alignment": NaN'), "", ""),
    ("an integer beyond a double", ('"arena_alignment": 256', '"arena_alignment": 9007199254740994'), "", ""),
    ("an integer of 2**53", ('"arena_alignment": 256', '"arena_alignment": 9007199254740992'), "", ""),
    ("a float for an int", ('"arena_alignment": 256', '"arena_alignment": 2.56e2'), "", ""),
    ("a byte order mark", None, "﻿", ""),
    ("not an object", None, "[", "]"),
    ("trailing data", None, "", "{}"),
]


def documents():
    """Every vector: ``{"base", "name", "accepted", "why"}`` and either ``"ops"`` and ``"rehash"`` (applied to the base document, then its
    ``plan_digest`` computed again when ``rehash``) or ``"text"`` (``{"replace": [old, new] or null, "prefix", "suffix"}``, applied once to the base
    file's text)."""
    records = []
    for base_name in BASES:
        directory = GOLDEN / base_name
        manifest = Manifest.load(directory)
        base_text = (directory / "plan.json").read_text()
        base = json.loads(base_text)
        made = []
        for name, mutate in mutations():
            doc = copy.deepcopy(base)
            mutate(doc)
            made.append((name, doc, False))
        made += random_mutations(base, seed=7 if base_name == "block" else 8, count=300)
        for name, doc, keep_digest in made:
            ops = diff(base, doc)
            rehash = isinstance(doc, dict) and "plan_digest" in doc and not keep_digest
            if rehash:
                doc["plan_digest"] = _plan.plan_digest(doc)
            accepted, why = verdict(json.dumps(doc, indent=1), manifest)
            records.append({"base": base_name, "name": name, "ops": ops, "rehash": rehash, "accepted": accepted, "why": why})
        for name, replace, prefix, suffix in TEXT_EDITS:
            text = prefix + (base_text if replace is None else base_text.replace(replace[0], replace[1], 1)) + suffix
            accepted, why = verdict(text, manifest)
            edit = {"replace": None if replace is None else list(replace), "prefix": prefix, "suffix": suffix}
            records.append({"base": base_name, "name": name, "text": edit, "accepted": accepted, "why": why})
    return records


def plan_vectors_json():
    return json.dumps({"documents": documents()}, indent=None, separators=(",", ":"), ensure_ascii=True) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    directory = pathlib.Path(argv[0])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "python_plans.json").write_text(plan_vectors_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
