"""The device time of every step of a launch plan (:func:`tilefish.profile_plan`, ``tilefish-explain DIR --graph G --profile OUT``).

A runtime path of its own, outside every artefact: it runs the plan as :func:`tilefish.run` runs it (the same verified plan, the same binaries, the same
arena and its order), repeatedly, with timing events between the steps, and writes what it measured as a ``tilefish.step-timing.v1`` document, which
the advisor (:mod:`tilefish._advise`) reads. Nothing it does is in a manifest, a plan or a digest, its runs write what a run writes, and what they
change that outlives them (the caller's ``InOut`` tensors, the carried buffers the plan writes) is copied aside before the first run and back after
the last.

* ``method="graph"`` (the default): the steps are captured once in a CUDA graph with an event record node (``CU_EVENT_RECORD_EXTERNAL``) before the
  first step and after each, and the graph is replayed ``reps`` times; a step's time is the time between the nodes around it. That is how a host that
  captures the plan runs it. The plan is also captured without the events and timed as a whole, so the document says what the events cost.
* ``method="events"``: the steps are launched as a run launches them, with an event recorded on the stream before the first and after each; a spinning
  kernel first keeps the device busy while the host enqueues them, so that they run back to back rather than as fast as the host launches them.

``docs/advise.md`` says how accurate it is (its overhead, and an nsys run of the same plan).
"""

from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any, Mapping, Sequence

from ._advise import TIMING_SCHEMA, _parse_bucket
from ._launch import TORCH_DTYPE_NAMES
from .errors import TilefishRuntimeError


def _stats(xs: "Sequence[float]") -> "dict[str, float]":
    s = sorted(xs)

    def q(p: float) -> float:
        return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]

    cut = len(s) // 10  # the mean of the middle 80%: the events' timestamps tick in steps (about 2 us on B300), which a median snaps to and a mean averages over
    middle = s[cut:len(s) - cut] or s
    return {"median": round(statistics.median(s), 3), "mean": round(statistics.fmean(s), 3), "trimmed": round(statistics.fmean(middle), 3), "min": round(s[0], 3),
            "p10": round(q(0.1), 3), "p90": round(q(0.9), 3)}


def synthesize(doc: "Mapping[str, Any]", values: "Mapping[str, int]", device: Any, seed: int = 0) -> "dict[str, Any]":
    """Tensors for the parameters of a plan at a bucket that are given at a call (``input``, ``weight``, ``inout``): floats from a seeded normal
    distribution (times 0.5: finite in every dtype), integers zero (a valid index and count). Times do not depend on the values but where a kernel
    branches on them; give the real tensors (``externals=``) where they do."""
    import torch

    from ._plan import buffer_shapes

    shapes = buffer_shapes(doc, values)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = {}
    for b in doc["buffers"]:
        if b["role"] not in ("input", "weight", "inout"):
            continue
        dt = getattr(torch, TORCH_DTYPE_NAMES[b["dtype"]].split(".")[1])
        shape = shapes[b["name"]][1]
        if b["dtype"] in ("i32", "i64", "u8"):
            out[b["name"]] = torch.zeros(shape, dtype=dt, device=device)
        else:
            out[b["name"]] = (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * 0.5).to(dt)
    return out


def copy_bandwidth(device: Any, nbytes: int = 256 << 20, reps: int = 10) -> float:
    """The bandwidth (GB/s, bytes read plus bytes written) of a device-to-device copy of ``nbytes``: the memory roofline the advisor estimates with."""
    import torch

    a = torch.empty(nbytes, dtype=torch.uint8, device=device)
    b = torch.empty_like(a)
    b.copy_(a)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        b.copy_(a)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    del a, b
    return round(2 * nbytes / statistics.median(times) / 1e9, 1)


def profile_plan(
    graph: Any,
    plan: Any = None,
    *,
    bucket: "str | Mapping[str, int] | None" = None,
    reps: int = 50,
    warmup: int = 5,
    method: str = "graph",
    mode: "str | None" = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    externals: "Mapping[str, Any] | None" = None,
    devscalars: "Mapping[str, Any] | None" = None,
    seed: int = 0,
    out: "str | pathlib.Path | None" = None,
) -> "dict[str, Any]":
    """The device time of every step of the plan of ``graph`` at one bucket, as a ``tilefish.step-timing.v1`` document (written to ``out`` too when given).

    ``plan``: an export directory (``mode="replay"``, with ``pin=`` or ``trust_directory=True`` as for :func:`tilefish.run`), a :class:`~tilefish.Plan` or
    ``None`` (``mode="jit"``). ``bucket``: ``"M=64"`` or ``{"M": 64}`` (needed when the plan has several and ``externals`` does not say). ``externals``:
    ``{name: tensor}`` for the parameters given at a call (default: :func:`synthesize`, seeded by ``seed``); ``devscalars``: ``{name: value}`` for every
    device scalar (no default: a count changes what the kernels do). ``reps`` timed runs after ``warmup``; ``method``: ``"graph"`` or ``"events"`` (the
    module's doc). The runs change what outlives them (the caller's ``InOut`` tensors, the carried buffers the plan writes); their bytes are copied
    aside before the first run and back after the last, and what the arena notes of the carried buffers is as before. A plan that reads a carried
    buffer before it writes it is refused, as a run of it alone is. Needs torch and a CUDA device of the plan's architecture."""
    import torch

    from . import __version__, _cuda, _planrun
    from ._plan import Plan

    if not isinstance(reps, int) or reps < 1 or not isinstance(warmup, int) or warmup < 0:
        raise TilefishRuntimeError(f"profile_plan: reps is an integer >= 1 and warmup >= 0, got {reps!r} and {warmup!r}")
    if mode is None:
        mode = "jit" if plan is None or isinstance(plan, Plan) else "replay"
    runner = _planrun._runner_for(graph, "tilefish.profile_plan", mode, plan, pin, trust_directory)
    doc = runner._doc
    combos = [b["values"] for b in doc["buckets"]]
    wanted = _parse_bucket(bucket)
    scalars = [b["name"] for b in doc["buffers"] if b["role"] == "devscalar"]
    devscalars = dict(devscalars or {})
    missing = [n for n in scalars if n not in devscalars]
    if missing:
        raise TilefishRuntimeError(f"profile_plan: give the value of every device scalar of '{graph.name}' with devscalars= ({', '.join(missing)} missing): it changes what the steps do")
    if externals is None:
        if wanted is None:
            if len(combos) != 1:
                raise TilefishRuntimeError(f"profile_plan: the plan of '{graph.name}' has {len(combos)} buckets: say which with bucket= ({combos[:6]}...)")
            wanted = dict(combos[0])
        if wanted not in combos:
            raise TilefishRuntimeError(f"profile_plan: the plan of '{graph.name}' has no bucket {wanted} ({combos[:6]}...)")
        device = torch.device("cuda", torch.cuda.current_device())
        externals = synthesize(doc, wanted, device, seed)
    kwargs = {**dict(externals), **devscalars}
    result = runner.profile((), kwargs, reps=reps, warmup=warmup, method=method)
    values = result.get("bucket") or wanted
    index = result.get("device", torch.cuda.current_device())  # the device the plan ran on (its tensors')
    device = torch.device("cuda", index)
    steps_doc = next((b["steps"] for b in doc["buckets"] if values is not None and b["values"] == values), None)
    steps = []
    for i, ts in enumerate(result["steps"]):
        st = steps_doc[i] if steps_doc is not None else {}
        steps.append({"index": i, "kind": st.get("kind"), "entry": st.get("entry"), "us": _stats(ts)})
    total = sum(s["us"]["trimmed"] for s in steps)
    plan_us = _stats(result["plan"])
    # what the events around a step cost it (measured on B300 at about 3 us a step: an event node waits for the step before it to drain, where
    # consecutive kernels of a graph overlap one's tail with the next one's launch): the steps' sum less the plan's own time, shared evenly. A step's
    # "net" is its trimmed mean less that: the steps' nets add up to the plan's time
    event_us = max(0.0, (total - plan_us["trimmed"]) / len(steps)) if steps else 0.0
    for s in steps:
        s["us"]["net"] = round(max(s["us"]["trimmed"] - event_us, 0.0), 3)
    major, minor = torch.cuda.get_device_capability(index)
    timing = {
        "schema": TIMING_SCHEMA,
        "tilefish": __version__,
        "graph": graph.name,
        "plan_digest": doc.get("plan_digest") or runner.plan.digest,
        "manifest_digest": doc.get("manifest_digest"),
        "arch": doc.get("arch") or f"sm_{major}{minor}",
        "bucket": dict(values or {}),
        "method": method,
        "reps": reps,
        "warmup": warmup,
        "device": {"name": torch.cuda.get_device_name(index), "arch": f"sm_{major}{minor}", "sm_count": _cuda.sm_count(index),
                   "clock_khz": _cuda.clock_khz(index), "copy_GBps": copy_bandwidth(device)},
        "steps": steps,
        "sum_us": round(total, 3),
        "plan_us": plan_us,
        "event_us": round(event_us, 3),
        "overhead": round(total / plan_us["trimmed"] - 1.0, 4) if plan_us["trimmed"] > 0 else None,
    }
    if out is not None:
        pathlib.Path(out).write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")
    return timing


__all__ = ["copy_bandwidth", "profile_plan", "synthesize"]
