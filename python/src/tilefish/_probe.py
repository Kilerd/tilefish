"""Intra-kernel instrumentation: probes at the phase boundaries a kernel declares, built into a variant of its own (``docs/instrument.md``).

A kernel marks a phase boundary with ``tilefish.probe("name")``: in a ``@tilefish.kernel`` body, a ``@triton.jit`` kernel or a ``@gluon.jit`` kernel
(an extern). In a production build a probe is nothing: Triton's frontend emits no operation for it, so the IR, the binary's machine code and every
digest are those of the kernel without it. In an **instrumented build** (``export(..., instrument=True)``, ``tilefish.run(..., probes=...)``) each probe is
a few PTX instructions in which lane 0 of every warp that reaches it writes one record of 16 bytes:

=========  ======  =====================================================================================================
offset     type    what
=========  ======  =====================================================================================================
0          u64     ``%globaltimer``: nanoseconds, one clock for the whole GPU (its resolution is the driver's: 32 ns measured
                   on a B300)
8          u32     ``%clock``: the cycle counter of the SM, the low 32 bits (fine-grained, but each SM has its own)
12         u16     the probe's id (0 ``@begin``, 1 ``@end``, 2 ``@calib``, then the kernel's probes in the order of first use)
14         u16     ``%smid``: the SM the warp runs on
=========  ======  =====================================================================================================

into its own slot of the **probe buffer** of the launch, which the host gives in the kernel's *profile scratch* parameter (the second of the two
scratch slots every Triton kernel has, null in a production launch, so the ABI does not change). The slot of warp ``w`` of the CTA of linear index
``c`` (``x + y * nx + z * nx * ny`` over the CUDA grid) is at ``(c * warps + w) * (16 + 16 * capacity)``: a header of 16 bytes (``u32`` magic
``0x31706674``, the CTA, the warp, the number of records the warp attempted), then ``capacity`` records. A warp keeps its first ``capacity`` records
and counts the rest: a count above the capacity says how many were dropped. A null buffer makes every probe skip its store.

The build adds three probes of its own to every kernel it instruments: ``@begin`` at the entry of every warp (warp-specialised workers too),
``@calib`` twice right after it, from the same instructions (the cycles from the first to the second are what one probe costs that warp, its code
in the instruction cache), and ``@end`` before every ``ret`` (which writes the count). They are
put in by rewriting the PTX of the entry, between Triton's ``ptx`` and ``cubin`` stages (:func:`instrument_ptx`); the kernel's probes are inline
assembly that refers to the registers the rewrite declares (:func:`probe_asm`).

Nothing here imports Triton or torch: :func:`probe` imports Triton's language when Triton's code generator calls it.
"""

from __future__ import annotations

import contextlib
import re
import struct
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .errors import TilefishBuildError, TilefishTypeError

FORMAT = "tilefish.probes.v1"  # the record format, the buffer layout and the probes the build adds: the ``format`` of an entry's ``instrument``
MAGIC = 0x31706674  # the first word of a warp's header: the bytes "tfp1"
HEADER_BYTES = 16  # magic, CTA, warp, count (u32 each)
RECORD_BYTES = 16  # globaltimer (u64), clock (u32), id (u16), SM (u16)
BEGIN, END, CALIB = 0, 1, 2  # the probes every instrumented kernel has
IMPLICIT = {BEGIN: "@begin", END: "@end", CALIB: "@calib"}
FIRST_ID = 3  # the id of a kernel's first probe
MAX_ID = 0xFFFF
DEFAULT_CAPACITY = 256  # records per warp
MAX_CAPACITY = 1 << 16
REWRITE = 1  # the version of the PTX rewrite and of the probe's instructions (in the key of an instrumented build)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/\-]{0,63}\Z")


def check_name(name: Any) -> str:
    """``name`` when it is the name of a probe: a string of at most 64 characters, a letter or ``_`` and then letters, digits and ``_ . : / -``."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise TilefishTypeError(
            f"tilefish.probe: the name of a probe is a string of 1 to 64 characters, a letter or '_' first and then letters, digits and '_.:/-' "
            f"(load.wait, mma/issue), got {name!r}"
        )
    return name


@dataclass(frozen=True)
class Instrument:
    """What an instrumented build records (``tilefish.instrument(...)``; ``instrument=True`` is the default one).

    ``capacity`` is the records each warp keeps per launch (the rest are counted, not kept); ``ops`` the names of the operators to instrument
    (``None``: every Triton, Gluon and ``@tilefish.kernel`` binary; a prebuilt PTX or cubin and a library op are never instrumented)."""

    capacity: int = DEFAULT_CAPACITY
    ops: "tuple[str, ...] | None" = None

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or not 1 <= self.capacity <= MAX_CAPACITY:
            raise TilefishTypeError(f"tilefish.instrument: capacity is the records a warp keeps, an integer from 1 to {MAX_CAPACITY}, got {self.capacity!r}")
        if self.ops is not None and (not isinstance(self.ops, tuple) or not all(isinstance(o, str) and o for o in self.ops)):
            raise TilefishTypeError(f"tilefish.instrument: ops is a list of operator names, got {self.ops!r}")

    def covers(self, op_name: str) -> bool:
        """Whether the binaries of the operator ``op_name`` are instrumented."""
        return self.ops is None or op_name in self.ops

    @property
    def block_bytes(self) -> int:
        """The bytes of one warp's slot of the probe buffer."""
        return HEADER_BYTES + RECORD_BYTES * self.capacity

    def key(self) -> "dict[str, Any]":
        """What a build key holds of it (an instrumented binary is never served for a production build, nor the other way round)."""
        return {"format": FORMAT, "capacity": self.capacity, "rewrite": REWRITE}


def instrument(*, capacity: int = DEFAULT_CAPACITY, ops: "Sequence[str] | None" = None) -> Instrument:
    """An instrumented build's settings, for ``export(..., instrument=...)`` and ``tilefish.Probes(...)``: ``capacity`` records per warp and launch
    (default 256), ``ops`` the operators to instrument (default: all)."""
    return Instrument(capacity, None if ops is None else tuple(ops))


def config(value: Any) -> "Instrument | None":
    """The :class:`Instrument` of an ``instrument=`` argument: ``True`` the default one, ``False`` or ``None`` none."""
    if value is None or value is False:
        return None
    if value is True:
        return Instrument()
    if isinstance(value, Instrument):
        return value
    raise TilefishTypeError(f"instrument= is True, False or tilefish.instrument(capacity=..., ops=...), got {value!r}")


# -- the frontend: tilefish.probe ------------------------------------------------------------------------------------------------------------

_state = threading.local()  # .config: the Instrument of the build this thread runs (None: production); .registry: probe ids of the frontend run in progress


def active() -> "Instrument | None":
    """The settings of the instrumented build this thread is running, or ``None``."""
    return getattr(_state, "config", None)


@contextlib.contextmanager
def instrumenting(cfg: "Instrument | None") -> "Iterator[None]":
    """Within the block, this thread's builds (the frontend, which emits the probes, and the PTX stage, which adds the rest) are of the
    instrumented variant ``cfg`` (``None``: production)."""
    previous = getattr(_state, "config", None)
    _state.config = cfg
    try:
        yield
    finally:
        _state.config = previous


@contextlib.contextmanager
def frontend() -> "Iterator[None]":
    """One run of Triton's (or Gluon's) frontend: in an instrumented build a fresh table of probe ids, so that every run of the frontend on a kernel
    gives each probe the same id (ids in the order the code generator meets the names)."""
    previous = getattr(_state, "registry", None)
    _state.registry = {} if active() is not None else None
    try:
        yield
    finally:
        _state.registry = previous


def probe(name: str, _semantic: Any = None) -> None:
    """``tilefish.probe("name")``: a phase boundary of the kernel, recorded by lane 0 of every warp that reaches it in an instrumented build and
    nothing at all in a production one (``docs/instrument.md``).

    In a ``@triton.jit`` or ``@gluon.jit`` kernel (Triton's code generator calls it as a builtin: ``import tilefish`` and write
    ``tilefish.probe("mma.issue")``), and in a ``@tilefish.kernel`` body, where it marks the point between the ops written before it and after it
    (after the loads of the inputs, before the stores of the results). The time between two consecutive probes of a warp is the phase
    ``"a -> b"``. A probe records when its warp issues it: put one after a wait (``mbarrier.wait``) to time the wait, not the asynchronous work it
    waits for."""
    if _semantic is not None:  # Triton's code generator: a builtin of the kernel being compiled
        _emit(name, _semantic)
        return None
    from .trace import current_context

    ctx = current_context()
    if ctx is None:
        raise TilefishTypeError(
            f"tilefish.probe({name!r}) marks a phase boundary inside a kernel: a @tilefish.kernel body, a @triton.jit or a @gluon.jit kernel; it was "
            "called outside one"
        )
    check_name(name)
    if ctx.kind != "kernel":
        raise TilefishTypeError(
            f"tilefish.probe({name!r}) is in a {ctx.kind} body, which has no kernel of its own: put probes in the kernels it calls (every kernel of an "
            "instrumented build records @begin and @end, docs/instrument.md)"
        )
    from .trace import _written_at

    ctx.probes.append((name, len(ctx.nodes), _written_at(ctx.body_code)))
    return None


probe.__triton_builtin__ = True  # type: ignore[attr-defined]  # Triton's code generator calls a builtin with _semantic=


def generated_probe(name: str, _semantic: Any = None) -> None:
    """The probe a lowered kernel's generated source calls (``_tf_probe("name")``, where its body has ``tilefish.probe("name")``): the probe of
    the compile when Triton's code generator calls it, and nothing when the generated function runs as Python (an interpreter of it)."""
    if _semantic is not None:
        _emit(name, _semantic)


generated_probe.__triton_builtin__ = True  # type: ignore[attr-defined]


def _emit(name: Any, semantic: Any) -> None:
    """The probe ``name`` in the kernel Triton's code generator is building: an inline assembly statement in an instrumented build, nothing in a
    production one."""
    import triton.language as tl

    name = check_name(tl.core._unwrap_if_constexpr(name))
    registry = getattr(_state, "registry", None)
    if registry is None:
        return
    pid = registry.get(name)
    if pid is None:
        pid = FIRST_ID + len(registry)
        if pid > MAX_ID:
            raise TilefishTypeError(f"tilefish.probe: a kernel has at most {MAX_ID - FIRST_ID + 1} probe names")
        registry[name] = pid
    tl.core.inline_asm_elementwise(probe_asm(pid, name) + "mov.u32 $0, 0;\n", "=r", [], dtype=tl.int32, is_pure=False, pack=1, _semantic=semantic)


def probe_asm(pid: int, name: str) -> str:
    """The PTX of probe ``pid``: lane 0 of the warp (``%tfp_cap`` is 0 in every other lane, and in every lane when the buffer is null) writes the
    record at index ``%tfp_n`` of its slot if that is below the capacity, and every lane counts it. The registers ``%tfp_*`` are the entry's
    (:func:`instrument_ptx`). The first line names the probe (in the PTX, the IR and the trace)."""
    return (
        f"// tilefish.probe {pid} {name}\n"
        "{\n"
        ".reg .pred %tfq;\n"
        ".reg .b32 %tfc, %tfs, %tfm, %tfl, %tfh;\n"
        ".reg .b64 %tft, %tfa;\n"
        "mov.u64 %tft, %globaltimer;\n"
        "mov.u32 %tfc, %clock;\n"
        "mov.u32 %tfs, %smid;\n"
        "setp.lt.u32 %tfq, %tfp_n, %tfp_cap;\n"
        "mad.wide.u32 %tfa, %tfp_n, 16, %tfp_rec;\n"
        "shl.b32 %tfm, %tfs, 16;\n"
        f"or.b32 %tfm, %tfm, {pid};\n"
        "mov.b64 {%tfl, %tfh}, %tft;\n"
        "@%tfq st.global.v4.b32 [%tfa], {%tfl, %tfh, %tfc, %tfm};\n"
        "add.u32 %tfp_n, %tfp_n, 1;\n"
        "}\n"
    )


# -- the PTX stage: the entry's registers, @begin, @calib and @end -------------------------------------------------------------------------

_ENTRY = re.compile(r"\.entry\s+([A-Za-z_$%][\w$]*)\s*\(")
_FUNC = re.compile(r"\.func\s+(?:\([^)]*\)\s*)?([A-Za-z_$%][\w$]*)\s*\(")
_RET = re.compile(r"^([ \t]*)(?:@(!?)(%[\w$]+)[ \t]+)?(ret|exit)(?:\.uni)?[ \t]*;[ \t]*(//.*)?$", re.M)
_MARK = "// tilefish.probe "


def _matching_brace(text: str, start: int) -> int:
    """The index of the ``}`` that closes the ``{`` at ``start`` (PTX: ``//`` and ``/* */`` comments are skipped)."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise TilefishBuildError("the PTX of the kernel has an entry whose body does not end (unbalanced braces): it cannot be instrumented")


def _record(pid: int) -> str:
    return "\t" + probe_asm(pid, IMPLICIT[pid]).replace("\n", "\n\t").rstrip("\t")


def _init(capacity: int, scratch: str) -> str:
    """The registers of the probes and this warp's slot, then ``@begin`` and ``@calib``: the first statements of the entry."""
    block = HEADER_BYTES + RECORD_BYTES * capacity
    return (
        "\n\t// tilefish probes (docs/instrument.md): this warp's slot of the probe buffer, which the profile scratch parameter points to\n"
        "\t.reg .b64 %tfp_hdr, %tfp_rec;\n"
        "\t.reg .b32 %tfp_n, %tfp_cap, %tfp_k;\n"
        "\t.reg .pred %tfp_again;\n"
        "\t{\n"
        "\t.reg .pred %tfon, %tflead;\n"
        "\t.reg .b64 %tfa, %tfo;\n"
        "\t.reg .b32 %tfx, %tfy, %tfz, %tfnx, %tfny, %tfnt, %tfnw, %tfcta, %tft, %tfw, %tfl, %tfslot;\n"
        f"\tld.param.u64 %tfa, [{scratch}];\n"
        "\tmov.u32 %tfx, %ctaid.x;\n"
        "\tmov.u32 %tfy, %ctaid.y;\n"
        "\tmov.u32 %tfz, %ctaid.z;\n"
        "\tmov.u32 %tfnx, %nctaid.x;\n"
        "\tmov.u32 %tfny, %nctaid.y;\n"
        "\tmad.lo.u32 %tfcta, %tfz, %tfny, %tfy;\n"
        "\tmad.lo.u32 %tfcta, %tfcta, %tfnx, %tfx;\n"
        "\tmov.u32 %tft, %tid.x;\n"
        "\tshr.u32 %tfw, %tft, 5;\n"
        "\tand.b32 %tfl, %tft, 31;\n"
        "\tmov.u32 %tfnt, %ntid.x;\n"
        "\tadd.u32 %tfnt, %tfnt, 31;\n"
        "\tshr.u32 %tfnw, %tfnt, 5;\n"
        "\tmad.lo.u32 %tfslot, %tfcta, %tfnw, %tfw;\n"
        f"\tmul.wide.u32 %tfo, %tfslot, {block};\n"
        "\tadd.u64 %tfp_hdr, %tfa, %tfo;\n"
        f"\tadd.u64 %tfp_rec, %tfp_hdr, {HEADER_BYTES};\n"
        "\tsetp.ne.u64 %tfon, %tfa, 0;\n"
        "\tsetp.eq.u32 %tflead, %tfl, 0;\n"
        "\tand.pred %tfon, %tfon, %tflead;\n"
        f"\tselp.u32 %tfp_cap, {capacity}, 0, %tfon;\n"
        "\tmov.u32 %tfp_n, 0;\n"
        f"\t@%tfon st.global.v4.b32 [%tfp_hdr], {{{MAGIC}, %tfcta, %tfw, 0}};\n"
        "\t}\n"
        + _record(BEGIN)
        # @calib twice, from the same instructions: the second's cycles after the first are what one probe costs the warp once its code is in the
        # instruction cache (the first runs it cold)
        + "\tmov.u32 %tfp_k, 0;\n$L__tfp_calib:\n"
        + _record(CALIB)
        + "\tadd.u32 %tfp_k, %tfp_k, 1;\n\tsetp.lt.u32 %tfp_again, %tfp_k, 2;\n\t@%tfp_again bra $L__tfp_calib;\n"
    )


def _end() -> str:
    """``@end`` and the count of the warp's records (by its lane 0, when the buffer is not null), before a ``ret``."""
    return _record(END) + (
        "\t{\n\t.reg .pred %tfe;\n\tsetp.ne.u32 %tfe, %tfp_cap, 0;\n"
        f"\t@%tfe st.global.b32 [%tfp_hdr+{HEADER_BYTES - 4}], %tfp_n;\n\t}}\n"
    )


def _function_with_probe(ptx: str) -> str:
    """The name of a ``.func`` of ``ptx`` whose body has a probe (``"?"`` when none is found)."""
    for f in _FUNC.finditer(ptx):
        start = ptx.find("{", f.end())
        semicolon = ptx.find(";", f.end())
        if start < 0 or 0 <= semicolon < start:  # a declaration, without a body
            continue
        if _MARK in ptx[start : _matching_brace(ptx, start)]:
            return f.group(1)
    return "?"


def instrument_ptx(ptx: str, capacity: int) -> str:
    """The PTX of the instrumented variant of a Triton kernel: the registers of the probes and this warp's slot, ``@begin`` and ``@calib`` first in
    the entry, and ``@end`` with the count before each ``ret`` (or ``exit``) of it. The probe buffer is the entry's last parameter, the profile
    scratch of a Triton kernel (a ``.u64``). Raises :class:`~tilefish.errors.TilefishBuildError` when the PTX is not one entry of that shape, or a
    probe is in a function the entry calls (a ``noinline`` function: its registers are not the entry's)."""
    m = _ENTRY.search(ptx)
    if m is None or _ENTRY.search(ptx, m.end()) is not None:
        raise TilefishBuildError("the PTX of the kernel is not one .entry: it cannot be instrumented")
    close = ptx.find(")", m.end())
    if close < 0:
        raise TilefishBuildError(f"the PTX entry '{m.group(1)}' has no end of its parameter list")
    params = [p.strip() for p in ptx[m.end() : close].split(",") if p.strip()]
    if not params or not params[-1].startswith(".param .u64"):
        raise TilefishBuildError(
            f"the PTX entry '{m.group(1)}' does not end with the 64-bit profile scratch parameter of a Triton kernel: it cannot be instrumented"
        )
    scratch = params[-1].split()[-1]
    if not re.fullmatch(r"[A-Za-z_$%][\w$]*", scratch):
        raise TilefishBuildError(f"the PTX entry '{m.group(1)}' has a last parameter {params[-1]!r} that is not a plain name")
    open_brace = ptx.find("{", close)
    if open_brace < 0:
        raise TilefishBuildError(f"the PTX entry '{m.group(1)}' has no body")
    end_brace = _matching_brace(ptx, open_brace)
    if _MARK in ptx[:open_brace] + ptx[end_brace:]:
        raise TilefishBuildError(
            f"a tilefish.probe is in the function '{_function_with_probe(ptx)}', which the kernel calls (a noinline function): a probe records into "
            "registers of the kernel's entry, so it must be in code Triton inlines into the kernel"
        )
    body = ptx[open_brace + 1 : end_brace]
    labels = [0]

    def ret(mm: "re.Match[str]") -> str:
        indent, negate, pred, op = mm.group(1), mm.group(2), mm.group(3), mm.group(4)
        if pred is None:
            return _end() + f"{indent}{op};"
        k = labels[0]
        labels[0] += 1
        skip = f"$L__tfp_skip_{k}"
        inverse = "" if negate else "!"
        return f"{indent}@{inverse}{pred} bra {skip};\n" + _end() + f"{indent}{op};\n{skip}:"

    rewritten, count = _RET.subn(ret, body)
    if count == 0:
        raise TilefishBuildError(f"the PTX entry '{m.group(1)}' has no ret or exit, before which a warp writes the count of its records")
    return ptx[: open_brace + 1] + _init(capacity, scratch) + rewritten + ptx[end_brace:]


def rewrite_ptx_stage(ptx: str) -> str:
    """What the PTX stage of a compile in this thread returns: ``ptx`` itself, or in an instrumented build its instrumented variant."""
    cfg = active()
    return ptx if cfg is None else instrument_ptx(ptx, cfg.capacity)


def _stages_hook(original: Any) -> Any:
    """``CUDABackend.add_stages`` wrapped: the ``ptx`` stage it registers goes through :func:`rewrite_ptx_stage` (which changes nothing in a thread
    that is not building an instrumented variant)."""

    def add_stages(self: Any, stages: Any, options: Any, *rest: Any, **kw: Any) -> Any:
        result = original(self, stages, options, *rest, **kw)
        make_ptx = stages.get("ptx")
        if make_ptx is not None and not getattr(make_ptx, "_tilefish_probes", False):

            def ptx_stage(src: Any, metadata: Any) -> Any:
                return rewrite_ptx_stage(make_ptx(src, metadata))

            ptx_stage._tilefish_probes = True  # type: ignore[attr-defined]
            stages["ptx"] = ptx_stage
        return result

    add_stages.__wrapped__ = original  # type: ignore[attr-defined]
    add_stages._tilefish_probes = True  # type: ignore[attr-defined]
    return add_stages


def install_stage_hook() -> None:
    """Wrap ``CUDABackend.add_stages`` once per process (under :data:`tilefish._ir.LOCK`, which every compile holds), so that an instrumented build's
    PTX is rewritten between Triton's ``ptx`` and ``cubin`` stages. Nothing when Triton's NVIDIA backend is not importable."""
    from . import _ir

    try:
        from triton.backends.nvidia.compiler import CUDABackend
    except ImportError:
        return
    with _ir.LOCK:
        current = CUDABackend.add_stages
        if not getattr(current, "_tilefish_probes", False):
            CUDABackend.add_stages = _stages_hook(current)


# -- where each probe is: the frontend's IR with its locations -------------------------------------------------------------------------------

_PROBE_OP = re.compile(r'tt\.elementwise_inline_asm\s+"// tilefish\.probe (\d+) ([A-Za-z0-9_.:/\-]+)\\0A')
_TT_FUNC = re.compile(r'^\s*tt\.func\s+(?:(public|private)\s+)?@("(?:[^"\\]|\\.)*"|[\w.$]+)')
_TT_CALL = re.compile(r'tt\.call\s+@("(?:[^"\\]|\\.)*"|[\w.$]+)')
_DEFAULT = re.compile(r"^\s*default\s*\{")
_PARTITION = re.compile(r"^\s*partition(\d+)\s*\(")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def _function_name(symbol: str) -> str:
    """The Python name of a function of the IR: ``ws_kernels._scale_load`` of ``ws_kernels._scale_load__TDfp16S64_64S...`` (its specialisation
    dropped)."""
    symbol = symbol.strip('"')
    module, dot, fn = symbol.rpartition(".")
    m = re.match(r"(_*[^_].*?)__", fn)
    return f"{module}{dot}{m.group(1) if m else fn}"


def probe_sites(source_ir: "str | None") -> "list[dict[str, Any]]":
    """Every probe of the frontend's IR (with its locations: Triton's ``asm["source"]``) of an instrumented build, as the manifest's table:
    ``{"id", "name", "sites": [{"loc": "file:line", "role": ..., "function": ...}]}``, by id. ``role`` is the partition of a warp-specialised
    kernel the probe runs in (``"default"``, ``"worker 0"``, ...; the code outside the partitions runs on the default warps), ``""`` in a kernel that
    Gluon does not partition; ``function`` the jit function it is written in."""
    if not source_ir:
        return []
    from ._tf32guard import _FILE_LOC, _LOC_TRAILER, _locations, _resolve

    table = _locations(source_ir)
    funcs: "dict[str, dict[str, Any]]" = {}
    public: "list[str]" = []
    current: "str | None" = None
    func_depth = depth = 0
    regions: "list[tuple[str, int]]" = []
    for line in source_ir.splitlines():
        bare = _STRING.sub('""', line)  # (the braces of a string, an inline assembly's, are not the module's)
        f = _TT_FUNC.match(line)
        if f is not None and current is None:
            current = f.group(2).strip('"')
            funcs[current] = {"probes": [], "calls": [], "ws": False}
            if f.group(1) != "private":
                public.append(current)
            func_depth = depth
        if current is not None:
            if _DEFAULT.match(bare):
                regions.append(("default", depth))
                funcs[current]["ws"] = True
            else:
                p = _PARTITION.match(bare)
                if p is not None:
                    regions.append((f"worker {int(p.group(1))}", depth))
                    funcs[current]["ws"] = True
            region = regions[-1][0] if regions else None
            op = _PROBE_OP.search(line)
            if op is not None:
                trailer = _LOC_TRAILER.search(line[op.end() :])
                where = ""
                if trailer is not None:
                    fl = _FILE_LOC.search(_resolve(trailer.group(1), table))
                    if fl is not None:
                        where = f"{fl.group(1)}:{fl.group(2)}"
                funcs[current]["probes"].append((int(op.group(1)), op.group(2), where, region))
            for c in _TT_CALL.finditer(line):
                funcs[current]["calls"].append((c.group(1).strip('"'), region))
        depth += bare.count("{") - bare.count("}")
        while regions and depth <= regions[-1][1]:
            regions.pop()
        if current is not None and depth <= func_depth:  # the function's body is closed (or it was a declaration, without one)
            current = None
    # the roles of each function: the kernel's (outside a partition: the default warps of a warp-specialised kernel), then down the calls
    roles: "dict[str, set[str]]" = {name: set() for name in funcs}
    todo: "list[str]" = []
    for name in public:
        roles[name].add("default" if funcs[name]["ws"] else "")
        todo.append(name)
    while todo:
        name = todo.pop()
        here = roles[name]
        for callee, region in funcs[name]["calls"]:
            if callee not in roles:
                continue
            new = {region} if region is not None else set(here)
            if not new <= roles[callee]:
                roles[callee] |= new
                todo.append(callee)
    by_id: "dict[int, dict[str, Any]]" = {}
    for name, f in funcs.items():
        for pid, pname, where, region in f["probes"]:
            entry = by_id.setdefault(pid, {"id": pid, "name": pname, "sites": []})
            for role in sorted({region} if region is not None else roles[name] or {""}):
                site = {"loc": where, "role": role, "function": _function_name(name)}
                if site not in entry["sites"]:
                    entry["sites"].append(site)
    return [by_id[k] for k in sorted(by_id)]


def entry_field(cfg: Instrument, sites: "Sequence[Mapping[str, Any]]") -> "dict[str, Any]":
    """The ``instrument`` of an instrumented binary's manifest entry (manifest schema 3, section 4.8)."""
    return {"format": FORMAT, "capacity": cfg.capacity, "probes": [dict(s) for s in sites]}


def check_field(who: str, value: Any) -> "str | None":
    """What is wrong with an entry's ``instrument`` (``None``: nothing), as both readers judge it: ``format`` this format, ``capacity`` an integer
    from 1 to 65536, ``probes`` a list of ``{"id", "name", "sites"}`` with ids from 3 to 65535 and names of probes, each once, and every site
    ``{"loc", "role", "function"}`` of strings; no other field anywhere."""
    if not isinstance(value, dict) or set(value) != {"format", "capacity", "probes"}:
        return f"{who}: instrument must be an object of format, capacity and probes, got {value!r}"
    if value["format"] != FORMAT:
        return f"{who}: instrument.format is {value['format']!r}, which this tilefish does not read (it reads {FORMAT!r})"
    cap = value["capacity"]
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_CAPACITY:
        return f"{who}: instrument.capacity must be an integer from 1 to {MAX_CAPACITY}, got {cap!r}"
    probes = value["probes"]
    if not isinstance(probes, list):
        return f"{who}: instrument.probes must be a list, got {probes!r}"
    ids: "set[int]" = set()
    names: "set[str]" = set()
    for k, p in enumerate(probes):
        if not isinstance(p, dict) or set(p) != {"id", "name", "sites"}:
            return f"{who}: instrument.probes[{k}] must be an object of id, name and sites, got {p!r}"
        pid, name, sites = p["id"], p["name"], p["sites"]
        if isinstance(pid, bool) or not isinstance(pid, int) or not FIRST_ID <= pid <= MAX_ID or pid in ids:
            return f"{who}: instrument.probes[{k}].id must be an integer from {FIRST_ID} to {MAX_ID}, each once, got {pid!r}"
        if not isinstance(name, str) or not _NAME.match(name) or name in names:
            return f"{who}: instrument.probes[{k}].name must be the name of a probe, each once, got {name!r}"
        if not isinstance(sites, list) or not all(
            isinstance(s, dict) and set(s) == {"loc", "role", "function"} and all(isinstance(v, str) for v in s.values()) for s in sites
        ):
            return f"{who}: instrument.probes[{k}].sites must be a list of {{loc, role, function}} strings, got {sites!r}"
        ids.add(pid)
        names.add(name)
    return None


# -- the probe buffer of a launch ------------------------------------------------------------------------------------------------------------


def buffer_bytes(capacity: int, warps: int, ctas: int) -> int:
    """The bytes of the probe buffer of a launch of ``ctas`` CTAs of ``warps`` warps."""
    return ctas * warps * (HEADER_BYTES + RECORD_BYTES * capacity)


@dataclass(frozen=True)
class WarpRecords:
    """What one warp of a launch recorded: its CTA (linear index over the CUDA grid), its warp, the records it attempted (``count``; more than
    the capacity: the rest were dropped) and the records kept, each ``(id, sm, globaltimer, clock)``."""

    cta: int
    warp: int
    count: int
    records: "tuple[tuple[int, int, int, int], ...]"

    @property
    def dropped(self) -> int:
        return max(0, self.count - len(self.records))


def decode(data: "bytes | bytearray | memoryview", capacity: int, warps: int, ctas: int) -> "tuple[list[WarpRecords], list[tuple[int, int]]]":
    """The records of a launch's probe buffer (its first :func:`buffer_bytes` bytes, read back from the device): every warp whose header is intact
    (the magic, its own CTA and warp), and the ``(cta, warp)`` slots that were not (a warp that never wrote its header: no kernel ran there, or it
    is not an instrumented kernel's buffer)."""
    block = HEADER_BYTES + RECORD_BYTES * capacity
    need = ctas * warps * block
    view = memoryview(data)
    if len(view) < need:
        raise TilefishBuildError(f"the probe buffer has {len(view)} bytes, but {ctas} CTAs of {warps} warps of {capacity} records need {need}")
    warps_out: "list[WarpRecords]" = []
    missing: "list[tuple[int, int]]" = []
    for c in range(ctas):
        for w in range(warps):
            off = (c * warps + w) * block
            magic, cta, warp, count = struct.unpack_from("<IIII", view, off)
            if magic != MAGIC or cta != c or warp != w:
                missing.append((c, w))
                continue
            kept = min(count, capacity)
            body = view[off + HEADER_BYTES : off + HEADER_BYTES + kept * RECORD_BYTES]
            recs = tuple((word & 0xFFFF, word >> 16, t, clk) for t, clk, word in struct.iter_unpack("<QII", body))
            warps_out.append(WarpRecords(c, w, count, recs))
    return warps_out, missing


__all__ = [
    "BEGIN",
    "CALIB",
    "DEFAULT_CAPACITY",
    "END",
    "FIRST_ID",
    "FORMAT",
    "HEADER_BYTES",
    "IMPLICIT",
    "Instrument",
    "MAGIC",
    "RECORD_BYTES",
    "WarpRecords",
    "buffer_bytes",
    "check_field",
    "check_name",
    "config",
    "decode",
    "entry_field",
    "instrument",
    "instrument_ptx",
    "probe",
    "probe_asm",
    "probe_sites",
]
