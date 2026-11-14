"""The TF32 guard: a kernel that asks for IEEE f32 dots is not built into TF32 tensor-core MMAs.

An f32 ``tl.dot(a, b, input_precision="ieee")`` asks for every product and sum in f32. A tensor core has no f32 multiply: an f32 MMA is a TF32
one (``tcgen05.mma ... kind::tf32``, ``wgmma ... .tf32``, ``mma.sync ... .tf32``), which keeps 10 of the 23 bits of each input's mantissa. Triton
3.5.1 nevertheless lowers such a dot to ``tcgen05.mma ... kind::tf32`` on sm_100 and sm_103 (measured on a B300): its MMAv5 path skips the
precision check of the other tensor-core paths when an operand derives from a load narrower than 32 bits (an f16, bf16 or i8 tensor converted to
f32) or from ``tl.join``, and the tile fits the instruction (``M % 64 == 0``, ``N % 16 == 0``, ``K >= 8``, 4 or 8 warps). The bits change and
nothing says so; the same kernel on sm_90 or sm_120 stays on FMAs and is exact.

The guard does not copy that rule, which a later Triton may change: it reads what the compiler made of the kernel.

* **Asked.** The IR Triton's frontend builds (the text whose digest identifies the binary, :mod:`tilefish._ir`) has a dot of f32 operands whose
  input precision is IEEE: a ``tt.dot`` that prints no ``inputPrecision`` (IEEE is the default, which MLIR leaves out) or ``inputPrecision =
  ieee``, or a ``ttng.warp_group_dot`` (Gluon's ``warpgroup_mma``; Triton 3.5.1 refuses an IEEE one of f32 operands at its lowering) likewise. A
  dot that says ``tf32`` or ``tf32x3`` asks for TF32, and a kernel without an IEEE f32 dot is never refused.
* **Got.** The PTX of the binary has a TF32 tensor-core instruction (:func:`ptx_tf32_mmas`), or its TTGIR has an op that multiplies f32 operands
  on tensor cores (:func:`tensor_core_ops`: a ``ttng.tc_gen5_mma`` of f32 operands, a ``ttng.warp_group_dot``, a ``tt.dot`` with an
  ``nvidia_mma`` result layout).

A kernel whose f32 dots all ask for IEEE (and that has no tensor-core op of its own, as a Gluon kernel may) is refused when it got any TF32 MMA:
nothing else in it could have made one. A kernel that mixes IEEE dots with dots that allow TF32 (or with Gluon's explicit ``tcgen05_mma``) is
judged op by op on its TTGIR. An op that keeps the dot's precision (``warp_group_dot``, ``tt.dot``) says it; a ``tc_gen5_mma`` does not, and is
traced back to the dot it was made from by its source location (the TTIR and the TTGIR of one compile print the same location, the call sites
included, for a dot and for the MMA made of it). It is refused when that dot asked for IEEE or when it cannot be traced to a dot that allowed
TF32; in a Gluon kernel a ``tc_gen5_mma`` is the kernel's own (Gluon has no pass that makes an MMA of a dot). A PTX with TF32 MMAs whose TTGIR
shows no tensor-core op at all is refused too: that is a Triton whose TTGIR this module does not know.

Pure: text in, verdict out; no Triton.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# InputPrecision of Triton 3.5.1 (TritonAttrDefs.td), as an attribute dictionary prints it (``{inputPrecision = 0 : i32}``)
_PRECISION_NAMES = {"0": "tf32", "1": "tf32x3", "2": "ieee"}
IEEE = "ieee"
# the ops: a dot that carries an input precision, Triton's ``tt.dot`` (not ``tt.dot_scaled``) and Hopper's ``ttng.warp_group_dot`` (not
# ``warp_group_dot_wait``), and an MMAv5 op, ``ttng.tc_gen5_mma`` (not ``tc_gen5_mma_scaled``, whose operands are 4 to 8 bits), which has none
_OP = re.compile(r"(?<![\w.])(tt\.dot|ttng\.warp_group_dot|ttng\.tc_gen5_mma)(?![\w.])")
_DOTS = ("tt.dot", "ttng.warp_group_dot")
_PRECISION = re.compile(r"\binputPrecision\s*=\s*(\w+)")
# the shape and element type of a tensor or memory descriptor type, and its encoding: ``tensor<64x16xf32, #blocked>`` -> ``64x16xf32``, ``#blocked``
_TYPE = re.compile(r"(?:tensor|!ttg\.memdesc)<([^,<>]*)(?:,\s*([^,>]+(?:<[^>]*>)?))?")
_LOC_TRAILER = re.compile(r"\sloc\((.*)\)\s*$")
_LOC_DEF = re.compile(r"^(#loc\d*)\s*=\s*loc\((.*)\)\s*$", re.M)
_LOC_REF = re.compile(r"#loc\d*(?!\w)")
_FILE_LOC = re.compile(r'"([^"]+)":(\d+):(\d+)')
_MMA_ALIAS = re.compile(r"^(#[\w.]+)\s*=\s*#ttg\.nvidia_mma<", re.M)
# a tensor-core instruction of PTX: tcgen05 (sm_100+), wgmma (sm_90), mma.sync and mma.sp (sm_80+), as one opcode token
_PTX_MMA = re.compile(r"(?<![\w.])(tcgen05\.mma[\w.:]*|wgmma\.mma_async[\w.:]*|mma(?:\.sp(?:::ordered_metadata)?)?\.sync[\w.:]*)")


@dataclass(frozen=True)
class Op:
    """One dot or MMA of an IR text: its op name, its input precision (``None``: the op has none, a ``tc_gen5_mma``), whether both operands
    are f32, whether it runs on tensor cores (by its op, or a ``tt.dot`` by its result layout), and its source location with every alias
    resolved (``None`` when the text has none)."""

    name: str
    precision: "str | None"
    f32: bool
    tensor_core: bool
    loc: "str | None"

    @property
    def where(self) -> str:
        """The source position of the op (``file:line:col`` of the innermost location, the dot itself), or ``""``."""
        if self.loc is None:
            return ""
        m = _FILE_LOC.search(self.loc)
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}" if m else ""


def _locations(text: str) -> "dict[str, str]":
    return {m.group(1): m.group(2) for m in _LOC_DEF.finditer(text)}


def _resolve(loc: str, table: "dict[str, str]") -> str:
    """``loc`` with every ``#locN`` alias replaced by what it stands for, recursively (a location prints the same in any module then)."""
    for _ in range(64):  # aliases nest a few levels (a call site of a named location of a file location); a cycle is not MLIR
        resolved = _LOC_REF.sub(lambda m: f"({table[m.group(0)]})" if m.group(0) in table else m.group(0), loc)
        if resolved == loc:
            break
        loc = resolved
    return loc


def _element(shape: str) -> str:
    return shape.strip().split("x")[-1]


def _ops(text: str) -> "list[Op]":
    """The dots and the MMAv5 ops of an MLIR text (TTIR or TTGIR, custom form, as Triton prints it: one op per line), in order."""
    table = _locations(text)
    mma_aliases = {m.group(1) for m in _MMA_ALIAS.finditer(text)}
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        matches = list(_OP.finditer(line))
        for i, m in enumerate(matches):
            name = m.group(1)
            body = line[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(line)]
            trailer = _LOC_TRAILER.search(body)
            if trailer:
                body = body[: trailer.start()]
            types = _TYPE.findall(body)  # the operand types in order: a, b, then the accumulator or the result
            f32 = len(types) >= 2 and _element(types[0][0]) == "f32" and _element(types[1][0]) == "f32"
            if name in _DOTS:
                p = _PRECISION.search(body)
                precision: "str | None" = IEEE if p is None else _PRECISION_NAMES.get(p.group(1), p.group(1))
                if name == "ttng.warp_group_dot":
                    tensor_core = True
                else:  # a tt.dot runs on tensor cores when its result has an nvidia_mma layout (MMAv2), on FMAs otherwise
                    encoding = types[2][1].strip() if len(types) >= 3 else ""
                    tensor_core = encoding in mma_aliases or encoding.startswith("#ttg.nvidia_mma")
            else:
                precision, tensor_core = None, True
            loc = _resolve(trailer.group(1), table) if trailer else None
            found.append(Op(name=name, precision=precision, f32=f32, tensor_core=tensor_core, loc=loc))
    return found


def f32_dots(text: str) -> "list[Op]":
    """The dots of f32 operands in an IR text (``tt.dot``, ``ttng.warp_group_dot``), with their input precision (``"ieee"``, ``"tf32"``,
    ``"tf32x3"``)."""
    return [op for op in _ops(text) if op.f32 and op.precision is not None]


def tensor_core_ops(text: str) -> "list[Op]":
    """The ops of an IR text that multiply f32 operands on tensor cores, which is in TF32: a ``ttng.tc_gen5_mma`` of f32 operands, a
    ``ttng.warp_group_dot`` of f32 operands, a ``tt.dot`` of f32 operands with an ``nvidia_mma`` result layout."""
    return [op for op in _ops(text) if op.f32 and op.tensor_core]


def ptx_tf32_mmas(ptx: str) -> "list[str]":
    """The TF32 tensor-core instructions of a PTX text, one opcode per instruction (``tcgen05.mma.cta_group::1.kind::tf32``,
    ``wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32``, ``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``). Comments are skipped."""
    found = []
    for line in ptx.splitlines():
        code = line.split("//", 1)[0]
        for m in _PTX_MMA.finditer(code):
            opcode = m.group(1)
            if ("kind::tf32" in opcode) if opcode.startswith("tcgen05") else (".tf32" in opcode):
                found.append(opcode)
    return found


def _counted(items: "list[str]") -> str:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return ", ".join(f"{n} x {item}" for item, n in counts.items())


def problem(frontend_ir: str, ptx: str, *, language: str = "triton", ttir: "str | None" = None, ttgir: "str | None" = None) -> "str | None":
    """What is wrong with a binary whose kernel asked for IEEE f32 dots (``None``: nothing). Pure.

    ``frontend_ir`` is the IR the frontend built (:func:`tilefish._ir.ttir_text`; a Gluon kernel's TTGIR), ``ptx`` the binary's PTX, and
    ``ttir`` and ``ttgir`` the TTIR and TTGIR of the compile with their locations (``compiled.asm``), when there was one: a build served by the
    cache has its PTX only, which judges every kernel but one that mixes IEEE dots with dots that allow TF32 (that one was judged, with its
    TTGIR, by the build that stored the entry). The message says what was asked, what was found and where, without the fixes
    (:func:`tilefish._build.check_tf32` adds them).
    """
    ops = _ops(frontend_ir)
    asked = [op for op in ops if op.f32 and op.precision == IEEE]
    if not asked:
        return None
    # what else in the kernel may make a TF32 MMA: a dot that says tf32 or tf32x3, a Gluon kernel's own tcgen05_mma or TF32 warpgroup_mma
    allowed = [op for op in ops if op.f32 and op.precision != IEEE]
    in_ptx = ptx_tf32_mmas(ptx)
    compiled = tensor_core_ops(ttgir) if ttgir is not None else []
    what = f"{len(asked)} f32 dot{'s' if len(asked) != 1 else ''} with IEEE input precision"
    if not allowed:  # every TF32 MMA of the binary is made of a dot that asked for IEEE: nothing else in the kernel makes one
        if not in_ptx and not compiled:
            return None
        found = [f"{_counted(in_ptx)} in its PTX"] if in_ptx else []
        if compiled:
            found.append(f"{_counted([op.name + ' of f32 operands' for op in compiled])} in its TTGIR")
        return f"it asks for {what}, and was compiled into TF32 tensor-core MMAs: {'; '.join(found)}{_located(compiled, ttir, language)}"
    if ttgir is None:
        return None
    if in_ptx and not compiled:
        return (
            f"it asks for {what} next to dots that allow TF32, its PTX has TF32 tensor-core MMAs ({_counted(in_ptx)}), and its TTGIR has no "
            "tensor-core op this check knows, so it cannot tell which dots they were made of"
        )
    sources = f32_dots(ttir) if ttir is not None else []
    trapped: list[Op] = []
    untraced: list[Op] = []
    for op in compiled:
        if op.precision is not None:  # the op keeps the dot's precision (a warp_group_dot, a tt.dot with an MMA layout)
            if op.precision == IEEE:
                trapped.append(op)
            continue
        if language == "gluon":  # a tc_gen5_mma of a Gluon kernel is its own tcgen05_mma: Gluon makes no MMA of a dot
            continue
        precisions = {d.precision for d in sources if op.loc is not None and d.loc == op.loc}
        if precisions == {IEEE}:
            trapped.append(op)
        elif not precisions or IEEE in precisions:  # made of no dot this check sees, or of a location shared with an IEEE dot
            untraced.append(op)
    if trapped:
        where = sorted({d.where for d in sources if d.precision == IEEE and d.where and any(d.loc == op.loc for op in trapped)})
        return (
            f"it asks for {what}, and {len(trapped)} TF32 tensor-core MMA{'s' if len(trapped) != 1 else ''} of its TTGIR "
            f"{'are' if len(trapped) != 1 else 'is'} made of {'them' if len(asked) != 1 else 'it'}: "
            f"{_counted([op.name + ' of f32 operands' for op in trapped])}{(' (the dot at ' + ', '.join(where) + ')') if where else ''}"
        )
    if untraced:
        return (
            f"it asks for {what} next to dots that allow TF32, and {len(untraced)} TF32 tensor-core MMA{'s' if len(untraced) != 1 else ''} of its "
            f"TTGIR cannot be traced, by source location, to a dot that allowed TF32: {_counted([op.name for op in untraced])}"
        )
    return None


def _located(compiled: "list[Op]", ttir: "str | None", language: str) -> str:
    """Where the IEEE dots that became the ``compiled`` MMAs are in the source, as far as the TTIR says (``""`` when it does not)."""
    if ttir is None or language == "gluon":
        return ""
    where = sorted({d.where for d in f32_dots(ttir) if d.precision == IEEE and d.where and any(d.loc == op.loc for op in compiled)})
    return f" (the dot{'s' if len(where) != 1 else ''} at {', '.join(where)})" if where else ""
