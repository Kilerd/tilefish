"""The CUDA math library a lowered kernel's bits reproduce, and the guard on it (``docs/numerics.md``; manifest schema 3, section 4.7).

Some ops of a lowered kernel are, by their numerics contract, the bits of a function of CUDA's math library: in IEEE math ``exp`` and ``silu``
are built on ``expf``, ``rsqrt`` is ``rsqrtf`` and ``erf`` is ``erff`` (inline PTX read from NVRTC), and in either math ``sin``, ``cos`` and
``tanh`` are libdevice's ``sinf``, ``cosf`` and ``tanhf`` (notes 10, 13 to 15 of :mod:`tilefish.lower`). Those functions are not the same in
every CUDA release (CUDA 13.2 changed ``erff``), and libdevice is a file Triton links. So:

* the generated source of a kernel says which of these functions it uses, and whether it links libdevice, by the code tilefish writes for
  them (:data:`MARKERS`: ``_tf_exp_ieee(``, ``libdevice.tanh(``, ``tl.sin(``, ...; :func:`of_source`, which a test holds to :func:`of_ops` of
  every op in every math). The source, and so its digest, is not changed for it;
* a build reads it from the source it compiles and records, on the manifest entry, ``cuda_math``: the functions, the CUDA releases on which
  tilefish measured each to be the CUDA C++ function's bits over every f32 (:data:`VERIFIED`, the releases every listed function was measured
  on), and the sha256 of the libdevice it linked;
* a run checks it: where torch's CUDA release (``torch.version.cuda``, whose math library torch's own ops are built with) is not one the
  functions were measured on, or a replay's process would link another libdevice than the entry's, it warns with
  :class:`~tilefish.errors.TilefishCudaMathWarning` (an error under ``warnings.simplefilter("error", TilefishCudaMathWarning)``). The binary is
  still the binary that was exported; what the warning says is that torch's ops, and the reference mode built on them, may not be its bits here.
  The Rust host refuses a mismatch with ``Manifest::check_cuda_math``.

:data:`VERIFIED` is kept by the test procedure of ``docs/numerics.md`` (``pytest -m cuda_math``, with ``TILEFISH_NVRTC_PATHS`` naming the NVRTC
of each release): a release is listed for a function only when that function compiled by its NVRTC is the op's bits on every f32.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import sys
import warnings
from typing import Any, Iterable, Mapping

from .errors import TilefishCudaMathWarning

# function -> the CUDA releases on which the op built on it is the bits of the CUDA C++ function (nvcc's defaults) on every f32, measured
# against NVRTC of each release on a B300 (sm_103): tests/test_gpu_cuda_math.py
VERIFIED: "dict[str, tuple[str, ...]]" = {
    "cosf": ("12.8", "12.9", "13.0", "13.1", "13.2"),
    "erff": ("12.8", "12.9", "13.0", "13.1"),  # 13.2 changed erff: other coefficients, one ULP on 353 987 744 inputs
    "expf": ("12.8", "12.9", "13.0", "13.1", "13.2"),
    "rsqrtf": ("12.8", "12.9", "13.0", "13.1", "13.2"),
    "sinf": ("12.8", "12.9", "13.0", "13.1", "13.2"),
    "tanhf": ("12.8", "12.9", "13.0", "13.1", "13.2"),
}

# op -> {math: the CUDA math library function its bits are, by its contract}
FUNCTIONS: "dict[str, dict[str, str]]" = {
    "exp": {"ieee": "expf"},
    "silu": {"ieee": "expf"},
    "rsqrt": {"ieee": "rsqrtf"},
    "erf": {"ieee": "erff"},
    "sin": {"fast": "sinf", "ieee": "sinf"},
    "cos": {"fast": "cosf", "ieee": "cosf"},
    "tanh": {"fast": "tanhf", "ieee": "tanhf"},
}
# op -> the maths in which Triton links libdevice for it
LIBDEVICE: "dict[str, tuple[str, ...]]" = {
    "sin": ("fast", "ieee"),
    "cos": ("fast", "ieee"),
    "tanh": ("fast", "ieee"),
    "gelu": ("fast", "ieee"),
    "erf": ("fast",),
}

# what the generated source of each op is made of (tilefish.lower): a module-level helper of IEEE math (its name, perhaps with a suffix when the
# kernel has a parameter of that name), or the call Triton links libdevice for; -> (the CUDA math function, whether it links libdevice)
MARKERS: "tuple[tuple[re.Pattern[str], str | None, bool], ...]" = (
    (re.compile(r"\b_tf_exp_ieee(_[0-9]+)?\("), "expf", False),
    (re.compile(r"\b_tf_silu_ieee(_[0-9]+)?\("), "expf", False),
    (re.compile(r"\b_tf_rsqrt_ieee(_[0-9]+)?\("), "rsqrtf", False),
    (re.compile(r"\b_tf_erf_ieee(_[0-9]+)?\("), "erff", False),
    (re.compile(r"\btl\.sin\("), "sinf", True),
    (re.compile(r"\btl\.cos\("), "cosf", True),
    (re.compile(r"\blibdevice\.tanh\("), "tanhf", True),
    (re.compile(r"\btl\.math\.erf\("), None, True),
)
_FUNCTION = re.compile(r"[a-z_][a-z0-9_]*\Z")
_RELEASE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def of_ops(ops: "Iterable[tuple[str, str]]") -> "tuple[tuple[str, ...], bool]":
    """The CUDA math functions ``(op, math)`` pairs reproduce (sorted, once each), and whether any links libdevice."""
    functions: set[str] = set()
    libdevice = False
    for op, math in ops:
        f = FUNCTIONS.get(op, {}).get(math)
        if f is not None:
            functions.add(f)
        libdevice = libdevice or math in LIBDEVICE.get(op, ())
    return tuple(sorted(functions)), libdevice


def of_source(source: str) -> "tuple[tuple[str, ...], bool]":
    """The CUDA math functions (sorted, once each) a generated source's code is made of, and whether it links libdevice (:data:`MARKERS`)."""
    functions: set[str] = set()
    libdevice = False
    for pattern, function, links in MARKERS:
        if pattern.search(source):
            if function is not None:
                functions.add(function)
            libdevice = libdevice or links
    return tuple(sorted(functions)), libdevice


def _release_key(r: str) -> "tuple[int, int]":
    major, minor = r.split(".")
    return int(major), int(minor)


def verified(functions: "Iterable[str]") -> "tuple[str, ...]":
    """The CUDA releases on which every one of ``functions`` was measured (sorted by release)."""
    common: "set[str] | None" = None
    for f in functions:
        releases = set(VERIFIED.get(f, ()))
        common = releases if common is None else common & releases
    return tuple(sorted(common or (), key=_release_key))


@functools.lru_cache(maxsize=16)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def libdevice_path(options: "Mapping[str, Any] | None" = None) -> "str | None":
    """The libdevice file Triton links for compile options ``options``: theirs (``extern_libs``), else ``TRITON_LIBDEVICE_PATH`` (the knob), else
    the one Triton ships; ``None`` without Triton."""
    libs = dict((options or {}).get("extern_libs") or {})
    if libs.get("libdevice"):
        return str(libs["libdevice"])
    try:
        from triton import knobs
        import triton.backends.nvidia as nvidia
    except ImportError:
        return None
    knob = getattr(getattr(knobs, "nvidia", None), "libdevice_path", None)
    if knob:
        return str(knob)
    return os.path.join(os.path.dirname(nvidia.__file__), "lib", "libdevice.10.bc")


def libdevice_sha256(options: "Mapping[str, Any] | None" = None) -> "str | None":
    """The sha256 of :func:`libdevice_path` (``None`` where there is no Triton or the file cannot be read)."""
    path = libdevice_path(options)
    if path is None:
        return None
    try:
        st = os.stat(path)
        return _file_sha256(path, st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def entry_field(source: str, options: "Mapping[str, Any] | None" = None) -> "dict[str, Any] | None":
    """The ``cuda_math`` of the manifest entry of a binary built from the generated ``source`` (``None``: the kernel uses none of these functions)."""
    functions, libdevice = of_source(source)
    if not functions and not libdevice:
        return None
    field: dict[str, Any] = {"functions": list(functions), "cuda": list(verified(functions))}
    if libdevice:
        sha = libdevice_sha256(options)
        if sha is not None:
            field["libdevice"] = sha
    return field


def check_field(who: str, field: Any) -> "str | None":
    """What is wrong with an entry's ``cuda_math`` (``None``: nothing), the rules both readers hold."""
    if not isinstance(field, dict):
        return f"{who}: cuda_math must be an object, got {field!r}"
    extra = sorted(set(field) - {"functions", "cuda", "libdevice"})
    if extra:
        return f"{who}: cuda_math has unknown fields {extra}"
    functions, cuda = field.get("functions"), field.get("cuda")
    if not isinstance(functions, list) or not all(isinstance(f, str) and _FUNCTION.match(f) for f in functions):
        return f"{who}: cuda_math.functions must be a list of function names, got {functions!r}"
    if functions != sorted(set(functions)):
        return f"{who}: cuda_math.functions must be sorted, each once, got {functions!r}"
    if not isinstance(cuda, list) or not all(isinstance(r, str) and _RELEASE.match(r) for r in cuda):
        return f"{who}: cuda_math.cuda must be a list of CUDA releases such as \"13.0\", got {cuda!r}"
    if cuda != sorted(set(cuda), key=_release_key):
        return f"{who}: cuda_math.cuda must be sorted by release, each once, got {cuda!r}"
    if not functions and cuda:
        return f"{who}: cuda_math.cuda lists releases {cuda!r} for no function"
    libdevice = field.get("libdevice")
    if libdevice is not None and not (isinstance(libdevice, str) and _SHA256.match(libdevice)):
        return f"{who}: cuda_math.libdevice must be a sha256 (64 lowercase hex digits), got {libdevice!r}"
    if not functions and libdevice is None:
        return f"{who}: cuda_math names no function and no libdevice (an entry without either has no cuda_math)"
    return None


def torch_release() -> "str | None":
    """The CUDA release torch was built with (``"12.8"``), when torch is imported; ``None`` otherwise."""
    torch = sys.modules.get("torch")
    version = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(version, str):
        return None
    m = re.match(r"(\d+)\.(\d+)", version)
    return f"{int(m.group(1))}.{int(m.group(2))}" if m else None


def check(who: str, field: "Mapping[str, Any] | None", *, libdevice_here: "str | None" = None) -> None:
    """Warn (:class:`~tilefish.errors.TilefishCudaMathWarning`) where ``field`` (an entry's ``cuda_math``) does not hold in this process: torch's
    CUDA release is not one its functions were measured on, or ``libdevice_here`` (the libdevice a build here would link) is not the one it
    records."""
    if not field:
        return
    release = torch_release()
    functions = list(field.get("functions") or ())
    if functions and release is not None and release not in (field.get("cuda") or ()):
        warnings.warn(
            f"{who}: its {', '.join(functions)} {'is' if len(functions) == 1 else 'are'} the bits of CUDA's math library on CUDA "
            f"{', '.join(field.get('cuda') or ()) or 'no release measured'}, and torch here is built with CUDA {release}: torch's own ops (and the "
            "reference mode, which uses them) may not be the binary's bits. Re-run the CUDA math tests on this CUDA (docs/numerics.md)",
            TilefishCudaMathWarning,
            stacklevel=3,
        )
    recorded = field.get("libdevice")
    if recorded is not None and libdevice_here is not None and libdevice_here != recorded:
        warnings.warn(
            f"{who}: the binary links the libdevice with sha256 {recorded}, and a build here would link another ({libdevice_here}): a jit or a "
            "new export of this kernel here has other bits for its libdevice functions than this binary (Triton or TRITON_LIBDEVICE_PATH changed)",
            TilefishCudaMathWarning,
            stacklevel=3,
        )


def check_graph(who: str, graph: Any, kernel_math: str) -> None:
    """The torch half of :func:`check` for a kernel's IR (the ``reference`` mode, which runs torch's ops)."""
    functions, _ = of_ops((n.op, str(n.attrs.get("math", kernel_math))) for n in graph.nodes)
    if functions:
        check(who, {"functions": list(functions), "cuda": list(verified(functions))})
