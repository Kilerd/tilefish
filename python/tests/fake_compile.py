"""A stand-in for Triton's compiler, enough for ``tilefish.aot.build`` to run without Triton (not collected).

``install(monkeypatch)`` puts a fake ``triton`` (and the submodules ``build`` imports) in ``sys.modules`` and returns the
:class:`FakeCompiler` behind it. Its ``compile`` does what the real one is trusted to do for the cache and the digests, no
more: it "compiles" what the *process holds* (it runs the kernel function, so a module attribute read in memory goes into the
code bytes of the cubin, as a ``tl.constexpr`` is compiled into a real one), and it counts the calls. The cubin is a small
ELF file (``elf_builder``) whose code differs with what was compiled; the PTX has the entry point the spec's ABI asks for.
"""

import hashlib
import sys
import types
from collections import namedtuple

import fake_jit
import fake_triton
from elf_builder import cubin_like

GPUTarget = namedtuple("GPUTarget", "backend arch warp_size")


class ASTSource:
    def __init__(self, fn, signature, constexprs=None, attrs=None):
        self.fn, self.signature, self.attrs, self.constants = fn, signature, attrs, {}


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def ptx_of(signature, arch):
    """The PTX of a kernel with this signature (name -> Triton type): the entry point ``tf_k`` with the parameters its ABI has."""
    params = [".param .u64 p%d" % i if t.startswith("*") else ".param .u32 p%d" % i for i, t in enumerate(t for t in signature.values() if t != "constexpr")]
    params += [".param .u64 scratch0", ".param .u64 scratch1"]
    return f".version 8.8\n.target sm_{arch}a\n.visible .entry tf_k(\n\t" + ",\n\t".join(params) + "\n)\n{\n"


class FakeCompiler:
    def __init__(self, version="3.5.1"):
        self.version = version
        self.calls = []  # (the function's name, the options)
        self.evaluate = True  # run the kernel function to see the values the process holds
        self.shared = 16
        self.fail = None  # an exception to raise from ``compile``

    @property
    def count(self):
        return len(self.calls)

    def values_held(self, src):
        """What a compile of ``src`` takes from the process: the lines of every jit function it reaches (as imported), and the
        kernel function's result on dummy arguments (the values of the module attributes it reads)."""
        from tilefish import _source

        found = _source.reachable(src.fn)
        text = "".join("".join(f.raw_src) for f in (found.functions if found else [src.fn]))
        if not self.evaluate:
            return text, None
        try:
            return text, src.fn(*[1] * len(src.fn.arg_names))
        except Exception:  # noqa: BLE001 - a generated kernel needs the real tl; its source text is what is compiled
            return text, None

    def compile(self, src, target=None, options=None):
        if self.fail is not None:
            raise self.fail
        options = dict(options or {})
        self.calls.append((src.fn.__name__, options))
        text = repr(self.values_held(src)) + repr(sorted(options.items())) + repr(target.arch)
        code = hashlib.sha256(text.encode()).digest() * 2
        ptx = ptx_of(src.signature, target.arch)
        md = types.SimpleNamespace(
            name="tf_k",
            shared=self.shared,
            target=target,
            global_scratch_size=0,
            profile_scratch_size=0,
            launch_pdl=False,
            cluster_dims=(1, 1, 1),
            num_ctas=1,
            launch_cooperative_grid=False,
            **options,
        )
        return types.SimpleNamespace(metadata=md, asm={"cubin": cubin_like(code=code, shared=1024), "ptx": ptx})


def install(monkeypatch, version="3.5.1"):
    """Fake ``triton`` for ``build``; ``@triton.jit`` makes the ``FakeJit`` of ``fake_jit``. Returns the :class:`FakeCompiler`."""
    compiler = FakeCompiler(version)
    fake_triton.install(monkeypatch)
    triton = fake_triton.triton
    monkeypatch.setattr(triton, "jit", fake_jit.jit, raising=False)
    monkeypatch.setattr(triton, "__version__", version, raising=False)
    monkeypatch.setattr(triton, "compile", lambda *a, **k: compiler.compile(*a, **k), raising=False)
    knobs = types.SimpleNamespace(
        compilation=types.SimpleNamespace(always_compile=False),
        nvidia=types.SimpleNamespace(ptxas=types.SimpleNamespace(path="/fake/ptxas")),
    )
    monkeypatch.setattr(triton, "knobs", knobs, raising=False)
    monkeypatch.setitem(sys.modules, "triton.runtime", _module("triton.runtime"))
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", _module("triton.runtime.jit", JITFunction=fake_jit.FakeJit))
    monkeypatch.setitem(sys.modules, "triton.backends", _module("triton.backends"))
    monkeypatch.setitem(sys.modules, "triton.backends.compiler", _module("triton.backends.compiler", GPUTarget=GPUTarget))
    monkeypatch.setitem(sys.modules, "triton.compiler", _module("triton.compiler", ASTSource=ASTSource))
    from tilefish import _build

    monkeypatch.setattr(_build, "_ptxas_version", lambda path: "Cuda compilation tools, release 12.9, V12.9.86")
    return compiler
