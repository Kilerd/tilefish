"""A stand-in for Triton's frontend and compiler, enough for ``tilefish.aot.build`` to run without Triton (not collected).

``install(monkeypatch)`` puts a fake ``triton`` (and the submodules ``build`` and ``tilefish._ir`` import) in ``sys.modules`` and
returns the :class:`FakeCompiler` behind it. What it models is what the identity of a binary rests on:

* the **frontend** (``ASTSource.make_ir``) turns the kernel function *as the process holds it* into an IR text by running it on
  symbolic arguments, so every value the function reads, by any route (a module attribute, a closure, ``getattr``, a class
  attribute), is folded into the text, and comments, blank lines, the path of the file and the names of locals are not in it;
  ``str_nodebug()`` is that text. (A function that cannot run on symbols, a generated kernel that needs the real ``tl``, is
  represented by the dump of its syntax tree, which has the same two properties.)
* the **compiler** (``triton.compile``) calls the frontend, then makes a small ELF file whose code is a function of the IR text, the
  options and the architecture, and counts the calls.

It is not Triton: it checks nothing a real compile checks. The real thing is checked on a GPU (``test_gpu_identity.py``).
"""

import ast
import hashlib
import sys
import textwrap
import threading
import time
import types
from collections import namedtuple

import fake_jit
import fake_triton
from elf_builder import cubin_like

GPUTarget = namedtuple("GPUTarget", "backend arch warp_size")


class Sym:
    """A symbolic value: the operations on it build the expression that stands for the IR."""

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text

    def _bin(self, op, other, flip=False):
        a, b = (repr(other), self.text) if flip else (self.text, repr(other))
        return Sym(f"({a} {op} {b})")

    def __add__(self, o):
        return self._bin("+", o)

    def __radd__(self, o):
        return self._bin("+", o, True)

    def __sub__(self, o):
        return self._bin("-", o)

    def __rsub__(self, o):
        return self._bin("-", o, True)

    def __mul__(self, o):
        return self._bin("*", o)

    def __rmul__(self, o):
        return self._bin("*", o, True)

    def __truediv__(self, o):
        return self._bin("/", o)

    def __rtruediv__(self, o):
        return self._bin("/", o, True)


class FakeModule:
    def __init__(self, text):
        self.text = text

    def str_nodebug(self):
        return self.text

    def __str__(self):
        return "#loc = loc(fake)\n" + self.text


class ASTSource:
    def __init__(self, fn, signature, constexprs=None, attrs=None):
        self.fn, self.signature, self.attrs, self.constants = fn, signature, attrs, {}

    def parse_options(self):
        return {}

    def make_ir(self, target, options, codegen_fns, module_map, context):
        names = list(self.fn.arg_names)
        constants = {index[0]: value for index, value in self.constants.items()}
        args = [constants[i] if i in constants else Sym(names[i]) for i in range(len(names))]
        try:
            body = repr(self.fn(*args))
        except Exception:  # noqa: BLE001 - a function that needs the real tl: its syntax tree is what is compiled
            body = ast.dump(ast.parse(textwrap.dedent("".join(self.fn.raw_src))))
        types_ = ",".join(f"{self.signature.get(n, '?')}" for n in names)
        return FakeModule(f"module {{ func @{self.fn.__name__}({types_}) attrs={sorted(map(str, (self.attrs or {}).items()))} {{ {body} }} }}")


class FakeBackend:
    def __init__(self, target):
        self.target = target

    def parse_options(self, options):
        return types.SimpleNamespace(**options)

    def load_dialects(self, context):
        pass

    def get_codegen_implementation(self, options):
        return {}

    def get_module_map(self):
        return {}


class FakeContext:
    def disable_multithreading(self):
        pass


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
        self.frontends = 0  # how many times the frontend ran (a compile runs it once, a digest once)
        self.shared = 16
        self.fail = None  # an exception to raise from ``compile``
        self.during = None  # a function to call when ``compile`` starts: where Triton would read the files an option names
        self.second_ir = None  # a text for the frontend to give the second time it runs (a frontend that is not deterministic)
        self.fail_frontend = None  # an exception for the frontend to raise
        self.delay = 0.0  # seconds a frontend takes (so that two that overlap are seen to)
        self.active = 0  # frontends running now, and the most there ever were at once
        self.max_active = 0
        self.lock = threading.Lock()

    @property
    def count(self):
        return len(self.calls)

    def compile(self, src, target=None, options=None):
        if self.during is not None:
            self.during()
        if self.fail is not None:
            raise self.fail
        options = dict(options or {})
        self.calls.append((src.fn.__name__, options))
        module = src.make_ir(target, types.SimpleNamespace(**options), {}, {}, FakeContext())
        text = module.str_nodebug() + repr(sorted(options.items())) + repr(target.arch)
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

    class CountingSource(ASTSource):
        def make_ir(self, *a, **k):
            with compiler.lock:
                compiler.frontends += 1
                compiler.active += 1
                compiler.max_active = max(compiler.max_active, compiler.active)
            try:
                if compiler.delay:
                    time.sleep(compiler.delay)
                if compiler.fail_frontend is not None:
                    raise compiler.fail_frontend
                module = super().make_ir(*a, **k)
            finally:
                with compiler.lock:
                    compiler.active -= 1
            if compiler.second_ir is not None and compiler.frontends > 1:
                return FakeModule(compiler.second_ir)
            return module

    monkeypatch.setattr(triton, "jit", fake_jit.jit, raising=False)
    monkeypatch.setattr(triton, "__version__", version, raising=False)
    monkeypatch.setattr(triton, "compile", lambda *a, **k: compiler.compile(*a, **k), raising=False)
    knobs = types.SimpleNamespace(
        compilation=types.SimpleNamespace(
            always_compile=False,
            override=False,
            use_ir_loc=None,
            enable_asan=False,
            disable_line_info=False,
            front_end_debugging=False,
            allow_non_constexpr_globals=False,
            enable_experimental_consan=False,
        ),
        language=types.SimpleNamespace(fp32_default=None, default_fp_fusion=True),
        nvidia=types.SimpleNamespace(
            ptxas=types.SimpleNamespace(path="/fake/ptxas"), disable_ptxas_opt=False, mock_ptx_version=None, libdevice_path=None
        ),
        runtime=types.SimpleNamespace(override_arch=None),
    )
    monkeypatch.setattr(triton, "knobs", knobs, raising=False)
    monkeypatch.setitem(sys.modules, "triton.runtime", _module("triton.runtime"))
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", _module("triton.runtime.jit", JITFunction=fake_jit.FakeJit))
    monkeypatch.setitem(sys.modules, "triton.backends", _module("triton.backends"))
    monkeypatch.setitem(sys.modules, "triton.backends.compiler", _module("triton.backends.compiler", GPUTarget=GPUTarget))
    monkeypatch.setitem(sys.modules, "triton.compiler", _module("triton.compiler", ASTSource=CountingSource))
    monkeypatch.setitem(sys.modules, "triton.compiler.compiler", _module("triton.compiler.compiler", make_backend=FakeBackend))
    ir = _module("triton._C.libtriton.ir", context=FakeContext, load_dialects=lambda context: None)
    monkeypatch.setitem(sys.modules, "triton._C", _module("triton._C"))
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", _module("triton._C.libtriton", ir=ir))
    monkeypatch.setitem(sys.modules, "triton._C.libtriton.ir", ir)
    from tilefish import _build

    monkeypatch.setattr(_build, "_ptxas_version", lambda path: "Cuda compilation tools, release 12.9, V12.9.86")
    return compiler
