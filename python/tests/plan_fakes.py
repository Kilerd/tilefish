"""Export a launch plan without a GPU: fake builds in a real manifest directory (not collected: the name has no ``test_`` prefix).

``export(plan, directory)`` does what ``tilefish.runtime.export`` does after the binaries are built: one ``BuiltKernel`` around a tiny ELF file per kernel entry and one
``BuiltLibrary`` (a made-up 64-byte algorithm) per GEMM go into ``manifest.json``, ``plan.json`` is written against its digest, and the plan is loaded back, verified.
The checks of the document and of the manifest are the real ones; what is not real is the machine code.
"""

from binary_fakes import fake_binary_build
from built_fakes import fake_build

from tilefish import _planrun
from tilefish._library import BuiltLibrary

ARCH = 103
FAKE_ALGO = "ab" * 64
FAKE_CUBLASLT = 120804
FAKE_SM_COUNT = 148  # the multiprocessors of a whole B300 (a fake pin says which device it is for like a real one)


def builds(plan, arch=ARCH):
    kernels, library = [], []
    for stem, callee in sorted(plan.callees.items()):
        spec = callee.by_stem[stem]
        if callee.kind == "library":
            library.append(BuiltLibrary(spec, f"sm_{arch}", FAKE_ALGO, FAKE_CUBLASLT, FAKE_SM_COUNT))
        elif callee.extern is not None and callee.extern.binary is not None:  # a prebuilt PTX or cubin: a fake ELF with its entry point
            kernels.append(fake_binary_build(spec, callee.extern.binary, arch=f"sm_{arch}"))
        else:
            kernels.append(fake_build(spec, arch=f"sm_{arch}"))
    return kernels, library


def export(plan, directory, *, arch=ARCH, overwrite=False):
    kernels, library = builds(plan, arch)
    return _planrun.write_export(plan, kernels, library, directory, arch, overwrite)
