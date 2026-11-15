"""Hosts generated from a manifest: the code that launches the binaries of a manifest directory without Python.

:mod:`tilefish.hosts.rust` writes typed Rust launchers for the ``tilefish`` crate (``crates/tilefish`` of the repository),
which loads a manifest directory, checks every cubin against the manifest's digests and launches it through the CUDA
driver. :mod:`tilefish.hosts.cpp` writes typed C++ launchers and plan runners for the C++ host (``crates/tilefish-capi``: the C ABI
``tilefish.h`` of that crate and its C++17 wrapper ``tilefish.hpp``). Nothing here needs torch, Triton or a GPU.
"""
