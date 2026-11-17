# Documentation

* [`scope.md`](scope.md): what tilefish is and is not, and how features are added.
* The formats a host reads:
  * [`manifest-schema-3.md`](manifest-schema-3.md): the manifest (kernel entries, the ABI, prebuilt binaries, pinned library calls);
  * [`plan-schema-1.md`](plan-schema-1.md): launch plans and plan sets.
* [`numerics.md`](numerics.md): the numerics contract's dependencies, and how to check it again after a CUDA or torch change.
* [`cute.md`](cute.md): CuTe DSL kernels as externs (the `nvidia-cutlass-dsl` pin and its licence).
* Tools:
  * [`explain.md`](explain.md): traces, `tilefish-explain` and SASS assertions;
  * [`instrument.md`](instrument.md): probes and the instrumented variant;
  * [`advise.md`](advise.md): the fusion advisor.
* Design records ([`rfcs/`](rfcs)): [0001, plan fusion](rfcs/0001-plan-fusion.md); [0002, the C++ host](rfcs/0002-cpp-host.md);
  [0003, epilogue and prologue fusion](rfcs/0003-epilogue-fusion.md).
* [`../CHANGELOG.md`](../CHANGELOG.md): what each release changed.
