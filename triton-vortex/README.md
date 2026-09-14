# triton-vortex — out-of-tree Triton backend for Vortex

P4-01 of the fullstack plan. Installed as a pip package registering the
`vortex` backend through the `triton.backends` entry point (Triton 3.8
discovers out-of-tree backends via entry points, not TRITON_PLUGIN_DIRS).

## Milestone status (v0.1)

| Piece | Status |
|---|---|
| Backend registration (entry point, GPUTarget "vortex") | done |
| Driver: libhip_vortex binding, device/stream, module load + launch | done — tests/test_driver_launch.py runs a compiled Triton-shaped kernel through it |
| Reference numerics via Triton interpreter | done — tests/test_interpreter.py (vecadd/softmax/layernorm) |
| Codegen: Triton IR → RISC-V vxbin | **not implemented** — the `bin` stage raises with the roadmap below |

## Codegen roadmap (the `bin` stage)

The compile pipeline reuses Triton's generic `ttir`/`ttgir`/`llir` stages;
what is missing is the final `llir -> vxbin` translation:

1. rewrite target-specific intrinsics the generic llir emits
   (program-id/lane-id math) onto the Vortex CSR reads from
   `sw/kernel/include/vx_spawn2.h`;
2. run the llir through VOLT clang (`ci/hipcc_vortex.py` already wraps
   `-x ir`-able compiles) with the riscv target triple + `+xvortex`;
3. link via the same two-pass `vx_start.S` + `vxbin.py` flow hipcc uses,
   attaching VXKMDATA arg-layout metadata derived from the Triton
   signature (this is where the P1-3 metadata ABI pays off);
4. cache key must include the Vortex ISA/config hash (hipcc --configs).

Until then, Triton kernels run through the interpreter for reference
numerics only; device execution uses the manually compiled kernels under
kernels/ via the driver below.

## Driver

`triton_vortex.hip` is a ctypes binding of libhip_vortex with
device-width pointer packing (rv32 device behind a 64-bit host — the
P2-02 finding). `tests/test_driver_launch.py` loads kernels/vecadd.vxbin
(a Triton-shaped `add(ptr, ptr, ptr, n, BLOCK)` kernel compiled by
hipcc-vortex) and checks numerics against the interpreter result.

## Import-order constraint (Triton plugin contract)

Plugin modules must not cold-start Triton: always `import triton` (or
something that pulls it in) **before** importing `triton_vortex.driver`
directly. Triton's backend discovery imports plugin modules while
`triton.backends` is still initializing; a direct cold import of the
plugin recurses through `triton.__init__` and the discovery then sees a
partially initialized module (0 concrete subclasses). `import triton_vortex`
(the package `__init__`) is safe — only `triton_vortex.driver`/`.compiler`
are affected.

## Environment

- `VORTEX_BUILD` — configured build tree (default: auto-discovered,
  preferring rv64 `build*` directories with `sw/hip` built).
- `VORTEX_DRIVER` — simx / rtlsim / xrt for the underlying runtime.
