# Stretch acceptance for P4-02: tt.reduce (axis 0) codegen.
#  case 1: rowmax — tl.max(x, 0) per row, bit-exact vs torch (comparisons
#          only; masked lanes load -inf which cannot change a max).
#  case 2: full tutorial softmax — two tt.reduce + math.exp; device expf
#          vs torch exp may differ by ~1 ulp per element, so the honest
#          tolerance is rtol/atol 2e-5 (documented, not weakened: the
#          comparison targets cross-library expf agreement, not the
#          compiler lowering, which case 1 already pins bit-exactly).
#
# Each case runs in its own subprocess: KMU images link at the fixed
# STARTUP_ADDR (0x80000000), so a second hipModuleLoad in the same
# process overlaps the first allocation (simx device memory allocator).
import os
import subprocess
import sys

PY = os.path.abspath(os.path.join(
    os.path.expanduser("~"), "miniconda3/envs/vortex/bin/python"))

if os.environ.get("__VTEX_CASE__"):
    import triton  # MUST be imported before any triton_vortex module
    import triton.language as tl
    import torch
    import ctypes
    from triton.compiler import ASTSource, compile as tri_compile
    from triton.backends.compiler import GPUTarget

    @triton.jit
    def rowmax_kernel(in_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(in_ptr + row * n_cols + offs, mask=mask,
                    other=float("-inf"))
        m = tl.max(x, 0)
        tl.store(out_ptr + row, m)

    @triton.jit
    def softmax_kernel(in_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(in_ptr + row * n_cols + offs, mask=mask,
                    other=float("-inf"))
        m = tl.max(x, 0)
        e = tl.exp(x - m)
        s = tl.sum(e, 0)
        tl.store(out_ptr + row * n_cols + offs, e / s, mask=mask)

    def run_case():
        sys.path.insert(0, "/home/guantp/aichip/vortex/triton-vortex")
        from triton_vortex import hip
        which = os.environ["__VTEX_CASE__"]
        torch.manual_seed(20260914)
        if which == "rowmax":
            rows, cols, block = 4, 29, 32
            x = torch.randn(rows, cols)
            ref, nbytes = x.max(dim=1).values, rows * 4
            kernel, rtol, atol = rowmax_kernel, 0.0, 0.0
        else:
            rows, cols, block = 3, 37, 64
            x = torch.randn(rows, cols)
            ref, nbytes = torch.softmax(x, dim=-1), rows * cols * 4
            kernel, rtol, atol = softmax_kernel, 2e-5, 2e-5

        k = tri_compile(
            ASTSource(fn=kernel,
                      signature={"in_ptr": "*fp32", "out_ptr": "*fp32",
                                 "n_cols": "i32", "BLOCK": "constexpr"},
                      constexprs={"BLOCK": block}),
            target=GPUTarget("vortex", 64, 4))
        md = k.metadata
        din = hip.malloc(x.numel() * 4)
        dout = hip.malloc(nbytes)
        hip.memcpy(din, x.numpy().ctypes.data, x.numel() * 4,
                   hip.HIP_MEMCPY_H2D)
        mod = hip.load_module(md.vxbin_path)
        fn = hip.get_function(mod, md.name)
        fmt = "".join(f for _, f in md.sig)
        blob = hip.pack_args(fmt, [din, dout, cols])
        assert len(blob) == md.args_size, (len(blob), md.args_size)
        hip.launch(fn, (rows, 1, 1), (md.num_warps * 4, 1, 1), blob,
                   shared=md.shared)
        hip.synchronize()
        out = (ctypes.c_char * nbytes)()
        hip.memcpy(ctypes.addressof(out), dout, nbytes, hip.HIP_MEMCPY_D2H)
        got = torch.frombuffer(bytearray(out), dtype=torch.float32) \
            .reshape(ref.shape)
        bad = (~torch.isclose(got, ref, rtol=rtol, atol=atol)).sum().item()
        max_abs = (got - ref).abs().max().item()
        max_rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
        print(f"{which}: rows={rows} cols={cols} BLOCK={block} "
              f"args_size={md.args_size} lmem={md.shared} "
              f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} bad={bad}")
        sys.exit(0 if bad == 0 else 1)

    run_case()


def main():
    failures = 0
    for case in ("rowmax", "softmax"):
        env = dict(os.environ, __VTEX_CASE__=case)
        r = subprocess.run([PY, os.path.abspath(__file__)], env=env)
        if r.returncode != 0:
            failures += 1
    if failures == 0:
        print("PASSED")
        return 0
    print(f"FAILED ({failures})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
