# Negative coverage for P4-02: everything outside the transpiler whitelist
# must fail with a NotImplementedError naming the offending TTIR op/type.
import sys
import triton  # MUST be imported before any triton_vortex module
import triton.language as tl
from triton.compiler import ASTSource, compile as tri_compile
from triton.backends.compiler import GPUTarget


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
               K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def loop_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, n, BLOCK):  # dynamic trip count -> scf.for survives
        offs = i + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(out_ptr + tl.arange(0, BLOCK), acc)


@triton.jit
def fp16_copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, a, mask=m)


@triton.jit
def twod_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(x_ptr + offs[:, None] * BLOCK + offs[None, :])
    tl.store(y_ptr + offs[:, None] * BLOCK + offs[None, :], a)


CASES = [
    ("tl.dot (2-D tensors)", dot_kernel,
     {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "constexpr",
      "N": "constexpr", "K": "constexpr"},
     {"M": 16, "N": 16, "K": 16}),
    ("dynamic loop (scf.for)", loop_kernel,
     {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
     {"BLOCK": 16}),
    ("fp16 pointers", fp16_copy,
     {"x_ptr": "*fp16", "y_ptr": "*fp16", "n": "i32", "BLOCK": "constexpr"},
     {"BLOCK": 16}),
    ("2-D broadcast", twod_kernel,
     {"x_ptr": "*fp32", "y_ptr": "*fp32", "BLOCK": "constexpr"},
     {"BLOCK": 16}),
]

failures = 0
for name, fn, sig, cexpr in CASES:
    try:
        tri_compile(ASTSource(fn=fn, signature=sig, constexprs=cexpr),
                    target=GPUTarget("vortex", 64, 4))
        print(f"{name}: compiled (UNEXPECTED — FAIL)")
        failures += 1
    except NotImplementedError as e:
        print(f"{name}: NotImplementedError: "
              f"{str(e).splitlines()[0].strip()}")
if failures == 0:
    print("PASSED")
    sys.exit(0)
print("FAILED")
sys.exit(1)
