// Copyright © 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// THE argument-block layout for torch-vortex kernels.
//
// This header is included by all three consumers:
//   - kernels/ops.hip           (device -- VOLT/clang, rv64)
//   - kernels/dnn.hip           (device -- VOLT/clang, rv64)
//   - src/vortex_ext.cpp        (host   -- the host C++ compiler, LP64)
//   - kernels/gen_metadata.cpp  (host   -- generates the VXKMDATA args_size)
//
// It exists because those were previously three independent hand-written
// copies of the same structs plus a hand-typed sizeof table in the Makefile,
// and the table had drifted: conv declared 104 bytes for an 88-byte struct and
// pool 80 for 72. The runtime copies exactly args_size bytes off the host
// blob (sw/runtime/common/queue.cpp), so conv was reading 16 bytes past the
// end of the host argument block.
//
// Addresses are uint64_t everywhere, never uintptr_t or a host pointer: the
// images are rv64 (LP64) and so is the host, so one width is correct on both
// sides. Keep this header free of HIP/ATen includes so the host compiler can
// include it directly.
//
// A pure-C header: no includes beyond stdint, so the device compiler and the
// host compiler see literally the same declarations.

#ifndef TORCH_VORTEX_KERNEL_ARGS_H
#define TORCH_VORTEX_KERNEL_ARGS_H

#include <stdint.h>

// ---- elementwise ----------------------------------------------------------
//
// Elementwise kernels are op-code driven: one kernel per arity, with the
// operation selected by a field, so adding an op is a row here rather than a
// kernel. The enum lives in this header because the host and the device must
// agree on the numbering, and a mismatch is silent wrong answers.

enum TorchUnaryOp {
    TORCH_UNARY_RELU = 0,
    TORCH_UNARY_NEG,
    TORCH_UNARY_ABS,
    TORCH_UNARY_EXP,
    TORCH_UNARY_LOG,
    TORCH_UNARY_SQRT,
    TORCH_UNARY_RSQRT,
    TORCH_UNARY_SIGMOID,
    TORCH_UNARY_TANH,
    TORCH_UNARY_RECIPROCAL,
    TORCH_UNARY_GELU,          // erf form, torch's default
    TORCH_UNARY_GELU_TANH,     // approximate="tanh"
    TORCH_UNARY_SILU,
    TORCH_UNARY_COUNT
};

enum TorchBinaryOp {
    TORCH_BINARY_ADD = 0,
    TORCH_BINARY_SUB,
    TORCH_BINARY_MUL,
    TORCH_BINARY_DIV,
    TORCH_BINARY_MAXIMUM,
    TORCH_BINARY_MINIMUM,
    TORCH_BINARY_COUNT
};

// binary_op_kernel: dst = a <TorchBinaryOp> b. `op` was previously padding,
// so this costs nothing in size or ABI.
struct binary_args_t {
    uint64_t dst;
    uint64_t a;
    uint64_t b;
    uint32_t n;
    uint32_t op;
};

// scalar_op_kernel: dst = a <TorchBinaryOp> value, value first if reverse.
struct scalar_args_t {
    uint64_t dst;
    uint64_t a;
    float value;
    uint32_t n;
    uint32_t op;
    uint32_t reverse;
    uint32_t pad;
};

// broadcast_op_kernel: dst = a <TorchBinaryOp> b for operands whose shapes
// broadcast against each other, up to 4 dimensions. The output is contiguous,
// so only the operands need strides; a broadcast dimension is a stride of 0,
// which is what `x + bias` needs and what the copy kernel already relies on.
//
// The host is responsible for having already checked that the shapes
// broadcast; this kernel computes offsets and nothing else.
struct broadcast_op_args_t {
    uint64_t dst;
    uint64_t a;
    uint64_t b;
    uint32_t op;
    uint32_t ndim;
    uint32_t total;
    uint32_t pad;
    uint32_t sizes[4];
    uint32_t a_strides[4];
    uint32_t b_strides[4];
};

// fill_kernel; relu_kernel (in-place) reuses the layout (dst + n)
struct fill_args_t {
    uint64_t dst;
    uint32_t n;
    float value;
    uint32_t pad;
};

// unary_op_kernel: dst = f(a) with f selected by `op`. Separate source and
// destination because aten::relu and friends must not run in place on their
// input; the in-place forms go through fill_args_t (relu) or scalar_args_t.
struct unary_args_t {
    uint64_t dst;
    uint64_t a;
    uint32_t n;
    uint32_t op;
};

// reduce_rows_kernel: one thread per row, reducing `cols` contiguous values.
//
// Row-wise and contiguous, with no LMEM and no tree, which is the simplest
// shape there is -- deliberate, given what an unusual kernel shape cost in
// tv_mm_kernel. Anything that is not "reduce the trailing dimension" is
// normalised on the host with movedim+contiguous before it gets here, so this
// kernel never has to know about strides.
//
// op: 0 = sum, 1 = mean, 2 = max.
struct reduce_args_t {
    uint64_t in;
    uint64_t out;
    uint32_t rows;
    uint32_t cols;
    uint32_t op;
    uint32_t pad;
};

// copy_strided_kernel: elementwise copy between two arbitrarily strided
// buffers of the same shape, up to 4 dimensions. A stride of 0 broadcasts,
// which is how `x.t().to("vortex")` and `contiguous()` on a transposed tensor
// are served without a second kernel.
//
// Sizes are the destination's; the source shares them. Offsets are computed in
// uint32 because the element count is checked to fit.
struct copy_strided_args_t {
    uint64_t dst;
    uint64_t src;
    uint32_t ndim;
    uint32_t total;
    uint32_t sizes[4];
    uint32_t dst_strides[4];
    uint32_t src_strides[4];
};

// ---- cnn ------------------------------------------------------------------

// tv_conv2d_kernel (NCHW, dilation 1, groups 1)
struct conv_args_t {
    uint64_t in, weight, bias, out;
    uint32_t n, ci, hi, wi, co, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw, has_bias;
};

// tv_pool2d_kernel (op 0 = max, 1 = avg)
struct pool_args_t {
    uint64_t in, out;
    uint32_t n, c, hi, wi, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw, op;
};

// tv_bn_affine_kernel: inference batch norm over NCHW.
//
// `hw` is the spatial span H*W and is passed explicitly rather than derived:
// the channel of element idx is (idx / (H*W)) % C, and deriving the divisor as
// total/c gives N*H*W, which happens to be right only when N == 1.
//
// The kernel computes rstd = 1/sqrt(var + eps) itself, so var is passed
// directly and no host-side sqrt round-trip is needed. has_affine selects
// weight/bias (both pointers must be valid when it is set).
struct bn_args_t {
    uint64_t in, mean, var, weight, bias, out;
    uint32_t total, c, hw;
    float eps;
    uint32_t has_affine;
};

// tv_mm_kernel: 16x16 tile per CTA, K in chunks of 8
//
// transb: 0 = b is [k][n] row-major, 1 = b is [n][k] (i.e. use b^T)
//
// Plain out = a @ b, deliberately with no epilogue. The alpha/beta/self
// scaling of torch.addmm lives in tv_mm_epilogue_kernel below, because
// putting that branch inside this kernel's epilogue makes VOLT emit code that
// accumulates wrongly (see docs/mydocs/pytorch_plan.md W5.4 and
// tests/test_matmul.py::test_addmm_kernel_is_plain_matmul). Keep this kernel
// shape-stable: it is the known-good form.
struct mm_args_t {
    uint64_t a, b, out;
    uint32_t m, n, k;
    uint32_t transb;
};

// tv_mm_epilogue_kernel: out = alpha*out + beta*self, applied in place.
//
// A separate elementwise pass rather than an epilogue in tv_mm_kernel (see
// above). Simple elementwise kernels are unaffected by the codegen problem.
//
// self_kind: 0 = no addend, 1 = self is (n,) broadcast across rows,
//            2 = self is the full (m, n) matrix.
// The `beta != 0` test is not an optimisation: torch.addmm with beta == 0 is
// defined to ignore self entirely, NaN and Inf included, and 0 * Inf is NaN.
struct mm_epilogue_args_t {
    uint64_t out, self;
    uint32_t m, n;
    float alpha, beta;
    uint32_t self_kind;
    uint32_t pad;
};

// tv_bias_add_kernel: out[i, j] += bias[j] (row broadcast)
struct bias_args_t {
    uint64_t out, bias;
    uint32_t m, n;
};

// ---- the kernel table -----------------------------------------------------
//
// One row per kernel entry point. Fields:
//   X(entry_point_name, arg_type, max_block_x, static_lmem_bytes)
// A max_block_x of 0 means "no max_block in the metadata" (the runtime then
// only applies its own NUM_THREADS*NUM_WARPS ceiling); static_lmem_bytes of 0
// means the kernel uses no static LMEM.
//
// gen_metadata.cpp walks this table to emit the VXKMDATA args_size values, the
// extension walks it to expose arg_sizes() for the init-time cross-check, and
// load_ops() walks it to resolve every entry point. Adding a kernel means
// adding one row.

#define TORCH_KERNEL_TABLE(X)                                                  \
    X(binary_op_kernel, binary_args_t, 0, 0)                                   \
    X(broadcast_op_kernel, broadcast_op_args_t, 0, 0)                         \
    X(scalar_op_kernel, scalar_args_t, 0, 0)                                   \
    X(unary_op_kernel, unary_args_t, 0, 0)                                     \
    X(copy_strided_kernel, copy_strided_args_t, 0, 0)                         \
    X(reduce_rows_kernel, reduce_args_t, 0, 0)                                \
    X(fill_kernel, fill_args_t, 0, 0)                                          \
    X(relu_kernel, fill_args_t, 0, 0)                                          \
    X(tv_conv2d_kernel, conv_args_t, 16, 0)                                    \
    X(tv_pool2d_kernel, pool_args_t, 16, 0)                                    \
    X(tv_bn_affine_kernel, bn_args_t, 0, 0)                                    \
    X(tv_mm_kernel, mm_args_t, 16, 1024)                                       \
    X(tv_mm_epilogue_kernel, mm_epilogue_args_t, 0, 0)                         \
    X(tv_bias_add_kernel, bias_args_t, 0, 0)

#endif  // TORCH_VORTEX_KERNEL_ARGS_H
