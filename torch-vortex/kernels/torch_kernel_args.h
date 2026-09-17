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
    X(tv_bias_add_kernel, bias_args_t, 0, 0)

#endif  // TORCH_VORTEX_KERNEL_ARGS_H
