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
//
// The unary operations are not here: they are sw/dl's (vx_prim_op), because
// that library's prim module implements them and this image would only be a
// second copy of the same switch. Binary operation labels are public in
// vortex/prim.h; their argument blocks and kernels live in sw/dl's prim image.

enum TorchCopyDType {
    TORCH_COPY_F32 = 0,
    TORCH_COPY_F64,
    TORCH_COPY_F16,
    TORCH_COPY_BF16,
    TORCH_COPY_I32,
    TORCH_COPY_I64,
    TORCH_COPY_BOOL,
};

// fill_kernel: dst[i] = value
struct fill_args_t {
    uint64_t dst;
    uint32_t n;
    float value;
    uint32_t pad;
};

// nll_loss_forward: input is [C] or [N,C], target is scalar or [N]. The
// kernel writes output (scalar for reductions, N values for none) and the
// scalar total weight; invalid targets set the device-side flag.
struct nll_loss_args_t {
    uint64_t dst;
    uint64_t total_weight;
    uint64_t input;
    uint64_t target;
    uint64_t weight;
    uint64_t invalid;
    uint32_t rows;
    uint32_t classes;
    uint32_t reduction;
    uint32_t pad;
    int64_t ignore_index;
};

struct arange_args_t {
    uint64_t dst;
    uint32_t n;
    uint32_t dtype;  // 0 = float32, 1 = int32, 2 = int64
    uint32_t pad;
    float start;
    float step;
    int64_t start_i;
    int64_t step_i;
};

// sort/topk: one row is one independent trailing-dimension sort. The kernel
// keeps the best k values in output order, so topk and full sort share one
// device implementation without a host-side index or value buffer.
struct sort_args_t {
    uint64_t dst;
    uint64_t indices;
    uint64_t src;
    uint32_t rows;
    uint32_t cols;
    uint32_t k;
    uint32_t descending;
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
    uint32_t dst_type;
    uint32_t src_type;
    uint32_t sizes[4];
    uint32_t dst_strides[4];
    uint32_t src_strides[4];
};

// widen_u32_i64_kernel: the DL reduction produces indices as uint32 (that is
// vx_prim_reduce's contract) and ATen's are int64. This is a representation
// change, not a computation -- a second reduction pass would be one.
struct widen_args_t {
    uint64_t dst;   // int64 output
    uint64_t src;   // uint32 input
    uint32_t n;
    uint32_t pad;
};

// cat/stack copy one contiguous input into a contiguous output. The logical
// layout is outer x (out_dim) x inner; src_dim is the input extent at the
// concatenation/stack dimension and dst_offset selects its output interval.
struct cat_args_t {
    uint64_t dst;
    uint64_t src;
    uint32_t outer;
    uint32_t src_dim;
    uint32_t out_dim;
    uint32_t inner;
    uint32_t dst_offset;
    uint32_t total;
};

// gather/scatter index one contiguous logical tensor. `sizes` describes the
// index/source shape, `strides` is the contiguous self/output stride, and
// index_type is 0 for int32 or 1 for int64 indices.
struct index_args_t {
    uint64_t dst;
    uint64_t src;
    uint64_t index;
    uint64_t invalid;
    uint32_t ndim;
    uint32_t dim;
    uint32_t dim_size;
    uint32_t index_type;
    uint32_t total;
    uint32_t sizes[4];
    uint32_t strides[4];
};

// index_add accumulates every source row into one output row. The kernel uses
// a deterministic scan over index rather than float atomics, which keeps the
// base configuration correct when ZACAS/float AMO is unavailable.
struct index_add_args_t {
    uint64_t dst;
    uint64_t self;
    uint64_t src;
    uint64_t index;
    uint64_t invalid;
    uint32_t ndim;
    uint32_t dim;
    uint32_t dim_size;
    uint32_t index_type;
    uint32_t index_count;
    uint32_t total;
    float alpha;
    uint32_t sizes[4];
    uint32_t self_strides[4];
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
    X(copy_strided_kernel, copy_strided_args_t, 0, 0)                         \
    X(widen_u32_i64_kernel, widen_args_t, 0, 0)                               \
    X(cat_kernel, cat_args_t, 0, 0)                                           \
    X(gather_kernel, index_args_t, 0, 0)                                      \
    X(scatter_kernel, index_args_t, 0, 0)                                     \
    X(index_add_kernel, index_add_args_t, 0, 0)                               \
    X(fill_kernel, fill_args_t, 0, 0)                                          \
    X(nll_loss_kernel, nll_loss_args_t, 0, 0)                                 \
    X(arange_kernel, arange_args_t, 0, 0)                                      \
    X(sort_kernel, sort_args_t, 0, 0)                                          \
    X(tv_bias_add_kernel, bias_args_t, 0, 0)

#endif  // TORCH_VORTEX_KERNEL_ARGS_H
