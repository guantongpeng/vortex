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

// Vortex DL prim/norm host layer (plan P3-02). Mirrors blas_host.cpp:
// loads the KMU image, packs device-width arg blocks, launches through
// vx_enqueue_launch. Unused grid/block dims are always 1 (never 0 —
// see the P3-01 finding).

#include <vortex/prim.h>
#include <string>

#include <cstring>

#include "prim_args.h"

// The op numbers are declared twice: once here for callers, once in
// prim_args.h for the device compiler, which cannot see <vortex/prim.h>. The
// unary entry passes the caller's value through unchanged and the reduce entry
// derives the kernel's op by subtracting, so a value that drifted between the
// two declarations would run a different operation with no diagnostic. This is
// the mechanism that keeps them equal -- without it the agreement is a
// coincidence that the next added op silently breaks.
#define PRIM_UNARY_OP_IS(name)                                                 \
    static_assert((int)VX_PRIM_OP_##name == (int)VX_PRIM_##name,               \
                  "public and kernel unary op numbers disagree: " #name)
PRIM_UNARY_OP_IS(RELU);
PRIM_UNARY_OP_IS(GELU_TANH);
PRIM_UNARY_OP_IS(SILU);
PRIM_UNARY_OP_IS(NEG);
PRIM_UNARY_OP_IS(ABS);
PRIM_UNARY_OP_IS(EXP);
PRIM_UNARY_OP_IS(LOG);
PRIM_UNARY_OP_IS(SQRT);
PRIM_UNARY_OP_IS(RSQRT);
PRIM_UNARY_OP_IS(SIGMOID);
PRIM_UNARY_OP_IS(TANH);
PRIM_UNARY_OP_IS(RECIPROCAL);
PRIM_UNARY_OP_IS(GELU_ERF);
#undef PRIM_UNARY_OP_IS

#define PRIM_BINARY_OP_IS(name)                                               \
    static_assert((int)VX_PRIM_BINARY_##name == (int)VX_PRIM_BIN_##name,      \
                  "public and kernel binary op numbers disagree: " #name)
PRIM_BINARY_OP_IS(ADD);
PRIM_BINARY_OP_IS(SUB);
PRIM_BINARY_OP_IS(MUL);
PRIM_BINARY_OP_IS(DIV);
PRIM_BINARY_OP_IS(MAXIMUM);
PRIM_BINARY_OP_IS(MINIMUM);
#undef PRIM_BINARY_OP_IS

// The reductions cannot share the unary numbers -- both live in vx_prim_op --
// so they are offset by VX_PRIM_OP_SUM and the host subtracts. These pin the
// offset per op rather than the values, which is the property vx_prim_reduce
// relies on.
#define PRIM_REDUCE_OP_IS(name)                                                \
    static_assert((int)VX_PRIM_OP_##name - (int)VX_PRIM_OP_SUM ==              \
                      (int)VX_PRIM_RED_##name,                                 \
                  "public and kernel reduce op numbers disagree: " #name)
PRIM_REDUCE_OP_IS(SUM);
PRIM_REDUCE_OP_IS(MAX);
PRIM_REDUCE_OP_IS(ARGMAX);
PRIM_REDUCE_OP_IS(MEAN);
#undef PRIM_REDUCE_OP_IS

namespace {

struct PrimState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h unary = nullptr;
    vx_kernel_h binary = nullptr;
    vx_kernel_h scalar = nullptr;
    vx_kernel_h broadcast = nullptr;
    vx_kernel_h reduce = nullptr;
    vx_kernel_h min = nullptr;
    vx_kernel_h min_index = nullptr;
    vx_kernel_h softmax = nullptr;
    vx_kernel_h layernorm = nullptr;
    vx_kernel_h rmsnorm = nullptr;
    std::string path;   // what init was called with
};

PrimState g_prim;

// Launch helper: canonical dims (unused = 1) + LMEM for row kernels.
vx_prim_status launch(vx_queue_h q, vx_kernel_h k, const void* args,
                      size_t args_size, uint32_t grid_x, uint32_t block_x,
                      uint32_t lmem) {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = 3;
    li.grid_dim[0] = grid_x;
    li.grid_dim[1] = 1;
    li.grid_dim[2] = 1;
    li.block_dim[0] = block_x;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = lmem;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_PRIM_OK
               : VX_PRIM_ERR_LAUNCH;
}

} // namespace

vx_prim_status vx_prim_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_PRIM_ERR_BAD_ARGS;
    // Idempotent only for the same device and image. It used to return OK
    // without looking at either, so a second caller with a different image
    // silently got the first caller's kernels.
    if (g_prim.module) {
        if (g_prim.dev != dev || g_prim.path != vxbin_path) {
            return VX_PRIM_ERR_ALREADY_INITIALIZED;
        }
        return VX_PRIM_OK;
    }
    if (vx_module_load_file(dev, vxbin_path, &g_prim.module) != VX_SUCCESS) {
        vx_prim_finalize();
        return VX_PRIM_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"prim_unary_kernel", &g_prim.unary},
        {"prim_binary_kernel", &g_prim.binary},
        {"prim_scalar_kernel", &g_prim.scalar},
        {"prim_broadcast_kernel", &g_prim.broadcast},
        {"prim_reduce_kernel", &g_prim.reduce},
        {"prim_min_kernel", &g_prim.min},
        {"prim_min_index_kernel", &g_prim.min_index},
        {"prim_softmax_kernel", &g_prim.softmax},
        {"prim_layernorm_kernel", &g_prim.layernorm},
        {"prim_rmsnorm_kernel", &g_prim.rmsnorm},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_prim.module, e.name, e.slot) != VX_SUCCESS) {
            vx_prim_finalize();
            return VX_PRIM_ERR_BAD_ARGS;
        }
    }
    g_prim.dev = dev;
    g_prim.path = vxbin_path;
    return VX_PRIM_OK;
}

vx_prim_status vx_prim_finalize(void) {
    if (!g_prim.module) return VX_PRIM_OK;
    vx_kernel_h ks[] = {g_prim.unary, g_prim.binary, g_prim.scalar,
                        g_prim.broadcast, g_prim.reduce, g_prim.min,
                        g_prim.min_index,
                        g_prim.softmax,
                        g_prim.layernorm, g_prim.rmsnorm};
    for (vx_kernel_h k : ks) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_prim.module);
    g_prim = PrimState{};
    return VX_PRIM_OK;
}

vx_prim_status vx_prim_unary(vx_queue_h q, vx_prim_op op,
                             uint64_t in, uint64_t out, uint32_t n) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || n == 0) return VX_PRIM_ERR_BAD_ARGS;
    // The unary half of vx_prim_op, which ends where the reductions start.
    if (op > VX_PRIM_OP_GELU_ERF) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_unary_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.n = n;
    args.op = (uint32_t)op;
    // Single-warp CTAs for pure elementwise work (P2-03 constraint).
    return launch(q, g_prim.unary, &args, sizeof(args),
                  (n + 3) / 4, 4, 0);
}

vx_prim_status vx_prim_binary(vx_queue_h q, vx_prim_binary_op op,
                              uint64_t dst, uint64_t a, uint64_t b,
                              uint32_t n) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!dst || !a || !b || n == 0 || op > VX_PRIM_BINARY_MINIMUM) {
        return VX_PRIM_ERR_BAD_ARGS;
    }
    vx_prim_binary_args_t args = {};
    args.dst = (vx_dl_ptr_t)dst;
    args.a = (vx_dl_ptr_t)a;
    args.b = (vx_dl_ptr_t)b;
    args.n = n;
    args.op = (uint32_t)op;
    return launch(q, g_prim.binary, &args, sizeof(args), (n + 3) / 4, 4, 0);
}

vx_prim_status vx_prim_scalar(vx_queue_h q, vx_prim_binary_op op,
                              uint64_t dst, uint64_t a, float value,
                              uint32_t n, uint32_t reverse) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!dst || !a || n == 0 || op > VX_PRIM_BINARY_MINIMUM || reverse > 1) {
        return VX_PRIM_ERR_BAD_ARGS;
    }
    vx_prim_scalar_args_t args = {};
    args.dst = (vx_dl_ptr_t)dst;
    args.a = (vx_dl_ptr_t)a;
    args.value = value;
    args.n = n;
    args.op = (uint32_t)op;
    args.reverse = reverse;
    return launch(q, g_prim.scalar, &args, sizeof(args), (n + 3) / 4, 4, 0);
}

vx_prim_status vx_prim_broadcast(vx_queue_h q, vx_prim_binary_op op,
                                 uint64_t dst, uint64_t a, uint64_t b,
                                 uint32_t total, uint32_t ndim,
                                 const uint32_t sizes[4],
                                 const uint32_t a_strides[4],
                                 const uint32_t b_strides[4]) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!dst || !a || !b || total == 0 || ndim > 4 ||
        op > VX_PRIM_BINARY_MINIMUM || !sizes || !a_strides || !b_strides) {
        return VX_PRIM_ERR_BAD_ARGS;
    }
    vx_prim_broadcast_args_t args = {};
    args.dst = (vx_dl_ptr_t)dst;
    args.a = (vx_dl_ptr_t)a;
    args.b = (vx_dl_ptr_t)b;
    args.op = (uint32_t)op;
    args.ndim = ndim;
    args.total = total;
    std::memcpy(args.sizes, sizes, sizeof(args.sizes));
    std::memcpy(args.a_strides, a_strides, sizeof(args.a_strides));
    std::memcpy(args.b_strides, b_strides, sizeof(args.b_strides));
    return launch(q, g_prim.broadcast, &args, sizeof(args), (total + 3) / 4,
                  4, 0);
}

static bool is_arg_op(vx_prim_op op) {
    return op == VX_PRIM_OP_ARGMAX || op == VX_PRIM_OP_ARGMIN;
}

vx_prim_status vx_prim_reduce(vx_queue_h q, vx_prim_op op,
                              uint64_t in, uint64_t out,
                              uint32_t rows, uint32_t cols) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || rows == 0 || cols == 0) return VX_PRIM_ERR_BAD_ARGS;
    if (op < VX_PRIM_OP_SUM || op > VX_PRIM_OP_MIN) return VX_PRIM_ERR_BAD_ARGS;
    if (is_arg_op(op)) return VX_PRIM_ERR_BAD_ARGS;   // use vx_prim_index_reduce
    vx_prim_reduce_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.rows = rows;
    args.cols = cols;
    args.op = (uint32_t)(op - VX_PRIM_OP_SUM);
    if (op == VX_PRIM_OP_MIN) {
        return launch(q, g_prim.min, &args, sizeof(args), rows, 16, 256);
    }
    return launch(q, g_prim.reduce, &args, sizeof(args), rows, 16, 256);
}

vx_prim_status vx_prim_index_reduce(vx_queue_h q, vx_prim_op op, uint64_t in,
                                    uint64_t indices, uint64_t values,
                                    uint32_t rows, uint32_t cols) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !indices || rows == 0 || cols == 0) return VX_PRIM_ERR_BAD_ARGS;
    if (!is_arg_op(op)) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_reduce_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)indices;   // uint32 per row
    args.values = (vx_dl_ptr_t)values; // float per row, or 0
    args.rows = rows;
    args.cols = cols;
    args.op = (uint32_t)(op - VX_PRIM_OP_SUM);
    vx_kernel_h kernel = op == VX_PRIM_OP_ARGMIN ? g_prim.min_index : g_prim.reduce;
    return launch(q, kernel, &args, sizeof(args), rows, 16, 256);
}

// The softmax family, one row entry per op. They share a kernel, so the only
// difference between the three is the selector and -- for logsumexp -- that
// `out` holds one float per row rather than cols of them.
static vx_prim_status row_entry(vx_queue_h q, vx_prim_row_op_e op,
                                uint64_t in, uint64_t out, uint32_t rows,
                                uint32_t cols) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || rows == 0 || cols == 0) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_rowargs_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.rows = rows;
    args.cols = cols;
    args.op = (uint32_t)op;
    return launch(q, g_prim.softmax, &args, sizeof(args), rows, 16, 256);
}

vx_prim_status vx_prim_softmax(vx_queue_h q, uint64_t in, uint64_t out,
                               uint32_t rows, uint32_t cols) {
    return row_entry(q, VX_PRIM_ROW_SOFTMAX, in, out, rows, cols);
}

// out is rows floats: log(sum(exp(x - max))) + max, one per row.
vx_prim_status vx_prim_logsumexp(vx_queue_h q, uint64_t in, uint64_t out,
                                 uint32_t rows, uint32_t cols) {
    return row_entry(q, VX_PRIM_ROW_LOGSUMEXP, in, out, rows, cols);
}

vx_prim_status vx_prim_log_softmax(vx_queue_h q, uint64_t in, uint64_t out,
                                   uint32_t rows, uint32_t cols) {
    return row_entry(q, VX_PRIM_ROW_LOG_SOFTMAX, in, out, rows, cols);
}

vx_prim_status vx_prim_layernorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                                 uint64_t beta, uint64_t out, uint64_t mean,
                                 uint64_t rstd, uint32_t rows, uint32_t cols,
                                 float eps) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || rows == 0 || cols == 0) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_norm_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.gamma = (vx_dl_ptr_t)gamma;
    args.beta = (vx_dl_ptr_t)beta;
    args.out = (vx_dl_ptr_t)out;
    args.mean = (vx_dl_ptr_t)mean;
    args.rstd = (vx_dl_ptr_t)rstd;
    args.rows = rows;
    args.cols = cols;
    args.eps = eps;
    return launch(q, g_prim.layernorm, &args, sizeof(args), rows, 16, 256);
}

vx_prim_status vx_prim_rmsnorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                               uint64_t out, uint32_t rows, uint32_t cols,
                               float eps) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || rows == 0 || cols == 0) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_rmsnorm_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.gamma = (vx_dl_ptr_t)gamma;
    args.out = (vx_dl_ptr_t)out;
    args.rows = rows;
    args.cols = cols;
    args.eps = eps;
    return launch(q, g_prim.rmsnorm, &args, sizeof(args), rows, 16, 256);
}
