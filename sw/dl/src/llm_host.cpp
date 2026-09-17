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

// Vortex DL LLM support-ops host layer (plan P7). Mirrors quant_host:
// loads the llm.vxbin KMU image, packs device-width arg blocks, launches
// through vx_enqueue_launch with canonical dims (unused = 1). All four
// kernels are grid-stride elementwise/row-copy work, so single-warp
// 4-thread CTAs and lmem_size 0 suffice.

#include <vortex/llm.h>
#include <string>

#include "llm_args.h"

// The VXKMDATA args_size records baked into llm.vxbin must equal the C
// sizeof of these structs; assert the layout at compile time so a drift
// fails the build instead of corrupting device arg blocks.
#if defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
static_assert(sizeof(vx_llm_rope_args_t) == 12, "meta args_size mismatch");
static_assert(sizeof(vx_llm_swiglu_args_t) == 16, "meta args_size mismatch");
static_assert(sizeof(vx_llm_embedding_args_t) == 20, "meta args_size mismatch");
static_assert(sizeof(vx_llm_kv_append_args_t) == 20, "meta args_size mismatch");
#else
static_assert(sizeof(vx_llm_rope_args_t) == 16, "meta args_size mismatch");
static_assert(sizeof(vx_llm_swiglu_args_t) == 32, "meta args_size mismatch");
static_assert(sizeof(vx_llm_embedding_args_t) == 32, "meta args_size mismatch");
static_assert(sizeof(vx_llm_kv_append_args_t) == 32, "meta args_size mismatch");
#endif

namespace {

struct LlmState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h rope = nullptr;
    vx_kernel_h swiglu = nullptr;
    vx_kernel_h embed = nullptr;
    vx_kernel_h kv_append = nullptr;
    std::string path;   // what init was called with
};

LlmState g_l;

vx_llm_status launch(vx_queue_h q, vx_kernel_h k, const void* args,
                     size_t args_size, uint32_t total) {
    if (total == 0) return VX_LLM_ERR_BAD_ARGS;
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = 3;
    li.grid_dim[0] = (total + 3) / 4;  // one 4-thread warp per 4 elements
    li.grid_dim[1] = 1;
    li.grid_dim[2] = 1;
    li.block_dim[0] = 4;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = 0;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_LLM_OK
               : VX_LLM_ERR_LAUNCH;
}

} // namespace

vx_llm_status vx_llm_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_LLM_ERR_BAD_ARGS;
    // Idempotent only for the same device and image. It used to return OK
    // without looking at either, so a second caller with a different image
    // silently got the first caller's kernels.
    if (g_l.module) {
        if (g_l.dev != dev || g_l.path != vxbin_path) {
            return VX_LLM_ERR_ALREADY_INITIALIZED;
        }
        return VX_LLM_OK;
    }
    if (vx_module_load_file(dev, vxbin_path, &g_l.module) != VX_SUCCESS) {
        vx_llm_finalize();
        return VX_LLM_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"llm_rope_kernel", &g_l.rope},
        {"llm_swiglu_kernel", &g_l.swiglu},
        {"llm_embedding_kernel", &g_l.embed},
        {"llm_kv_append_kernel", &g_l.kv_append},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_l.module, e.name, e.slot) != VX_SUCCESS) {
            vx_llm_finalize();
            return VX_LLM_ERR_BAD_ARGS;
        }
    }
    g_l.dev = dev;
    g_l.path = vxbin_path;
    return VX_LLM_OK;
}

vx_llm_status vx_llm_finalize(void) {
    if (!g_l.module) return VX_LLM_OK;
    vx_kernel_h ks[] = {g_l.rope, g_l.swiglu, g_l.embed, g_l.kv_append};
    for (vx_kernel_h k : ks) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_l.module);
    g_l = LlmState{};
    return VX_LLM_OK;
}

vx_llm_status vx_llm_rope(vx_queue_h q, uint64_t x, uint32_t seq,
                          uint32_t dim) {
    if (!g_l.module) return VX_LLM_ERR_NOT_INITIALIZED;
    if (!x || seq == 0 || dim < 2) return VX_LLM_ERR_BAD_ARGS;
    vx_llm_rope_args_t args = {};
    args.x = (vx_dl_ptr_t)x;
    args.seq = seq;
    args.dim = dim;
    return launch(q, g_l.rope, &args, sizeof(args), seq * (dim / 2));
}

vx_llm_status vx_llm_swiglu(vx_queue_h q, uint64_t a, uint64_t b,
                            uint64_t out, uint32_t n) {
    if (!g_l.module) return VX_LLM_ERR_NOT_INITIALIZED;
    if (!a || !b || !out || n == 0) return VX_LLM_ERR_BAD_ARGS;
    vx_llm_swiglu_args_t args = {};
    args.a = (vx_dl_ptr_t)a;
    args.b = (vx_dl_ptr_t)b;
    args.out = (vx_dl_ptr_t)out;
    args.n = n;
    return launch(q, g_l.swiglu, &args, sizeof(args), n);
}

vx_llm_status vx_llm_embedding(vx_queue_h q, uint64_t table, uint64_t ids,
                               uint64_t out, uint32_t dim, uint32_t n) {
    if (!g_l.module) return VX_LLM_ERR_NOT_INITIALIZED;
    if (!table || !ids || !out || dim == 0 || n == 0) {
        return VX_LLM_ERR_BAD_ARGS;
    }
    vx_llm_embedding_args_t args = {};
    args.table = (vx_dl_ptr_t)table;
    args.ids = (vx_dl_ptr_t)ids;
    args.out = (vx_dl_ptr_t)out;
    args.dim = dim;
    args.n = n;
    return launch(q, g_l.embed, &args, sizeof(args), n * dim);
}

vx_llm_status vx_llm_kv_append(vx_queue_h q, uint64_t cache, uint64_t new_k,
                               uint32_t capacity, uint32_t base,
                               uint32_t new_tokens, uint32_t dim) {
    if (!g_l.module) return VX_LLM_ERR_NOT_INITIALIZED;
    if (!cache || !new_k || capacity == 0 || new_tokens == 0 || dim == 0 ||
        base >= capacity ||
        (uint64_t)base + (uint64_t)new_tokens > (uint64_t)capacity) {
        return VX_LLM_ERR_BAD_ARGS;
    }
    vx_llm_kv_append_args_t args = {};
    args.cache = (vx_dl_ptr_t)cache;
    args.new_k = (vx_dl_ptr_t)new_k;
    args.base = base;
    args.new_tokens = new_tokens;
    args.dim = dim;
    return launch(q, g_l.kv_append, &args, sizeof(args), new_tokens * dim);
}
