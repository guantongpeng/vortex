ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Vortex DL operator libraries (plan §7.1): host dispatch layers over the
# vortex2.h ABI plus the KMU kernel images compiled by hipcc-vortex.
# v1 delivers libvortex_dl.so (BLAS); prim/dnn/attention/rng/quant follow
# the same pattern under src/.

SRC_DIR := $(VORTEX_HOME)/sw/dl/src
INC_DIR := $(VORTEX_HOME)/sw/dl/include
RUNTIME_DIR := $(ROOT_DIR)/sw/runtime

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror -fPIC
CXXFLAGS += -I$(INC_DIR) -I$(SRC_DIR) -I$(VORTEX_HOME)/sw/runtime/include
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN)

LDFLAGS += -shared -pthread
LDFLAGS += -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,$(RUNTIME_DIR)
LDLIBS += -L$(RUNTIME_DIR) -lvortex

PROJECT := libvortex_dl.so

OBJS := blas_host.o prim_host.o quant_host.o dnn_host.o mxfp8_host.o \
         nvfp4_host.o sparse24_host.o attn_host.o llm_host.o mamba_host.o \
         rng_host.o

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

# vx_blas_gemm_args_t includes three byte strides and batch metadata for the
# grid-z batched GEMM path.
ifeq ($(XLEN),64)
ARGS_SIZE := 80
else
ARGS_SIZE := 72
endif
KERNEL_META := [{"name": "sgemm_f32_kernel", "args_size": $(ARGS_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}, {"name": "hgemm_f16_kernel", "args_size": $(ARGS_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}, {"name": "bgemm_bf16_kernel", "args_size": $(ARGS_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}]

# prim/norm arg-block sizes (device-width pointers; see prim_args.h).
# reduce gained rows/cols/op/pad when it became row-wise, and `values` when
# the arg ops started returning the extreme alongside the index: 3 ptr + 4 u32.
# layernorm gained mean/rstd as optional outputs: 6 ptr + 4 u32.
# rowargs gained an op selector when softmax became a family: 2 ptr + 4 u32.
ifeq ($(XLEN),64)
PRIM_UNARY_SIZE := 24
PRIM_BINARY_SIZE := 32
PRIM_SCALAR_SIZE := 40
PRIM_BROADCAST_SIZE := 88
PRIM_REDUCE_SIZE := 40
PRIM_ROWARGS_SIZE := 32
PRIM_NORM_SIZE := 64
PRIM_RMSNORM_SIZE := 40
else
PRIM_UNARY_SIZE := 16
PRIM_BINARY_SIZE := 20
PRIM_SCALAR_SIZE := 28
PRIM_BROADCAST_SIZE := 76
PRIM_REDUCE_SIZE := 28
PRIM_ROWARGS_SIZE := 24
PRIM_NORM_SIZE := 40
PRIM_RMSNORM_SIZE := 28
endif
PRIM_META := [{"name": "prim_unary_kernel", "args_size": $(PRIM_UNARY_SIZE)}, {"name": "prim_binary_kernel", "args_size": $(PRIM_BINARY_SIZE)}, {"name": "prim_scalar_kernel", "args_size": $(PRIM_SCALAR_SIZE)}, {"name": "prim_broadcast_kernel", "args_size": $(PRIM_BROADCAST_SIZE)}, {"name": "prim_reduce_kernel", "args_size": $(PRIM_REDUCE_SIZE), "static_lmem_bytes": 256, "max_block": [16, 1, 1]}, {"name": "prim_min_kernel", "args_size": $(PRIM_REDUCE_SIZE), "static_lmem_bytes": 256, "max_block": [16, 1, 1]}, {"name": "prim_min_index_kernel", "args_size": $(PRIM_REDUCE_SIZE), "static_lmem_bytes": 256, "max_block": [16, 1, 1]}, {"name": "prim_softmax_kernel", "args_size": $(PRIM_ROWARGS_SIZE), "static_lmem_bytes": 256, "max_block": [16, 1, 1]}, {"name": "prim_layernorm_kernel", "args_size": $(PRIM_NORM_SIZE), "static_lmem_bytes": 256, "max_block": [16, 1, 1]}, {"name": "prim_rmsnorm_kernel", "args_size": $(PRIM_RMSNORM_SIZE), "static_lmem_bytes": 256, "max_block": [16, 1, 1]}]

include $(ROOT_DIR)/sw/common/module_slots.mk

# Give every image its own link address so several can be loaded into one
# process (see sw/common/module_slots.mk).
$(eval blas.vxbin: HIPCC_FLAGS += $(call module_flags_of,blas))
$(eval prim.vxbin: HIPCC_FLAGS += $(call module_flags_of,prim))
$(eval dnn.vxbin: HIPCC_FLAGS += $(call module_flags_of,dnn))
$(eval quant.vxbin: HIPCC_FLAGS += $(call module_flags_of,quant))
$(eval mxfp8.vxbin: HIPCC_FLAGS += $(call module_flags_of,mxfp8))
$(eval nvfp4.vxbin: HIPCC_FLAGS += $(call module_flags_of,nvfp4))
$(eval sparse24.vxbin: HIPCC_FLAGS += $(call module_flags_of,sparse24))
$(eval attn.vxbin: HIPCC_FLAGS += $(call module_flags_of,attn))
$(eval llm.vxbin: HIPCC_FLAGS += $(call module_flags_of,llm))
$(eval mamba.vxbin: HIPCC_FLAGS += $(call module_flags_of,mamba))
$(eval rng.vxbin: HIPCC_FLAGS += $(call module_flags_of,rng))

all: $(PROJECT) blas.vxbin prim.vxbin quant.vxbin dnn.vxbin mxfp8.vxbin \
     nvfp4.vxbin sparse24.vxbin attn.vxbin llm.vxbin mamba.vxbin rng.vxbin

blas.vxbin: $(SRC_DIR)/blas_kernels.hip $(SRC_DIR)/blas_args.h \
            $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(KERNEL_META)' > blas_meta.json
	VX_KERNEL_METADATA=$$PWD/blas_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

prim.vxbin: $(SRC_DIR)/prim_kernels.hip $(SRC_DIR)/prim_args.h \
            $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(PRIM_META)' > prim_meta.json
	VX_KERNEL_METADATA=$$PWD/prim_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

blas_host.o: $(SRC_DIR)/blas_host.cpp $(SRC_DIR)/blas_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

prim_host.o: $(SRC_DIR)/prim_host.cpp $(SRC_DIR)/prim_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# quant arg-block sizes (device-width pointers).
# pack4/unpack4: 3 ptr + 3 u32 -> rv64 36/align8=40, rv32 24
# gemm w4a16: 4 ptr + 4 u32 -> rv64 48, rv32 32
# gemm w8a8: 4 ptr + float + u32 + 3 u32 -> rv64 48, rv32 32
ifeq ($(XLEN),64)
Q_PACK_SIZE := 40
Q_GEMM4_SIZE := 48
Q_GEMM8_SIZE := 56
else
Q_PACK_SIZE := 24
Q_GEMM4_SIZE := 32
Q_GEMM8_SIZE := 36
endif
# gemm fp8: 3 ptr + 4 u32 -> rv64 40, rv32 28
ifeq ($(XLEN),64)
Q_FP8_SIZE := 40
else
Q_FP8_SIZE := 28
endif
QUANT_META := [{"name": "quant_scales4_kernel", "args_size": $(Q_PACK_SIZE)}, {"name": "quant_pack4_kernel", "args_size": $(Q_PACK_SIZE)}, {"name": "quant_unpack4_kernel", "args_size": $(Q_PACK_SIZE)}, {"name": "quant_gemm_w4a16_kernel", "args_size": $(Q_GEMM4_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}, {"name": "quant_gemm_w8a8_kernel", "args_size": $(Q_GEMM8_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}, {"name": "quant_gemm_fp8_kernel", "args_size": $(Q_FP8_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}]

quant.vxbin: $(SRC_DIR)/quant_kernels.hip $(SRC_DIR)/quant_args.h \
             $(SRC_DIR)/quant_fp8.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(QUANT_META)' > quant_meta.json
	VX_KERNEL_METADATA=$$PWD/quant_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

quant_host.o: $(SRC_DIR)/quant_host.cpp $(SRC_DIR)/quant_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# dnn arg-block sizes, MEASURED from the structs rather than counted by hand.
#
# They were hand-written and had drifted: conv declared 96 for an 88-byte
# struct, pool 88 for 72, and bn 56 for 72 -- the last one too small, which is
# the dangerous direction, because the runtime copies exactly args_size bytes
# and the kernel would read the rest as whatever was next in memory. (That one
# appeared when bn_args_t gained has_affine and this table did not follow.)
#
# Measure with the host compiler, which shares LP64 with rv64; for rv32 define
# VX_CFG_XLEN=32 and no __VORTEX__, which selects the header's uint32_t pointer
# branch. torch-vortex/tests/test_dl_bridge.py re-measures both and fails if
# these drift again.
#
# The DL host passes its own sizeof() at launch, so the metadata below is not
# what these kernels are launched with -- it is what a caller using the
# kernelParams path would get, which is why the drift was latent rather than
# an active over-read.
ifeq ($(XLEN),64)
D_CONV_SIZE := 104
D_POOL_SIZE := 88
D_BN_SIZE := 72
else
D_CONV_SIZE := 84
D_POOL_SIZE := 76
D_BN_SIZE := 44
endif
ifeq ($(XLEN),64)
D_RESIZE_SIZE := 96
else
D_RESIZE_SIZE := 84
endif
DNN_META := [{"name": "dnn_conv2d_kernel", "args_size": $(D_CONV_SIZE), "max_block": [16, 1, 1]}, {"name": "dnn_pool2d_kernel", "args_size": $(D_POOL_SIZE), "max_block": [16, 1, 1]}, {"name": "dnn_bn_affine_kernel", "args_size": $(D_BN_SIZE), "max_block": [16, 1, 1]}, {"name": "dnn_resize_kernel", "args_size": $(D_RESIZE_SIZE), "max_block": [16, 1, 1]}]

dnn.vxbin: $(SRC_DIR)/dnn_kernels.hip $(SRC_DIR)/dnn_args.h \
           $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(DNN_META)' > dnn_meta.json
	VX_KERNEL_METADATA=$$PWD/dnn_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

dnn_host.o: $(SRC_DIR)/dnn_host.cpp $(SRC_DIR)/dnn_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- new modules (workflow batch): metadata args_size = true struct sizeof --
# mxfp8 pack/unpack: 3p+3u -> 40/24; gemm: 5p+4u -> 56/36
# nvfp4 pack: 3p+f+3u -> 40/28; unpack/gemm same family 40/56 (rv64) with
#   rv32 28/44 (unpack: 3p+f+u; gemm: 4p+f+3u)
# sparse24 prune: 3p+2u -> 32/20; gemm: 4p+4u -> 48/32
# attn: 4p+5u+f -> 56/40
# llm rope 1p+3u+f -> 16/12; swiglu 3p+u -> 32/16; embed 2p+2u -> 24/12;
#   kv_append 2p+3u+f -> 24/16 (check struct; kept in meta per kernel)
# mamba scan: 6p+4u -> 64/40
# rng: 1p+u+2xu64 -> 32/20
ifeq ($(XLEN),64)
MX_PACK := 40
MX_GEMM := 56
N4_PACK := 40
N4_GEMM := 56
SP_PRUNE := 32
SP_GEMM := 48
ATTN_SZ := 56
LLM_ROPE := 16
LLM_SWI := 32
LLM_EMB := 32
LLM_KV := 32
MAMBA_SZ := 64
MAMBA_STATE_SZ := 80
RNG_SZ := 32
else
MX_PACK := 24
MX_GEMM := 36
N4_PACK := 28
N4_GEMM := 36
SP_PRUNE := 20
SP_GEMM := 28
ATTN_SZ := 40
LLM_ROPE := 12
LLM_SWI := 16
LLM_EMB := 20
LLM_KV := 20
MAMBA_SZ := 40
MAMBA_STATE_SZ := 48
RNG_SZ := 28
endif
VXBIN_RULE = echo '$(2)' > $(1)_meta.json && \
	VX_KERNEL_METADATA=$$PWD/$(1)_meta.json $(HIPCC) $(HIPCC_FLAGS) $(3) -o $@

MXFP8_META := [{"name": "mxfp8_scales_kernel", "args_size": $(MX_PACK)}, {"name": "mxfp8_pack_kernel", "args_size": $(MX_PACK)}, {"name": "mxfp8_dequant_kernel", "args_size": $(MX_PACK)}, {"name": "mxfp8_gemm_kernel", "args_size": $(MX_GEMM), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}]
NVFP4_META := [{"name": "nvfp4_scales_kernel", "args_size": $(N4_PACK)}, {"name": "nvfp4_pack_kernel", "args_size": $(N4_PACK)}, {"name": "nvfp4_unpack_kernel", "args_size": $(N4_PACK)}, {"name": "nvfp4_gemm_kernel", "args_size": $(N4_GEMM), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}]
SPARSE24_META := [{"name": "sparse24_prune_kernel", "args_size": $(SP_PRUNE)}, {"name": "sparse24_gemm_kernel", "args_size": $(SP_GEMM), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}]
ATTN_META := [{"name": "attn_forward_kernel", "args_size": $(ATTN_SZ)}]
LLM_META := [{"name": "llm_rope_kernel", "args_size": $(LLM_ROPE)}, {"name": "llm_swiglu_kernel", "args_size": $(LLM_SWI)}, {"name": "llm_embedding_kernel", "args_size": $(LLM_EMB)}, {"name": "llm_kv_append_kernel", "args_size": $(LLM_KV)}]
MAMBA_META := [{"name": "mamba_scan_kernel", "args_size": $(MAMBA_SZ), "static_lmem_bytes": 64, "max_block": [16, 1, 1]}, {"name": "mamba_state_kernel", "args_size": $(MAMBA_STATE_SZ), "static_lmem_bytes": 64, "max_block": [16, 1, 1]}]
RNG_META := [{"name": "rng_philox4x32_uniform_kernel", "args_size": $(RNG_SZ)}]

mxfp8.vxbin: $(SRC_DIR)/mxfp8_kernels.hip $(SRC_DIR)/mxfp8_args.h $(SRC_DIR)/quant_fp8.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(MXFP8_META)' > mxfp8_meta.json
	VX_KERNEL_METADATA=$$PWD/mxfp8_meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

nvfp4.vxbin: $(SRC_DIR)/nvfp4_kernels.hip $(SRC_DIR)/nvfp4_args.h $(SRC_DIR)/quant_fp8.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(NVFP4_META)' > nvfp4_meta.json
	VX_KERNEL_METADATA=$$PWD/nvfp4_meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

sparse24.vxbin: $(SRC_DIR)/sparse24_kernels.hip $(SRC_DIR)/sparse24_args.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(SPARSE24_META)' > sparse24_meta.json
	VX_KERNEL_METADATA=$$PWD/sparse24_meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

attn.vxbin: $(SRC_DIR)/attn_kernels.hip $(SRC_DIR)/attn_args.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(ATTN_META)' > attn_meta.json
	VX_KERNEL_METADATA=$$PWD/attn_meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

llm.vxbin: $(SRC_DIR)/llm_kernels.hip $(SRC_DIR)/llm_args.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(LLM_META)' > llm_meta.json
	VX_KERNEL_METADATA=$$PWD/llm_meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

mamba.vxbin: $(SRC_DIR)/mamba_kernels.hip $(SRC_DIR)/mamba_args.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(MAMBA_META)' > mamba_meta.json
	VX_KERNEL_METADATA=$$PWD/mamba_meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

rng.vxbin: $(SRC_DIR)/rng_kernels.hip $(SRC_DIR)/rng_args.h $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(RNG_META)' > rng_meta.json
	VX_KERNEL_METADATA=$$PWD/rng_meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

mxfp8_host.o: $(SRC_DIR)/mxfp8_host.cpp $(SRC_DIR)/mxfp8_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
nvfp4_host.o: $(SRC_DIR)/nvfp4_host.cpp $(SRC_DIR)/nvfp4_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
sparse24_host.o: $(SRC_DIR)/sparse24_host.cpp $(SRC_DIR)/sparse24_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
attn_host.o: $(SRC_DIR)/attn_host.cpp $(SRC_DIR)/attn_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
llm_host.o: $(SRC_DIR)/llm_host.cpp $(SRC_DIR)/llm_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
mamba_host.o: $(SRC_DIR)/mamba_host.cpp $(SRC_DIR)/mamba_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
rng_host.o: $(SRC_DIR)/rng_host.cpp $(SRC_DIR)/rng_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PROJECT): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

clean:
	rm -f *.o $(PROJECT) *.vxbin *_meta.json *.hipcc_tmp*

.PHONY: all clean
