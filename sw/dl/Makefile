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

OBJS := blas_host.o

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

# vx_blas_gemm_args_t size: rv64 = 3*8 + 3*4 + 3*4 + pad -> 48 (align 8);
# rv32 = 3*4 + 3*4 + 3*4 -> 36 (align 4).
ifeq ($(XLEN),64)
ARGS_SIZE := 48
else
ARGS_SIZE := 36
endif
KERNEL_META := [{"name": "sgemm_f32_kernel", "args_size": $(ARGS_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}, {"name": "hgemm_f16_kernel", "args_size": $(ARGS_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}, {"name": "bgemm_bf16_kernel", "args_size": $(ARGS_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}]

all: $(PROJECT) blas.vxbin

blas.vxbin: $(SRC_DIR)/blas_kernels.hip $(SRC_DIR)/blas_args.h \
            $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(KERNEL_META)' > blas_meta.json
	VX_KERNEL_METADATA=$$PWD/blas_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

blas_host.o: $(SRC_DIR)/blas_host.cpp $(SRC_DIR)/blas_args.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PROJECT): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

clean:
	rm -f *.o $(PROJECT) blas.vxbin blas_meta.json *.hipcc_tmp*

.PHONY: all clean
