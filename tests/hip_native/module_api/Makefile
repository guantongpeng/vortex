ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := module_api

# Host sources live in the source tree, like tests/hip/* does it. A bare
# `main.cpp` silently compiled whatever stale copy a build tree held.
SRC_DIR := $(VORTEX_HOME)/tests/hip_native/$(PROJECT)

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

# Kernel arg block size for the VXKMDATA sidecar consumed by
# hipModuleLaunchKernel's kernelParams path: 3 device-width pointers +
# u32 count + u32 pad. rv64: 8*3 + 4 + 4 = 32; rv32: 4*3 + 4 + 4 = 20.
# The host main.cpp packs with the same widths (HIP_TEST_DEV_PTR_WIDTH).
ifeq ($(XLEN),64)
ARGS_SIZE := 32
else
ARGS_SIZE := 20
endif

# Headers from the source tree, library from the configured build tree.
HIP_INC := $(VORTEX_HOME)/sw/hip/include
HIP_LIB := $(ROOT_DIR)/sw/hip

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(HIP_INC) -DHIP_TEST_DEV_PTR_WIDTH=$(XLEN)

LDFLAGS += -L$(HIP_LIB) -lhip_vortex
LDFLAGS += -L$(ROOT_DIR)/sw/runtime
LDFLAGS += -Wl,-rpath,$(HIP_LIB) -Wl,-rpath,$(ROOT_DIR)/sw/runtime

VORTEX_DRIVER ?= simx

all: $(PROJECT) kernel.vxbin

kernel.vxbin: $(SRC_DIR)/kernel.hip $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '[{"name": "vecadd_kernel", "args_size": $(ARGS_SIZE)}]' > kernel_meta.json
	VX_KERNEL_METADATA=$$PWD/kernel_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(PROJECT): $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-simx: $(PROJECT) kernel.vxbin
	VORTEX_DRIVER=simx LD_LIBRARY_PATH=$(ROOT_DIR)/sw/runtime \
	  ./$(PROJECT) ./kernel.vxbin

run-rtlsim: $(PROJECT) kernel.vxbin
	VORTEX_DRIVER=rtlsim LD_LIBRARY_PATH=$(ROOT_DIR)/sw/runtime \
	  ./$(PROJECT) ./kernel.vxbin

clean:
	rm -f *.o $(PROJECT) kernel.vxbin kernel_meta.json *.hipcc_tmp*

.PHONY: all run run-simx run-rtlsim clean
