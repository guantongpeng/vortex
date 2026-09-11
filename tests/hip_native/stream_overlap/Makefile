ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := stream_overlap

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

# scale_args_t: 3 device-width pointers + u32 + u32 + u32(+pad to align).
# rv64: 8*3 + 4*3 = 33 -> 40 (align 8); rv32: 4*3 + 4*3 = 24 (align 4).
ifeq ($(XLEN),64)
ARGS_SIZE := 40
else
ARGS_SIZE := 24
endif

HIP_INC := $(VORTEX_HOME)/sw/hip/include
HIP_LIB := $(ROOT_DIR)/sw/hip

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(HIP_INC) -DHIP_TEST_DEV_PTR_WIDTH=$(XLEN)

LDFLAGS += -L$(HIP_LIB) -lhip_vortex
LDFLAGS += -L$(ROOT_DIR)/sw/runtime
LDFLAGS += -Wl,-rpath,$(HIP_LIB) -Wl,-rpath,$(ROOT_DIR)/sw/runtime

all: $(PROJECT) kernel.vxbin

kernel.vxbin: kernel.hip $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '[{"name": "scale_kernel", "args_size": $(ARGS_SIZE)}]' > kernel_meta.json
	VX_KERNEL_METADATA=$$PWD/kernel_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(PROJECT): main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-simx: $(PROJECT) kernel.vxbin
	VORTEX_DRIVER=simx LD_LIBRARY_PATH=$(ROOT_DIR)/sw/runtime \
	  ./$(PROJECT) ./kernel.vxbin

run-rtlsim: $(PROJECT) kernel.vxbin
	VORTEX_DRIVER=rtlsim LD_LIBRARY_PATH=$(ROOT_DIR)/sw/runtime \
	  ./$(PROJECT) ./kernel.vxbin

clean:
	rm -f *.o $(PROJECT) kernel.vxbin kernel_meta.json *.hipcc_tmp*

.PHONY: all run-simx run-rtlsim clean
