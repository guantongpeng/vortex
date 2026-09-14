ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifeq ($(XLEN),64)
ARGS_SIZE := 4816
else
ARGS_SIZE := 4812
endif

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(VORTEX_HOME)/sw/hip/include
LDFLAGS += -L$(ROOT_DIR)/sw/hip -lhip_vortex -L$(ROOT_DIR)/sw/runtime
LDFLAGS += -Wl,-rpath,$(ROOT_DIR)/sw/hip -Wl,-rpath,$(ROOT_DIR)/sw/runtime

all: main kernel.vxbin

kernel.vxbin: kernel.hip $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '[{"name": "bigargs_kernel", "args_size": $(ARGS_SIZE)}]' > meta.json
	VX_KERNEL_METADATA=$$PWD/meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

main: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-simx: all
	VORTEX_DRIVER=simx LD_LIBRARY_PATH=$(ROOT_DIR)/sw/runtime ./main ./kernel.vxbin

run-rtlsim: all
	VORTEX_DRIVER=rtlsim LD_LIBRARY_PATH=$(ROOT_DIR)/sw/runtime ./main ./kernel.vxbin

clean:
	rm -f main kernel.vxbin meta.json *.hipcc_tmp*

.PHONY: all run-simx run-rtlsim clean
