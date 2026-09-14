ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifeq ($(XLEN),64)
ARGS_SIZE := 32
else
ARGS_SIZE := 20
endif

all: vecadd.vxbin

vecadd.vxbin: vecadd.hip $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '[{"name": "add_kernel", "args_size": $(ARGS_SIZE)}]' > meta.json
	VX_KERNEL_METADATA=$$PWD/meta.json $(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -f vecadd.vxbin meta.json *.hipcc_tmp*
