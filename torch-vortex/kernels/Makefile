ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# torch-vortex op image (plan P5-01): FP32 add/mul/fill kernels.

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

# binary_args_t: 3 device pointers + u32 + pad.
# fill_args_t: 1 pointer + u32 + float + pad.
ifeq ($(XLEN),64)
BIN_SIZE := 32
FILL_SIZE := 24
else
BIN_SIZE := 20
FILL_SIZE := 16
endif

all: torch_ops.vxbin

torch_ops.vxbin: ops.hip $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '[{"name": "add_kernel", "args_size": $(BIN_SIZE)}, {"name": "mul_kernel", "args_size": $(BIN_SIZE)}, {"name": "fill_kernel", "args_size": $(FILL_SIZE)}]' > ops_meta.json
	VX_KERNEL_METADATA=$$PWD/ops_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -f torch_ops.vxbin ops_meta.json *.hipcc_tmp*

.PHONY: all clean
