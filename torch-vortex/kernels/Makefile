ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# torch-vortex op images (P5-01/P5-02). Everything links into ONE vxbin:
# the simx backend loads modules at a fixed base address, so multiple
# images would overlap (measured: second hipModuleLoad fails with an
# address-range conflict).

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

# struct sizes (C layout, align 8 on rv64 / 4 on rv32):
# binary: 3p+2u = 32/20; fill: p+u+f+u = 24/16
# conv: 4p+17u = 104/84; pool: 2p+15u = 80/68; bn: 6p+2u = 56/32
# mm: 3p+4u = 40/28; bias: 2p+2u = 24/16
ifeq ($(XLEN),64)
BIN_SIZE := 32
FILL_SIZE := 24
else
BIN_SIZE := 20
FILL_SIZE := 16
endif
ifeq ($(XLEN),64)
CONV_SIZE := 104
POOL_SIZE := 80
BN_SIZE := 56
MM_SIZE := 40
BIAS_SIZE := 24
else
CONV_SIZE := 84
POOL_SIZE := 68
BN_SIZE := 32
MM_SIZE := 28
BIAS_SIZE := 12
endif
ALL_META := [{"name": "add_kernel", "args_size": $(BIN_SIZE)}, {"name": "mul_kernel", "args_size": $(BIN_SIZE)}, {"name": "fill_kernel", "args_size": $(FILL_SIZE)}, {"name": "relu_kernel", "args_size": $(FILL_SIZE)}, {"name": "tv_conv2d_kernel", "args_size": $(CONV_SIZE), "max_block": [16, 1, 1]}, {"name": "tv_pool2d_kernel", "args_size": $(POOL_SIZE), "max_block": [16, 1, 1]}, {"name": "tv_bn_affine_kernel", "args_size": $(BN_SIZE)}, {"name": "tv_mm_kernel", "args_size": $(MM_SIZE), "static_lmem_bytes": 1024, "max_block": [16, 1, 1]}, {"name": "tv_bias_add_kernel", "args_size": $(BIAS_SIZE)}]

all: torch_all.vxbin

torch_all.vxbin: ops.hip dnn.hip $(VORTEX_HOME)/ci/hipcc_vortex.py
	echo '$(ALL_META)' > all_meta.json
	$(HIPCC) $(HIPCC_FLAGS) ops.hip -c -o ops.o
	$(HIPCC) $(HIPCC_FLAGS) dnn.hip -c -o dnn.o
	VX_KERNEL_METADATA=$$PWD/all_meta.json \
	  $(HIPCC) $(HIPCC_FLAGS) ops.o dnn.o -o $@
	rm -f ops.o dnn.o

clean:
	rm -f torch_all.vxbin all_meta.json *.hipcc_tmp* ops.o dnn.o

.PHONY: all clean
