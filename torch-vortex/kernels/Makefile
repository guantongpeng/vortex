ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk
include $(ROOT_DIR)/sw/common/module_slots.mk

# torch-vortex op images (P5-01/P5-02). The ops and dnn kernels share one
# vxbin because they share a module; that is a packaging choice, not a
# constraint any more. Each image gets its own link address from
# sw/common/module_slots.mk, so several images can be loaded into one process
# (this used to be impossible: every image linked at 0x80000000 and the second
# hipModuleLoad failed with an address-range conflict).
#
# The argument-block sizes in the VXKMDATA footer are generated from measured
# sizeof() by gen_metadata, not typed here: they used to be hand-maintained and
# had drifted (conv declared 104 bytes for an 88-byte struct).

ifeq ($(filter $(XLEN),64),)
$(error torch-vortex kernel images require XLEN=64: the host extension's \
	argument blocks are LP64 and an rv32 image would misread every pointer)
endif

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) \
               --offload-arch=vortex$(XLEN) --kernel-lib=vortex2
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

CXX ?= c++
VXBIN := torch_all.vxbin
META := all_meta.json
SIDECAR := $(VXBIN).meta.json

all: $(VXBIN)

gen_metadata: gen_metadata.cpp torch_kernel_args.h
	$(CXX) -std=c++17 -I. -o $@ gen_metadata.cpp

# One invocation writes both: the VXKMDATA records for hipcc, and the sidecar
# the extension cross-checks its own compiled-in sizes against at init time.
$(META): gen_metadata torch_kernel_args.h
	./gen_metadata --xlen=$(XLEN) --meta=$(META) --sidecar=$(SIDECAR)

$(SIDECAR): $(META)

$(VXBIN): HIPCC_FLAGS += $(call module_flags_of,torch_all)

$(VXBIN): ops.hip dnn.hip torch_kernel_args.h $(META) $(VORTEX_HOME)/ci/hipcc_vortex.py
	$(HIPCC) $(HIPCC_FLAGS) -I$(CURDIR) ops.hip -c -o ops.o
	$(HIPCC) $(HIPCC_FLAGS) -I$(CURDIR) dnn.hip -c -o dnn.o
	VX_KERNEL_METADATA=$$PWD/$(META) \
	  $(HIPCC) $(HIPCC_FLAGS) ops.o dnn.o -o $@
	rm -f ops.o dnn.o

clean:
	rm -f $(VXBIN) $(META) $(SIDECAR) gen_metadata *.hipcc_tmp* ops.o dnn.o

.PHONY: all clean
