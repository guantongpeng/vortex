ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := reduction

HIPCC := python3 $(VORTEX_HOME)/ci/hipcc_vortex.py
HIPCC_FLAGS := --tooldir=$(TOOLDIR) --build-dir=$(ROOT_DIR) --offload-arch=vortex$(XLEN)
ifneq ($(strip $(CONFIGS)),)
HIPCC_FLAGS += --configs='$(CONFIGS)'
endif

SIMX_BIN   := $(ROOT_DIR)/sim/simx/simx
RTLSIM_BIN := $(ROOT_DIR)/sim/rtlsim/rtlsim

all: $(PROJECT).vxbin

$(PROJECT).vxbin: $(PROJECT).hip $(VORTEX_HOME)/ci/hipcc_vortex.py
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(SIMX_BIN):
	$(MAKE) -C $(ROOT_DIR)/sim/simx

$(RTLSIM_BIN):
	$(MAKE) -C $(ROOT_DIR)/sim/rtlsim

run-simx: $(PROJECT).vxbin $(SIMX_BIN)
	$(SIMX_BIN) $(PROJECT).vxbin

run-rtlsim: $(PROJECT).vxbin $(RTLSIM_BIN)
	$(RTLSIM_BIN) $(PROJECT).vxbin

clean:
	rm -rf *.vxbin *.elf *.o *.dump *.log *.hipcc_tmp*
