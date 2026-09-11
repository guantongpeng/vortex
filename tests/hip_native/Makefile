ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Native HIP tests compiled by ci/hipcc_vortex.py against
# third_party/hip-vortex headers. Kept out of the default tests `all`
# target because they additionally require a TOOLDIR with llvm-vortex,
# riscv sysroots, picolibc and compiler-rt (see ci/hip_native_probe.py).

# Default-capable tests: run on the default (A-off) config.
TESTS := vecadd module_api reduction sgemm stream_overlap

# atomics additionally needs a simulator built with
# -DVX_CFG_EXT_A_ENABLE — run `run-simx-a` from an A-enabled build tree.
A_ONLY_TESTS := atomics

BACKENDS := simx rtlsim

.PHONY: all clean $(addprefix run-,$(BACKENDS)) run-simx-a

all: $(addprefix build-,$(TESTS) $(A_ONLY_TESTS))
clean: $(addprefix clean-,$(TESTS) $(A_ONLY_TESTS))

define backend_rule
.PHONY: run-$(1)
run-$(1): $$(addprefix run-$(1)-,$$(TESTS))
endef
$(foreach b,$(BACKENDS),$(eval $(call backend_rule,$(b))))

run-simx-a: $(addprefix run-simx-,$(A_ONLY_TESTS))

build-%:
	$(MAKE) -C $* all

clean-%:
	$(MAKE) -C $* clean

run-%:
	$(MAKE) -C $(word 2,$(subst -, ,$*)) run-$(word 1,$(subst -, ,$*))

.PHONY: build-% clean-%
