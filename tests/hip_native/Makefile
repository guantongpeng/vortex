ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Native HIP tests compiled by ci/hipcc_vortex.py against
# third_party/hip-vortex headers. Kept out of the default tests `all`
# target because they additionally require a TOOLDIR with llvm-vortex,
# riscv sysroots, picolibc and compiler-rt (see ci/hip_native_probe.py).

TESTS := vecadd module_api

BACKENDS := simx rtlsim

.PHONY: all clean $(addprefix run-,$(BACKENDS))

all: $(addprefix build-,$(TESTS))
clean: $(addprefix clean-,$(TESTS))

define backend_rule
.PHONY: run-$(1)
run-$(1): $$(addprefix run-$(1)-,$$(TESTS))
endef
$(foreach b,$(BACKENDS),$(eval $(call backend_rule,$(b))))

build-%:
	$(MAKE) -C $* all

clean-%:
	$(MAKE) -C $* clean

run-%:
	$(MAKE) -C $(word 2,$(subst -, ,$*)) run-$(word 1,$(subst -, ,$*))

.PHONY: build-% clean-%
