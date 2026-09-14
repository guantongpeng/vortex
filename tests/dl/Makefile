ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Vortex DL operator tests (plan §7.1). On-demand like tests/hip_native:
# they need a TOOLDIR with the native HIP toolchain inputs plus a
# configured build tree with sw/dl built.

TESTS := blas prim quant dnn

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
