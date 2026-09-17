# Slot assignment for KMU kernel images.
#
# Every image produced by ci/hipcc_vortex.py links at a fixed base address, and
# the runtime's module loader reserves [min_vma, max_vma) for each image it
# loads (sw/runtime/common/module.cpp). Two images linked at the same base
# therefore cannot coexist in one process: the second hipModuleLoad fails with
# "address range exceeds the reserved range". Nothing in the runtime is
# responsible for this -- the loader places each image wherever its own header
# says, and the launch PC comes from the loaded module's base -- so distinct
# bases are the whole fix, and this table is what hands them out.
#
# Adding an image means adding a row. An image that is not in the table is a
# hard build error rather than a silent fall back to slot 0, because that
# silent fall back is exactly what made the collision invisible before.

MODULE_REGION_BASE ?= 0x80000000
MODULE_SLOT_STRIDE ?= 0x00040000   # 256 KiB; the largest image today is < 20 KiB

MODULE_SLOT_torch_all := 0
MODULE_SLOT_blas      := 1
MODULE_SLOT_prim      := 2
MODULE_SLOT_dnn       := 3
MODULE_SLOT_quant     := 4
MODULE_SLOT_mxfp8     := 5
MODULE_SLOT_nvfp4     := 6
MODULE_SLOT_sparse24  := 7
MODULE_SLOT_attn      := 8
MODULE_SLOT_llm       := 9
MODULE_SLOT_mamba     := 10
MODULE_SLOT_rng       := 11

# $(call module_base_of,<image>) -> the link address for that image.
module_slot_of = $(if $(MODULE_SLOT_$(1)),$(MODULE_SLOT_$(1)),\
  $(error no module slot assigned to image '$(1)'; add MODULE_SLOT_$(1) to sw/common/module_slots.mk))
module_base_of = $(shell printf '0x%x' \
  $$(( $(MODULE_REGION_BASE) + $(call module_slot_of,$(1)) * $(MODULE_SLOT_STRIDE) )))

# $(call module_flags_of,<image>) -> what to append to a link command.
module_flags_of = --image-base=$(call module_base_of,$(1))

# Duplicate slots would put two images back on top of each other, which is the
# failure this table exists to prevent. That check lives in
# torch-vortex/tests/test_abi.py rather than here: a target defined in this
# fragment would become the default goal of every makefile that includes it.
