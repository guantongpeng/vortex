ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: unittest kernel regression opencl riscv

unittest:
	$(MAKE) -C unittest

kernel:
	$(MAKE) -C kernel

regression:
	$(MAKE) -C regression

opencl:
	$(MAKE) -C opencl

riscv:
	$(MAKE) -C riscv

hip:
	$(MAKE) -C hip

# graphics (gfx_* EXT tests) and mpi (multi-process tests) are driven
# on-demand by ci/regression.sh (blackbox per-image args / MPI=1 NP=4),
# so they stay out of the default `all` build — but remain buildable and
# are scrubbed by `clean`.
graphics:
	$(MAKE) -C graphics

mpi:
	$(MAKE) -C mpi

# native HIP (hipcc-vortex) tests: need a TOOLDIR with llvm-vortex,
# riscv sysroots, picolibc + compiler-rt (probed by ci/hip_native_probe.py).
hip_native:
	$(MAKE) -C hip_native

clean:
	$(MAKE) -C unittest clean
	$(MAKE) -C kernel clean
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean
	$(MAKE) -C riscv clean
	$(MAKE) -C hip clean
	$(MAKE) -C graphics clean
	$(MAKE) -C mpi clean
	$(MAKE) -C hip_native clean

.PHONY: all unittest kernel regression opencl riscv hip hip_native graphics mpi