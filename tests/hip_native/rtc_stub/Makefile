ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rtc_stub
SRC_DIR := $(VORTEX_HOME)/tests/hip_native/$(PROJECT)

HIP_INC := $(VORTEX_HOME)/sw/hip/include
HIP_LIB := $(ROOT_DIR)/sw/hip

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror -I$(HIP_INC)
LDFLAGS += -L$(HIP_LIB) -lhip_vortex -L$(ROOT_DIR)/sw/runtime
LDFLAGS += -Wl,-rpath,$(HIP_LIB) -Wl,-rpath,$(ROOT_DIR)/sw/runtime

all: $(PROJECT)

$(PROJECT): $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-simx: $(PROJECT)
	VORTEX_DRIVER=simx LD_LIBRARY_PATH=$(ROOT_DIR)/sw/runtime ./$(PROJECT)

run-rtlsim: run-simx

clean:
	rm -f $(PROJECT)

.PHONY: all run-simx run-rtlsim clean
