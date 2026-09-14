ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mxfp8_test

DL_DIR := $(ROOT_DIR)/sw/dl
RUNTIME_DIR := $(ROOT_DIR)/sw/runtime

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(VORTEX_HOME)/sw/dl/include -I$(VORTEX_HOME)/sw/dl/src -I$(VORTEX_HOME)/sw/runtime/include

LDFLAGS += -L$(DL_DIR) -lvortex_dl -L$(RUNTIME_DIR) -lvortex
LDFLAGS += -Wl,-rpath,$(DL_DIR) -Wl,-rpath,$(RUNTIME_DIR)

all: $(PROJECT)

$(DL_DIR)/mxfp8.vxbin $(DL_DIR)/libvortex_dl.so:
	$(MAKE) -C $(DL_DIR)

$(PROJECT): main.cpp $(DL_DIR)/libvortex_dl.so
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-simx: $(PROJECT) $(DL_DIR)/mxfp8.vxbin
	VORTEX_DRIVER=simx LD_LIBRARY_PATH=$(RUNTIME_DIR) \
	  ./$(PROJECT) $(DL_DIR)/mxfp8.vxbin

run-rtlsim: $(PROJECT) $(DL_DIR)/mxfp8.vxbin
	VORTEX_DRIVER=rtlsim LD_LIBRARY_PATH=$(RUNTIME_DIR) \
	  ./$(PROJECT) $(DL_DIR)/mxfp8.vxbin

clean:
	rm -f $(PROJECT)

.PHONY: all run-simx run-rtlsim clean
