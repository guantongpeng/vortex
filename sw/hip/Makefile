ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# libhip_vortex: HIP host API subset over the vortex2.h ABI.
# Links libvortex.so (the backend dispatcher) — no private backend path.

SRC_DIR := $(VORTEX_HOME)/sw/hip/src
INC_DIR := $(VORTEX_HOME)/sw/hip/include
RUNTIME_DIR := $(ROOT_DIR)/sw/runtime

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -fPIC
CXXFLAGS += -I$(INC_DIR) -I$(VORTEX_HOME)/sw/runtime/include

LDFLAGS += -shared -pthread
# Libraries must follow the objects on the link line (as-needed drops
# them otherwise), so they are kept separate from LDFLAGS.
LDLIBS += -L$(RUNTIME_DIR) -lvortex
# Find libvortex.so next to this library, and in the runtime build dir.
LDFLAGS += -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,$(RUNTIME_DIR)

SRCS := $(SRC_DIR)/hip_vortex.cpp

PROJECT := libhip_vortex.so

OBJS := $(patsubst %.cpp, %.o, $(notdir $(SRCS)))

all: $(PROJECT)

$(PROJECT): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

hip_vortex.o: $(SRC_DIR)/hip_vortex.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(PROJECT)

.PHONY: all clean
