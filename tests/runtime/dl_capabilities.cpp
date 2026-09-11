// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <vortex2.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

struct Cap {
  const char* name;
  uint32_t id;
};

constexpr Cap kCaps[] = {
    {"version", VX_CAPS_VERSION},
    {"num_threads", VX_CAPS_NUM_THREADS},
    {"num_warps", VX_CAPS_NUM_WARPS},
    {"num_cores", VX_CAPS_NUM_CORES},
    {"cache_line_size", VX_CAPS_CACHE_LINE_SIZE},
    {"global_mem_size", VX_CAPS_GLOBAL_MEM_SIZE},
    {"local_mem_size", VX_CAPS_LOCAL_MEM_SIZE},
    {"isa_flags", VX_CAPS_ISA_FLAGS},
    {"num_mem_banks", VX_CAPS_NUM_MEM_BANKS},
    {"mem_bank_size", VX_CAPS_MEM_BANK_SIZE},
    {"num_clusters", VX_CAPS_NUM_CLUSTERS},
    {"socket_size", VX_CAPS_SOCKET_SIZE},
    {"issue_width", VX_CAPS_ISSUE_WIDTH},
    {"clock_rate_mhz", VX_CAPS_CLOCK_RATE},
    {"peak_mem_bw_mb_s", VX_CAPS_PEAK_MEM_BW},
    {"vm_support", VX_CAPS_VM_SUPPORT},
    {"vm_pinned_size", VX_CAPS_VM_PINNED_SIZE},
    {"vm_pinned_free", VX_CAPS_VM_PINNED_FREE},
    {"cp_num_queues", VX_CAPS_CP_NUM_QUEUES},
    {"cp_ring_size_log2", VX_CAPS_CP_RING_SIZE_LOG2},
    {"cp_axi_tid_width", VX_CAPS_CP_AXI_TID_WIDTH},
    {"cp_supports_draw", VX_CAPS_CP_SUPPORTS_DRAW},
    {"cp_supports_qmd", VX_CAPS_CP_SUPPORTS_QMD},
    {"cp_mmu_fault_report", VX_CAPS_CP_MMU_FAULT_REPORT},
};

void print_json_string(const char* value) {
  std::printf("\"%s\"", value);
}

}  // namespace

int main() {
  uint32_t count = 0;
  if (vx_device_count(&count) != VX_SUCCESS || count == 0) {
    std::fprintf(stderr, "no Vortex device available\n");
    return 1;
  }

  vx_device_h dev = nullptr;
  vx_result_t result = vx_device_open(0, &dev);
  if (result != VX_SUCCESS) {
    std::fprintf(stderr, "vx_device_open failed: %s\n",
                 vx_result_string(result));
    return 1;
  }

  std::printf("{\n  \"device_count\": %u,\n  \"capabilities\": {", count);
  for (std::size_t i = 0; i < sizeof(kCaps) / sizeof(kCaps[0]); ++i) {
    uint64_t value = 0;
    result = vx_device_query(dev, kCaps[i].id, &value);
    if (result != VX_SUCCESS) {
      std::fprintf(stderr, "capability %s failed: %s\n", kCaps[i].name,
                   vx_result_string(result));
      vx_device_release(dev);
      return 1;
    }
    if (i != 0) std::printf(",");
    std::printf("\n    ");
    print_json_string(kCaps[i].name);
    std::printf(": %llu", static_cast<unsigned long long>(value));
  }
  std::printf("\n  }\n}\n");

  result = vx_device_release(dev);
  if (result != VX_SUCCESS) {
    std::fprintf(stderr, "vx_device_release failed: %s\n",
                 vx_result_string(result));
    return 1;
  }
  return 0;
}
