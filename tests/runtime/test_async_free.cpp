// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <vortex2.h>

#include <cstdio>

#define CHECK(expr) do { \
  vx_result_t result = (expr); \
  if (result != VX_SUCCESS) { \
    std::fprintf(stderr, "%s failed: %s\n", #expr, vx_result_string(result)); \
    return 1; \
  } \
} while (0)

int main() {
  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));

  vx_queue_info_t info = {};
  info.struct_size = sizeof(info);
  info.priority = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h queue = nullptr;
  CHECK(vx_queue_create(dev, &info, &queue));

  vx_buffer_h buffer = nullptr;
  CHECK(vx_buffer_create(dev, 4096, VX_MEM_READ_WRITE, &buffer));
  vx_event_h free_event = nullptr;
  CHECK(vx_enqueue_free(queue, buffer, 0, nullptr, &free_event));

  // Drop the caller's reference before the worker runs. The command's
  // retained reference must keep the allocation valid until it retires.
  CHECK(vx_buffer_release(buffer));
  CHECK(vx_event_wait_value(free_event, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_release(free_event));
  CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));
  CHECK(vx_queue_release(queue));
  CHECK(vx_device_release(dev));
  std::printf("PASSED\n");
  return 0;
}
