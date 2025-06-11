// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "timepoint_bridge.h"

#include "iree/base/api.h"
#include "iree/hal/utils/semaphore_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#ifdef __cplusplus
extern "C" {
namespace iree {
namespace {

#endif  // __cplusplus

static void iree_hal_test_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {};
static iree_status_t iree_hal_test_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  return iree_ok_status();
};

const iree_hal_semaphore_vtable_t iree_hal_test_semaphore_vtable = {
    iree_hal_test_semaphore_destroy,
    NULL,
    iree_hal_test_semaphore_signal,
    NULL,
    NULL,
    NULL,
    NULL,
};

typedef struct iree_hal_test_semaphore_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_semaphore_t base;
};

static iree_hal_test_semaphore_t iree_hal_test_semaphore_create() {
  iree_hal_test_semaphore_t out_semaphore;
  iree_hal_semaphore_initialize(&iree_hal_test_semaphore_vtable,
                                &out_semaphore.base);
  return out_semaphore;
}

iree_status_t multi_wait(const iree_hal_semaphore_list_t* semaphore_list,
                         iree_timeout_t timeout, uint64_t* out_values) {
  return iree_ok_status();
};

// Tests initialization of the timepoint bridge.
TEST(TimepointBridge, Initialize) {
  iree_hal_vulkan_timepoint_bridge_t bridge;
  auto allocator = iree_allocator_system();
  iree_hal_test_semaphore_t null_semaphore = iree_hal_test_semaphore_create();
  iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait_fn =
      &multi_wait;

  IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_initialize(
      iree_allocator_system(), (iree_hal_semaphore_t*)&null_semaphore,
      multi_wait_fn, &bridge));
  ASSERT_EQ(bridge.state, IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_SUSPENDED);
  ASSERT_EQ(bridge.allocator.ctl, iree_allocator_system().ctl);

  ASSERT_EQ(bridge.signal_set.capacity, 16);
  ASSERT_EQ(bridge.wait_set.capacity, 16);

  ASSERT_EQ(bridge.signal_set.handles.count, 1);
  ASSERT_EQ(bridge.wait_set.handles.count, 1);
  ASSERT_NE(bridge.wait_set.handles.payload_values, nullptr);
  ASSERT_EQ(bridge.wait_set.handles.semaphores[0],
            (iree_hal_semaphore_t*)&null_semaphore);
  ASSERT_EQ(bridge.wait_set.multi_wait, &multi_wait);

  iree_hal_vulkan_timepoint_bridge_free(&bridge);
}

// Tests insertion and erasure of timepoints of the timepoint bridge.
TEST(TimepointBridge, InsertErase) {
  {
    iree_hal_vulkan_timepoint_bridge_t bridge;
    auto allocator = iree_allocator_system();

    iree_event_t event;
    iree_hal_test_semaphore_t null_semaphore = iree_hal_test_semaphore_create();
    iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait_fn =
        &multi_wait;

    IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_initialize(
        iree_allocator_system(), (iree_hal_semaphore_t*)&null_semaphore,
        multi_wait_fn, &bridge));
    int items = 1;

    IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_insert(
        &bridge, (iree_hal_semaphore_t*)&null_semaphore, 2, &event));
    ++items;

    ASSERT_EQ(bridge.wait_set.handles.payload_values[items - 1], 2);
    iree_hal_vulkan_timepoint_bridge_erase(&bridge, 0);
    --items;

    ASSERT_EQ(bridge.signal_set.handles.count, items);
    ASSERT_EQ(bridge.wait_set.handles.count, items);

    IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_insert(
        &bridge, (iree_hal_semaphore_t*)&null_semaphore, 1, &event));
    ++items;
    IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_insert(
        &bridge, (iree_hal_semaphore_t*)&null_semaphore, 1, &event));
    ++items;
    IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_insert(
        &bridge, (iree_hal_semaphore_t*)&null_semaphore, 3, &event));
    ++items;
    ASSERT_EQ(bridge.signal_set.handles.count, items);
    ASSERT_EQ(bridge.wait_set.handles.count, items);
    ASSERT_EQ(bridge.wait_set.handles.payload_values[items - 1], 3);

    iree_hal_vulkan_timepoint_bridge_erase(&bridge, 1);
    --items;
    ASSERT_EQ(bridge.wait_set.handles.payload_values[1], 3);
    ASSERT_EQ(bridge.signal_set.handles.count, items);
    ASSERT_EQ(bridge.wait_set.handles.count, items);
    ASSERT_EQ(bridge.wait_set.handles.payload_values[items - 1], 1);

    iree_hal_vulkan_timepoint_bridge_free(&bridge);
  }
  {
    auto allocator = iree_allocator_system();
    iree_event_t event;
    iree_hal_test_semaphore_t null_semaphore = iree_hal_test_semaphore_create();
    iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait_fn =
        &multi_wait;
    iree_hal_vulkan_timepoint_bridge_t bridge;

    IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_initialize(
        iree_allocator_system(), (iree_hal_semaphore_t*)&null_semaphore,
        multi_wait_fn, &bridge));

    for (int i = 0; i < 15; ++i) {
      IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_insert(
          &bridge, (iree_hal_semaphore_t*)&null_semaphore, 1, &event));
    }

    iree_hal_vulkan_timepoint_bridge_erase(&bridge, 0);

    IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_insert(
        &bridge, (iree_hal_semaphore_t*)&null_semaphore, 1, &event));

    iree_hal_vulkan_timepoint_bridge_free(&bridge);
  }
}

// Tests import into the timepoint bridge.
TEST(TimepointBridge, Import) {
  auto allocator = iree_allocator_system();

  iree_event_t event;
  iree_event_initialize(0, &event);

  iree_hal_test_semaphore_t null_semaphore = iree_hal_test_semaphore_create();
  iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait_fn =
      &multi_wait;

  iree_hal_vulkan_timepoint_bridge_t bridge;

  IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_initialize(
      iree_allocator_system(), (iree_hal_semaphore_t*)&null_semaphore,
      multi_wait_fn, &bridge));

  IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_import(
      &bridge, (iree_hal_semaphore_t*)&null_semaphore, 1, &event));

  iree_hal_vulkan_timepoint_bridge_pump(&bridge);
  ASSERT_EQ(bridge.wait_set.handles.count, 2);

  // // IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_import(
  // //     &bridge, (iree_hal_semaphore_t*)&null_semaphore, 1, &event));
  iree_hal_vulkan_timepoint_bridge_free(&bridge);
}

#ifdef __cplusplus
}
}
}  // extern "C"
#endif  // __cplusplus
