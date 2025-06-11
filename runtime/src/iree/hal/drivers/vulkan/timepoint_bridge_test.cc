// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "timepoint_bridge.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#ifdef __cplusplus
extern "C" {
namespace iree {
namespace {

#endif  // __cplusplus

TEST(TimepointBridge, Initialize) {
  iree_hal_vulkan_timepoint_bridge_t bridge;
  auto allocator = iree_allocator_system();
  IREE_ASSERT_OK(iree_hal_vulkan_timepoint_bridge_allocate(
      iree_allocator_system(), &bridge));
  ASSERT_EQ(bridge.state, IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_SUSPENDED);
  ASSERT_EQ(bridge.allocator.ctl, iree_allocator_system().ctl);
  ASSERT_EQ(bridge.signal_set.capacity, 16);
  ASSERT_EQ(bridge.wait_set.capacity, 16);
  ASSERT_EQ(bridge.signal_set.handles.count, 0);
  ASSERT_EQ(bridge.wait_set.handles.count, 0);
  ASSERT_NE(bridge.signal_set.handles.payload_values, nullptr);
  ASSERT_NE(bridge.wait_set.handles.payload_values, nullptr);

  iree_hal_vulkan_timepoint_bridge_free(&bridge);
}

TEST(TimepointBridge, InsertErase) {}

TEST(TimepointBridge, Pump) {}

#ifdef __cplusplus
}
}
}  // extern "C"
#endif  // __cplusplus
