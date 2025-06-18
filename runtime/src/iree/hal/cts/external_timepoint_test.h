// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_EXTERNAL_TIMEPOINT_TEST_H_
#define IREE_HAL_CTS_EXTERNAL_TIMEPOINT_TEST_H_

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

class ExternalTimepointTest : public CTSTestBase<> {};

// Test export timepoint with immediate value.
TEST_F(ExternalTimepointTest, ExportTimepointImmediate) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 10ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Export a timepoint for a value that's already been reached.
  iree_hal_external_timepoint_t external_timepoint;
  iree_status_t status = iree_hal_semaphore_export_timepoint(
      semaphore, 5ull, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
      IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE, &external_timepoint);

  if (iree_status_is_unimplemented(status)) {
    // Driver doesn't support external timepoint export - skip test.
    iree_status_ignore(status);
    iree_hal_semaphore_release(semaphore);
    GTEST_SKIP() << "Driver does not support external timepoint export";
    return;
  }
  IREE_ASSERT_OK(status);

  // Should be a wait primitive type
  EXPECT_EQ(IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
            external_timepoint.type);

  // Should be immediately ready since value 5 < current value 10
  iree_wait_handle_t wait_handle;
  iree_wait_handle_wrap_primitive(
      external_timepoint.handle.wait_primitive.type,
      external_timepoint.handle.wait_primitive.value, &wait_handle);
  EXPECT_TRUE(iree_wait_handle_is_immediate(wait_handle));

  iree_hal_semaphore_release(semaphore);
}

// Test export timepoint with future value
TEST_F(ExternalTimepointTest, ExportTimepointFuture) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 5ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Export a timepoint for a future value
  iree_hal_external_timepoint_t external_timepoint;
  iree_status_t status = iree_hal_semaphore_export_timepoint(
      semaphore, 10ull, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
      IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE, &external_timepoint);

  if (iree_status_is_unimplemented(status)) {
    // Driver doesn't support external timepoint export - skip test
    iree_status_ignore(status);
    iree_hal_semaphore_release(semaphore);
    GTEST_SKIP() << "Driver does not support external timepoint export";
    return;
  }
  IREE_ASSERT_OK(status);

  // Should be a wait primitive type
  EXPECT_EQ(IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
            external_timepoint.type);

  // Should not be immediately ready since value 10 > current value 5
  iree_wait_handle_t wait_handle;
  iree_wait_handle_wrap_primitive(
      external_timepoint.handle.wait_primitive.type,
      external_timepoint.handle.wait_primitive.value, &wait_handle);
  EXPECT_FALSE(iree_wait_handle_is_immediate(wait_handle));

  // Signal the semaphore to make the timepoint ready
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 10ull));

  // Now the wait handle should be ready
  IREE_ASSERT_OK(iree_wait_one(&wait_handle, IREE_TIME_INFINITE_PAST));

  iree_hal_semaphore_release(semaphore);
}

// Test import timepoint with immediate value
TEST_F(ExternalTimepointTest, ImportTimepointImmediate) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Create an immediate external timepoint
  iree_hal_external_timepoint_t external_timepoint;
  external_timepoint.type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE;
  external_timepoint.flags = IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE;
  external_timepoint.compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT;
  external_timepoint.handle.wait_primitive = iree_wait_primitive_immediate();

  // Import the immediate timepoint - should signal semaphore immediately
  iree_status_t status = iree_hal_semaphore_import_timepoint(
      semaphore, 42ull, IREE_HAL_QUEUE_AFFINITY_ANY, external_timepoint);

  if (iree_status_is_unimplemented(status)) {
    // Driver doesn't support external timepoint import - skip test
    iree_status_ignore(status);
    iree_hal_semaphore_release(semaphore);
    GTEST_SKIP() << "Driver does not support external timepoint import";
    return;
  }
  IREE_ASSERT_OK(status);

  // Semaphore should now be at value 42
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(42ull, value);

  iree_hal_semaphore_release(semaphore);
}

// Test import timepoint with future value (using external poller)
TEST_F(ExternalTimepointTest, ImportTimepointFuture) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Create a wait primitive that we'll signal later
  iree_event_t external_event;
  IREE_ASSERT_OK(iree_event_initialize(false, &external_event));

  // Create external timepoint from the wait handle
  iree_hal_external_timepoint_t external_timepoint;
  external_timepoint.type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE;
  external_timepoint.flags = IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE;
  external_timepoint.compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT;
  external_timepoint.handle.wait_primitive.type = external_event.type;
  external_timepoint.handle.wait_primitive.value = external_event.value;

  // Import the timepoint - should not signal semaphore yet
  iree_status_t status = iree_hal_semaphore_import_timepoint(
      semaphore, 100ull, IREE_HAL_QUEUE_AFFINITY_ANY, external_timepoint);

  if (iree_status_is_unimplemented(status)) {
    // Driver doesn't support external timepoint import - skip test
    iree_status_ignore(status);
    iree_event_deinitialize(&external_event);
    iree_hal_semaphore_release(semaphore);
    GTEST_SKIP() << "Driver does not support external timepoint import";
    return;
  }
  IREE_ASSERT_OK(status);

  // Semaphore should still be at initial value
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(0ull, value);

  // Signal the external wait primitive
  iree_event_set(&external_event);

  // Wait for the semaphore to be signaled by the poller
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 100ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  // Verify final value
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(100ull, value);

  iree_event_deinitialize(&external_event);
  iree_hal_semaphore_release(semaphore);
}

// Test round-trip: export from one semaphore, import to another
TEST_F(ExternalTimepointTest, ExportImportRoundTrip) {
  iree_hal_semaphore_t* source_semaphore = nullptr;
  iree_hal_semaphore_t* dest_semaphore = nullptr;

  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &source_semaphore));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &dest_semaphore));

  // Export a timepoint from source semaphore
  iree_hal_external_timepoint_t external_timepoint;
  iree_status_t export_status = iree_hal_semaphore_export_timepoint(
      source_semaphore, 50ull, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
      IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE, &external_timepoint);

  if (iree_status_is_unimplemented(export_status)) {
    // Driver doesn't support external timepoint export - skip test
    iree_status_ignore(export_status);
    iree_hal_semaphore_release(source_semaphore);
    iree_hal_semaphore_release(dest_semaphore);
    GTEST_SKIP() << "Driver does not support external timepoint export";
    return;
  }
  IREE_ASSERT_OK(export_status);

  // Import the timepoint to destination semaphore
  iree_status_t import_status = iree_hal_semaphore_import_timepoint(
      dest_semaphore, 200ull, IREE_HAL_QUEUE_AFFINITY_ANY, external_timepoint);

  if (iree_status_is_unimplemented(import_status)) {
    // Driver doesn't support external timepoint import - skip test
    iree_status_ignore(import_status);
    iree_hal_semaphore_release(source_semaphore);
    iree_hal_semaphore_release(dest_semaphore);
    GTEST_SKIP() << "Driver does not support external timepoint import";
    return;
  }
  IREE_ASSERT_OK(import_status);

  // Both semaphores should still be at 0
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(source_semaphore, &value));
  EXPECT_EQ(0ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(dest_semaphore, &value));
  EXPECT_EQ(0ull, value);

  // Signal source semaphore
  IREE_ASSERT_OK(iree_hal_semaphore_signal(source_semaphore, 50ull));

  // Wait for destination semaphore to be signaled
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      dest_semaphore, 200ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  // Verify both semaphores have expected values
  IREE_ASSERT_OK(iree_hal_semaphore_query(source_semaphore, &value));
  EXPECT_EQ(50ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(dest_semaphore, &value));
  EXPECT_EQ(200ull, value);

  iree_hal_semaphore_release(source_semaphore);
  iree_hal_semaphore_release(dest_semaphore);
}

// Test error cases
TEST_F(ExternalTimepointTest, ErrorCases) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Test export with unsupported timepoint type
  iree_hal_external_timepoint_t external_timepoint;
  iree_status_t status = iree_hal_semaphore_export_timepoint(
      semaphore, 10ull, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_CUDA_EVENT,  // Unsupported
      IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE, &external_timepoint);

  // Should either be unimplemented (no external timepoint support) or invalid
  // argument (unsupported type)
  EXPECT_TRUE(iree_status_is_unimplemented(status) ||
              iree_status_is_invalid_argument(status));
  iree_status_ignore(status);

  // Test import with unsupported timepoint type
  external_timepoint.type =
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT;  // Unsupported
  external_timepoint.flags = IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE;
  status = iree_hal_semaphore_import_timepoint(
      semaphore, 10ull, IREE_HAL_QUEUE_AFFINITY_ANY, external_timepoint);

  // Should either be unimplemented (no external timepoint support) or invalid
  // argument (unsupported type)
  EXPECT_TRUE(iree_status_is_unimplemented(status) ||
              iree_status_is_invalid_argument(status));
  iree_status_ignore(status);

  iree_hal_semaphore_release(semaphore);
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_EXTERNAL_TIMEPOINT_TEST_H_
