// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_EXTERNAL_TIMEPOINT_POLLER_H_
#define IREE_HAL_UTILS_EXTERNAL_TIMEPOINT_POLLER_H_

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_external_timepoint_poller_t
    iree_hal_external_timepoint_poller_t;

// Represents a pending import waiting for an external timepoint
typedef struct iree_hal_pending_external_import_t {
  // The HAL semaphore to signal when ready
  iree_hal_semaphore_t* semaphore;
  // Value to signal the semaphore to
  uint64_t target_value;
  // External wait handle to wait on
  iree_wait_handle_t wait_handle;
  // Intrusive linked list
  struct iree_hal_pending_external_import_t* next;
} iree_hal_pending_external_import_t;

// Creates a global external timepoint poller for HAL drivers
// This poller can be shared across multiple drivers in the same process
iree_status_t iree_hal_external_timepoint_poller_create(
    iree_allocator_t host_allocator,
    iree_hal_external_timepoint_poller_t** out_poller);

// Destroys the external timepoint poller and stops the background thread
void iree_hal_external_timepoint_poller_destroy(
    iree_hal_external_timepoint_poller_t* poller);

// Adds a new external timepoint import to be monitored by the poller
// When the external timepoint becomes ready, the semaphore will be signaled
// to the target_value
iree_status_t iree_hal_external_timepoint_poller_add_import(
    iree_hal_external_timepoint_poller_t* poller,
    iree_hal_semaphore_t* semaphore, uint64_t target_value,
    iree_wait_handle_t wait_handle);

// Cancels all pending imports for a specific semaphore (called during cleanup)
// This should be called when a semaphore is being destroyed to avoid
// dangling references
void iree_hal_external_timepoint_poller_cancel_imports(
    iree_hal_external_timepoint_poller_t* poller,
    iree_hal_semaphore_t* semaphore);

//===----------------------------------------------------------------------===//
// Global external timepoint poller singleton
//===----------------------------------------------------------------------===//

// Returns the global external timepoint poller singleton.
// This poller is shared across all HAL drivers in the process and is
// automatically created on first access and destroyed at process exit.
//
// This function is thread-safe and may be called from any thread.
// The returned poller remains valid for the lifetime of the process.
iree_hal_external_timepoint_poller_t* iree_hal_external_timepoint_poller_global(
    void);

// Adds an external timepoint import to the global poller.
// This is a convenience function equivalent to:
//   iree_hal_external_timepoint_poller_add_import(
//       iree_hal_external_timepoint_poller_global(), ...)
//
// Thread-safe and may be called from any thread.
static inline iree_status_t iree_hal_external_timepoint_import_global(
    iree_hal_semaphore_t* semaphore, uint64_t target_value,
    iree_wait_handle_t wait_handle) {
  return iree_hal_external_timepoint_poller_add_import(
      iree_hal_external_timepoint_poller_global(), semaphore, target_value,
      wait_handle);
}

// Cancels all pending imports for a semaphore from the global poller.
// This is a convenience function equivalent to:
//   iree_hal_external_timepoint_poller_cancel_imports(
//       iree_hal_external_timepoint_poller_global(), ...)
//
// Thread-safe and may be called from any thread.
static inline void iree_hal_external_timepoint_cancel_imports_global(
    iree_hal_semaphore_t* semaphore) {
  iree_hal_external_timepoint_poller_cancel_imports(
      iree_hal_external_timepoint_poller_global(), semaphore);
}

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_UTILS_EXTERNAL_TIMEPOINT_POLLER_H_
