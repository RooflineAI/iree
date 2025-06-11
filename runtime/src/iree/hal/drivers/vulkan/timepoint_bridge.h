// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_TIMEPOINT_BRIDGE_H_
#define IREE_HAL_DRIVERS_VULKAN_TIMEPOINT_BRIDGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// This thread waits on a set of vulkan semaphore timeline timepoints (there is
// potential to generalize this), and notifies a corresponding wait_primitive
// once signaled. The goal is to make timepoints exportable for usage ooutside
// of a specific driver. This is somewhat analogous to the task poller thread.

typedef enum iree_hal_vulkan_timepoint_bridge_state_e : int32_t {
  // Wait thread has been created in a suspended state and must be resumed to
  // wake for the first time.
  IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_SUSPENDED = 0,
  // Wait thread is running and servicing wait tasks.
  IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_RUNNING = 1,
  // Wait thread should exit (or is exiting) and will soon enter the zombie
  // state.
  IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_EXITING = 2,
  // Wait thread has exited and entered a 🧟 state (waiting for join).
  // The thread handle is still valid and must be destroyed.
  IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_ZOMBIE = 3,
} iree_hal_vulkan_timepoint_bridge_state_t;

// Represents a pending import waiting for an external timepoint
typedef struct iree_hal_vulkan_timepoint_bridge_external_import_t {
  // The HAL semaphore to signal when ready
  iree_hal_semaphore_t *semaphore;
  uint64_t value;
  iree_event_t *event;
  struct iree_hal_vulkan_timepoint_bridge_external_import_t *next;
} iree_hal_vulkan_timepoint_bridge_external_import_t;

// TODO deadlines

typedef struct iree_hal_event_list_t {
  iree_host_size_t count;
  iree_event_t **events;
} iree_hal_event_list_t;

typedef struct iree_hal_vulkan_timepoint_bridge_signal_set {
  iree_hal_event_list_t handles;
  size_t capacity;
} iree_hal_vulkan_timepoint_bridge_signal_set_t;

typedef iree_status_t (*iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t)(
    const iree_hal_semaphore_list_t *semaphore_list, iree_timeout_t timeout,
    uint64_t *out_values);

typedef struct iree_hal_vulkan_timepoint_bridge_wait_set {
  iree_hal_semaphore_list_t handles;
  size_t capacity;
  iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait;
} iree_hal_vulkan_timepoint_bridge_wait_set_t;

// Represents a bridge for Vulkan timepoints to HAL events.
typedef struct iree_hal_vulkan_timepoint_bridge {
  iree_allocator_t allocator;
  // thread waiting on timepoints.
  iree_thread_t *thread;
  // Mutex guarding access to pending_imports.
  iree_slim_mutex_t mutex;
  // Pending pair of timepoint and event to be inserted into the bridge.
  iree_hal_vulkan_timepoint_bridge_external_import_t pending_imports
      IREE_GUARDED_BY(mutex);
  // Set of timepoints currently being waited on.
  iree_hal_vulkan_timepoint_bridge_wait_set_t wait_set;
  // Set of timepoints to be signaled.
  iree_hal_vulkan_timepoint_bridge_signal_set_t signal_set;
  iree_atomic_int32_t state;
} iree_hal_vulkan_timepoint_bridge_t;

// Initializes a wait set for Vulkan timepoint bridge semaphores.
iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_initialize(
    iree_allocator_t allocator,
    iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t,
    iree_hal_vulkan_timepoint_bridge_wait_set_t *out_wait_set);

// Initializes a signal set for Vulkan timepoint bridge events.
// |out_wait_set|: Initialized signal set.
iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_initialize(
    iree_allocator_t allocator,
    iree_hal_vulkan_timepoint_bridge_signal_set_t *out_wait_set);

// Inserts a semaphore and value into the wait set.
// |wait_set|: The wait set to insert into.
// |semaphore|: Semaphore to wait on.
// |value|: Value to wait for.
iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_insert(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_hal_semaphore_t *semaphore, uint64_t value);

// Inserts an event and value into the signal set.
// |wait_set|: The signal set to insert into.
// |event|: Event to signal.
iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_insert(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set,
    iree_event_t *event);

// Removes an entry from the wait set by index.
iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_erase(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set, size_t index);

// Signals an event in the signal set by index.
// |index|: Index of the event to signal.
iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_signal(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *signal_set, size_t index);

// Removes an entry from the signal set by index.
iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_erase(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set, size_t index);

// Waits for any semaphore in the wait set to be signaled.
// |wait_set|: The wait set to wait on.
// |earliest_deadline_ns|: Timeout deadline in nanoseconds.
// |resolved_values|: Output array for values.
void vulkan_iree_hal_vulkan_timepoint_bridge_wait_set_wait_any(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_timeout_t earliest_deadline_ns, uint64_t *resolved_values);

// Inserts a semaphore, value, and event into the bridge for tracking.
// |semaphore|: Semaphore to track.
// |value|: Value to wait for.
// |event|: Event to signal when ready.
iree_status_t iree_hal_vulkan_timepoint_bridge_insert(
    iree_hal_vulkan_timepoint_bridge_t *bridge, iree_hal_semaphore_t *semaphore,
    uint64_t value, iree_event_t *event);

// Imports a semaphore, event pair into the bridge.
iree_status_t iree_hal_vulkan_timepoint_bridge_import(
    iree_hal_vulkan_timepoint_bridge_t *bridge, iree_hal_semaphore_t *semaphore,
    uint64_t value, iree_event_t *event);

// Initializes a Vulkan timepoint bridge.
// |allocator|: Allocator for memory management.
// |wake_event|: Semaphore to wake the bridge thread.
// |multi_wait|: Function pointer for multi-wait implementation.
// |out_bridge|: Output pointer for the initialized bridge.
iree_status_t iree_hal_vulkan_timepoint_bridge_initialize(
    iree_allocator_t allocator, iree_hal_semaphore_t *wake_event,
    iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait,
    iree_hal_vulkan_timepoint_bridge_t *out_bridge);

// Creates a bridge thread and starts it.
iree_status_t iree_hal_vulkan_timepoint_bridge_create(
    iree_hal_vulkan_timepoint_bridge_t *out_bridge);

// Joins the bridge thread and releases it.
void iree_hal_vulkan_timepoint_bridge_destroy(
    iree_hal_vulkan_timepoint_bridge_t *out_bridge);

// Frees memory associated with a Vulkan timepoint bridge.
void iree_hal_vulkan_timepoint_bridge_free(
    iree_hal_vulkan_timepoint_bridge_t *bridge);

// Pumps the bridge to process pending timepoints.
iree_status_t iree_hal_vulkan_timepoint_bridge_pump(
    iree_hal_vulkan_timepoint_bridge_t *bridge);

// Removes an entry from the bridge by index.
iree_status_t iree_hal_vulkan_timepoint_bridge_erase(
    iree_hal_vulkan_timepoint_bridge_t *bridge, size_t index);

// Continuously pumps the bridge until an exit is requested.
void iree_hal_vulkan_timepoint_bridge_pump_until_exit(
    iree_hal_vulkan_timepoint_bridge_t *bridge);

// Main thread function for the timepoint bridge.
// Returns 0 on success.
int vulkan_iree_hal_vulkan_timepoint_bridge_main(
    iree_hal_vulkan_timepoint_bridge_t *bridge);

// Requests the bridge thread to exit.
// |bridge|: The timepoint bridge bridge.
void iree_hal_vulkan_timepoint_bridge_request_exit(
    iree_hal_vulkan_timepoint_bridge_t *bridge);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif
