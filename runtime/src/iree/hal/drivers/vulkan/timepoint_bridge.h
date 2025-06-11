// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_POLLER_H_
#define IREE_HAL_DRIVERS_VULKAN_POLLER_H_

#include "iree/base/api.h"
// #include "iree/base/internal/atomics.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_semaphore.h"

using namespace iree::hal::vulkan;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// iree_event_set
// iree_notification_post

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

// TODO iree_hal_semaphore_timepoint_t and timepoint deadline

typedef struct iree_hal_event_list_t {
  iree_host_size_t count;
  iree_event_t **events;
  uint64_t *payload_values;
} iree_hal_event_list_t;

typedef struct iree_hal_vulkan_timepoint_bridge_signal_set {
  iree_hal_event_list_t handles;
  size_t capacity;
} iree_hal_vulkan_timepoint_bridge_signal_set_t;

typedef struct iree_hal_vulkan_timepoint_bridge_wait_set {
  iree_hal_semaphore_list_t handles;
  size_t capacity;
} iree_hal_vulkan_timepoint_bridge_wait_set_t;

typedef struct iree_hal_vulkan_timepoint_bridge {
  iree_allocator_t allocator;
  iree_hal_vulkan_timepoint_bridge_wait_set_t wait_set;
  iree_hal_vulkan_timepoint_bridge_signal_set_t signal_set;
  // all semaphores are vulkan semaphores (VPI1)
  // TODO have one waitable that is used in case o empty wait_set, and might be
  // more efficient to wait on

  iree_atomic_int32_t state;
} iree_hal_vulkan_timepoint_bridge_t;

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_initialize(
    iree_allocator_t space_allocator,
    iree_hal_vulkan_timepoint_bridge_wait_set_t *out_wait_set);

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_initialize(
    iree_allocator_t space_allocator,
    iree_hal_vulkan_timepoint_bridge_signal_set_t *out_wait_set);

iree_status_t set_initialize(iree_allocator_t space_allocator,
                             uint64_t **payload_values, void *items,
                             size_t item_size, iree_host_size_t initial_size);

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_insert(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_hal_semaphore_t *semaphore, uint64_t value);

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_insert(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set,
    iree_event_t *event, uint64_t value);

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_erase(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set, size_t index);

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_signal(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *signal_set, size_t index);

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_erase(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set, size_t index);

void vulkan_iree_hal_vulkan_timepoint_bridge_wait_set_wait_any(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_timeout_t earliest_deadline_ns, uint64_t *resolved_values);

iree_status_t iree_hal_vulkan_timepoint_bridge_insert(
    iree_hal_vulkan_timepoint_bridge_t *poller, iree_hal_semaphore_t *semaphore,
    uint64_t value, iree_event_t *event);

iree_status_t iree_hal_vulkan_timepoint_bridge_allocate(
    iree_allocator_t space_allocator,
    iree_hal_vulkan_timepoint_bridge_t *out_poller);

void iree_hal_vulkan_timepoint_bridge_free(
    iree_hal_vulkan_timepoint_bridge_t *poller);
// TODO need a method to yeet a specific semaphore out?

iree_status_t iree_hal_vulkan_timepoint_bridge_pump(
    iree_hal_vulkan_timepoint_bridge_t *poller);

iree_status_t iree_hal_vulkan_timepoint_bridge_erase(
    iree_hal_vulkan_timepoint_bridge_t *poller, size_t index);

void iree_hal_vulkan_timepoint_bridge_pump_until_exit(
    iree_hal_vulkan_timepoint_bridge_t *poller);

void iree_task_poller_request_exit(iree_hal_vulkan_timepoint_bridge_t *poller);

void vulkan_iree_hal_vulkan_timepoint_bridge_commit_cancel(
    iree_hal_semaphore_list_t semaphores_list);

void vulkan_iree_hal_vulkan_timepoint_bridge_main(
    iree_hal_vulkan_timepoint_bridge_t *poller);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif
