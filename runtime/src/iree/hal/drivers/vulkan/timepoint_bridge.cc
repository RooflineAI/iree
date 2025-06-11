// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "timepoint_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define VULKAN_TIMEPOINT_BRIDGE_POOL_SIZE 16

// TODO iree_hal_semaphore_timepoint_t and timepoint deadline
// TODO multi import
// TODO freeing of event
// TODO disable by default

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_initialize(
    iree_allocator_t allocator,
    iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait,
    iree_hal_vulkan_timepoint_bridge_wait_set *out_wait_set) {
  memset(out_wait_set, 0, sizeof(*out_wait_set));

  iree_status_t status = iree_allocator_malloc(
      allocator, VULKAN_TIMEPOINT_BRIDGE_POOL_SIZE * sizeof(uint64_t),
      (void **)&out_wait_set->handles.payload_values);

  IREE_CHECK_OK(status);

  status = iree_allocator_malloc(
      allocator,
      VULKAN_TIMEPOINT_BRIDGE_POOL_SIZE * sizeof(iree_hal_semaphore_t *),
      (void **)&out_wait_set->handles.semaphores);

  IREE_CHECK_OK(status);

  out_wait_set->capacity = VULKAN_TIMEPOINT_BRIDGE_POOL_SIZE;
  out_wait_set->multi_wait = multi_wait;
  return status;
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_initialize(
    iree_allocator_t allocator,
    iree_hal_vulkan_timepoint_bridge_signal_set_t *out_wait_set) {
  memset(out_wait_set, 0, sizeof(*out_wait_set));

  iree_status_t status = iree_allocator_malloc(
      allocator, VULKAN_TIMEPOINT_BRIDGE_POOL_SIZE * sizeof(iree_event_t *),
      (void **)&out_wait_set->handles.events);

  out_wait_set->capacity = VULKAN_TIMEPOINT_BRIDGE_POOL_SIZE;
  return status;
}

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_insert(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_hal_semaphore_t *semaphore, uint64_t value) {
  if (wait_set->capacity > wait_set->handles.count) {
    // keep the semaphore alive while it is waited for
    iree_hal_semaphore_retain(semaphore);

    iree_host_size_t wait_set_tail_index = wait_set->handles.count;
    wait_set->handles.semaphores[wait_set_tail_index] = semaphore;
    wait_set->handles.payload_values[wait_set_tail_index] = value;
    ++(wait_set->handles.count);

    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "realloc not impl");
}

iree_status_t iree_hal_vulkan_timepoint_bridge_import(
    iree_hal_vulkan_timepoint_bridge_t *bridge, iree_hal_semaphore_t *semaphore,
    uint64_t value, iree_event_t *event) {
  IREE_ASSERT_ARGUMENT(bridge);
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(event);
  iree_hal_semaphore_retain(semaphore);
  iree_slim_mutex_lock(&(bridge->mutex));
  bridge->pending_imports.semaphore = semaphore;
  bridge->pending_imports.value = value;
  bridge->pending_imports.event = event;
  iree_slim_mutex_unlock(&(bridge->mutex));
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_insert(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set,
    iree_event_t *event) {
  if (wait_set->capacity > wait_set->handles.count) {
    iree_host_size_t wait_set_tail_index = wait_set->handles.count;
    wait_set->handles.events[wait_set_tail_index] = event;
    ++(wait_set->handles.count);

    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "realloc not impl");
}

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_erase(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set, size_t index) {
  assert(wait_set->handles.count > 0);
  iree_hal_semaphore_release(wait_set->handles.semaphores[index]);

  wait_set->handles.semaphores[index] = NULL;
  wait_set->handles.payload_values[index] = 0;
  // swap with last

  size_t tail_index = wait_set->handles.count - 1;
  if (index != tail_index) {
    wait_set->handles.semaphores[index] =
        wait_set->handles.semaphores[tail_index];
    wait_set->handles.payload_values[index] =
        wait_set->handles.payload_values[tail_index];
  }

  wait_set->handles.count = tail_index;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_signal(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *signal_set, size_t index) {
  assert(signal_set->handles.count > index);
  if (signal_set->handles.events[index] != NULL)
    iree_event_set(signal_set->handles.events[index]);
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_erase(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set, size_t index) {
  assert(wait_set->handles.count > 0);

  wait_set->handles.events[index] = NULL;
  // swap with last

  size_t tail_index = wait_set->handles.count - 1;
  if (index != tail_index) {
    wait_set->handles.events[index] = wait_set->handles.events[tail_index];
  }

  wait_set->handles.count = tail_index;
  return iree_ok_status();
}

void vulkan_iree_hal_vulkan_timepoint_bridge_wait_set_wait_any(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_timeout_t earliest_deadline_ns, uint64_t *resolved_values) {
  IREE_ASSERT_ARGUMENT(wait_set);

  if (wait_set->handles.count > 0) {
    iree_status_t status = wait_set->multi_wait(
        &wait_set->handles, earliest_deadline_ns, resolved_values);
    // TODO handle status

    // check which to notify
    for (iree_host_size_t i = 0; i < wait_set->handles.count; ++i) {
      if (resolved_values[i] >= wait_set->handles.payload_values[i]) {
        resolved_values[i] = true;
      } else {
        resolved_values[i] = false;
      }
    }
  }
}

iree_status_t iree_hal_vulkan_timepoint_bridge_initialize(
    iree_allocator_t allocator, iree_hal_semaphore_t *wake_event,
    iree_hal_vulkan_timepoint_bridge_multi_wait_any_fn_t multi_wait,
    iree_hal_vulkan_timepoint_bridge_t *out_bridge) {
  IREE_ASSERT_ARGUMENT(out_bridge);

  iree_status_t status = iree_hal_vulkan_timepoint_bridge_wait_set_initialize(
      allocator, multi_wait, &out_bridge->wait_set);
  IREE_CHECK_OK(status);

  status = iree_hal_vulkan_timepoint_bridge_signal_set_initialize(
      allocator, &out_bridge->signal_set);
  IREE_CHECK_OK(status);

  status = iree_hal_vulkan_timepoint_bridge_insert(out_bridge, wake_event, 1,
                                                   nullptr);
  IREE_CHECK_OK(status);

  out_bridge->allocator = allocator;
  out_bridge->pending_imports.semaphore = NULL;
  out_bridge->pending_imports.event = NULL;

  out_bridge->state = IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_SUSPENDED;

  iree_slim_mutex_initialize(&out_bridge->mutex);

  IREE_CHECK_OK(status);
  return status;
}

void iree_hal_vulkan_timepoint_bridge_free(
    iree_hal_vulkan_timepoint_bridge_t *bridge) {
  IREE_ASSERT_ARGUMENT(bridge);

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = bridge->allocator;
  for (iree_host_size_t i = 0; i < bridge->wait_set.handles.count; ++i) {
    iree_hal_semaphore_release(bridge->wait_set.handles.semaphores[i]);
  }

  if (bridge->pending_imports.semaphore)
    iree_hal_semaphore_release(bridge->pending_imports.semaphore);

  iree_allocator_free(allocator, bridge->wait_set.handles.payload_values);
  iree_allocator_free(allocator, bridge->wait_set.handles.semaphores);
  iree_allocator_free(allocator, bridge->signal_set.handles.events);

  iree_slim_mutex_deinitialize(&bridge->mutex);
  IREE_TRACE_ZONE_END(z0);

#ifndef NDEBUG
  memset(bridge, 0, sizeof(*bridge));
#endif
}

iree_status_t iree_hal_vulkan_timepoint_bridge_insert(
    iree_hal_vulkan_timepoint_bridge_t *bridge, iree_hal_semaphore_t *semaphore,
    uint64_t value, iree_event_t *event) {
  IREE_ASSERT_ARGUMENT(bridge);
  IREE_ASSERT_ARGUMENT(semaphore);

  iree_status_t status = iree_hal_vulkan_timepoint_bridge_wait_set_insert(
      &bridge->wait_set, semaphore, value);
  IREE_CHECK_OK(status);
  status = iree_hal_vulkan_timepoint_bridge_signal_set_insert(
      &bridge->signal_set, event);
  IREE_CHECK_OK(status);

  iree_hal_semaphore_signal(bridge->wait_set.handles.semaphores[0],
                            bridge->wait_set.handles.payload_values[0] + 1);
  return status;
}

iree_status_t iree_hal_vulkan_timepoint_bridge_pump(
    iree_hal_vulkan_timepoint_bridge_t *bridge) {
  uint64_t *semaphore_values = (uint64_t *)iree_alloca(
      bridge->wait_set.handles.count * sizeof(uint64_t));

  vulkan_iree_hal_vulkan_timepoint_bridge_wait_set_wait_any(
      &bridge->wait_set, iree_infinite_timeout(), semaphore_values);

  // check which to notify
  for (iree_host_size_t i = 0; i < bridge->wait_set.handles.count; ++i) {
    if (semaphore_values[i]) {
      iree_hal_vulkan_timepoint_bridge_signal_set_signal(&bridge->signal_set,
                                                         i);
    }
  }

  for (iree_host_size_t i = 0; i < bridge->wait_set.handles.count; ++i) {
    if (semaphore_values[i]) {
      if (i == 0) {
        iree_slim_mutex_lock(&bridge->mutex);
        if (bridge->pending_imports.semaphore != NULL) {
          iree_hal_vulkan_timepoint_bridge_insert(
              bridge, bridge->pending_imports.semaphore,
              bridge->pending_imports.value, bridge->pending_imports.event);
          memset(&bridge->pending_imports, 0, sizeof(bridge->pending_imports));
        }
        iree_slim_mutex_unlock(&bridge->mutex);
        if (bridge->wait_set.handles.payload_values[i] != SEM_VALUE_MAX)
          ++bridge->wait_set.handles.payload_values[i];

      } else {
        iree_hal_vulkan_timepoint_bridge_signal_set_erase(&bridge->signal_set,
                                                          i);
      }
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_timepoint_bridge_erase(
    iree_hal_vulkan_timepoint_bridge_t *bridge, size_t index) {
  iree_hal_vulkan_timepoint_bridge_wait_set_erase(&bridge->wait_set, index);
  iree_hal_vulkan_timepoint_bridge_signal_set_erase(&bridge->signal_set, index);
  return iree_ok_status();
}

void iree_hal_vulkan_timepoint_bridge_pump_until_exit(
    iree_hal_vulkan_timepoint_bridge_t *bridge) {
  while (true) {
    if (iree_atomic_load(&bridge->state, iree_memory_order_acquire) ==
        IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_EXITING) {
      // Thread exit requested - cancel pumping.
      break;
    }

    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_vulkan_timepoint_bridge_pump_until_exit");

    iree_hal_vulkan_timepoint_bridge_pump(bridge);
    IREE_TRACE_ZONE_END(z0);
  }
}

int vulkan_iree_hal_vulkan_timepoint_bridge_main(
    iree_hal_vulkan_timepoint_bridge_t *bridge) {
  IREE_TRACE_ZONE_BEGIN(thread_zone);

  // iree_thread_request_affinity(bridge->thread,
  // bridge->ideal_thread_affinity);

  // Enter the running state immediately. Note that we could have been
  // requested to exit while suspended/still starting up, so check that here
  // before we mess with any data structures.
  const bool should_run =
      iree_atomic_exchange(&bridge->state,
                           IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_RUNNING,
                           iree_memory_order_acq_rel) !=
      IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_EXITING;
  if (IREE_LIKELY(should_run)) {
    // << work happens here >>
    iree_hal_vulkan_timepoint_bridge_pump_until_exit(bridge);
  }

  IREE_TRACE_ZONE_END(thread_zone);
  iree_atomic_store(&bridge->state,
                    IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_ZOMBIE,
                    iree_memory_order_release);

  return 0;
}

iree_status_t iree_hal_vulkan_timepoint_bridge_create(
    iree_hal_vulkan_timepoint_bridge_t *out_bridge) {
  iree_hal_vulkan_timepoint_bridge_t *bridge = NULL;

  iree_status_t status = iree_ok_status();

  // Start background thread
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t params;
    memset(&params, 0, sizeof(params));
    params.name = iree_make_cstring_view("iree_hal_vulkan_timepoint_bridge");
    params.create_suspended = false;
    params.priority_class = IREE_THREAD_PRIORITY_CLASS_NORMAL;
    status = iree_thread_create(
        (iree_thread_entry_t)vulkan_iree_hal_vulkan_timepoint_bridge_main,
        bridge, params, bridge->allocator, &bridge->thread);
  }

  return status;
}

void iree_hal_vulkan_timepoint_bridge_request_exit(
    iree_hal_vulkan_timepoint_bridge_t *bridge) {
  // Signal thread to exit
  iree_atomic_store(&bridge->state,
                    IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_EXITING,
                    iree_memory_order_acq_rel);
  iree_hal_semaphore_signal(bridge->wait_set.handles.semaphores[0],
                            SEM_VALUE_MAX);
}

void iree_hal_vulkan_timepoint_bridge_destroy(
    iree_hal_vulkan_timepoint_bridge_t *bridge) {
  // Wait for thread to exit
  if (bridge->thread) {
    iree_thread_join(bridge->thread);
    iree_thread_release(bridge->thread);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
