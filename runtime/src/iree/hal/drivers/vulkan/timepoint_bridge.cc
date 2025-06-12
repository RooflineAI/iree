// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "timepoint_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define INITIAL_POOL_SIZE 16

// TODO iree_hal_semaphore_timepoint_t and timepoint deadline

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_initialize(
    iree_allocator_t space_allocator,
    iree_hal_vulkan_timepoint_bridge_wait_set *out_wait_set) {
  memset(out_wait_set, 0, sizeof(*out_wait_set));

  iree_status_t status =
      set_initialize(space_allocator, &out_wait_set->handles.payload_values,
                     &out_wait_set->handles.semaphores,
                     sizeof(iree_hal_semaphore_t *), INITIAL_POOL_SIZE);

  out_wait_set->capacity = INITIAL_POOL_SIZE;
  return status;
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_initialize(
    iree_allocator_t space_allocator,
    iree_hal_vulkan_timepoint_bridge_signal_set_t *out_wait_set) {
  memset(out_wait_set, 0, sizeof(*out_wait_set));

  iree_status_t status = set_initialize(
      space_allocator, &out_wait_set->handles.payload_values,
      &out_wait_set->handles.events, sizeof(iree_event_t *), INITIAL_POOL_SIZE);

  out_wait_set->capacity = INITIAL_POOL_SIZE;
  return status;
}

iree_status_t set_initialize(iree_allocator_t space_allocator,
                             uint64_t **payload_values, void *items,
                             size_t item_size, iree_host_size_t initial_size) {
  iree_status_t status = iree_allocator_malloc(
      space_allocator, initial_size * sizeof(uint64_t),
      (void **)payload_values);  // TODO i've seen some things about alignment

  IREE_CHECK_OK(status);  // TODO maybe more graceful

  status = iree_allocator_malloc(space_allocator, initial_size * item_size,
                                 (void **)items);

  IREE_CHECK_OK(status);  // TODO maybe more graceful
  return status;
}

iree_status_t iree_hal_vulkan_timepoint_bridge_wait_set_insert(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_hal_semaphore_t *semaphore, uint64_t value) {
  // TODO assert sema type
  if (wait_set->capacity > wait_set->handles.count) {
    // keep the semaphore alive while it is waited for
    iree_hal_semaphore_retain(semaphore);

    iree_host_size_t wait_set_tail_index = wait_set->handles.count;
    wait_set->handles.semaphores[wait_set_tail_index] = semaphore;
    wait_set->handles.payload_values[wait_set_tail_index] = value;
    ++(wait_set->handles.count);
    // TODO max capacity?

    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "realloc not impl");
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_insert(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set,
    iree_event_t *event, uint64_t value) {
  // TODO assert signal type
  if (wait_set->capacity > wait_set->handles.count) {
    // keep the semaphore alive while it is waited for
    // TODO iree_event_retain(event);

    iree_host_size_t wait_set_tail_index = wait_set->handles.count;
    wait_set->handles.events[wait_set_tail_index] = event;
    wait_set->handles.payload_values[wait_set_tail_index] = value;
    ++(wait_set->handles.count);
    // TODO max capacity?

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

  --wait_set->handles.count;
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_signal(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *signal_set, size_t index) {
  assert(signal_set->handles.count > index);
  iree_event_set(signal_set->handles.events[index]);
}

iree_status_t iree_hal_vulkan_timepoint_bridge_signal_set_erase(
    iree_hal_vulkan_timepoint_bridge_signal_set_t *wait_set, size_t index) {
  assert(wait_set->handles.count > 0);
  // TODO iree_hal_semaphore_release(wait_set->handles.events[index]);

  wait_set->handles.events[index] = NULL;
  wait_set->handles.payload_values[index] = 0;
  // swap with last

  size_t tail_index = wait_set->handles.count - 1;
  if (index != tail_index) {
    wait_set->handles.events[index] = wait_set->handles.events[tail_index];
    wait_set->handles.payload_values[index] =
        wait_set->handles.payload_values[tail_index];
  }

  --wait_set->handles.count;
}

void vulkan_iree_hal_vulkan_timepoint_bridge_wait_set_wait_any(
    iree_hal_vulkan_timepoint_bridge_wait_set_t *wait_set,
    iree_timeout_t earliest_deadline_ns, uint64_t *resolved_values) {
  IREE_ASSERT_ARGUMENT(wait_set);

  if (wait_set->handles.count > 0) {
    VkSemaphoreWaitFlags flags = VK_SEMAPHORE_WAIT_ANY_BIT_KHR;

    // TODO slightly bad, because it does allocations, but on stack
    iree_status_t status = iree_hal_vulkan_native_semaphore_multi_wait_X(
        &wait_set->handles, earliest_deadline_ns, flags, resolved_values);
    // TODO status

    // check which to notify
    for (iree_host_size_t i = 0; i < wait_set->handles.count; ++i) {
      // TODO check immediate/failre? i think this should work
      if (resolved_values[i] >= wait_set->handles.payload_values[i]) {
        resolved_values[i] = true;

      } else {
        resolved_values[i] = false;
      }
    }
  }
}

iree_status_t iree_hal_vulkan_timepoint_bridge_allocate(
    iree_allocator_t space_allocator,
    iree_hal_vulkan_timepoint_bridge_t *out_poller) {
  IREE_ASSERT_ARGUMENT(out_poller);

  iree_status_t status = iree_hal_vulkan_timepoint_bridge_wait_set_initialize(
      space_allocator, &out_poller->wait_set);
  IREE_CHECK_OK(status);  // TODO maybe more graceful

  status = iree_hal_vulkan_timepoint_bridge_signal_set_initialize(
      space_allocator, &out_poller->signal_set);
  IREE_CHECK_OK(status);  // TODO maybe more graceful

  // iree_status_t status =
  //     iree_hal_vulkan_timepoint_bridge_insert(out_poller, semaphore, 1);

  out_poller->allocator = space_allocator;

  out_poller->state =
      IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_SUSPENDED;  // TODO cheating

  IREE_CHECK_OK(status);  // TODO maybe more graceful
  return status;
}

void iree_hal_vulkan_timepoint_bridge_free(
    iree_hal_vulkan_timepoint_bridge_t *poller) {
  IREE_ASSERT_ARGUMENT(poller);

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = poller->allocator;
  iree_allocator_free(allocator, poller->wait_set.handles.payload_values);
  iree_allocator_free(allocator, poller->wait_set.handles.semaphores);
  iree_allocator_free(allocator, poller->signal_set.handles.payload_values);
  iree_allocator_free(allocator, poller->signal_set.handles.events);
  IREE_TRACE_ZONE_END(z0);

#ifndef NDEBUG
  memset(poller, 0, sizeof(*poller));
#endif
}

iree_status_t iree_hal_vulkan_timepoint_bridge_insert(
    iree_hal_vulkan_timepoint_bridge_t *poller, iree_hal_semaphore_t *semaphore,
    uint64_t value, iree_event_t *event) {
  IREE_ASSERT_ARGUMENT(poller);
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(event);

  // inserting same semaphore will result in multiple retains
  // TODO jesus why is there a sema list but no wait source list
  // l_siree_hal_semaphore_await()

  iree_status_t status = iree_hal_vulkan_timepoint_bridge_wait_set_insert(
      &poller->wait_set, semaphore, value);
  IREE_CHECK_OK(status);
  status = iree_hal_vulkan_timepoint_bridge_signal_set_insert(
      &poller->signal_set, event, value);
  IREE_CHECK_OK(status);
  return status;
  // TODO max capa
}

// TODO need a method to yeet a specific semaphore out?

iree_status_t iree_hal_vulkan_timepoint_bridge_pump(
    iree_hal_vulkan_timepoint_bridge_t *poller) {
  uint64_t *semaphore_values = (uint64_t *)iree_alloca(
      poller->wait_set.handles.count * sizeof(uint64_t));

  vulkan_iree_hal_vulkan_timepoint_bridge_wait_set_wait_any(
      &poller->wait_set, iree_infinite_timeout(), semaphore_values);

  // check which to notify
  for (iree_host_size_t i = 0; i < poller->wait_set.handles.count; ++i) {
    // TODO check immediate/failre? i think this should work
    if (semaphore_values[i]) {
      iree_hal_vulkan_timepoint_bridge_signal_set_signal(&poller->signal_set,
                                                         i);  // TODO
                                                              // notify
                                                              // many?
    }
  }

  for (iree_host_size_t i = 0; i < poller->wait_set.handles.count; ++i) {
    if (semaphore_values[i]) {
      iree_hal_vulkan_timepoint_bridge_signal_set_erase(&poller->signal_set, i);
    }
  }
}

iree_status_t iree_hal_vulkan_timepoint_bridge_erase(
    iree_hal_vulkan_timepoint_bridge_t *poller, size_t index) {
  iree_hal_vulkan_timepoint_bridge_wait_set_erase(&poller->wait_set, index);
}

void iree_hal_vulkan_timepoint_bridge_pump_until_exit(
    iree_hal_vulkan_timepoint_bridge_t *poller) {
  while (true) {
    if (iree_atomic_load(&poller->state, iree_memory_order_acquire) ==
        IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_EXITING) {
      // Thread exit requested - cancel pumping.
      break;
    }

    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_vulkan_timepoint_bridge_pump_until_exit");

    iree_hal_vulkan_timepoint_bridge_pump(poller);
    IREE_TRACE_ZONE_END(z0);
  }
}

void iree_task_poller_request_exit(iree_hal_vulkan_timepoint_bridge_t *poller) {
  iree_hal_vulkan_timepoint_bridge_state_t prev_state =
      (iree_hal_vulkan_timepoint_bridge_state_t)iree_atomic_exchange(
          &poller->state, IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_EXITING,
          iree_memory_order_acq_rel);

  // TODO we could abuse the 0th semaphore for internal signaling
}

void vulkan_iree_hal_vulkan_timepoint_bridge_commit_cancel(
    iree_hal_semaphore_list_t semaphores_list) {
  iree_hal_semaphore_list_fail(semaphores_list,
                               iree_make_status(IREE_STATUS_CANCELLED));
}

void vulkan_iree_hal_vulkan_timepoint_bridge_main(
    iree_hal_vulkan_timepoint_bridge_t *poller) {
  IREE_TRACE_ZONE_BEGIN(thread_zone);

  // Enter the running state immediately. Note that we could have been
  // requested to exit while suspended/still starting up, so check that here
  // before we mess with any data structures.
  const bool should_run =
      iree_atomic_exchange(&poller->state,
                           IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_RUNNING,
                           iree_memory_order_acq_rel) !=
      IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_EXITING;
  if (IREE_LIKELY(should_run)) {
    // << work happens here >>
    iree_hal_vulkan_timepoint_bridge_pump_until_exit(poller);
  }

  vulkan_iree_hal_vulkan_timepoint_bridge_commit_cancel(
      poller->wait_set.handles);
  IREE_TRACE_ZONE_END(thread_zone);
  iree_atomic_store(&poller->state,
                    IREE_HAL_VULKAN_TIMEPOINT_BRIDGE_STATE_ZOMBIE,
                    iree_memory_order_release);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
