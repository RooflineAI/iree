// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/native_semaphore.h"

#include <cstddef>

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/drivers/vulkan/dynamic_symbol_tables.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/timepoint_bridge.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"
#include "iree/hal/utils/semaphore_base.h"

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_native_semaphore_t {
  iree_hal_semaphore_t base;
  VkDeviceHandle* logical_device;
  VkSemaphore handle;
  iree_atomic_intptr_t failure_status;
} iree_hal_vulkan_native_semaphore_t;

namespace {
extern const iree_hal_semaphore_vtable_t
    iree_hal_vulkan_native_semaphore_vtable;
}  // namespace

static iree_hal_vulkan_native_semaphore_t*
iree_hal_vulkan_native_semaphore_cast(iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_semaphore_vtable);
  return (iree_hal_vulkan_native_semaphore_t*)base_value;
}

iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = NULL;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = initial_value;

  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &timeline_create_info;
  create_info.flags = 0;
  VkSemaphore handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateSemaphore(
                                  *logical_device, &create_info,
                                  logical_device->allocator(), &handle),
                              "vkCreateSemaphore"));

  iree_hal_vulkan_native_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      logical_device->host_allocator(), sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_vulkan_native_semaphore_vtable,
                                  &semaphore->base);
    semaphore->logical_device = logical_device;
    semaphore->handle = handle;
    iree_atomic_store(&semaphore->failure_status, 0, iree_memory_order_release);
    *out_semaphore = &semaphore->base;
  } else {
    logical_device->syms()->vkDestroySemaphore(*logical_device, handle,
                                               logical_device->allocator());
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore((iree_status_t)iree_atomic_load(
      &semaphore->failure_status, iree_memory_order_acquire));

  semaphore->logical_device->syms()->vkDestroySemaphore(
      *semaphore->logical_device, semaphore->handle,
      semaphore->logical_device->allocator());

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_native_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is(&semaphore->resource,
                              &iree_hal_vulkan_native_semaphore_vtable);
}

VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  return semaphore->handle;
}

static iree_status_t iree_hal_vulkan_native_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  *out_value = 0;

  // Query from Vulkan source-of-truth.
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(VK_RESULT_TO_STATUS(
      semaphore->logical_device->syms()->vkGetSemaphoreCounterValue(
          *semaphore->logical_device, semaphore->handle, &value),
      "vkGetSemaphoreCounterValue"));

  // If the semaphore failed then clone the status so we can report it.
  if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    iree_status_t failure_status = (iree_status_t)iree_atomic_load(
        &semaphore->failure_status, iree_memory_order_acquire);
    if (iree_status_is_ok(failure_status)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "overflowed timeline semaphore max value");
    }
    iree_hal_semaphore_notify(&semaphore->base, value,
                              iree_status_code(failure_status));
    return iree_status_clone(failure_status);
  }

  // Notify timepoints on the query as we aren't notified by Vulkan when a
  // device-side signal occurs. This helps us keep latencies lower by flushing
  // timepoints without needing waits at the risk of making queries slower.
  iree_hal_semaphore_notify(&semaphore->base, value, IREE_STATUS_OK);

  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_native_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);

  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = new_value;
  iree_status_t status =
      VK_RESULT_TO_STATUS(semaphore->logical_device->syms()->vkSignalSemaphore(
                              *semaphore->logical_device, &signal_info),
                          "vkSignalSemaphore");

  // Notify of semaphore reaching a new timepoint.
  iree_hal_semaphore_notify(&semaphore->base, new_value,
                            iree_status_code(status));

  return status;
}

static void iree_hal_vulkan_native_semaphore_fail(
    iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  iree_status_code_t status_code = iree_status_code(status);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  iree_status_t old_status = iree_ok_status();
  if (!iree_atomic_compare_exchange_strong(
          &semaphore->failure_status, (intptr_t*)&old_status, (intptr_t)status,
          iree_memory_order_acq_rel,
          iree_memory_order_relaxed /* old_status is unused */)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
    return;
  }

  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  // NOTE: we don't care about the result in case of failures as we are
  // failing and the caller will likely be tearing everything down anyway.
  semaphore->logical_device->syms()->vkSignalSemaphore(
      *semaphore->logical_device, &signal_info);

  // Notify of semaphore failure.
  iree_hal_semaphore_notify(&semaphore->base, signal_info.value, status_code);
}

iree_status_t iree_hal_vulkan_native_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout,
    VkSemaphoreWaitFlags wait_flags, uint64_t* out_values) {
  if (semaphore_list->count == 0) return iree_ok_status();

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  uint64_t timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    timeout_ns = UINT64_MAX;
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    timeout_ns = 0;
  } else {
    iree_time_t now_ns = iree_time_now();
    timeout_ns = deadline_ns < now_ns ? 0 : (uint64_t)(deadline_ns - now_ns);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphore* semaphore_handles =
      (VkSemaphore*)iree_alloca(semaphore_list->count * sizeof(VkSemaphore));
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    semaphore_handles[i] =
        iree_hal_vulkan_native_semaphore_handle(semaphore_list->semaphores[i]);
  }

  VkSemaphoreWaitInfo wait_info;
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.pNext = nullptr;
  wait_info.flags = wait_flags;
  wait_info.semaphoreCount = semaphore_list->count;
  wait_info.pSemaphores = semaphore_handles;
  wait_info.pValues = semaphore_list->payload_values;
  static_assert(
      sizeof(wait_info.pValues[0]) == sizeof(semaphore_list->payload_values[0]),
      "payload value type must match vulkan expected size");

  // NOTE: this may fail with a timeout (VK_TIMEOUT) or in the case of a
  // device loss event may return either VK_SUCCESS *or* VK_ERROR_DEVICE_LOST.
  // We may want to explicitly query for device loss after a successful wait
  // to ensure we consistently return errors.
  VkResult result = logical_device->syms()->vkWaitSemaphores(
      *logical_device, &wait_info, timeout_ns);

  IREE_TRACE_ZONE_END(z0);

  // Poll semaphores and notify of state changes.
  // We aren't notified of device-side signals by Vulkan and need to check to
  // see if any signals were made.
  //
  // TODO(benvanik): on success optimize this to notify of reaching the new
  // values instead of a full poll; it'll avoid a bunch of additional API
  // queries.
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    uint64_t value = 0;
    iree_status_ignore(iree_hal_vulkan_native_semaphore_query(
        semaphore_list->semaphores[i], &value));
    if (out_values) {
      out_values[i] = value;
    }
  }

  if (result == VK_SUCCESS) {
    return iree_ok_status();
  } else if (result == VK_ERROR_DEVICE_LOST) {
    // Nothing we do now matters.
    return VK_RESULT_TO_STATUS(result, "vkWaitSemaphores");
  } else if (result == VK_TIMEOUT) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  return VK_RESULT_TO_STATUS(result, "vkWaitSemaphores");
}

static iree_status_t iree_hal_vulkan_native_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  iree_hal_semaphore_list_t semaphore_list = {
      /*.count=*/1,
      /*.semaphores=*/&base_semaphore,
      /*.payload_values=*/&value,
  };
  return iree_hal_vulkan_native_semaphore_multi_wait(
      semaphore->logical_device, &semaphore_list, timeout, 0);
}

iree_status_t iree_hal_vulkan_native_semaphore_multi_wait_any(
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout,
    uint64_t* out_values) {
  VkSemaphoreWaitFlags flags = VK_SEMAPHORE_WAIT_ANY_BIT_KHR;

  if (semaphore_list->count == 0) return iree_ok_status();
  return iree_hal_vulkan_native_semaphore_multi_wait(
      iree_hal_vulkan_native_semaphore_cast(semaphore_list->semaphores[0])
          ->logical_device,
      semaphore_list, timeout, flags, out_values);
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore, VkSemaphore* out_handle) {
  IREE_ASSERT_ARGUMENT(base_semaphore);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  *out_handle = semaphore->handle;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_vulkan_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  IREE_ASSERT_ARGUMENT(out_external_timepoint);
  // TODO use all inputs
  //  Only host-visible snapshot via wait-primitive for now.
  if ((requested_type & IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE) == 0) {
    iree_hal_vulkan_native_semaphore_t* semaphore =
        iree_hal_vulkan_native_semaphore_cast(base_semaphore);

    iree_hal_vulkan_timepoint_bridge_t* bridge =
        semaphore->logical_device->timepoint_bridge;

    iree_event_t* exp_event;
    // allocate event
    iree_status_t status = iree_allocator_malloc(
        bridge->allocator, sizeof(*exp_event), (void**)&exp_event);

    // TODO pool events
    iree_event_initialize(false, exp_event);

    // TODO shrtcut with iree_wait_primitive_immediate (req lock?) if signaled
    // or failure
    iree_hal_vulkan_timepoint_bridge_import(bridge, base_semaphore, value,
                                            exp_event);

    // TODO in the future: iree_event_deinitialize(exp_event);
    // who deinitializes it? the caller?
    // when is it removed?

    iree_wait_primitive_t wp =
        iree_make_wait_primitive(exp_event->type, exp_event->value);

    // type-erase into external timepoint
    *out_external_timepoint = (iree_hal_external_timepoint_t){
        .type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
        .flags = requested_flags,
        .compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
                         IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT,
        .handle = {.wait_primitive = wp}};

    return iree_ok_status();
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VULKAN only supports the export of WAIT_PRIMITIVE timepoints");
  }

  //   IREE_ASSERT_ARGUMENT(out_external_timepoint);
  //   memset(out_external_timepoint, 0x00, sizeof(*out_external_timepoint));
  //   // if ((requested_type &
  //   IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_VULKAN_TIMEPOINT)
  //   ==
  //   //     0) {  // TODO iree_all_bits_set is in runtime
  //   //   return iree_make_status(
  //   //       IREE_STATUS_INVALID_ARGUMENT,
  //   //       "VULKAN only supports the export of DEVICE_WAIT timepoints");
  //   // }

  //   // iree_hal_vulkan_native_semaphore_t* semaphore =
  //   //     iree_hal_vulkan_native_semaphore_cast(base_semaphore);

  //   // IREE_TRACE_ZONE_BEGIN(z0);
  //   // // TODO fast-path: if the semaphore is already satisfied?

  //   // // type-erase into external timepoint
  //   // *out_external_timepoint = (iree_hal_external_timepoint_t){
  //   //     .type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_VULKAN_TIMEPOINT,
  //   //     .flags = requested_flags,
  //   //     .handle = {.vulkan_semaphore = semaphore},
  //   //     .compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
  //   //                      IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT};

  //   // // NOTE: does __not__ transfer ownership; the caller must ensure the
  //   // // semaphore lives until it has waited/imported the time-point.
  //   // return iree_ok_status();

  //   // Only host-visible snapshot via wait-primitive for now.
  //   if ((requested_type & IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE) ==
  //   0) {
  //     return iree_make_status(
  //         IREE_STATUS_INVALID_ARGUMENT,
  //         "VULKAN only supports the export of WAIT_PRIMITIVE timepoints");
  //   }
  //   // // Make sure the semaphore **will** reach |value|: fast-poll first,
  //   then
  //   // // block (host wait) so vkGetSemaphoreFdKHR never hands out a fd that
  //   will
  //   // // never signal.
  //   // IREE_RETURN_IF_ERROR(
  //   //     iree_hal_semaphore_wait(base_semaphore, value,
  //   //     iree_infinite_timeout()));

  //   // Vulkan → POSIX fd

  //   iree_hal_vulkan_native_semaphore_t* sem =
  //       iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  //   // needs to be locked?
  //   iree_slim_mutex_lock(&sem->base.timepoint_mutex);

  //   VkSemaphoreGetFdInfoKHR get_fd_info = {
  //       .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
  //       .pNext = NULL,
  //       .semaphore = sem->handle,
  //       .handleType =
  //           VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,  //
  //           IREE_HAVE_WAIT_TYPE_EVENTFD,
  //                                                             //
  //                                                             IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE,
  //                                                             //
  //                                                             IREE_VULKAN_TIMELINE_HANDLE_TYPE,
  //   };
  //   int fd = -1;
  //   IREE_RETURN_IF_ERROR(
  //       VK_RESULT_TO_STATUS(sem->logical_device->syms()->vkGetSemaphoreFdKHR(
  //                               *sem->logical_device, &get_fd_info, &fd),
  //                           "vkGetSemaphoreFdKHR"));

  //   // Wrap fd in an IREE wait-primitive so higher layers can poll/wait.
  //   iree_wait_primitive_t wp = iree_make_wait_primitive(
  //       IREE_HAVE_WAIT_TYPE_EVENTFD, iree_wait_primitive_value_t{.event =
  //       fd});

  //   // type-erase into external timepoint
  //   *out_external_timepoint = (iree_hal_external_timepoint_t){
  //       .type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
  //       .flags = requested_flags,
  //       .compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
  //                        IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT,
  //       .handle = {.wait_primitive = wp}};

  //   // iree_hal_semaphore_retain(base_semaphore); ??
  //   iree_slim_mutex_unlock(&sem->base.timepoint_mutex);

  //   return iree_ok_status();
}

namespace {
const iree_hal_semaphore_vtable_t iree_hal_vulkan_native_semaphore_vtable = {
    /*.destroy=*/iree_hal_vulkan_native_semaphore_destroy,
    /*.query=*/iree_hal_vulkan_native_semaphore_query,
    /*.signal=*/iree_hal_vulkan_native_semaphore_signal,
    /*.fail=*/iree_hal_vulkan_native_semaphore_fail,
    /*.wait=*/iree_hal_vulkan_native_semaphore_wait,
    /*.import_timepoint=*/iree_hal_vulkan_semaphore_import_timepoint,
    /*.export_timepoint=*/iree_hal_vulkan_semaphore_export_timepoint,
};
}  // namespace
