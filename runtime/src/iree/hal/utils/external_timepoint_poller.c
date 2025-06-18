// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/external_timepoint_poller.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/call_once.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/string_view.h"

struct iree_hal_external_timepoint_poller_t {
  iree_allocator_t host_allocator;

  // Background thread that polls for ready imports
  iree_thread_t* thread;
  iree_atomic_int32_t should_exit;

  // Event to wake the poller when new imports are added
  iree_event_t wake_event;

  // Mutex protecting the pending imports list
  iree_slim_mutex_t mutex;

  // Linked list of pending imports
  iree_hal_pending_external_import_t* pending_imports;

  // Wait set for efficient multi-wait
  iree_wait_set_t* wait_set;
};

// Main thread function for the external timepoint poller
static int iree_hal_external_timepoint_poller_main(void* arg) {
  iree_hal_external_timepoint_poller_t* poller =
      (iree_hal_external_timepoint_poller_t*)arg;

  while (!iree_atomic_load(&poller->should_exit, iree_memory_order_acquire)) {
    // Clear wait set and rebuild from current pending imports
    iree_wait_set_clear(poller->wait_set);

    // Always include the wake event in the wait set
    {
      iree_status_t status_insert =
          iree_wait_set_insert(poller->wait_set, poller->wake_event);
      if (!iree_status_is_ok(status_insert)) {
        // Should never happen given small wait-set size. Log and break to
        // avoid spinning forever.
        break;
      }
    }

    // Add all pending imports to wait set
    iree_slim_mutex_lock(&poller->mutex);
    iree_host_size_t import_count = 0;
    for (iree_hal_pending_external_import_t* import = poller->pending_imports;
         import != NULL; import = import->next) {
      iree_wait_set_insert(poller->wait_set, import->wait_handle);
      import_count++;
    }
    iree_slim_mutex_unlock(&poller->mutex);

    if (import_count == 0) {
      // No imports to wait on, just wait for wake event or exit
      iree_wait_handle_t wake_handle;
      iree_wait_any(poller->wait_set, IREE_TIME_INFINITE_FUTURE, &wake_handle);
      iree_event_reset(&poller->wake_event);
      continue;
    }

    // Wait for any import to become ready
    iree_wait_handle_t wake_handle;
    iree_status_t status = iree_wait_any(
        poller->wait_set, IREE_TIME_INFINITE_FUTURE, &wake_handle);

    if (!iree_status_is_ok(status)) {
      // Handle error - for now just continue
      continue;
    }

    // Check if wake event was signaled (wake_handle == poller->wake_event)
    if (wake_handle.type == poller->wake_event.type &&
        memcmp(&wake_handle.value, &poller->wake_event.value,
               sizeof(wake_handle.value)) == 0) {
      // Wake event was signaled - reset it and continue
      iree_event_reset(&poller->wake_event);
      continue;
    }

    // Find and detach the ready import **while holding the mutex**
    iree_slim_mutex_lock(&poller->mutex);
    iree_hal_pending_external_import_t* prev = NULL;
    iree_hal_pending_external_import_t* current = poller->pending_imports;
    iree_hal_pending_external_import_t* ready_import = NULL;

    while (current != NULL) {
      if (current->wait_handle.type == wake_handle.type &&
          memcmp(&current->wait_handle.value, &wake_handle.value,
                 sizeof(wake_handle.value)) == 0) {
        // Detach from list.
        if (prev) {
          prev->next = current->next;
        } else {
          poller->pending_imports = current->next;
        }
        ready_import = current;
        break;
      }
      prev = current;
      current = current->next;
    }
    iree_slim_mutex_unlock(&poller->mutex);

    // If we found a matching import, perform the (potentially expensive)
    // semaphore operation outside the mutex to avoid lock-ordering issues.
    if (ready_import) {
      iree_hal_semaphore_signal(ready_import->semaphore,
                                ready_import->target_value);
      iree_hal_semaphore_release(ready_import->semaphore);
      iree_allocator_free(poller->host_allocator, ready_import);
    }
  }

  return 0;
}

iree_status_t iree_hal_external_timepoint_poller_create(
    iree_allocator_t host_allocator,
    iree_hal_external_timepoint_poller_t** out_poller) {
  iree_hal_external_timepoint_poller_t* poller = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*poller), (void**)&poller));

  poller->host_allocator = host_allocator;
  iree_atomic_store(&poller->should_exit, 0, iree_memory_order_release);
  poller->pending_imports = NULL;

  iree_status_t status = iree_ok_status();

  // Initialize synchronization primitives
  iree_slim_mutex_initialize(&poller->mutex);
  if (iree_status_is_ok(status)) {
    status =
        iree_event_initialize(/*initial_state=*/false, &poller->wake_event);
  }

  // Create wait set
  if (iree_status_is_ok(status)) {
    status = iree_wait_set_allocate(/*capacity=*/128, host_allocator,
                                    &poller->wait_set);
  }

  // Start background thread
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t params;
    memset(&params, 0, sizeof(params));
    params.name = iree_make_cstring_view("external_timepoint_poller");
    params.create_suspended = false;
    params.priority_class = IREE_THREAD_PRIORITY_CLASS_NORMAL;
    status = iree_thread_create(
        (iree_thread_entry_t)iree_hal_external_timepoint_poller_main, poller,
        params, host_allocator, &poller->thread);
  }

  if (iree_status_is_ok(status)) {
    *out_poller = poller;
  } else {
    iree_hal_external_timepoint_poller_destroy(poller);
  }

  return status;
}

void iree_hal_external_timepoint_poller_destroy(
    iree_hal_external_timepoint_poller_t* poller) {
  if (!poller) return;

  // Signal thread to exit
  iree_atomic_store(&poller->should_exit, 1, iree_memory_order_release);
  iree_event_set(&poller->wake_event);

  // Wait for thread to exit
  if (poller->thread) {
    iree_thread_join(poller->thread);
    iree_thread_release(poller->thread);
  }

  // Clean up remaining imports
  iree_slim_mutex_lock(&poller->mutex);
  iree_hal_pending_external_import_t* current = poller->pending_imports;
  while (current) {
    iree_hal_pending_external_import_t* next = current->next;
    iree_hal_semaphore_release(current->semaphore);
    iree_allocator_free(poller->host_allocator, current);
    current = next;
  }
  iree_slim_mutex_unlock(&poller->mutex);

  // Clean up synchronization primitives
  iree_wait_set_free(poller->wait_set);
  iree_event_deinitialize(&poller->wake_event);
  iree_slim_mutex_deinitialize(&poller->mutex);

  iree_allocator_free(poller->host_allocator, poller);
}

iree_status_t iree_hal_external_timepoint_poller_add_import(
    iree_hal_external_timepoint_poller_t* poller,
    iree_hal_semaphore_t* semaphore, uint64_t target_value,
    iree_wait_handle_t wait_handle) {
  // Allocate new import entry
  iree_hal_pending_external_import_t* import = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(poller->host_allocator,
                                             sizeof(*import), (void**)&import));

  import->semaphore = semaphore;
  import->target_value = target_value;
  import->wait_handle = wait_handle;
  import->next = NULL;

  // Retain semaphore
  iree_hal_semaphore_retain(semaphore);

  // Add to list
  iree_slim_mutex_lock(&poller->mutex);
  import->next = poller->pending_imports;
  poller->pending_imports = import;
  iree_slim_mutex_unlock(&poller->mutex);

  // Wake the poller thread
  iree_event_set(&poller->wake_event);

  return iree_ok_status();
}

void iree_hal_external_timepoint_poller_cancel_imports(
    iree_hal_external_timepoint_poller_t* poller,
    iree_hal_semaphore_t* semaphore) {
  iree_slim_mutex_lock(&poller->mutex);

  iree_hal_pending_external_import_t* prev = NULL;
  iree_hal_pending_external_import_t* current = poller->pending_imports;

  while (current) {
    if (current->semaphore == semaphore) {
      // Remove from list
      if (prev) {
        prev->next = current->next;
      } else {
        poller->pending_imports = current->next;
      }

      iree_hal_pending_external_import_t* to_free = current;
      current = current->next;

      // Cleanup
      iree_hal_semaphore_release(to_free->semaphore);
      iree_allocator_free(poller->host_allocator, to_free);
    } else {
      prev = current;
      current = current->next;
    }
  }

  iree_slim_mutex_unlock(&poller->mutex);

  // Wake poller to rebuild wait set
  iree_event_set(&poller->wake_event);
}

//===----------------------------------------------------------------------===//
// Global external timepoint poller singleton
//===----------------------------------------------------------------------===//

static iree_hal_external_timepoint_poller_t*
    iree_hal_external_timepoint_poller_global_ = NULL;
static iree_once_flag iree_hal_external_timepoint_poller_global_flag_ =
    IREE_ONCE_FLAG_INIT;

// Called at process exit to clean up the global poller (useful for tools that
// spin up multiple IREE instances in the same process).
static void iree_hal_external_timepoint_poller_global_shutdown(void) {
  if (iree_hal_external_timepoint_poller_global_) {
    iree_hal_external_timepoint_poller_destroy(
        iree_hal_external_timepoint_poller_global_);
    iree_hal_external_timepoint_poller_global_ = NULL;
  }
}

static void iree_hal_external_timepoint_poller_global_initialize(void) {
  // Create the global poller using the system allocator
  // NOTE: This will never be destroyed - it lives for the process lifetime
  iree_status_t status = iree_hal_external_timepoint_poller_create(
      iree_allocator_system(), &iree_hal_external_timepoint_poller_global_);
  if (!iree_status_is_ok(status)) {
    // If we can't create the global poller, we'll just have to live without it
    // This shouldn't happen in practice unless we're out of memory or threads
    iree_hal_external_timepoint_poller_global_ = NULL;
  } else {
    // Register an at-exit handler so that sanitizers / memory leak detectors
    // do not report this as a leak when the process terminates.
    (void)atexit(iree_hal_external_timepoint_poller_global_shutdown);
  }
}

iree_hal_external_timepoint_poller_t* iree_hal_external_timepoint_poller_global(
    void) {
  iree_call_once(&iree_hal_external_timepoint_poller_global_flag_,
                 iree_hal_external_timepoint_poller_global_initialize);
  return iree_hal_external_timepoint_poller_global_;
}
