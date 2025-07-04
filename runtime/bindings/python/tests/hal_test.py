# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List
import iree.runtime
import iree.compiler

import gc
import numpy as np
import threading
import time
import unittest


class NonDeviceHalTest(unittest.TestCase):
    def testMemoryEnums(self):
        print("MemoryType:", iree.runtime.MemoryType)
        print("HOST_VISIBLE:", int(iree.runtime.MemoryType.HOST_VISIBLE))

        # Enum and/or operations on BufferCompatibility.
        self.assertEqual(
            iree.runtime.BufferCompatibility.IMPORTABLE
            | iree.runtime.BufferCompatibility.EXPORTABLE,
            int(iree.runtime.BufferCompatibility.IMPORTABLE)
            | int(iree.runtime.BufferCompatibility.EXPORTABLE),
        )
        self.assertEqual(
            iree.runtime.BufferCompatibility.EXPORTABLE
            & iree.runtime.BufferCompatibility.EXPORTABLE,
            int(iree.runtime.BufferCompatibility.EXPORTABLE),
        )

        # Enum and/or operations on BufferUsage.
        self.assertEqual(
            iree.runtime.BufferUsage.TRANSFER | iree.runtime.BufferUsage.MAPPING,
            int(iree.runtime.BufferUsage.TRANSFER)
            | int(iree.runtime.BufferUsage.MAPPING),
        )
        self.assertEqual(
            iree.runtime.BufferUsage.TRANSFER & iree.runtime.BufferUsage.TRANSFER,
            int(iree.runtime.BufferUsage.TRANSFER),
        )

        # Enum and/or operations on MemoryAccess.
        self.assertEqual(
            iree.runtime.MemoryAccess.READ | iree.runtime.MemoryAccess.WRITE,
            int(iree.runtime.MemoryAccess.READ) | int(iree.runtime.MemoryAccess.WRITE),
        )
        self.assertEqual(
            iree.runtime.MemoryAccess.ALL & iree.runtime.MemoryAccess.READ,
            int(iree.runtime.MemoryAccess.READ),
        )

        # Enum and/or operations on MemoryType.
        self.assertEqual(
            iree.runtime.MemoryType.DEVICE_LOCAL | iree.runtime.MemoryType.HOST_VISIBLE,
            int(iree.runtime.MemoryType.DEVICE_LOCAL)
            | int(iree.runtime.MemoryType.HOST_VISIBLE),
        )
        self.assertEqual(
            iree.runtime.MemoryType.OPTIMAL & iree.runtime.MemoryType.OPTIMAL,
            int(iree.runtime.MemoryType.OPTIMAL),
        )

    def testElementTypeEnums(self):
        i8 = iree.runtime.HalElementType.INT_8
        i4 = iree.runtime.HalElementType.INT_4
        self.assertTrue(iree.runtime.HalElementType.is_byte_aligned(i8))
        self.assertFalse(iree.runtime.HalElementType.is_byte_aligned(i4))
        self.assertEqual(1, iree.runtime.HalElementType.dense_byte_count(i8))


class DeviceHalTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.device = iree.runtime.get_device("local-task")
        self.allocator = self.device.allocator
        gc.collect()

    def testTrim(self):
        self.allocator.trim()
        # Just running is sufficient.

    def testProfilingDefaults(self):
        self.device.begin_profiling()
        self.device.flush_profiling()
        self.device.end_profiling()
        # Just running is sufficient.

    def testProfilingOptions(self):
        self.device.begin_profiling(mode="queue", file_path="foo.rdc")
        self.device.end_profiling()
        # Just running is sufficient.

    def testProfilingInvalidOptions(self):
        with self.assertRaisesRegex(ValueError, "unrecognized profiling mode"):
            self.device.begin_profiling(mode="SOMETHING THAT DOESN'T EXIST")

    def testStatistics(self):
        stats_dict = self.allocator.statistics
        stats_str = self.allocator.formatted_statistics
        if self.allocator.has_statistics:
            self.assertIn("host_bytes_peak", stats_dict)
            self.assertIn("host_bytes_allocated", stats_dict)
            self.assertIn("host_bytes_freed", stats_dict)
            self.assertIn("device_bytes_peak", stats_dict)
            self.assertIn("device_bytes_allocated", stats_dict)
            self.assertIn("device_bytes_freed", stats_dict)
            self.assertIn("HOST_LOCAL", stats_str)

    def testQueryCompatibility(self):
        compat = self.allocator.query_buffer_compatibility(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            intended_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=1024,
        )
        print("COMPAT:", compat)
        self.assertTrue(
            bool(compat & int(iree.runtime.BufferCompatibility.ALLOCATABLE)),
            "should be allocatable",
        )
        self.assertTrue(
            bool(compat & int(iree.runtime.BufferCompatibility.IMPORTABLE)),
            "should be importable",
        )
        self.assertTrue(
            bool(compat & int(iree.runtime.BufferCompatibility.EXPORTABLE)),
            "should be exportable",
        )

    def testAllocateBuffer(self):
        buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        print("BUFFER:", buffer)

    def testBufferViewConstructor(self):
        buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        bv = iree.runtime.HalBufferView(
            buffer, (1, 2), iree.runtime.HalElementType.INT_16
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(bv),
            "<HalBufferView (1, 2), element_type=0x10000010, 13 bytes (at offset 0 into 13), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )
        self.assertEqual(4, bv.byte_length)

    def testBufferMap(self):
        buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        m = buffer.map()
        self.assertIsInstance(m, iree.runtime.MappedMemory)

    def testAllocateBufferCopy(self):
        ary = np.zeros([3, 4], dtype=np.int32) + 2
        buffer = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=ary,
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(buffer),
            "<HalBuffer 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )

    def testAllocateBufferCopyCreateView(self):
        ary = np.zeros([3, 4], dtype=np.int32) + 2
        buffer = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=ary,
            element_type=iree.runtime.HalElementType.SINT_32,
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(buffer),
            "<HalBufferView (3, 4), element_type=0x20000011, 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )

    def testAllocateBufferViewCopy(self):
        # Crerate an ndarray, copy it, and check if the copy is equal
        ary = np.random.randint(low=0, high=666, size=[3, 4], dtype=np.int32)
        buffer = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=ary,
        )

        assert isinstance(buffer, iree.runtime.HalBuffer)
        bv = iree.runtime.HalBufferView(
            buffer, (3, 4), iree.runtime.HalElementType.INT_32
        )

        bv_copy1 = self.allocator.allocate_buffer_view_copy(
            iree.runtime.MemoryType.DEVICE_LOCAL,
            iree.runtime.BufferUsage.DEFAULT,
            self.device,
            self.device,
            bv,
        )

        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(bv_copy1),
            "<HalBufferView (3, 4), element_type=0x20000010, 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=READ|WRITE, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )
        self.assertEqual(48, bv.byte_length)
        ary_copy = iree.runtime.DeviceArray(self.device, bv_copy1).to_host()
        np.testing.assert_array_equal(ary, ary_copy)

        # Create a slice of the copied buffer and copy it, check for equality
        bv_slice = iree.runtime.HalBufferView(
            bv_copy1.get_buffer(), (1, 4), iree.runtime.HalElementType.INT_32
        )

        bv_slice_copy = self.allocator.allocate_buffer_view_copy(
            iree.runtime.MemoryType.DEVICE_LOCAL,
            iree.runtime.BufferUsage.DEFAULT,
            self.device,
            self.device,
            bv_slice,
        )

        self.assertEqual(
            repr(bv_slice_copy),
            "<HalBufferView (1, 4), element_type=0x20000010, 16 bytes (at offset 0 into 16), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=READ|WRITE, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )
        ary_copy = iree.runtime.DeviceArray(self.device, bv_slice_copy).to_host()
        np.testing.assert_array_equal(ary[0, 0:4], ary_copy[0])

    def testAllocateHostStagingBufferCopy(self):
        buffer = self.allocator.allocate_host_staging_buffer_copy(
            self.device, np.int32(0)
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(buffer),
            "<HalBuffer 4 bytes (at offset 0 into 4), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|MAPPING|MAPPING_PERSISTENT>",
        )

    def testSemaphore(self):
        sem0 = self.device.create_semaphore(0)
        self.assertEqual(sem0.query(), 0)
        sem1 = self.device.create_semaphore(1)
        self.assertEqual(sem1.query(), 1)
        sem1.signal(2)
        self.assertEqual(sem1.query(), 2)

    def testSemaphoreSignal(self):
        sem = self.device.create_semaphore(0)
        self.assertFalse(sem.wait(1, deadline=0))
        sem.signal(1)
        self.assertTrue(sem.wait(1, deadline=0))

    def testSynchronousSemaphoreFailed(self):
        sem = self.device.create_semaphore(0)
        sem.fail("TEST FAILURE")
        with self.assertRaisesRegex(
            RuntimeError, "^synchronous semaphore failure.*TEST FAILURE"
        ):
            sem.wait(1, deadline=0)

    def testAsynchronousSemaphoreFailed(self):
        sem = self.device.create_semaphore(0)
        exceptions = []

        def run():
            print("SIGNALLING ASYNC FAILURE")
            time.sleep(0.2)
            sem.fail("TEST FAILURE")
            print("SIGNALLED")

        def wait():
            print("WAITING")
            try:
                sem.wait(1)
            except RuntimeError as e:
                exceptions.append(e)

        runner = threading.Thread(target=run)
        waiter = threading.Thread(target=wait)
        waiter.start()
        runner.start()
        waiter.join()
        runner.join()
        self.assertTrue(exceptions)
        print(exceptions)
        # Note: It is impossible to 100% guarantee that this sequences such as to
        # report an asynchronous vs synchronous failure, although we tip the odds in
        # this favor with the sleep in the signalling thread. Therefore, we do not
        # check the "asynchronous" vs "synchronous" message prefix to avoid flaky
        # test races.
        self.assertIn("TEST FAILURE", str(exceptions[0]))

    def testTrivialQueueAlloc(self):
        sem = self.device.create_semaphore(0)
        buf = self.device.queue_alloca(
            1024, wait_semaphores=[(sem, 0)], signal_semaphores=[(sem, 1)]
        )
        self.assertIsInstance(buf, iree.runtime.HalBuffer)
        self.device.queue_dealloca(
            buf, wait_semaphores=[(sem, 1)], signal_semaphores=[]
        )

    def testAllocAcceptsFences(self):
        # Also tests HalFence, HalFence.insert, HalFence.wait (infinite)
        sem = self.device.create_semaphore(0)
        fence0 = iree.runtime.HalFence(1)
        fence0.insert(sem, 0)
        fence1 = iree.runtime.HalFence(1)
        fence1.insert(sem, 1)
        fence2 = iree.runtime.HalFence(2)
        fence2.insert(sem, 2)
        buf = self.device.queue_alloca(
            1024, wait_semaphores=fence0, signal_semaphores=fence1
        )
        self.assertIsInstance(buf, iree.runtime.HalBuffer)
        self.device.queue_dealloca(
            buf, wait_semaphores=fence1, signal_semaphores=fence2
        )
        self.assertTrue(fence2.wait())
        self.assertEqual(sem.query(), 2)

    def testFenceCreateAt(self):
        sem = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence.create_at(sem, 1)
        self.assertFalse(fence.wait(deadline=0))
        sem.signal(1)
        self.assertTrue(fence.wait(deadline=0))

    def testFenceSignal(self):
        sem = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence.create_at(sem, 1)
        self.assertFalse(fence.wait(deadline=0))
        fence.signal()
        self.assertTrue(fence.wait(deadline=0))

    def testSynchronousFenceFailed(self):
        sem = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence.create_at(sem, 1)
        fence.fail("TEST FAILURE")
        with self.assertRaisesRegex(
            RuntimeError, "^synchronous fence failure.*TEST FAILURE"
        ):
            fence.wait(deadline=0)

    def testAsynchronousFenceFailed(self):
        sem = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence.create_at(sem, 1)
        exceptions = []

        def run():
            print("SIGNALLING ASYNC FAILURE")
            time.sleep(0.2)
            fence.fail("TEST FAILURE")
            print("SIGNALLED")

        def wait():
            print("WAITING")
            try:
                fence.wait()
            except RuntimeError as e:
                exceptions.append(e)

        runner = threading.Thread(target=run)
        waiter = threading.Thread(target=wait)
        waiter.start()
        runner.start()
        waiter.join()
        runner.join()
        self.assertTrue(exceptions)
        print(exceptions)
        # Note: It is impossible to 100% guarantee that this sequences such as to
        # report an asynchronous vs synchronous failure, although we tip the odds in
        # this favor with the sleep in the signalling thread. Therefore, we do not
        # check the "asynchronous" vs "synchronous" message prefix to avoid flaky
        # test races.
        self.assertIn("TEST FAILURE", str(exceptions[0]))

    def testFenceJoin(self):
        sem1 = self.device.create_semaphore(0)
        sem2 = self.device.create_semaphore(0)
        fence1 = iree.runtime.HalFence.create_at(sem1, 1)
        fence2 = iree.runtime.HalFence.create_at(sem2, 1)
        fence = iree.runtime.HalFence.join([fence1, fence2])
        self.assertEqual(fence.timepoint_count, 2)

    def testFenceInsert(self):
        sem1 = self.device.create_semaphore(0)
        sem2 = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence(2)
        fence.insert(sem1, 1)
        self.assertEqual(fence.timepoint_count, 1)
        fence.insert(sem1, 2)
        self.assertEqual(fence.timepoint_count, 1)
        fence.insert(sem2, 2)
        self.assertEqual(fence.timepoint_count, 2)

    def testFenceExtend(self):
        sem1 = self.device.create_semaphore(0)
        sem2 = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence(2)
        fence.insert(sem1, 1)
        self.assertEqual(fence.timepoint_count, 1)
        fence.extend(iree.runtime.HalFence.create_at(sem2, 2))
        self.assertEqual(fence.timepoint_count, 2)

    def testRoundTripQueueCopy(self):
        original_ary = np.zeros([3, 4], dtype=np.int32) + 2
        source_bv = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=original_ary,
            element_type=iree.runtime.HalElementType.SINT_32,
        )
        source_buffer = source_bv.get_buffer()
        target_buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=source_buffer.byte_length(),
        )
        sem = self.device.create_semaphore(0)
        self.device.queue_copy(
            source_buffer,
            target_buffer,
            wait_semaphores=iree.runtime.HalFence.create_at(sem, 0),
            signal_semaphores=iree.runtime.HalFence.create_at(sem, 1),
        )
        iree.runtime.HalFence.create_at(sem, 1).wait()
        copy_ary = target_buffer.map().asarray(original_ary.shape, original_ary.dtype)
        np.testing.assert_array_equal(original_ary, copy_ary)

    def testIncompatibleSizeQueueCopy(self):
        source_buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        target_buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        sem = self.device.create_semaphore(0)
        with self.assertRaisesRegex(ValueError, "length must be less than"):
            self.device.queue_copy(
                source_buffer,
                target_buffer,
                wait_semaphores=iree.runtime.HalFence.create_at(sem, 0),
                signal_semaphores=iree.runtime.HalFence.create_at(sem, 1),
            )

    def testCommandBufferStartsByDefault(self):
        cb = iree.runtime.HalCommandBuffer(self.device)
        with self.assertRaisesRegex(RuntimeError, "FAILED_PRECONDITION"):
            cb.begin()
        cb = iree.runtime.HalCommandBuffer(self.device, begin=False)
        cb.begin()

    def testCommandBufferCopy(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        buffer2 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        cb.copy(buffer1, buffer2, end=True)
        with self.assertRaisesRegex(RuntimeError, "FAILED_PRECONDITION"):
            cb.end()

    def testCommandBufferFill(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        cb.fill(buffer1, np.int32(1), 0, 12, end=True)
        with self.assertRaisesRegex(RuntimeError, "FAILED_PRECONDITION"):
            cb.end()

    def testCommandBufferExecute(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        cb.fill(buffer1, np.int32(1), 0, 12, end=True)

        sem = self.device.create_semaphore(0)
        self.device.queue_execute(
            cb, wait_semaphores=[(sem, 0)], signal_semaphores=[(sem, 1)]
        )
        iree.runtime.HalFence.create_at(sem, 1).wait()

    def testCommandBufferExecuteAcceptsFence(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        cb.fill(buffer1, np.int32(1), 0, 12, end=True)

        sem = self.device.create_semaphore(0)
        self.device.queue_execute(
            cb,
            wait_semaphores=iree.runtime.HalFence.create_at(sem, 0),
            signal_semaphores=iree.runtime.HalFence.create_at(sem, 1),
        )
        iree.runtime.HalFence.create_at(sem, 1).wait()


class DeviceDLPackTest(unittest.TestCase):
    """Tests low level DLPack import/export against the CPU HAL backend.

    This test leverages the fact that numpy is a reasonable dlpack
    producer/consumer. It has the caveat that our low level support does not
    allow import of non page aligned data, so we have to take some extra
    steps to prep it. For pure CPU/Numpy import/export, we have better
    supported paths than this, but we leverage it here for its testing
    value, as it exercises code paths that are otherwise only accessible
    on devices.
    """

    def setUp(self):
        super().setUp()
        self.device = iree.runtime.get_device("local-task")
        self.allocator = self.device.allocator
        gc.collect()

    def roundtrip(self, input_array, element_type):
        # We have to copy the input array into our own buffer to ensure
        # alignment (dlpack import/export require aligned data).
        orig_bv = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=input_array,
            element_type=element_type,
        )
        aligned_input_array = orig_bv.map().asarray(
            input_array.shape, input_array.dtype
        )

        # Export the __dlpack__ capsule from numpy, which should be a plain
        # view over the buffer we originally allocated (therefore, aligned
        # and importable).
        input_capsule = aligned_input_array.__dlpack__()
        aligned_input_array = None
        gc.collect()
        imported_bv = self.device.from_dlpack_capsule(input_capsule)

        # Export a capsule from this imported buffer view and create a new
        # array out of it.
        class DummyProducer:
            def __dlpack__(_, stream=None):
                capsule = self.device.create_dlpack_capsule(imported_bv, 1, 0)
                return capsule

            def __dlpack_device__(self):
                return (1, 0)  # CPU, id 0

        reimported_array = np.from_dlpack(DummyProducer())
        imported_bv = None
        gc.collect()
        np.testing.assert_array_equal(input_array, reimported_array)

    def testImportExportF64(self):
        self.roundtrip(np.random.rand(3, 4), iree.runtime.HalElementType.FLOAT_64)

    def testImportExportF32(self):
        self.roundtrip(
            np.random.rand(3, 4, 16, 32, 1, 5, 2).astype(np.float32),
            iree.runtime.HalElementType.FLOAT_32,
        )

    def testImportExportF16(self):
        self.roundtrip(
            np.random.rand(3, 4).astype(np.float16),
            iree.runtime.HalElementType.FLOAT_16,
        )

    def testImportExportSI8(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.int8),
            iree.runtime.HalElementType.SINT_8,
        )

    def testImportExportSI16(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.int16),
            iree.runtime.HalElementType.SINT_16,
        )

    def testImportExportSI32(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.int32),
            iree.runtime.HalElementType.SINT_32,
        )

    def testImportExportSI64(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.int64),
            iree.runtime.HalElementType.SINT_64,
        )

    def testImportExportUI8(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.uint8),
            iree.runtime.HalElementType.UINT_8,
        )

    def testImportExportUI16(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.uint16),
            iree.runtime.HalElementType.UINT_16,
        )

    def testImportExportUI32(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.uint32),
            iree.runtime.HalElementType.UINT_32,
        )

    def testImportExportUI64(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.uint64),
            iree.runtime.HalElementType.UINT_64,
        )

    def testImportExportBool(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.bool_),
            iree.runtime.HalElementType.BOOL_8,
        )

    def testImportExportUI64(self):
        self.roundtrip(
            (np.random.rand(3, 4) * 255.0).astype(np.uint64),
            iree.runtime.HalElementType.UINT_64,
        )

    def testImportExportComplex64(self):
        shape = (3, 1, 5, 6, 12, 2, 3)
        self.roundtrip(
            np.random.uniform(-1, 1, shape) + 1.0j * np.random.uniform(-1, 1, shape),
            iree.runtime.HalElementType.COMPLEX_64,
        )

    def testImportExportComplex64(self):
        shape = (3, 1, 5, 6, 12, 2, 3)
        self.roundtrip(
            (
                np.random.uniform(-1, 1, shape) + 1.0j * np.random.uniform(-1, 1, shape)
            ).astype(np.complex128),
            iree.runtime.HalElementType.COMPLEX_64,
        )


class HalModuleDebugSinkTest(unittest.TestCase):
    COMPILED_TRACE_TENSOR: bytes

    @classmethod
    def compile_trace_tensor(cls):
        if not hasattr(cls, "COMPILED_TRACE_TENSOR"):
            cls.COMPILED_TRACE_TENSOR = iree.compiler.compile_str(
                """
                func.func @trace_args(%arg0: tensor<2xi32>, %arg1: tensor<3xi32>) {
                    flow.tensor.trace "debug_sink_test" = [
                        %arg0: tensor<2xi32>,
                        %arg1: tensor<3xi32>
                    ]
                    return
                }
                """,
                target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
            )
        return cls.COMPILED_TRACE_TENSOR

    def testHalModuleBufferViewTraceCallback(self):
        """Check that the trace tensor callback gets called with the expected
        arguments."""
        program_bytes = HalModuleDebugSinkTest.compile_trace_tensor()

        arg0 = np.array([1, 2], dtype=np.int32)
        arg1 = np.array([3, 4, 5], dtype=np.int32)

        callback_key: str = None
        callback_buffer_views = None

        def callback(key: str, buffer_views: List[iree.runtime.HalBufferView]):
            nonlocal callback_key
            callback_key = key
            nonlocal callback_buffer_views
            callback_buffer_views = buffer_views

        instance = iree.runtime.VmInstance()
        device = iree.runtime.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)
        hal_module = iree.runtime.create_hal_module(
            instance, device, debug_sink=iree.runtime.HalModuleDebugSink(callback)
        )
        program_module = iree.runtime.VmModule.copy_buffer(instance, program_bytes)
        context = iree.runtime.VmContext(instance)
        context.register_modules([hal_module, program_module])
        fn = program_module.lookup_function("trace_args")
        fn_invoker = iree.runtime.FunctionInvoker(context, device, fn)
        fn_invoker(arg0, arg1)

        assert callback_key == "debug_sink_test"
        assert len(callback_buffer_views) == 2
        actual_arg0 = iree.runtime.DeviceArray(
            device, callback_buffer_views[0]
        ).to_host()
        actual_arg1 = iree.runtime.DeviceArray(
            device, callback_buffer_views[1]
        ).to_host()
        np.testing.assert_equal(actual_arg0, arg0)
        np.testing.assert_equal(actual_arg1, arg1)

    def testNoneHalModuleDebugSink(self):
        device = iree.runtime.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)
        instance = iree.runtime.VmInstance()
        hal_module = iree.runtime.create_hal_module(
            instance,
            device,
            debug_sink=None,
        )

    def testExceptionInHalModuleBufferViewTraceCallback(self):
        """When an exception occurs in the callback check that it properly propagates
        through the bindings and results in a IREE module function failed invocation.
        """
        program_bytes = HalModuleDebugSinkTest.compile_trace_tensor()

        arg0 = np.array([1, 2], dtype=np.int32)
        arg1 = np.array([3, 4, 5], dtype=np.int32)

        device = iree.runtime.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)

        class TestException(Exception):
            def __init__(self, msg: str):
                super().__init__(msg)

        def callback(key: str, buffer_views: List[iree.runtime.HalBufferView]):
            raise TestException("This is a test exception")

        instance = iree.runtime.VmInstance()
        hal_module = iree.runtime.create_hal_module(
            instance, device, debug_sink=iree.runtime.HalModuleDebugSink(callback)
        )
        program_module = iree.runtime.VmModule.copy_buffer(instance, program_bytes)
        context = iree.runtime.VmContext(instance)
        context.register_modules([hal_module, program_module])
        fn = program_module.lookup_function("trace_args")
        fn_invoker = iree.runtime.FunctionInvoker(context, device, fn)
        # TODO: once IREE status chains messages test for the actual message we raise
        # within the callback.
        self.assertRaisesRegex(RuntimeError, "UNKNOWN", fn_invoker, arg0, arg1)

    def testHalModuleBufferViewTraceCallbackReferencingItselfDoesNotLeak(self):
        """Check that if we do not hold reference to the HAL module or VM context,
        but we hold a reference to the debug sink in the callback, the callback object
        does not leak.
        """
        is_callback_destroyed: bool = False

        class Callback:
            def __del__(self):
                nonlocal is_callback_destroyed
                is_callback_destroyed = True

            def __call__(
                self, key: str, buffer_views: List[iree.runtime.HalBufferView]
            ):
                pass

        callback = Callback()
        debug_sink = iree.runtime.HalModuleDebugSink(callback)
        setattr(callback, "debug_sink", debug_sink)

        device = iree.runtime.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)

        vm_instance = iree.runtime.VmInstance()
        hal_module = iree.runtime.create_hal_module(
            vm_instance,
            device,
            debug_sink=debug_sink,
        )
        vm_context = iree.runtime.VmContext(vm_instance)
        vm_context.register_modules([hal_module])
        assert not is_callback_destroyed

        del callback
        del debug_sink
        del hal_module
        del vm_instance
        del vm_context
        gc.collect()
        assert is_callback_destroyed


if __name__ == "__main__":
    unittest.main()
