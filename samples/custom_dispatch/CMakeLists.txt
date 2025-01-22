# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_add_all_subdirs()
 
#TODO: CUDA breaks with 'python runner.py configure'
add_subdirectory(cpu)
add_subdirectory(hip)
add_subdirectory(vulkan)