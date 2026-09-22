# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for the trees under src/common/ that both architectures share. Each
# subdirectory mirrors one of them, and support/ holds the fixtures and
# dlopen'd stand-ins its siblings link.

# The two runtime directories come first: each defines the object libraries and
# helpers for its runtime, and platform/ builds a pair of its cases against both
# runtimes.
add_subdirectory(host_build_graph)
add_subdirectory(tensormap_and_ringbuffer)

add_subdirectory(platform)
add_subdirectory(log)
add_subdirectory(utils)
add_subdirectory(worker)
add_subdirectory(runtime_status)
add_subdirectory(task_interface)
add_subdirectory(hierarchical)

# platform_comm's only case needs a device to run and CANN to build.
if(SIMPLER_ENABLE_HARDWARE_TESTS)
    add_subdirectory(platform_comm)
else()
    simpler_ut_skip_dir(platform_comm)
endif()
