# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# wsesim Platform - Thread-Based Simulation
#
# This platform simulates the Host/direct PE execution model using threads.
# It provides the same API as the simpler platform API for testing and development.
#
# Build: cmake -B build && cmake --build build
# Or use from examples: cmake -DCUSTOM_INCLUDE_DIRS=... -DCUSTOM_SOURCE_DIRS=...

cmake_minimum_required(VERSION 3.16.3)

project(wsesim_platform LANGUAGES CXX)

# Add subdirectories
add_subdirectory(host)
add_subdirectory(aicore)
