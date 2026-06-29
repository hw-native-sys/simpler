# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# nanobind Python bindings for task_interface and distributed runtime

set(BINDING_SOURCES
    task_interface.cpp
)

list(TRANSFORM BINDING_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

set(HIERARCHICAL_SRC ${CMAKE_SOURCE_DIR}/src/common/hierarchical)

set(HIERARCHICAL_SOURCES
    ${HIERARCHICAL_SRC}/types.cpp
    ${HIERARCHICAL_SRC}/tensormap.cpp
    ${HIERARCHICAL_SRC}/ring.cpp
    ${HIERARCHICAL_SRC}/scope.cpp
    ${HIERARCHICAL_SRC}/remote_wire.cpp
    ${HIERARCHICAL_SRC}/remote_endpoint.cpp
    ${HIERARCHICAL_SRC}/orchestrator.cpp
    ${HIERARCHICAL_SRC}/worker_manager.cpp
    ${HIERARCHICAL_SRC}/scheduler.cpp
    ${HIERARCHICAL_SRC}/worker.cpp
)

nanobind_add_module(_task_interface ${BINDING_SOURCES} ${HIERARCHICAL_SOURCES})

target_sources(_task_interface PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/worker/chip_worker.cpp
    # Host-side assert_impl / get_stacktrace for the unified Tensor's
    # always_assert (Tensor::make -> init_external). The binding does not link
    # the runtime orchestration/common.cpp that provides these for runtime .so.
    ${CMAKE_SOURCE_DIR}/src/common/task_interface/assert_compat.cpp
)

target_include_directories(_task_interface PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/task_interface
    ${CMAKE_SOURCE_DIR}/src/common/worker
    ${CMAKE_SOURCE_DIR}/src/common/hierarchical
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_task_interface PRIVATE ${CMAKE_DL_LIBS} pthread)

# POSIX shm_open / shm_unlink (used by worker_manager.cpp broadcast_register) live
# in librt on glibc < 2.34 and in libSystem on macOS. glibc 2.34+ folded librt
# into libc, so -lrt is a no-op there but harmless (and required for older glibc).
if(UNIX AND NOT APPLE)
    target_link_libraries(_task_interface PRIVATE rt)
endif()

if(SKBUILD_MODE)
    install(TARGETS _task_interface DESTINATION .)
else()
    set_target_properties(_task_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    )
endif()
