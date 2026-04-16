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

set(DIST_SRC ${CMAKE_SOURCE_DIR}/src/common/distributed)

set(DIST_SOURCES
    ${DIST_SRC}/dist_types.cpp
    ${DIST_SRC}/dist_tensormap.cpp
    ${DIST_SRC}/dist_ring.cpp
    ${DIST_SRC}/dist_scope.cpp
    ${DIST_SRC}/dist_orchestrator.cpp
    ${DIST_SRC}/dist_worker_manager.cpp
    ${DIST_SRC}/dist_scheduler.cpp
    ${DIST_SRC}/dist_worker.cpp
)

nanobind_add_module(_task_interface ${BINDING_SOURCES} ${DIST_SOURCES})

target_sources(_task_interface PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/worker/chip_worker.cpp
)

target_include_directories(_task_interface PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/task_interface
    ${CMAKE_SOURCE_DIR}/src/common/worker
    ${CMAKE_SOURCE_DIR}/src/common/distributed
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_task_interface PRIVATE ${CMAKE_DL_LIBS} pthread)

if(SKBUILD_MODE)
    install(TARGETS _task_interface DESTINATION .)
else()
    set_target_properties(_task_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    )
endif()
