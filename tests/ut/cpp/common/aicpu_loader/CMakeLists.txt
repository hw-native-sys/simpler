# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Test dispatcher path selection and publication with a CPU-only CANN context stub.
macro(aicpu_loader_case src)
    simpler_ut_plain_case(${src}
        SOURCES ${SIMPLER_SRC}/common/aicpu_loader/device/aicpu_dispatcher.cpp
                ${SIMPLER_SRC}/common/platform/sim/aicpu/cache_ops.cpp
        INCLUDES ${SIMPLER_SRC}/common ${SIMPLER_SRC}/common/platform/include
                 ${CMAKE_CURRENT_SOURCE_DIR}/support LABEL no_hardware)
endmacro()
# One actual TMR registration/cleanup fixture per architecture. The loader
# transport is shared; no separate backend or dual-owner test family.
foreach(arch a2a3 a5)
    set(module tmr_registration_${arch})
    add_library(${module} SHARED support/tmr_registration_module.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/onboard/aicpu/orch_so_file.cpp)
    simpler_ut_includes(${arch} tensormap_and_ringbuffer aicpu includes)
    target_include_directories(${module} PRIVATE ${includes}
        ${SIMPLER_SRC}/${arch}/runtime/tensormap_and_ringbuffer/aicpu
        ${SIMPLER_SRC}/${arch}/runtime)
    target_compile_options(${module} PRIVATE -fvisibility=hidden -ffunction-sections -fdata-sections)
    target_link_libraries(${module} PRIVATE ${CMAKE_DL_LIBS})
    if(APPLE)
        target_link_options(${module} PRIVATE -Wl,-dead_strip
            -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/support/tmr_registration_module.exports)
    else()
        target_link_options(${module} PRIVATE -Wl,--gc-sections
            -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/support/tmr_registration_module.map)
    endif()
endforeach()
foreach(value 4 17)
    add_library(tmr_orchestration_${value} SHARED support/tmr_orchestration.cpp)
    target_compile_definitions(tmr_orchestration_${value} PRIVATE TEST_Y_CONFIG=${value})
endforeach()
foreach(arch a2a3 a5)
    simpler_ut_add_target(NAME test_${arch}_loader_lifecycle
        SOURCES test_loader_lifecycle.cpp
            ${SIMPLER_SRC}/common/aicpu_loader/host/load_aicpu_op.cpp
            ${SIMPLER_SRC}/common/aicpu_loader/device/aicpu_dispatcher.cpp
            ${SIMPLER_SRC}/common/platform/sim/aicpu/cache_ops.cpp
        ARCH ${arch} RUNTIME tensormap_and_ringbuffer
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/support ${SIMPLER_SRC}/common/worker
        INCLUDES_BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/support/loader
        DEFINES TEST_RUNTIME_IMAGE="$<TARGET_FILE:tmr_registration_${arch}>"
            TEST_ORCHESTRATION="$<TARGET_FILE:tmr_orchestration_4>"
            TEST_SECOND_ORCHESTRATION="$<TARGET_FILE:tmr_orchestration_17>"
        LINK ${CMAKE_DL_LIBS}
        DEPENDS tmr_registration_${arch} tmr_orchestration_4 tmr_orchestration_17
        LABEL no_hardware PRIVATE_WORKDIR)
endforeach()
simpler_ut_glob_cases(aicpu_loader_case)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(test_so_upload PRIVATE "LINKER:--wrap=write" "LINKER:--wrap=statvfs"
        "LINKER:--wrap=close" "LINKER:--wrap=link")
endif()
