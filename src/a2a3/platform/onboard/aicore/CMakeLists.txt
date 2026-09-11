# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build AICore kernel using kernel.cpp + common folder + custom includes and sources (for `Task`)
# BISHENG_CC and BISHENG_LD are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake
cmake_minimum_required(VERSION 3.16.3)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/log/include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common")
# The kernel is an add_custom_target, so the profiling header reaches it as an
# include directory in this list rather than through target_include_directories.
include("${SIMPLER_CMAKE_DIR}/profiling_config.cmake")
simpler_generate_profiling_header(OUT_DIR SIMPLER_PROFILING_INCLUDE_DIR)
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${SIMPLER_PROFILING_INCLUDE_DIR}")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build with `Task` implementation")
endif()

# Build complete source list: kernel.cpp + CUSTOM_SOURCE_DIRS
set(ALL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp")
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB_RECURSE DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND ALL_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_SOURCE_DIRS to build with `Task` implementation")
endif()
list(LENGTH ALL_SOURCES NUM_SOURCES)
message(STATUS "AICore kernel: ${NUM_SOURCES} source files")
message(VERBOSE "AICore kernel sources: ${ALL_SOURCES}")

# Output files
set(AICORE_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/aicore_kernel.o)

# Convert include dirs to -I flags for compiler command
set(CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "")
foreach(INC_DIR ${CMAKE_CUSTOM_INCLUDE_DIRS})
    list(APPEND CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "-I${INC_DIR}")
endforeach()

# Compiler flags
set(AICORE_FLAGS
    "-c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}"
)
separate_arguments(AICORE_FLAGS)

# Step 1: Compile each source file to individual .o files for AIC and AIV
set(ALL_AIC_OBJECTS "")
set(ALL_AIV_OBJECTS "")

foreach(SRC_FILE ${ALL_SOURCES})
    get_filename_component(SRC_NAME ${SRC_FILE} NAME_WE)
    set(SOURCE_WIRE_HEADERS "")
    if(SRC_FILE MATCHES "/tensormap_and_ringbuffer/aicore/aicore_executor\\.cpp$")
        set(SOURCE_WIRE_HEADERS
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface/tmr_kernel_context.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface/tmr_kernel_control.h"
        )
    endif()
    set(OBJ_AIC "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aic.o")
    set(OBJ_AIV "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aiv.o")

    list(APPEND ALL_AIC_OBJECTS ${OBJ_AIC})
    list(APPEND ALL_AIV_OBJECTS ${OBJ_AIV})

    # Compile for AIC architecture
    add_custom_command(
        OUTPUT ${OBJ_AIC}
        COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS} --cce-aicore-arch=dav-c220-cube
                -o ${OBJ_AIC} ${SRC_FILE}
        DEPENDS ${SRC_FILE} ${SOURCE_WIRE_HEADERS}
        COMMENT "Compiling ${SRC_NAME} for AIC"
    )

    # Compile for AIV architecture
    add_custom_command(
        OUTPUT ${OBJ_AIV}
        COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS} --cce-aicore-arch=dav-c220-vec
                -o ${OBJ_AIV} ${SRC_FILE}
        DEPENDS ${SRC_FILE} ${SOURCE_WIRE_HEADERS}
        COMMENT "Compiling ${SRC_NAME} for AIV"
    )
endforeach()

# Step 2: Combine all .o files using ld -r (relocatable link)
set(COMBINED_AIC_OBJ "${CMAKE_CURRENT_BINARY_DIR}/aicore_aic_combined.o")
set(COMBINED_AIV_OBJ "${CMAKE_CURRENT_BINARY_DIR}/aicore_aiv_combined.o")

add_custom_command(
    OUTPUT ${COMBINED_AIC_OBJ}
    COMMAND ${BISHENG_LD} -r -o ${COMBINED_AIC_OBJ} ${ALL_AIC_OBJECTS}
    DEPENDS ${ALL_AIC_OBJECTS}
    COMMENT "Combining ${NUM_SOURCES} AIC object files into single .o"
)

add_custom_command(
    OUTPUT ${COMBINED_AIV_OBJ}
    COMMAND ${BISHENG_LD} -r -o ${COMBINED_AIV_OBJ} ${ALL_AIV_OBJECTS}
    DEPENDS ${ALL_AIV_OBJECTS}
    COMMENT "Combining ${NUM_SOURCES} AIV object files into single .o"
)

# Step 3: Final link
add_custom_target(aicore_kernel ALL
    DEPENDS ${COMBINED_AIC_OBJ} ${COMBINED_AIV_OBJ}

    COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n
    -o ${AICORE_KERNEL_OBJ} ${COMBINED_AIC_OBJ} ${COMBINED_AIV_OBJ}

    BYPRODUCTS ${AICORE_KERNEL_OBJ}
    COMMENT "Linking AICore kernel binary"
)

# The separate kernel-mode entry is applicable only to the TMR executor.
# Never link kernel.cpp into this ELF: the runtime launches entry ID 0, and
# the old program entry uses that same ID with a different argument layout.
foreach(SRC_FILE ${ALL_SOURCES})
    if(SRC_FILE MATCHES "/tensormap_and_ringbuffer/aicore/aicore_executor\\.cpp$")
        set(TMR_KERNEL_MODE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernel_mode_kernel.cpp")
        set(TMR_KERNEL_MODE_AIC "${CMAKE_CURRENT_BINARY_DIR}/kernel_mode_kernel_aic.o")
        set(TMR_KERNEL_MODE_AIV "${CMAKE_CURRENT_BINARY_DIR}/kernel_mode_kernel_aiv.o")
        set(TMR_KERNEL_MODE_OBJ "${CMAKE_CURRENT_BINARY_DIR}/aicore_kernel_mode.o")
        set(TMR_KERNEL_MODE_OBJECTS ${ALL_AIC_OBJECTS} ${ALL_AIV_OBJECTS})
        list(REMOVE_ITEM TMR_KERNEL_MODE_OBJECTS
            "${CMAKE_CURRENT_BINARY_DIR}/kernel_aic.o"
            "${CMAKE_CURRENT_BINARY_DIR}/kernel_aiv.o"
        )
        set(TMR_KERNEL_WIRE_HEADERS
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface/tmr_kernel_context.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface/tmr_kernel_control.h"
        )
        add_custom_command(
            OUTPUT ${TMR_KERNEL_MODE_AIC}
            COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}
                    --cce-aicore-arch=dav-c220-cube -o ${TMR_KERNEL_MODE_AIC} ${TMR_KERNEL_MODE_SRC}
            DEPENDS ${TMR_KERNEL_MODE_SRC} ${TMR_KERNEL_WIRE_HEADERS}
            COMMENT "Compiling kernel-mode entry for AIC"
        )
        add_custom_command(
            OUTPUT ${TMR_KERNEL_MODE_AIV}
            COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}
                    --cce-aicore-arch=dav-c220-vec -o ${TMR_KERNEL_MODE_AIV} ${TMR_KERNEL_MODE_SRC}
            DEPENDS ${TMR_KERNEL_MODE_SRC} ${TMR_KERNEL_WIRE_HEADERS}
            COMMENT "Compiling kernel-mode entry for AIV"
        )
        add_custom_target(aicore_kernel_mode ALL
            DEPENDS ${TMR_KERNEL_MODE_OBJECTS} ${TMR_KERNEL_MODE_AIC} ${TMR_KERNEL_MODE_AIV}
            COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n
                    -o ${TMR_KERNEL_MODE_OBJ} ${TMR_KERNEL_MODE_OBJECTS}
                    ${TMR_KERNEL_MODE_AIC} ${TMR_KERNEL_MODE_AIV}
            BYPRODUCTS ${TMR_KERNEL_MODE_OBJ}
            COMMENT "Linking TMR kernel-mode AICore binary"
        )
        break()
    endif()
endforeach()

# One-shot SDMA control-path warmup: a separate VECTOR-ONLY ELF, not part of
# aicore_kernel.o above. It is its own binary because the executor ELF is a
# resident loop launched per run, while this must run once at worker init, and
# because it needs the PTO headers -- kept on this one compile command so the
# executor's include set is unaffected. Skipped (not fatal) without PTO_ISA_ROOT:
# the warmup is a pure latency optimization and the host launcher degrades to a
# warning when the binary is absent.
set(SDMA_WARMUP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sdma_warmup_kernel.cpp")
if(DEFINED PTO_ISA_ROOT AND EXISTS ${SDMA_WARMUP_SRC})
    set(SDMA_WARMUP_OBJ_AIV "${CMAKE_CURRENT_BINARY_DIR}/sdma_warmup_kernel_aiv.o")
    set(SDMA_WARMUP_OBJ "${CMAKE_CURRENT_BINARY_DIR}/sdma_warmup_kernel.o")

    add_custom_command(
        OUTPUT ${SDMA_WARMUP_OBJ_AIV}
        COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}
                -I${PTO_ISA_ROOT}/include -I${PTO_ISA_ROOT}/include/pto
                --cce-aicore-arch=dav-c220-vec
                -o ${SDMA_WARMUP_OBJ_AIV} ${SDMA_WARMUP_SRC}
        DEPENDS ${SDMA_WARMUP_SRC}
        COMMENT "Compiling sdma_warmup_kernel for AIV"
    )

    add_custom_target(sdma_warmup_kernel ALL
        DEPENDS ${SDMA_WARMUP_OBJ_AIV}

        COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n
        -o ${SDMA_WARMUP_OBJ} ${SDMA_WARMUP_OBJ_AIV}

        BYPRODUCTS ${SDMA_WARMUP_OBJ}
        COMMENT "Linking SDMA warmup kernel binary (vector-only)"
    )
else()
    message(STATUS "SDMA warmup kernel: skipped (PTO_ISA_ROOT undefined or source missing)")
endif()
