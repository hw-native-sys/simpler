# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Compiles producer.cce into a relocatable AICore ELF that the host registers
# via rtRegisterAllKernel(). The compile mirrors what
# simpler_setup/runtime_compiler.py emits for a2a3 AICore kernels — same
# flags, same arch — so the resulting .o is byte-compatible with the
# production path.
#
# Default arch is dav-c220-cube (a3). Override with -DCCE_AICORE_ARCH=...
# at configure time to retarget (e.g. dav-c220-vec, dav-c310-cube for a5).

cmake_minimum_required(VERSION 3.16)
project(fin_order_producer LANGUAGES NONE)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

set(CCEC "${ASCEND_HOME_PATH}/bin/ccec")
set(LD_LLD "${ASCEND_HOME_PATH}/bin/ld.lld")
if(NOT EXISTS "${CCEC}")
    message(FATAL_ERROR "ccec not found at ${CCEC}")
endif()

if(NOT DEFINED CCE_AICORE_ARCH)
    set(CCE_AICORE_ARCH "dav-c220-cube")
endif()
# The mix pair needs both halves; derive the vector arch from the cube one.
string(REPLACE "-cube" "-vec" CCE_AICORE_ARCH_VEC "${CCE_AICORE_ARCH}")
set(CCE_AICORE_ARCH_CUBE "${CCE_AICORE_ARCH}")

set(CCE_INCLUDE_DIRS
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/external"
    "${ASCEND_HOME_PATH}/include/c_api"
    "${ASCEND_HOME_PATH}/include/c_api/internal"
    # AscendC kernel headers live under the arch subtree on CANN 9.x, not under
    # the toolkit root's include/. kernel_operator.h pulls impl/ and interface/
    # by bare name, so all three are needed.
    "${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw"
    "${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/impl"
    "${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/interface"
)
set(CCE_INCLUDE_FLAGS "")
foreach(d ${CCE_INCLUDE_DIRS})
    list(APPEND CCE_INCLUDE_FLAGS "-I${d}")
endforeach()

set(PRODUCER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/producer.cce)
set(PRODUCER_AIC ${CMAKE_CURRENT_BINARY_DIR}/fin_order_producer_aic.o)
set(PRODUCER_AIV ${CMAKE_CURRENT_BINARY_DIR}/fin_order_producer_aiv.o)
set(PRODUCER_OBJ ${CMAKE_CURRENT_BINARY_DIR}/fin_order_producer.o)

set(CCE_COMMON_FLAGS
    -c -O2 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
)

add_custom_command(
    OUTPUT ${PRODUCER_AIC}
    COMMAND ${CCEC} ${CCE_COMMON_FLAGS} --cce-aicore-arch=${CCE_AICORE_ARCH_CUBE}
            ${CCE_INCLUDE_FLAGS} -o ${PRODUCER_AIC} ${PRODUCER_SRC}
    DEPENDS ${PRODUCER_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/../shared/handshake.h
    COMMENT "ccec compiling AICore producer, cube half (${CCE_AICORE_ARCH_CUBE})"
    VERBATIM
)

add_custom_command(
    OUTPUT ${PRODUCER_AIV}
    COMMAND ${CCEC} ${CCE_COMMON_FLAGS} --cce-aicore-arch=${CCE_AICORE_ARCH_VEC}
            ${CCE_INCLUDE_FLAGS} -o ${PRODUCER_AIV} ${PRODUCER_SRC}
    DEPENDS ${PRODUCER_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/../shared/handshake.h
    COMMENT "ccec compiling AICore producer, vector half (${CCE_AICORE_ARCH_VEC})"
    VERBATIM
)

# rtRegisterAllKernel takes a linked device ELF carrying BOTH halves of the
# mix-kernel symbol pair (`*_0_mix_aic` and `*_0_mix_aiv`); a raw `ccec -c`
# object, or one holding only the cube half, is rejected with 107000. Same link
# line as src/a2a3/platform/onboard/aicore/CMakeLists.txt.
add_custom_command(
    OUTPUT ${PRODUCER_OBJ}
    COMMAND ${LD_LLD} -m aicorelinux -Ttext=0 -static -n
            -o ${PRODUCER_OBJ} ${PRODUCER_AIC} ${PRODUCER_AIV}
    DEPENDS ${PRODUCER_AIC} ${PRODUCER_AIV}
    COMMENT "Linking AICore producer mix pair into a registrable ELF"
    VERBATIM
)

add_custom_target(fin_order_producer ALL DEPENDS ${PRODUCER_OBJ})

# Export the artifact path so the parent CMake (or host launcher) can
# locate it for embedding via xxd or runtime ifstream.
set_property(GLOBAL PROPERTY FIN_ORDER_PRODUCER_OBJ ${PRODUCER_OBJ})
