# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# The stand-ins every directory shares, as one archive per arch.
#
# STATIC rather than OBJECT, and that is the whole point. An object library
# puts every one of its objects into whatever links it, so a single symbol the
# target also defines makes the library unusable and the target has to give up
# all of it — which is what NO_OBJS was for. An archive member is pulled only
# to resolve a symbol nothing else has defined, so a case that compiles the
# real src/common/log/unified_log_host.cpp simply never pulls the log stub, and
# says nothing about it.
#
# That only works because each file here holds one group of symbols: the
# linker's unit is the object file, so a file bundling four groups is pulled
# for any one of them and brings the other three with it.
#
# device_time_stub.cpp is here for everyone who needs get_sys_cnt_aicpu and has
# no definition of their own, and is *also* compiled straight into the tmr
# object libraries. That is not redundant: the tmr runtime defines the symbol
# weakly as `return 0` (orchestrator.cpp, runtime_core.cpp), a weak definition
# satisfies the reference, and the linker then never searches an archive for
# the real one. Compiling it in is what beats the weak fallback; the archive
# copy is simply not pulled there, because the symbol is already defined.
#
# The symptom of relying on the archive alone was a hang, not a link error: the
# runtime's 500 ms reclaim backstop counts ticks that then never advance.
#
# Per arch because device_time_stub.cpp reads PLATFORM_PROF_SYS_CNT_FREQ out of
# that arch's platform_config.h, and platform_regs_stub.cpp compiles against
# that arch's RegId.
foreach(arch a2a3 a5)
    add_library(${arch}_ut_support STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/unified_log_stubs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device_time_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cache_maintenance_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/weak_link_placeholders.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/assert_stubs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform_regs_stub.cpp
    )
    simpler_ut_common_includes(${arch} aicpu _support_includes)
    target_include_directories(${arch}_ut_support PUBLIC ${_support_includes})
endforeach()
