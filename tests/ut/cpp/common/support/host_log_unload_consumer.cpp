/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

// Stands in for a dlopened module that compiles the host logger and is unloaded
// while the process continues — the sim AICPU SO, the sim AICore SO, and the
// generated orchestration SO are all this shape.

#include "host_log.h"

extern "C" __attribute__((visibility("default"))) int test_host_log_unload_bind(SimplerHostLogState *state) {
    return HostLogger::get_instance().bind_state(state);
}

extern "C" __attribute__((visibility("default"))) void test_host_log_unload_emit(int count) {
    for (int index = 0; index < count; ++index) {
        // INFO rides this module's own stdio buffer: severity below WARN selects
        // no write-through, which is what leaves a tail to lose at unload.
        HostLogger::get_instance().log(simpler::log::LogLevel::INFO, "unloaded_module", "record=%d", index);
    }
}
