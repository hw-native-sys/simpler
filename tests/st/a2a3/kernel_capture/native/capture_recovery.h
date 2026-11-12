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

#pragma once

#include <cstdio>
#include <cstdlib>

#include "capture_submit_fault.h"
#include "runtime_c_api.h"

namespace simpler::capture_test {

inline void check(int rc, const char *stage) {
    std::printf("stage=%s rc=%d\n", stage, rc);
    std::fflush(stdout);
    if (rc != 0) std::exit(1);
}

inline void require(bool condition, const char *stage) { check(condition ? 0 : -1, stage); }

size_t prepare_snapshot(DeviceContextHandle context, int scenario);
void check_prepare(DeviceContextHandle context, int scenario, size_t before);
bool needs_publications(int scenario);
void check_publications(int scenario, const CapturedSubmission &submission);
void inject_boot_failure();
void restore_boot_failure();
void check_boot_failure();
void check_progress_timeout();

}  // namespace simpler::capture_test
