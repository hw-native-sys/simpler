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

struct CaptureSubmitFaultResult {
    int accepted_core = 0;
    int cpu_calls = 0;
    int submissions_after_failure = 0;
    int boundary_rc = 0;
    int conflict_rc = 0;
    bool fired = false;
};

void arm_capture_submit_fault(void *caller, void *event);
CaptureSubmitFaultResult finish_capture_submit_fault();

void begin_capture_arena_probe();
void *finish_capture_arena_probe();
