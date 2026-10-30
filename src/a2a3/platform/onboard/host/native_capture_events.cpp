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
#include "device_runner_base.h"
#include "acl/error_codes/rt_error_codes.h"

extern "C" int runtime_native_capture_supported_impl() { return 1; }

// Acquisition is process-wide: RTS has one event work mode per process, and
// capture needs hardware events (mode 1). The mode is raised at kernel-context
// initialization and is not restored at close; program-mode work in the same
// process observes the same setting. A context that already sees mode 1 does
// nothing, so the switch is idempotent and only ever escalates 0 -> 1.
int DeviceRunnerBase::configure_hbg_kernel_events() {
    if (runtime_native_capture_supported_impl() == 0) return 0;
    uint8_t mode = 0;
    int rc = rtEventWorkModeGet(&mode);
    if (rc == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) return 0;
    if (rc != 0 || mode == 1) return rc;
    rc = rtEventWorkModeSet(1);
    if (rc == 0) return 0;
    if (rtEventWorkModeGet(&mode) == 0 && mode == 1) return 0;
    return rc;
}
