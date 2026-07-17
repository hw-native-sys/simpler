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

/**
 * ACL / RT Error-Code Names and Triage Hints (host)
 *
 * Names the CANN return codes this project actually hits, so a host failure line carries
 * more than a bare integer. Only the codes we have seen and can say something useful about
 * are listed; anything else is reported as unknown rather than guessed at.
 *
 * Values mirror CANN's acl/error_codes/rt_error_codes.h. They are spelled as literals here
 * so this header stays free of any CANN include and can be unit-tested on a host without a
 * toolkit installed.
 *
 * Full triage steps: docs/troubleshooting/device-error-codes.md
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_ACL_ERROR_NAMES_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_ACL_ERROR_NAMES_H_

#include <cstdint>

// a5 DeviceRunner fail-fast sentinel: the device was marked unusable and the runner refused
// to run. Not a CANN code -- it never leaves this project.
#define SIMPLER_DEVICE_UNUSABLE (-1)

// Symbolic name of a CANN return code, or nullptr when the code is not one we have
// characterised. Callers use nullptr to suppress the annotation instead of guessing.
static inline const char *acl_error_name(int32_t rc) {
    switch (rc) {
    case SIMPLER_DEVICE_UNUSABLE:
        return "SIMPLER_DEVICE_UNUSABLE";
    case 107022:
        return "ACL_ERROR_RT_DEVICE_TASK_ABORT";
    case 207001:
        return "ACL_ERROR_RT_MEMORY_ALLOCATION";
    case 507000:
        return "ACL_ERROR_RT_INTERNAL_ERROR";
    case 507014:
        return "ACL_ERROR_RT_AICORE_TIMEOUT";
    case 507015:
        return "ACL_ERROR_RT_AICORE_EXCEPTION";
    case 507017:
        return "ACL_ERROR_RT_AICPU_TIMEOUT";
    case 507018:
        return "ACL_ERROR_RT_AICPU_EXCEPTION";
    case 507046:
        return "ACL_ERROR_RT_STREAM_SYNC_TIMEOUT";
    case 507899:
        return "ACL_ERROR_RT_DRV_INTERNAL_ERROR";
    default:
        return nullptr;
    }
}

// One-sentence meaning of a CANN return code, in terms of what it means *for this project*.
static inline const char *acl_error_desc(int32_t rc) {
    switch (rc) {
    case SIMPLER_DEVICE_UNUSABLE:
        return "the DeviceRunner had already marked this device unusable and refused to run";
    case 107022:
        return "the device task was aborted, usually as collateral damage after another task on the "
               "same device faulted";
    case 207001:
        return "device memory allocation failed (out of memory)";
    case 507000:
        return "runtime internal error -- on a5 this is what an AICPU op timeout surfaces as at "
               "stream sync";
    case 507014:
        return "an AICore task exceeded its execution timeout";
    case 507015:
        return "an AICore task raised an exception";
    case 507017:
        return "an AICPU task exceeded its execution timeout";
    case 507018:
        return "an AICPU task raised an exception; several distinct on-device mechanisms all surface "
               "as this one generic code";
    case 507046:
        return "the host timed out waiting on a stream sync; the device never reported completion";
    case 507899:
        return "driver internal error -- typically a sticky error returned by every later call on an "
               "already-poisoned context, so it is a symptom of an earlier failure, not the cause";
    default:
        return nullptr;
    }
}

// What to do next about a CANN return code. nullptr when we have nothing specific to say.
static inline const char *acl_error_hint(int32_t rc) {
    switch (rc) {
    case SIMPLER_DEVICE_UNUSABLE:
        return "look for the earlier failure that marked the device unusable; this line is a "
               "consequence of it";
    case 107022:
    case 507899:
        return "look further up the log for the first failure on this device -- this code is fallout, "
               "not the root cause";
    case 207001:
        return "reduce the workload's device memory footprint, or check whether another process is "
               "holding memory on this device";
    case 507000:
    case 507014:
    case 507017:
    case 507018:
    case 507046:
        return "do NOT read this as 'deadlock' or 'OOM' on its own -- it is a generic host-side code. "
               "Grep the host log for 'orch_error_code=' / 'sched_error_code=' / 'sub_class=' to get "
               "the device-classified reason, and read the device log for the detector that fired";
    case 507015:
        return "find the faulting AICore kernel in the device log; this is a kernel-side fault, not a "
               "capacity or scheduling problem";
    default:
        return nullptr;
    }
}

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_ACL_ERROR_NAMES_H_
