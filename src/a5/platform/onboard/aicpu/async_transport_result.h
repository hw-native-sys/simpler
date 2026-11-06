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

#include <cstdint>

#include "common/device_run_result.h"
#include "host_build_graph/async_retirement.h"
#include "host_build_graph/runtime_status.h"

// Runtime-private capability query. Platforms without an audited async image
// provide the weak false implementation; no Runtime layout crosses this call.
extern "C" bool aicpu_async_transport_enabled(const void *runtime, uint64_t run_epoch);

// Call after the native final owner returns. Non-final callbacks return zero
// without reading the region that the last callback may still be publishing.
// This changes only CANN transport: the Error record and Host last-use gates
// retain their independent meanings.
inline int32_t a5_async_transport_result(bool async_run, uint64_t result_base, uint64_t epoch, int32_t rc) {
    if (!async_run || result_base == 0 || epoch == 0 || rc == 0 || result_base % alignof(DeviceRunResultRegion) != 0 ||
        result_base > UINT64_MAX - sizeof(DeviceRunResultRegion))
        return rc;
    if (rc != -SIMPLER_ERROR_SCHEDULER_TIMEOUT && rc != -SIMPLER_ERROR_ASYNC_COMPLETION_INVALID &&
        rc != -SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW && rc != -SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED)
        return rc;
    const auto &region = *reinterpret_cast<const DeviceRunResultRegion *>(result_base);
    const auto terminal = device_run_result_terminal(region, epoch);
    if (terminal.state != DeviceRunTerminalState::Failed || terminal.source != DeviceRunCodeSource::Header ||
        terminal.code != rc || !hbg_async_cores_retired(region.payload, region.payload_bytes, epoch))
        return rc;
    return 0;
}
