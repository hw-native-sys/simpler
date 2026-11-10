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

#include "context_preparation.h"

#include "host/host_regs.h"
#include "host/memory_allocator.h"

#include "acl/acl.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "common/kernel_args.h"
#include "common/platform_config.h"
#include "runtime_c_api.h"

namespace simpler::a5 {
int ensure_ctrl_table(MemoryAllocator &allocator, int device_id, uint64_t &table, bool &committed) {
    if (committed) return 0;
    const int rc = init_aicore_register_addresses(&table, static_cast<uint64_t>(device_id), allocator);
    if (rc == 0) committed = true;
    return rc;
}

int release_aicpu_query_result(MemoryAllocator &allocator, rtStream_t stream, void *&scratch, bool &pending) {
    if (scratch == nullptr) return pending ? PTO_RUNTIME_ERR_INVALID_STATE : 0;
    if (pending) {
        if (stream == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
        const int rc = aclrtSynchronizeStreamWithTimeout(stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
        if (rc != 0) return rc;
        pending = false;
    }
    const int rc = allocator.free(scratch);
    if (rc == 0) scratch = nullptr;
    return rc;
}

int query_device_aicpu_occupancy(
    MemoryAllocator &allocator, host::LoadAicpuOp &loader, rtStream_t stream, void *&scratch, bool &pending,
    AicpuDeviceOccupancy &out
) {
    if (stream == nullptr || pending) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (scratch == nullptr) scratch = allocator.alloc(sizeof(AicpuTopologyQueryResult));
    if (scratch == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    const AicpuTopologyQueryResult zero{};
    int rc = rtMemcpy(scratch, sizeof(zero), &zero, sizeof(zero), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) return rc;
    AicpuTopologyQueryArgs args{};
    args.result_addr = reinterpret_cast<uint64_t>(scratch);
    // Even a failed enqueue can leave an execution-visible writer.
    pending = true;
    rc = loader.LaunchBuiltInOp(stream, &args, sizeof(args), 1, "simpler_aicpu_query_topology");
    if (rc != 0) return rc;
    rc = aclrtSynchronizeStreamWithTimeout(stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc != 0) return rc;
    pending = false;
    AicpuTopologyQueryResult result{};
    rc = rtMemcpy(&result, sizeof(result), scratch, sizeof(result), RT_MEMCPY_DEVICE_TO_HOST);
    if (rc != 0) return rc;
    if (result.occupy_rc != 0 || result.occupy == 0) return PTO_RUNTIME_ERR_INTERNAL;
    rc = release_aicpu_query_result(allocator, stream, scratch, pending);
    if (rc != 0) return rc;
    out.occupy = result.occupy;
    out.pf_occupy = result.pf_occupy;
    out.os_sched = result.os_sched;
    out.occupy_valid = result.occupy_rc == 0;
    out.pf_occupy_valid = result.pf_occupy_rc == 0;
    out.os_sched_valid = result.os_sched_rc == 0;
    return 0;
}
}  // namespace simpler::a5
