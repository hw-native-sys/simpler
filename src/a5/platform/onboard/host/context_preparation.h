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

#include "aicpu_topology_probe.h"
#include "runtime/rt.h"

class MemoryAllocator;
namespace host {
class LoadAicpuOp;
}

namespace pto::a5 {
int ensure_ctrl_table(MemoryAllocator &allocator, int device_id, uint64_t &table, bool &committed);
// Scratch and pending-write state belong to the calling DeviceRunner.
int query_device_aicpu_occupancy(
    MemoryAllocator &allocator, host::LoadAicpuOp &loader, rtStream_t stream, void *&scratch, bool &pending,
    AicpuDeviceOccupancy &out
);
int release_aicpu_query_result(MemoryAllocator &allocator, rtStream_t stream, void *&scratch, bool &pending);
}  // namespace pto::a5
