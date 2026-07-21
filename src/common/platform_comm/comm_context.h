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
 * CommContext — device-side distributed communication context.
 *
 * This struct is the ABI contract between host (comm_hccl.cpp / comm_sim.cpp)
 * and device kernels. PTO communication instructions (TREDUCE, TGET, TPUT)
 * access remote data through the GVA addresses in windowsIn[]/windowsOut[]
 * via MTE2 DMA.
 *
 * Host fills the struct from scratch:
 *   - comm_hccl.cpp (Path D): allocates a per-rank symmetric pool via the
 *     public ACL IPC primitives (aclrtMalloc + aclrtIpcMemGetExportKey +
 *     SetImportPid + ImportByKey), then writes rankId / rankNum / winSize /
 *     windowsIn[]. No HCCL-private struct is reinterpret_cast'd here; the
 *     layout is owned end-to-end by simpler.
 *   - comm_sim.cpp: same shape, filled with malloc'd host pointers.
 *
 * The earlier "HCCL MESH reinterpret_cast / RING reverse-parse of
 * HcclOpResParam" paths have been deleted along with their internal-ABI
 * coupling -- see .docs/28.l3-comm/ext.01.pr-774-review.md for context.
 *
 * Long-term private-ization of this struct (decouple from the pto-isa
 * HcclDeviceContext parallel declaration) is tracked as F4 in the same
 * doc; this file is intentionally left at the current shared layout
 * until that decision lands.
 */

#pragma once

#include <cstddef>
#include <cstdint>

static constexpr uint32_t COMM_MAX_RANK_NUM = 64;

struct CommContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[COMM_MAX_RANK_NUM];
    uint64_t windowsOut[COMM_MAX_RANK_NUM];
};

// The struct itself lives in this repo, so on the surface these asserts look
// like they only check that we do not contradict ourselves. Their real value
// is that this layout is consumed by *two* out-of-band parties that never see
// this header at the same time:
//
//   1. The pto-isa repo carries a parallel declaration (HcclDeviceContext)
//      that must be byte-equivalent to this struct -- pto-isa kernels read
//      windowsIn[]/winSize/rankId via that mirror. Any insert/reorder here
//      that is not matched in pto-isa silently shifts the device-side field
//      offsets and corrupts MTE2 reads. The locks below pin our side; the
//      pto-isa side should add its own mirror asserts.
//
//   2. Device kernels (AICore / AICPU) compiled with CCEC may apply slightly
//      different alignment rules than host gcc. A host-side sizeof/offset
//      lock is a necessary-but-not-sufficient guard.
//
// Treat the numbers below as a tripwire: changing them is a deliberate act
// that forces the editor to coordinate the matching change on the pto-isa
// side, not a routine "oh I just added a field" edit.
static_assert(sizeof(CommContext) == 1056, "CommContext size shifted");
static_assert(offsetof(CommContext, workSpace) == 0, "CommContext layout drift");
static_assert(offsetof(CommContext, workSpaceSize) == 8, "CommContext layout drift");
static_assert(offsetof(CommContext, rankId) == 16, "CommContext layout drift");
static_assert(offsetof(CommContext, rankNum) == 20, "CommContext layout drift");
static_assert(offsetof(CommContext, winSize) == 24, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsIn) == 32, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsOut) == 544, "CommContext layout drift");
