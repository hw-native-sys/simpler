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

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "task_interface/data_type.h"

struct RdmaTransferArgsV1 {
    uint64_t comm_context;
    uint64_t peer_rank;
    uint64_t local_offset;
    uint64_t remote_offset;
    uint64_t nbytes;
    uint64_t qp_index;
};

static_assert(std::is_trivially_copyable_v<RdmaTransferArgsV1> && std::is_standard_layout_v<RdmaTransferArgsV1>);
static_assert(sizeof(RdmaTransferArgsV1) == 48 && alignof(RdmaTransferArgsV1) == 8);
static_assert(offsetof(RdmaTransferArgsV1, comm_context) == 0);
static_assert(offsetof(RdmaTransferArgsV1, peer_rank) == 8);
static_assert(offsetof(RdmaTransferArgsV1, local_offset) == 16);
static_assert(offsetof(RdmaTransferArgsV1, remote_offset) == 24);
static_assert(offsetof(RdmaTransferArgsV1, nbytes) == 32);
static_assert(offsetof(RdmaTransferArgsV1, qp_index) == 40);

PTO_DEVICE_FUNC inline bool rdma_request_in_bounds(
    const RdmaTransferArgsV1 &args, uint32_t my_rank, uint32_t rank_count, uint32_t qp_num, uint64_t window_size
) {
    return args.comm_context != 0 && args.comm_context % 8 == 0 && my_rank < rank_count &&
           args.peer_rank < rank_count && args.peer_rank != my_rank && args.qp_index < qp_num && args.nbytes != 0 &&
           args.nbytes <= INT32_MAX && args.nbytes % 64 == 0 && args.local_offset % 64 == 0 &&
           args.remote_offset % 64 == 0 && args.local_offset <= window_size && args.remote_offset <= window_size &&
           args.nbytes <= window_size - args.local_offset && args.nbytes <= window_size - args.remote_offset;
}
