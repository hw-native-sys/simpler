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

namespace pto2::urma_backend {

enum class UrmaDbMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB = 1,
};

struct UrmaInfo {
    uint32_t qp_num;
    uint32_t rank_count;
    uint64_t sq_ptr;
    uint64_t rq_ptr;
    uint64_t scq_ptr;
    uint64_t rcq_ptr;
    uint64_t mem_ptr;
    uint64_t notify_pool_ptr;
    uint32_t layout;
    uint32_t jetty_count;
    uint32_t jetties_per_core;
};

struct UrmaWqCtx {
    uint32_t wqn;
    uint64_t buf_addr;
    uint32_t wqe_shift_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    UrmaDbMode db_mode;
    uint64_t db_addr;
    uint32_t sl;
    uint32_t submitted_wqe_count;
};

struct UrmaCqCtx {
    uint32_t cqn;
    uint64_t buf_addr;
    uint32_t cqe_shift_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    UrmaDbMode db_mode;
    uint64_t db_addr;
};

}  // namespace pto2::urma_backend
