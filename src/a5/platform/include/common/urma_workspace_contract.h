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

#include <cstddef>
#include <pto/comm/async/urma/urma_types.hpp>
#include "common/urma_workspace_abi.h"

namespace pto2::urma_backend {
static_assert(static_cast<uint32_t>(pto::comm::urma::UrmaLayout::PER_PEER) == 0);
static_assert(sizeof(UrmaInfo) == sizeof(pto::comm::urma::UrmaInfo));
static_assert(alignof(UrmaInfo) == alignof(pto::comm::urma::UrmaInfo));
static_assert(offsetof(UrmaInfo, qp_num) == offsetof(pto::comm::urma::UrmaInfo, rowsPerPeer));
static_assert(offsetof(UrmaInfo, rank_count) == offsetof(pto::comm::urma::UrmaInfo, rankCount));
static_assert(offsetof(UrmaInfo, sq_ptr) == offsetof(pto::comm::urma::UrmaInfo, sqPtr));
static_assert(offsetof(UrmaInfo, rq_ptr) == offsetof(pto::comm::urma::UrmaInfo, rqPtr));
static_assert(offsetof(UrmaInfo, scq_ptr) == offsetof(pto::comm::urma::UrmaInfo, scqPtr));
static_assert(offsetof(UrmaInfo, rcq_ptr) == offsetof(pto::comm::urma::UrmaInfo, rcqPtr));
static_assert(offsetof(UrmaInfo, mem_ptr) == offsetof(pto::comm::urma::UrmaInfo, memPtr));
static_assert(offsetof(UrmaInfo, notify_pool_ptr) == offsetof(pto::comm::urma::UrmaInfo, notifyPoolPtr));
static_assert(offsetof(UrmaInfo, layout) == offsetof(pto::comm::urma::UrmaInfo, layout));
static_assert(offsetof(UrmaInfo, jetty_count) == offsetof(pto::comm::urma::UrmaInfo, jettyCount));
static_assert(offsetof(UrmaInfo, jetties_per_core) == offsetof(pto::comm::urma::UrmaInfo, jettiesPerCore));
static_assert(sizeof(UrmaWqCtx) == sizeof(pto::comm::urma::UrmaWQCtx));
static_assert(alignof(UrmaWqCtx) == alignof(pto::comm::urma::UrmaWQCtx));
static_assert(offsetof(UrmaWqCtx, wqn) == offsetof(pto::comm::urma::UrmaWQCtx, wqn));
static_assert(offsetof(UrmaWqCtx, buf_addr) == offsetof(pto::comm::urma::UrmaWQCtx, bufAddr));
static_assert(offsetof(UrmaWqCtx, wqe_shift_size) == offsetof(pto::comm::urma::UrmaWQCtx, wqeShiftSize));
static_assert(offsetof(UrmaWqCtx, depth) == offsetof(pto::comm::urma::UrmaWQCtx, depth));
static_assert(offsetof(UrmaWqCtx, head_addr) == offsetof(pto::comm::urma::UrmaWQCtx, headAddr));
static_assert(offsetof(UrmaWqCtx, tail_addr) == offsetof(pto::comm::urma::UrmaWQCtx, tailAddr));
static_assert(offsetof(UrmaWqCtx, db_mode) == offsetof(pto::comm::urma::UrmaWQCtx, dbMode));
static_assert(offsetof(UrmaWqCtx, db_addr) == offsetof(pto::comm::urma::UrmaWQCtx, dbAddr));
static_assert(offsetof(UrmaWqCtx, sl) == offsetof(pto::comm::urma::UrmaWQCtx, sl));
static_assert(offsetof(UrmaWqCtx, submitted_wqe_count) == offsetof(pto::comm::urma::UrmaWQCtx, submittedWqeCount));
static_assert(sizeof(UrmaCqCtx) == sizeof(pto::comm::urma::UrmaCqCtx));
static_assert(alignof(UrmaCqCtx) == alignof(pto::comm::urma::UrmaCqCtx));
static_assert(offsetof(UrmaCqCtx, cqn) == offsetof(pto::comm::urma::UrmaCqCtx, cqn));
static_assert(offsetof(UrmaCqCtx, buf_addr) == offsetof(pto::comm::urma::UrmaCqCtx, bufAddr));
static_assert(offsetof(UrmaCqCtx, cqe_shift_size) == offsetof(pto::comm::urma::UrmaCqCtx, cqeShiftSize));
static_assert(offsetof(UrmaCqCtx, depth) == offsetof(pto::comm::urma::UrmaCqCtx, depth));
static_assert(offsetof(UrmaCqCtx, head_addr) == offsetof(pto::comm::urma::UrmaCqCtx, headAddr));
static_assert(offsetof(UrmaCqCtx, tail_addr) == offsetof(pto::comm::urma::UrmaCqCtx, tailAddr));
static_assert(offsetof(UrmaCqCtx, db_mode) == offsetof(pto::comm::urma::UrmaCqCtx, dbMode));
static_assert(offsetof(UrmaCqCtx, db_addr) == offsetof(pto::comm::urma::UrmaCqCtx, dbAddr));
}  // namespace pto2::urma_backend
