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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"
#include "../records.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    const uint64_t begin_cycles = get_sys_cnt();
    const uint32_t logical = get_block_idx(args);
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *records = reinterpret_cast<__gm__ urma_test::TransferRecord *>(
        reinterpret_cast<__gm__ int64_t *>(input->buffer.addr) + input->start_offset
    );
    auto *observations = reinterpret_cast<__gm__ urma_test::ObservationRecord *>(
        reinterpret_cast<__gm__ int64_t *>(output->buffer.addr) + output->start_offset
    );
    auto &record = records[logical];
    auto &observed = observations[logical];
    observed.begin_cycles = begin_cycles;
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    auto *info =
        reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(comm_workspace<CommWorkspaceKind::URMA>(*comm).addr);
    observed.status = record.status == 1 && record.jetties > 0 && record.jetties <= urma_test::kMaxJetties ? 1 : -1;
    if (observed.status == 1) {
        for (uint32_t j = 0; j < record.jetties; ++j) {
            const uint32_t row = record.queues[j].row;
            if (row >= info->jettyCount) {
                observed.status = -2;
                break;
            }
            auto *wq = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(info->sqPtr) + row;
            auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(info->scqPtr) + row;
            observed.queues[j].completed_bb = pto2::urma_backend::load_urma_counter(wq->tailAddr);
            observed.queues[j].completed_cqe = pto2::urma_backend::load_urma_counter(cq->tailAddr);
        }
    }
    pto2::detail::defer_flush_range(&observed, sizeof(observed));
    dsb(DSB_DDR);
}
