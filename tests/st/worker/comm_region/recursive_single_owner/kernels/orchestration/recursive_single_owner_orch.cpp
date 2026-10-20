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

#include <stdint.h>

#include "aicpu/region_instance_view.h"
#include "common/region_task_args.h"
#include "orchestration_api.h"  // NOLINT(build/include_subdir)
#include "aicpu/region_endpoint_diagnostics.h"

namespace {

constexpr uint32_t kChannelData = 1;
constexpr uint32_t kChannelStop = 2;
constexpr uint32_t kTransformFuncId = 0;
constexpr int kExpectedArgCount = 16;

struct ChannelHeader {
    uint64_t seq;
    uint32_t opcode;
    uint32_t reserved;
};

void report_view_error(
    uint64_t session_bits, uint64_t transaction_id, uint64_t region_id, const RegionViewError &error,
    uint64_t counter_addr = 0, int32_t counter_operand = 0
) {
    region_report_endpoint_error(session_bits, transaction_id, region_id, error, counter_addr, counter_operand);
}

void report_decode_error(RegionTaskArgError error) { region_report_invalid_task_args(error); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void recursive_single_owner_orchestration(const ChipTaskArgs &orch_args) {
    uint64_t desc_scalars[REGION_TASK_SCALAR_COUNT] = {
        orch_args.scalar(0), orch_args.scalar(1), orch_args.scalar(2), orch_args.scalar(3),
        orch_args.scalar(4), orch_args.scalar(5), orch_args.scalar(6), orch_args.scalar(7),
    };
    RegionPartLocalSpan payload{};
    RegionPartLocalSpan counter{};
    uint64_t session_bits = 0;
    uint64_t transaction_id = 0;
    uint64_t region_id = 0;
    RegionTaskArgError decode_error = RegionTaskArgError::OK;
    if (!region_decode_task_args(
            desc_scalars, REGION_TASK_SCALAR_COUNT, payload, counter, session_bits, transaction_id, region_id,
            decode_error
        )) {
        report_decode_error(decode_error);
        return;
    }
    RegionInstanceView view(payload, counter);
    if (view.failed()) {
        report_view_error(session_bits, transaction_id, region_id, view.error());
        return;
    }

    const uint64_t input_offset = orch_args.scalar(8);
    const uint64_t output_offset = orch_args.scalar(9);
    const uint32_t numel = static_cast<uint32_t>(orch_args.scalar(10));
    const DataType dtype = static_cast<DataType>(orch_args.scalar(11));
    const uint64_t tensor_nbytes = orch_args.scalar(12);
    const float scalar = from_u64<float>(orch_args.scalar(13));
    const uint64_t data_ready_counter_offset = orch_args.scalar(14);
    const uint64_t completion_counter_offset = orch_args.scalar(15);
    const uint64_t timeout = 5000000000ULL;
    uint32_t shape[1] = {numel};
    const auto counter_addr_at = [&](uint64_t offset) -> uint64_t {
        if (region_add_overflows(counter.base, offset)) return 0;
        return counter.base + offset;
    };

    for (uint64_t seq = 1;; ++seq) {
        const int32_t signal_value = static_cast<int32_t>(seq);
        int32_t observed = 0;
        if (!view.counter().wait(data_ready_counter_offset, signal_value, RegionWaitCmp::GE, timeout, observed)) {
            report_view_error(
                session_bits, transaction_id, region_id, view.error(), counter_addr_at(data_ready_counter_offset),
                signal_value
            );
            return;
        }

        RegionPayloadView header_view{};
        if (!view.payload().read(0, sizeof(ChannelHeader), header_view)) {
            report_view_error(session_bits, transaction_id, region_id, view.error());
            return;
        }
        const ChannelHeader *header =
            reinterpret_cast<const ChannelHeader *>(static_cast<uintptr_t>(header_view.local_addr));
        if (header->seq != seq) {
            rt_report_fatal(
                SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "channel header seq mismatch expected=%llu got=%llu",
                static_cast<unsigned long long>(seq), static_cast<unsigned long long>(header->seq)
            );
            return;
        }
        if (header->opcode == kChannelStop) {
            return;
        }
        if (header->opcode != kChannelData) {
            rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "channel unknown opcode=%u", header->opcode);
            return;
        }

        RegionPayloadView input_view{};
        RegionPayloadView output_view{};
        if (!view.payload().read(input_offset, tensor_nbytes, input_view) ||
            !view.payload().read(output_offset, tensor_nbytes, output_view)) {
            report_view_error(session_bits, transaction_id, region_id, view.error());
            return;
        }

        simpler::tmr::Tensor input = simpler::tmr::make_tensor_external(
            reinterpret_cast<void *>(static_cast<uintptr_t>(input_view.local_addr)), shape, 1, dtype
        );
        simpler::tmr::Tensor output = simpler::tmr::make_tensor_external(
            reinterpret_cast<void *>(static_cast<uintptr_t>(output_view.local_addr)), shape, 1, dtype
        );
        CoreTaskArgs params;
        params.add_input(input);
        params.add_output(output);
        params.add_scalar(scalar);
        params.add_scalar(numel);
        rt_submit_aiv_task(kTransformFuncId, params);

        uint32_t first_index[1] = {0};
        (void)get_tensor_data<float>(output, 1, first_index);
        if (!view.counter().notify(completion_counter_offset, signal_value, RegionNotifyOp::Set)) {
            report_view_error(
                session_bits, transaction_id, region_id, view.error(), counter_addr_at(completion_counter_offset),
                signal_value
            );
            return;
        }
    }
}

}  // extern "C"
