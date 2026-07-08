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

#include "aicpu/l3_l2_orch_endpoint.h"
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr uint32_t kChannelData = 1;
constexpr uint32_t kChannelStop = 2;
constexpr uint32_t kTransformFuncId = 0;
constexpr int kExpectedArgCount = 14;

struct ChannelHeader {
    uint64_t seq;
    uint32_t opcode;
    uint32_t reserved;
};

void report_endpoint_error(const L3L2OrchEndpoint &endpoint) {
    const L3L2EndpointError &err = endpoint.error();
    rt_report_fatal(
        PTO2_ERROR_EXPLICIT_ORCH_FATAL,
        "L3-L2 endpoint error op=%s kind=%u region=%llu counter_addr=%llu counter_operand=%d observed_counter=%d "
        "msg=%s",
        l3_l2_endpoint_op_to_string(err.op), static_cast<unsigned>(err.kind),
        static_cast<unsigned long long>(err.region_id), static_cast<unsigned long long>(err.counter_addr),
        err.counter_operand, err.observed_counter, err.message
    );
}

bool has_endpoint_error(const L3L2OrchEndpoint &endpoint) {
    return endpoint.error().kind != L3L2EndpointErrorKind::NONE;
}

bool read_payload_or_fail(L3L2OrchEndpoint &endpoint, uint64_t offset, uint64_t nbytes, L3L2OrchPayloadView *out) {
    if (endpoint.payload_read(offset, nbytes, out)) {
        return true;
    }
    report_endpoint_error(endpoint);
    return false;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void l3_l2_orch_comm_orchestration(const L2TaskArgs &orch_args) {
    uint64_t desc_scalars[L3L2_ORCH_REGION_DESC_SCALAR_COUNT] = {
        orch_args.scalar(0), orch_args.scalar(1), orch_args.scalar(2),
        orch_args.scalar(3), orch_args.scalar(4), orch_args.scalar(5),
    };
    L3L2OrchEndpoint endpoint(desc_scalars, L3L2_ORCH_REGION_DESC_SCALAR_COUNT);
    if (has_endpoint_error(endpoint)) {
        report_endpoint_error(endpoint);
        return;
    }

    const uint64_t input_offset = orch_args.scalar(6);
    const uint64_t output_offset = orch_args.scalar(7);
    const uint32_t numel = static_cast<uint32_t>(orch_args.scalar(8));
    const DataType dtype = static_cast<DataType>(orch_args.scalar(9));
    const uint64_t tensor_nbytes = orch_args.scalar(10);
    const float scalar = from_u64<float>(orch_args.scalar(11));
    const uint64_t data_ready_counter_offset = orch_args.scalar(12);
    const uint64_t completion_counter_offset = orch_args.scalar(13);
    const uint64_t timeout = 5000000000ULL;
    uint32_t shape[1] = {numel};
    uint64_t data_ready_counter_addr = 0;
    uint64_t completion_counter_addr = 0;
    if (!endpoint.counter_addr(data_ready_counter_offset, &data_ready_counter_addr) ||
        !endpoint.counter_addr(completion_counter_offset, &completion_counter_addr)) {
        report_endpoint_error(endpoint);
        return;
    }

    for (uint64_t seq = 1;; ++seq) {
        const int32_t signal_value = static_cast<int32_t>(seq);
        int32_t observed = 0;
        if (!endpoint.signal_wait(data_ready_counter_addr, signal_value, L3L2OrchWaitCmp::GE, timeout, &observed)) {
            report_endpoint_error(endpoint);
            return;
        }

        L3L2OrchPayloadView header_view{};
        if (!read_payload_or_fail(endpoint, 0, sizeof(ChannelHeader), &header_view)) {
            return;
        }
        const ChannelHeader *header =
            reinterpret_cast<const ChannelHeader *>(static_cast<uintptr_t>(header_view.gm_addr));
        if (header->seq != seq) {
            rt_report_fatal(
                PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 channel header seq mismatch expected=%llu got=%llu",
                static_cast<unsigned long long>(seq), static_cast<unsigned long long>(header->seq)
            );
            return;
        }
        if (header->opcode == kChannelStop) {
            return;
        }
        if (header->opcode != kChannelData) {
            rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 channel unknown opcode=%u", header->opcode);
            return;
        }

        L3L2OrchPayloadView input_view{};
        L3L2OrchPayloadView output_view{};
        if (!read_payload_or_fail(endpoint, input_offset, tensor_nbytes, &input_view)) {
            return;
        }
        if (!read_payload_or_fail(endpoint, output_offset, tensor_nbytes, &output_view)) {
            return;
        }

        Tensor input =
            make_tensor_external(reinterpret_cast<void *>(static_cast<uintptr_t>(input_view.gm_addr)), shape, 1, dtype);
        Tensor output = make_tensor_external(
            reinterpret_cast<void *>(static_cast<uintptr_t>(output_view.gm_addr)), shape, 1, dtype
        );
        L0TaskArgs params;
        params.add_input(input);
        params.add_output(output);
        params.add_scalar(scalar);
        params.add_scalar(numel);
        rt_submit_aiv_task(kTransformFuncId, params);

        uint32_t first_index[1] = {0};
        (void)get_tensor_data<float>(output, 1, first_index);
        if (!endpoint.signal_notify(completion_counter_addr, signal_value, L3L2OrchNotifyOp::Set)) {
            report_endpoint_error(endpoint);
            return;
        }
    }
}

}  // extern "C"
