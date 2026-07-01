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

#include "aicpu/l3_l2_message_queue.h"
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr uint32_t kTransformFuncId = 0;
constexpr int kExpectedArgCount = 11;
constexpr uint64_t kQueueTimeoutNs = 5000000000ULL;
constexpr uint32_t kRows = 128;
constexpr uint32_t kCols = 128;
constexpr uint32_t kNumel = kRows * kCols;
constexpr uint64_t kTensorBytes = static_cast<uint64_t>(kNumel) * sizeof(float);

void report_queue_error(const L3L2QueueEndpoint &queue) {
    const L3L2QueueError &err = queue.error();
    rt_report_fatal(
        PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue error op=%s kind=%u region=%llu msg=%s",
        err.op ? err.op : "unknown", static_cast<unsigned>(err.kind), static_cast<unsigned long long>(err.region_id),
        err.message ? err.message : "unknown"
    );
}

bool has_queue_error(const L3L2QueueEndpoint &queue) { return queue.error().kind != L3L2QueueErrorKind::NONE; }

bool process_data_message(L3L2QueueEndpoint &queue, const L3L2QueueInputHandle &input, float scalar) {
    if (input.payload_nbytes != kTensorBytes) {
        rt_report_fatal(
            PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example expected %llu input bytes, got %llu",
            static_cast<unsigned long long>(kTensorBytes), static_cast<unsigned long long>(input.payload_nbytes)
        );
        return false;
    }

    L3L2QueueOutputReservation output{};
    if (!queue.output().reserve(kTensorBytes, kQueueTimeoutNs, &output)) {
        report_queue_error(queue);
        return false;
    }
    uint32_t shape[2] = {kRows, kCols};
    Tensor input_tensor = make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(input.payload.gm_addr)), shape, 2, DataType::FLOAT32
    );
    Tensor output_tensor = make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(output.payload.gm_addr)), shape, 2, DataType::FLOAT32
    );

    L0TaskArgs params;
    params.add_input(input_tensor);
    params.add_output(output_tensor);
    params.add_scalar(scalar);
    rt_submit_aiv_task(kTransformFuncId, params);

    uint32_t first_index[2] = {0, 0};
    (void)get_tensor_data<float>(output_tensor, 2, first_index);
    if (!queue.output().publish(output, L3L2QueueOpcode::DATA)) {
        report_queue_error(queue);
        return false;
    }
    if (!queue.input().release(input)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void l3_l2_message_queue_orchestration(const L2TaskArgs &orch_args) {
    L3L2OrchRegionDesc desc{
        orch_args.scalar(0), orch_args.scalar(1), orch_args.scalar(2),
        orch_args.scalar(3), orch_args.scalar(4), orch_args.scalar(5),
    };
    L3L2QueueArgs queue_args{
        orch_args.scalar(6),
        orch_args.scalar(7),
        orch_args.scalar(8),
        orch_args.scalar(9),
    };
    L3L2QueueEndpoint queue(desc, queue_args);
    if (has_queue_error(queue)) {
        report_queue_error(queue);
        return;
    }

    const float scalar = from_u64<float>(orch_args.scalar(10));
    for (;;) {
        L3L2QueueInputHandle input{};
        if (!queue.input().peek(kQueueTimeoutNs, &input)) {
            report_queue_error(queue);
            return;
        }
        if (input.opcode == L3L2QueueOpcode::STOP) {
            if (!queue.input().release(input)) {
                report_queue_error(queue);
            }
            return;
        }
        if (input.opcode != L3L2QueueOpcode::DATA) {
            rt_report_fatal(
                PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example unexpected input opcode=%llu",
                static_cast<unsigned long long>(input.opcode)
            );
            return;
        }
        if (!process_data_message(queue, input, scalar)) {
            return;
        }
    }
}

}  // extern "C"
