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
#include <string.h>

#include "aicpu/l3_l2_message_queue.h"
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int kExpectedArgCount = 12;
constexpr uint32_t kInputWindowComputeFuncId = 0;
constexpr uint64_t kQueueTimeoutNs = 5000000000ULL;
constexpr uint64_t kInputWindow = 4;
constexpr uint64_t kInputHeaderBytes = 64;
constexpr uint64_t kOutputHeaderBytes = 64;
constexpr uint32_t kTileRows = 128;
constexpr uint32_t kTileCols = 128;
constexpr uint64_t kTileBytes = static_cast<uint64_t>(kTileRows) * kTileCols * sizeof(float);

enum InputWindowOp : uint64_t {
    ADD_SCALAR = 1,
    ADD_TILES = 2,
};

struct InputHeader {
    uint64_t request_id;
    uint64_t mode;
};

struct OutputHeader {
    uint64_t request_id;
    uint64_t kind;
    uint64_t aux;
};

struct ActiveRequest {
    L3L2QueueInputHandle handle;
    InputHeader header;
};

void report_queue_error(const L3L2QueueEndpoint &queue) {
    const L3L2QueueError &err = queue.error();
    rt_report_fatal(
        PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 input-window queue error op=%s kind=%u region=%llu msg=%s",
        err.op ? err.op : "unknown", static_cast<unsigned>(err.kind), static_cast<unsigned long long>(err.region_id),
        err.message ? err.message : "unknown"
    );
}

bool has_queue_error(const L3L2QueueEndpoint &queue) { return queue.error().kind != L3L2QueueErrorKind::NONE; }

bool parse_input_header(const L3L2QueueInputHandle &input, InputHeader *header) {
    if (header == nullptr || input.payload_nbytes != kInputHeaderBytes + kTileBytes) {
        return false;
    }
    memcpy(header, reinterpret_cast<const void *>(static_cast<uintptr_t>(input.payload.gm_addr)), sizeof(*header));
    return true;
}

Tensor make_input_values_tensor(const L3L2QueueInputHandle &input) {
    uint32_t shape[2] = {kTileRows, kTileCols};
    void *values = reinterpret_cast<void *>(static_cast<uintptr_t>(input.payload.gm_addr + kInputHeaderBytes));
    return make_tensor_external(values, shape, 2, DataType::FLOAT32);
}

bool publish_aiv_output(
    L3L2QueueEndpoint &queue, const L3L2QueueInputHandle &first, const L3L2QueueInputHandle &second,
    uint64_t request_id, uint64_t kind, uint64_t aux, InputWindowOp op, float scalar
) {
    uint64_t nbytes = kOutputHeaderBytes + kTileBytes;
    L3L2QueueOutputReservation output{};
    if (!queue.output().reserve(nbytes, kQueueTimeoutNs, &output)) {
        report_queue_error(queue);
        return false;
    }

    OutputHeader header{request_id, kind, aux};
    uint8_t *dst = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(output.payload.gm_addr));
    memcpy(dst, &header, sizeof(header));

    Tensor first_tensor = make_input_values_tensor(first);
    Tensor second_tensor = make_input_values_tensor(second);
    uint32_t output_shape[2] = {kTileRows, kTileCols};
    Tensor output_tensor = make_tensor_external(dst + kOutputHeaderBytes, output_shape, 2, DataType::FLOAT32);

    L0TaskArgs params;
    params.add_input(first_tensor);
    params.add_input(second_tensor);
    params.add_output(output_tensor);
    params.add_scalar(static_cast<uint64_t>(op));
    params.add_scalar(to_u64<float>(scalar));
    rt_submit_aiv_task(kInputWindowComputeFuncId, params);

    uint32_t first_output_index[2] = {0, 0};
    (void)get_tensor_data<float>(output_tensor, 2, first_output_index);

    if (!queue.output().publish(output, L3L2QueueOpcode::DATA)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

bool collect_inputs_until_stop(L3L2QueueEndpoint &queue, ActiveRequest *active, uint64_t *active_count) {
    *active_count = 0;
    for (;;) {
        L3L2QueueInputHandle input{};
        if (!queue.input().peek(kQueueTimeoutNs, &input)) {
            report_queue_error(queue);
            return false;
        }
        if (input.opcode == L3L2QueueOpcode::STOP) {
            if (!queue.input().release(input)) {
                report_queue_error(queue);
                return false;
            }
            return true;
        }
        if (input.opcode != L3L2QueueOpcode::DATA || *active_count >= kInputWindow ||
            !parse_input_header(input, &active[*active_count].header)) {
            rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "invalid input-window example request");
            return false;
        }
        active[*active_count].handle = input;
        *active_count += 1;
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void l3_l2_message_queue_input_window_orchestration(const L2TaskArgs &orch_args
) {
    L3L2OrchRegionDesc desc{
        orch_args.scalar(0), orch_args.scalar(1), orch_args.scalar(2),
        orch_args.scalar(3), orch_args.scalar(4), orch_args.scalar(5),
    };
    L3L2QueueArgs queue_args{
        orch_args.scalar(6), orch_args.scalar(7),  orch_args.scalar(8),
        orch_args.scalar(9), orch_args.scalar(10), orch_args.scalar(11),
    };
    L3L2QueueEndpoint queue(desc, queue_args, L3L2QueueEndpointConfig{.max_l2_input_inflight = kInputWindow});
    if (has_queue_error(queue)) {
        report_queue_error(queue);
        return;
    }

    ActiveRequest active[kInputWindow]{};
    uint64_t active_count = 0;
    if (!collect_inputs_until_stop(queue, active, &active_count)) {
        return;
    }
    if (active_count != kInputWindow) {
        rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "input-window example expected four active requests");
        return;
    }

    if (!publish_aiv_output(
            queue, active[1].handle, active[1].handle, active[1].header.request_id, 20, 0, ADD_SCALAR, 20.0F
        )) {
        return;
    }
    if (!queue.input().release(active[1].handle)) {
        report_queue_error(queue);
        return;
    }

    if (!publish_aiv_output(
            queue, active[0].handle, active[0].handle, active[0].header.request_id, 10, 0, ADD_SCALAR, 10.0F
        ) ||
        !publish_aiv_output(
            queue, active[0].handle, active[0].handle, active[0].header.request_id, 11, 0, ADD_SCALAR, 11.0F
        )) {
        return;
    }
    if (!queue.input().release(active[0].handle)) {
        report_queue_error(queue);
        return;
    }

    if (!publish_aiv_output(
            queue, active[2].handle, active[3].handle, active[2].header.request_id, 30, active[3].header.request_id,
            ADD_TILES, 0.0F
        )) {
        return;
    }
    if (!queue.input().release(active[2].handle) || !queue.input().release(active[3].handle)) {
        report_queue_error(queue);
        return;
    }
    if (!queue.input().drained()) {
        rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "input-window example returned before drain");
    }
}

}  // extern "C"
