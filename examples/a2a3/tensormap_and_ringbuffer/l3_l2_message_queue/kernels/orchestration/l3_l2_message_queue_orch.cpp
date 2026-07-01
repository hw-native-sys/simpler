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
        PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue error op=%s kind=%u region=%llu msg=%s",
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
    memset(dst, 0, kOutputHeaderBytes);
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

bool release_input(L3L2QueueEndpoint &queue, const L3L2QueueInputHandle &input) {
    if (!queue.input().release(input)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

bool process_first_pair(L3L2QueueEndpoint &queue, ActiveRequest *active, const L3L2QueueInputHandle &input) {
    active[1].handle = input;
    if (!parse_input_header(input, &active[1].header) || active[0].header.request_id == 0) {
        rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "invalid L3-L2 queue example request");
        return false;
    }
    if (!publish_aiv_output(
            queue, active[1].handle, active[1].handle, active[1].header.request_id, 20, 0, ADD_SCALAR, 20.0F
        )) {
        return false;
    }
    if (!release_input(queue, active[1].handle)) {
        return false;
    }
    if (!publish_aiv_output(
            queue, active[0].handle, active[0].handle, active[0].header.request_id, 10, 0, ADD_SCALAR, 10.0F
        ) ||
        !publish_aiv_output(
            queue, active[0].handle, active[0].handle, active[0].header.request_id, 11, 0, ADD_SCALAR, 11.0F
        )) {
        return false;
    }
    return release_input(queue, active[0].handle);
}

bool remember_input_for_pair(ActiveRequest *active, const L3L2QueueInputHandle &input, const InputHeader &header) {
    if (active[2].header.request_id == 0) {
        active[2].handle = input;
        active[2].header = header;
        return true;
    }
    if (active[3].header.request_id == 0) {
        active[3].handle = input;
        active[3].header = header;
        return true;
    }
    rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example input window is full");
    return false;
}

bool process_data_message(L3L2QueueEndpoint &queue, const L3L2QueueInputHandle &input, ActiveRequest *active) {
    InputHeader header{};
    if (!parse_input_header(input, &header)) {
        rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "invalid L3-L2 queue example request");
        return false;
    }
    if (header.mode == 1) {
        active[0].handle = input;
        active[0].header = header;
        return true;
    }
    if (header.mode == 2) {
        return process_first_pair(queue, active, input);
    }
    if (header.mode == 3) {
        return remember_input_for_pair(active, input, header);
    }
    rt_report_fatal(
        PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example unexpected mode=%llu",
        static_cast<unsigned long long>(header.mode)
    );
    return false;
}

bool finish_pending_inputs(L3L2QueueEndpoint &queue, ActiveRequest *active) {
    if (active[2].header.request_id == 0 && active[3].header.request_id == 0) {
        return true;
    }
    if (active[2].header.request_id == 0 || active[3].header.request_id == 0) {
        rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example missing paired input");
        return false;
    }
    if (!publish_aiv_output(
            queue, active[2].handle, active[3].handle, active[2].header.request_id, 30, active[3].header.request_id,
            ADD_TILES, 0.0F
        )) {
        return false;
    }
    return release_input(queue, active[2].handle) && release_input(queue, active[3].handle);
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
        orch_args.scalar(6), orch_args.scalar(7),  orch_args.scalar(8),
        orch_args.scalar(9), orch_args.scalar(10), orch_args.scalar(11),
    };
    L3L2QueueEndpoint queue(desc, queue_args, L3L2QueueEndpointConfig{.max_l2_input_inflight = kInputWindow});
    if (has_queue_error(queue)) {
        report_queue_error(queue);
        return;
    }

    ActiveRequest active[kInputWindow]{};
    for (;;) {
        L3L2QueueInputHandle input{};
        if (!queue.input().peek(kQueueTimeoutNs, &input)) {
            report_queue_error(queue);
            return;
        }
        if (input.opcode == L3L2QueueOpcode::STOP) {
            if (!finish_pending_inputs(queue, active)) {
                return;
            }
            if (!queue.input().release(input)) {
                report_queue_error(queue);
                return;
            }
            if (!queue.input().drained()) {
                rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example returned before drain");
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
        if (!process_data_message(queue, input, active)) {
            return;
        }
    }
}

}  // extern "C"
