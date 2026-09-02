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
#include <utility>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "aicpu/region_instance_view.h"
#include "common/region_template.h"
#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int kExpectedArgCount = static_cast<int>(spsc_queue::kSpscQueueEndpointBindingScalarCount);
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

// The queue seq is transport ordering only. Payload headers carry the
// application-level request correlation needed for out-of-order and
// many-to-one outputs in this example.
struct InputHeader {
    uint64_t request_id;
    uint64_t mode;
};

struct OutputHeader {
    uint64_t request_id;
    uint64_t kind;
    uint64_t aux;
};

using QueueEndpoint = spsc_queue::SpscQueueEndpoint<RegionInstanceView, kInputWindow>;

struct ActiveRequest {
    spsc_queue::SpscQueueInputHandle handle;
    InputHeader header;
};

uint64_t spsc_queue_now_ns() { return sys_cnt_ticks_to_ns(device_time_now_ticks(), device_time_frequency_hz()); }

void report_queue_error(const QueueEndpoint &queue) {
    rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "%s", queue.error().message);
}

bool has_queue_error(const QueueEndpoint &queue) { return queue.error().kind != spsc_queue::SpscQueueErrorKind::NONE; }

bool parse_input_header(const spsc_queue::SpscQueueInputHandle &input, InputHeader *header) {
    if (header == nullptr || input.payload_nbytes != kInputHeaderBytes + kTileBytes) {
        return false;
    }
    memcpy(header, reinterpret_cast<const void *>(static_cast<uintptr_t>(input.payload.local_addr)), sizeof(*header));
    return true;
}

simpler::tmr::Tensor make_input_values_tensor(const spsc_queue::SpscQueueInputHandle &input) {
    uint32_t shape[2] = {kTileRows, kTileCols};
    void *values = reinterpret_cast<void *>(static_cast<uintptr_t>(input.payload.local_addr + kInputHeaderBytes));
    return simpler::tmr::make_tensor_external(values, shape, 2, DataType::FLOAT32);
}

bool publish_aiv_output(
    QueueEndpoint &queue, const spsc_queue::SpscQueueInputHandle &first, const spsc_queue::SpscQueueInputHandle &second,
    uint64_t request_id, uint64_t kind, uint64_t aux, InputWindowOp op, float scalar
) {
    uint64_t nbytes = kOutputHeaderBytes + kTileBytes;
    spsc_queue::SpscQueueOutputReservation output{};
    if (!queue.output().reserve(nbytes, kQueueTimeoutNs, output)) {
        report_queue_error(queue);
        return false;
    }

    OutputHeader header{request_id, kind, aux};
    uint8_t *dst = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(output.payload.local_addr));
    memset(dst, 0, kOutputHeaderBytes);
    memcpy(dst, &header, sizeof(header));

    simpler::tmr::Tensor first_tensor = make_input_values_tensor(first);
    simpler::tmr::Tensor second_tensor = make_input_values_tensor(second);
    uint32_t output_shape[2] = {kTileRows, kTileCols};
    simpler::tmr::Tensor output_tensor =
        simpler::tmr::make_tensor_external(dst + kOutputHeaderBytes, output_shape, 2, DataType::FLOAT32);

    CoreTaskArgs params;
    params.add_input(first_tensor);
    params.add_input(second_tensor);
    params.add_output(output_tensor);
    params.add_scalar(static_cast<uint64_t>(op));
    params.add_scalar(to_u64<float>(scalar));
    rt_submit_aiv_task(kInputWindowComputeFuncId, params);

    uint32_t first_output_index[2] = {0, 0};
    (void)get_tensor_data<float>(output_tensor, 2, first_output_index);
    cache_flush_range(dst, nbytes);

    if (!queue.output().publish(output, spsc_queue::SpscQueueOpcode::DATA)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

bool release_input(QueueEndpoint &queue, const spsc_queue::SpscQueueInputHandle &input) {
    if (!queue.input().release(input)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

bool process_first_pair(QueueEndpoint &queue, ActiveRequest *active, const spsc_queue::SpscQueueInputHandle &input) {
    active[1].handle = input;
    if (!parse_input_header(input, &active[1].header) || active[0].header.request_id == 0) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "invalid L3-L2 queue example request");
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
    active[1] = {};
    if (!publish_aiv_output(
            queue, active[0].handle, active[0].handle, active[0].header.request_id, 10, 0, ADD_SCALAR, 10.0F
        ) ||
        !publish_aiv_output(
            queue, active[0].handle, active[0].handle, active[0].header.request_id, 11, 0, ADD_SCALAR, 11.0F
        )) {
        return false;
    }
    if (!release_input(queue, active[0].handle)) {
        return false;
    }
    active[0] = {};
    return true;
}

bool remember_input_for_pair(
    ActiveRequest *active, const spsc_queue::SpscQueueInputHandle &input, const InputHeader &header
) {
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
    rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example input window is full");
    return false;
}

bool process_data_message(QueueEndpoint &queue, const spsc_queue::SpscQueueInputHandle &input, ActiveRequest *active) {
    InputHeader header{};
    if (!parse_input_header(input, &header)) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "invalid L3-L2 queue example request");
        return false;
    }
    if (header.mode == 1) {
        if (active[0].header.request_id != 0) {
            rt_report_fatal(
                SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example received mode=1 while a request is pending"
            );
            return false;
        }
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
        SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example unexpected mode=%llu",
        static_cast<unsigned long long>(header.mode)
    );
    return false;
}

bool finish_pending_inputs(QueueEndpoint &queue, ActiveRequest *active) {
    if (active[2].header.request_id == 0 && active[3].header.request_id == 0) {
        return true;
    }
    if (active[2].header.request_id == 0 || active[3].header.request_id == 0) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example missing paired input");
        return false;
    }
    if (!publish_aiv_output(
            queue, active[2].handle, active[3].handle, active[2].header.request_id, 30, active[3].header.request_id,
            ADD_TILES, 0.0F
        )) {
        return false;
    }
    if (!release_input(queue, active[2].handle)) {
        return false;
    }
    active[2] = {};
    if (!release_input(queue, active[3].handle)) {
        return false;
    }
    active[3] = {};
    return true;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void worker_chip_message_queue_orchestration(const ChipTaskArgs &orch_args) {
    uint64_t scalars[spsc_queue::kSpscQueueEndpointBindingScalarCount];
    for (size_t i = 0; i < spsc_queue::kSpscQueueEndpointBindingScalarCount; ++i) {
        scalars[i] = orch_args.scalar(static_cast<int32_t>(i));
    }
    spsc_queue::SpscQueueEndpointBinding binding{};
    if (!spsc_queue::decode_endpoint_binding(scalars, spsc_queue::kSpscQueueEndpointBindingScalarCount, &binding)) {
        rt_report_fatal(
            SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "SPSC queue endpoint error op=init kind=2 msg=invalid queue binding"
        );
        return;
    }
    RegionInstanceView view(
        RegionPartLocalSpan{binding.payload_base, binding.payload_bytes},
        RegionPartLocalSpan{binding.counter_base, binding.counter_bytes}
    );
    spsc_queue::MonotonicClock clock{&spsc_queue_now_ns};
    QueueEndpoint queue(binding, std::move(view), clock);
    if (!queue.live()) {
        report_queue_error(queue);
        return;
    }

    ActiveRequest active[kInputWindow]{};
    for (;;) {
        spsc_queue::SpscQueueInputHandle input{};
        if (!queue.input().peek(kQueueTimeoutNs, input)) {
            if (has_queue_error(queue)) {
                report_queue_error(queue);
                return;
            }
            continue;
        }
        if (input.opcode == spsc_queue::SpscQueueOpcode::STOP) {
            if (!finish_pending_inputs(queue, active)) {
                return;
            }
            if (!queue.input().release(input)) {
                report_queue_error(queue);
                return;
            }
            if (!queue.input().drained()) {
                rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example returned before drain");
            }
            return;
        }
        if (input.opcode != spsc_queue::SpscQueueOpcode::DATA) {
            rt_report_fatal(
                SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example unexpected input opcode=%llu",
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
