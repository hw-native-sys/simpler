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
#include "spsc_queue.h"
#include "orchestration_api.h"  // NOLINT(build/include_subdir)
#include "aicpu/region_endpoint_diagnostics.h"

namespace {

constexpr int kExpectedArgCount = static_cast<int>(REGION_TASK_SCALAR_COUNT + spsc_queue::kSpscQueueConfigScalarCount);
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

struct RegionIdentity {
    uint64_t session_bits;
    uint64_t transaction_id;
    uint64_t region_id;
};

struct ActiveRequest {
    spsc_queue::SpscQueueInputHandle handle;
    InputHeader header;
    bool occupied;
};

uint64_t spsc_queue_now_ns() { return sys_cnt_ticks_to_ns(device_time_now_ticks(), device_time_frequency_hz()); }

void report_queue_failure(const QueueEndpoint &queue, RegionIdentity identity) {
    const RegionViewError *region_error = queue.region_error();
    if (region_error != nullptr) {
        region_report_endpoint_error(
            identity.session_bits, identity.transaction_id, identity.region_id, *region_error,
            queue.region_error_counter_addr(), queue.region_error_counter_operand()
        );
        return;
    }
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
    QueueEndpoint &queue, RegionIdentity identity, const spsc_queue::SpscQueueInputHandle &first,
    const spsc_queue::SpscQueueInputHandle &second, uint64_t request_id, uint64_t kind, uint64_t aux, InputWindowOp op,
    float scalar
) {
    uint64_t nbytes = kOutputHeaderBytes + kTileBytes;
    spsc_queue::SpscQueueOutputReservation output{};
    if (!queue.output().reserve(nbytes, kQueueTimeoutNs, output)) {
        report_queue_failure(queue, identity);
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
        report_queue_failure(queue, identity);
        return false;
    }
    return true;
}

bool release_input(QueueEndpoint &queue, RegionIdentity identity, const spsc_queue::SpscQueueInputHandle &input) {
    if (!queue.input().release(input)) {
        report_queue_failure(queue, identity);
        return false;
    }
    return true;
}

bool process_first_pair(
    QueueEndpoint &queue, RegionIdentity identity, ActiveRequest *active, const spsc_queue::SpscQueueInputHandle &input
) {
    InputHeader header{};
    if (!parse_input_header(input, &header) || !active[0].occupied) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "invalid L3-L2 queue example request");
        return false;
    }
    if (!publish_aiv_output(queue, identity, input, input, header.request_id, 20, 0, ADD_SCALAR, 20.0F)) {
        return false;
    }
    if (!release_input(queue, identity, input)) {
        return false;
    }
    if (!publish_aiv_output(
            queue, identity, active[0].handle, active[0].handle, active[0].header.request_id, 10, 0, ADD_SCALAR, 10.0F
        ) ||
        !publish_aiv_output(
            queue, identity, active[0].handle, active[0].handle, active[0].header.request_id, 11, 0, ADD_SCALAR, 11.0F
        )) {
        return false;
    }
    if (!release_input(queue, identity, active[0].handle)) {
        return false;
    }
    active[0] = {};
    return true;
}

bool remember_input_for_pair(
    ActiveRequest *active, const spsc_queue::SpscQueueInputHandle &input, const InputHeader &header
) {
    if (!active[2].occupied) {
        active[2].handle = input;
        active[2].header = header;
        active[2].occupied = true;
        return true;
    }
    if (!active[3].occupied) {
        active[3].handle = input;
        active[3].header = header;
        active[3].occupied = true;
        return true;
    }
    rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example input window is full");
    return false;
}

bool process_data_message(
    QueueEndpoint &queue, RegionIdentity identity, const spsc_queue::SpscQueueInputHandle &input, ActiveRequest *active
) {
    InputHeader header{};
    if (!parse_input_header(input, &header)) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "invalid L3-L2 queue example request");
        return false;
    }
    if (header.mode == 1) {
        if (active[0].occupied) {
            rt_report_fatal(
                SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example received mode=1 while a request is pending"
            );
            return false;
        }
        active[0].handle = input;
        active[0].header = header;
        active[0].occupied = true;
        return true;
    }
    if (header.mode == 2) {
        return process_first_pair(queue, identity, active, input);
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

bool finish_pending_inputs(QueueEndpoint &queue, RegionIdentity identity, ActiveRequest *active) {
    if (!active[2].occupied && !active[3].occupied) {
        return true;
    }
    if (!active[2].occupied || !active[3].occupied) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "L3-L2 queue example missing paired input");
        return false;
    }
    if (!publish_aiv_output(
            queue, identity, active[2].handle, active[3].handle, active[2].header.request_id, 30,
            active[3].header.request_id, ADD_TILES, 0.0F
        )) {
        return false;
    }
    if (!release_input(queue, identity, active[2].handle)) {
        return false;
    }
    active[2] = {};
    if (!release_input(queue, identity, active[3].handle)) {
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
    constexpr size_t kTaskScalarCount = REGION_TASK_SCALAR_COUNT + spsc_queue::kSpscQueueConfigScalarCount;
    uint64_t scalars[kTaskScalarCount];
    for (size_t i = 0; i < kTaskScalarCount; ++i) {
        scalars[i] = orch_args.scalar(static_cast<int32_t>(i));
    }
    RegionPartLocalSpan payload{};
    RegionPartLocalSpan counter{};
    uint64_t session_bits = 0;
    uint64_t transaction_id = 0;
    uint64_t region_id = 0;
    RegionTaskArgError decode_error = RegionTaskArgError::OK;
    if (!region_decode_task_args(
            scalars, REGION_TASK_SCALAR_COUNT, payload, counter, session_bits, transaction_id, region_id, decode_error
        )) {
        region_report_invalid_task_args(decode_error);
        return;
    }
    spsc_queue::SpscQueueConfig config{scalars[8], scalars[9], scalars[10]};
    RegionInstanceView view(payload, counter);
    spsc_queue::MonotonicClock clock{&spsc_queue_now_ns};
    QueueEndpoint queue(std::move(view), config, clock);
    RegionIdentity identity{session_bits, transaction_id, region_id};
    if (queue.region_error() != nullptr || !queue.live()) {
        report_queue_failure(queue, identity);
        return;
    }

    ActiveRequest active[kInputWindow]{};
    for (;;) {
        spsc_queue::SpscQueueInputHandle input{};
        if (!queue.input().peek(kQueueTimeoutNs, input)) {
            if (has_queue_error(queue)) {
                report_queue_failure(queue, identity);
                return;
            }
            continue;
        }
        if (input.opcode == spsc_queue::SpscQueueOpcode::STOP) {
            if (!finish_pending_inputs(queue, identity, active)) {
                return;
            }
            if (!queue.input().release(input)) {
                report_queue_failure(queue, identity);
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
        if (!process_data_message(queue, identity, input, active)) {
            return;
        }
    }
}

}  // extern "C"
