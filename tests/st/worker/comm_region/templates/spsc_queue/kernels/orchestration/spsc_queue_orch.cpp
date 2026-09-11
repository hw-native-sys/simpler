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
constexpr uint32_t kComputeFuncId = 0;
constexpr uint64_t kQueueTimeoutNs = 5000000000ULL;
constexpr uint64_t kCmdEcho = 1;
constexpr uint64_t kCmdError = 2;
constexpr uint64_t kCmdCompute = 3;
constexpr uint64_t kHeaderBytes = 8;
constexpr uint32_t kTileRows = 128;
constexpr uint32_t kTileCols = 128;
constexpr uint64_t kTileBytes = static_cast<uint64_t>(kTileRows) * kTileCols * sizeof(float);
constexpr uint64_t kComputeBytes = kHeaderBytes + kTileBytes;
constexpr uint64_t kPostStopMarker = 0x11;
constexpr float kComputeScalar = 1.0F;

using QueueEndpoint = spsc_queue::SpscQueueEndpoint<RegionInstanceView>;

uint64_t spsc_queue_now_ns() { return sys_cnt_ticks_to_ns(device_time_now_ticks(), device_time_frequency_hz()); }

void report_queue_error(const QueueEndpoint &queue) {
    rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "%s", queue.error().message);
}

bool has_queue_error(const QueueEndpoint &queue) { return queue.error().kind != spsc_queue::SpscQueueErrorKind::NONE; }

bool copy_and_publish(
    QueueEndpoint &queue, const spsc_queue::SpscQueueInputHandle &input, spsc_queue::SpscQueueOpcode opcode
) {
    spsc_queue::SpscQueueOutputReservation output{};
    if (!queue.output().reserve(input.payload_nbytes, kQueueTimeoutNs, output)) {
        report_queue_error(queue);
        return false;
    }
    if (input.payload_nbytes != 0) {
        memcpy(
            reinterpret_cast<void *>(static_cast<uintptr_t>(output.payload.local_addr)),
            reinterpret_cast<const void *>(static_cast<uintptr_t>(input.payload.local_addr)), input.payload_nbytes
        );
        cache_flush_range(
            reinterpret_cast<void *>(static_cast<uintptr_t>(output.payload.local_addr)), input.payload_nbytes
        );
    }
    if (!queue.output().publish(output, opcode)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

bool publish_compute(QueueEndpoint &queue, const spsc_queue::SpscQueueInputHandle &input) {
    if (input.payload_nbytes != kComputeBytes) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "spsc queue ST compute payload size mismatch");
        return false;
    }
    spsc_queue::SpscQueueOutputReservation output{};
    if (!queue.output().reserve(kComputeBytes, kQueueTimeoutNs, output)) {
        report_queue_error(queue);
        return false;
    }

    uint8_t *dst = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(output.payload.local_addr));
    memcpy(dst, reinterpret_cast<const void *>(static_cast<uintptr_t>(input.payload.local_addr)), kHeaderBytes);

    uint32_t shape[2] = {kTileRows, kTileCols};
    void *src_tile = reinterpret_cast<void *>(static_cast<uintptr_t>(input.payload.local_addr + kHeaderBytes));
    void *dst_tile = dst + kHeaderBytes;
    simpler::tmr::Tensor in_tensor = simpler::tmr::make_tensor_external(src_tile, shape, 2, DataType::FLOAT32);
    simpler::tmr::Tensor out_tensor = simpler::tmr::make_tensor_external(dst_tile, shape, 2, DataType::FLOAT32);

    CoreTaskArgs params;
    params.add_input(in_tensor);
    params.add_output(out_tensor);
    params.add_scalar(to_u64<float>(kComputeScalar));
    rt_submit_aiv_task(kComputeFuncId, params);

    uint32_t first_index[2] = {0, 0};
    (void)get_tensor_data<float>(out_tensor, 2, first_index);
    cache_flush_range(dst, kComputeBytes);

    if (!queue.output().publish(output, spsc_queue::SpscQueueOpcode::DATA)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

bool publish_post_stop(QueueEndpoint &queue) {
    spsc_queue::SpscQueueOutputReservation output{};
    if (!queue.output().reserve(kHeaderBytes, kQueueTimeoutNs, output)) {
        report_queue_error(queue);
        return false;
    }
    uint64_t marker = kPostStopMarker;
    memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(output.payload.local_addr)), &marker, kHeaderBytes);
    cache_flush_range(reinterpret_cast<void *>(static_cast<uintptr_t>(output.payload.local_addr)), kHeaderBytes);
    if (!queue.output().publish(output, spsc_queue::SpscQueueOpcode::DATA)) {
        report_queue_error(queue);
        return false;
    }
    return true;
}

bool handle_data(QueueEndpoint &queue, const spsc_queue::SpscQueueInputHandle &input) {
    if (input.payload_nbytes == 0) {
        return copy_and_publish(queue, input, spsc_queue::SpscQueueOpcode::DATA);
    }
    if (input.payload_nbytes < kHeaderBytes) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "spsc queue ST data payload is shorter than the command");
        return false;
    }
    uint64_t command = 0;
    memcpy(&command, reinterpret_cast<const void *>(static_cast<uintptr_t>(input.payload.local_addr)), kHeaderBytes);
    if (command == kCmdError) {
        return copy_and_publish(queue, input, spsc_queue::SpscQueueOpcode::ERROR);
    }
    if (command == kCmdCompute) {
        return publish_compute(queue, input);
    }
    if (command != kCmdEcho) {
        rt_report_fatal(
            SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "spsc queue ST unexpected command=%llu",
            static_cast<unsigned long long>(command)
        );
        return false;
    }
    return copy_and_publish(queue, input, spsc_queue::SpscQueueOpcode::DATA);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void spsc_queue_orchestration(const ChipTaskArgs &orch_args) {
    uint64_t scalars[spsc_queue::kSpscQueueEndpointBindingScalarCount];
    for (size_t i = 0; i < spsc_queue::kSpscQueueEndpointBindingScalarCount; ++i) {
        scalars[i] = orch_args.scalar(static_cast<int32_t>(i));
    }
    spsc_queue::SpscQueueEndpointBinding binding{};
    if (!spsc_queue::decode_endpoint_binding(scalars, spsc_queue::kSpscQueueEndpointBindingScalarCount, &binding)) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "invalid queue binding");
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
            if (!publish_post_stop(queue)) {
                return;
            }
            if (!queue.input().release(input)) {
                report_queue_error(queue);
                return;
            }
            if (!queue.input().drained()) {
                rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "spsc queue ST returned before input drain");
            }
            return;
        }
        if (input.opcode == spsc_queue::SpscQueueOpcode::ERROR) {
            if (!copy_and_publish(queue, input, spsc_queue::SpscQueueOpcode::ERROR)) {
                return;
            }
        } else if (input.opcode == spsc_queue::SpscQueueOpcode::DATA) {
            if (!handle_data(queue, input)) {
                return;
            }
        } else {
            rt_report_fatal(
                SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "spsc queue ST unexpected input opcode=%llu",
                static_cast<unsigned long long>(input.opcode)
            );
            return;
        }
        if (!queue.input().release(input)) {
            report_queue_error(queue);
            return;
        }
    }
}

}  // extern "C"
