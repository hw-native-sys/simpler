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
#include <chrono>
#include <stdexcept>

#include "orchestration_api.h"
#include "platform_comm/ccu/ccu_device.h"
#include "../fault_handshake.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig fault_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}

extern "C" __attribute__((visibility("default"))) void fault_orchestration(const ChipTaskArgs &args) {
    const auto &source = args.tensor(0).ref();
    const auto &destination = args.tensor(1).ref();
    const auto &control = args.tensor(2).ref();
    const auto &results = args.tensor(3).ref();
    // Separate cache lines keep observation independent of the successor's result.
    const auto consumer_result = results.slice(0, 0, 16);
    const auto witness_results = results.slice(0, 16, results.shapes[0]);
    const uint64_t domain = args.scalar<uint64_t>(0);
    const uint64_t mismatch = args.scalar<uint64_t>(1);
    const uint64_t rank = args.scalar<uint64_t>(2);
    const uint64_t control_address = control.buffer.addr + control.start_offset * sizeof(uint32_t);
    const int32_t width = rt_available_aiv_count();
    const uint32_t submitted_index[] = {ccu_fault::kSubmitted};
    const uint32_t width_index[] = {ccu_fault::kWitnessWidth};
    const uint32_t successor_index[] = {ccu_fault::kSuccessorPublished};
    if (width <= 0 || width > static_cast<int32_t>(ccu_fault::kMaxAiv)) {
        throw std::runtime_error("CCU fault handshake has invalid witness width");
    }
    set_tensor_data(control, 1, width_index, static_cast<uint32_t>(width));
    ccu_detail::Sync(reinterpret_cast<void *>(control_address + width_index[0] * sizeof(uint32_t)), 64);
    SIMPLER_SCOPE() {
        CoreTaskArgs send;
        send.add_input(source);
        send.add_inout(destination);
        send.add_scalar(domain);
        // Control flags are a test handshake, deliberately outside task dependencies.
        send.add_scalar(control_address);
        send.add_scalar(mismatch);
        rt_submit_aiv_task(0, send);

        if (rank == 0 || mismatch == 0) {
            // The ready sync-start queue may outrank ordinary tasks. Admit the cohort
            // only after send has executed, without waiting for its deferred completion.
            const auto started = std::chrono::steady_clock::now();
            while (get_tensor_data<uint32_t>(control, 1, submitted_index) == 0) {
                if (std::chrono::steady_clock::now() - started >=
                    std::chrono::nanoseconds(ccu_fault::kHandshakeTimeoutNs)) {
                    throw std::runtime_error("CCU fault handshake did not observe request submission");
                }
            }
            if (get_tensor_data<uint32_t>(control, 1, submitted_index) != ccu_fault::kReady) {
                throw std::runtime_error("CCU fault handshake observed a rejected request");
            }
            CoreTaskArgs witness;
            witness.add_inout(witness_results);
            witness.add_scalar(domain);
            witness.add_scalar(control_address);
            witness.launch_spec.set_block_num(static_cast<int16_t>(width));
            witness.launch_spec.set_require_sync_start(true);
            rt_submit_aiv_task(1, witness);
        }

        // Failure observation requires the successor to be published before the peer submits.
        CoreTaskArgs consume;
        consume.add_input(destination);
        consume.add_inout(consumer_result);
        consume.add_scalar(control_address);
        rt_submit_aiv_task(2, consume);
        set_tensor_data(control, 1, successor_index, ccu_fault::kReady);
        ccu_detail::Sync(reinterpret_cast<void *>(control_address + successor_index[0] * sizeof(uint32_t)), 64);
    }
}
