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
#include <cstring>

#include "../protocol.h"
#include "task_interface/tmr_kernel_context.h"
#include "tensormap_and_ringbuffer/kernel_core_group.h"

namespace {
control_probe::Init prepared{};
control_probe::Result result{};
bool failed = false;

void publish_result() {
    auto *destination = reinterpret_cast<void *>(prepared.result);
    std::memcpy(destination, &result, sizeof(result));
    cache_flush_range(destination, sizeof(result));
}
}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    if (args == nullptr || prepared.descriptor != 0) return 1;
    control_probe::Init candidate{};
    std::memcpy(&candidate, args, sizeof(candidate));
    if (candidate.descriptor == 0 || candidate.registers == 0 || candidate.result == 0) return 1;
    const auto *descriptor = reinterpret_cast<const simpler::tmr::TmrKernelContextDescriptor *>(candidate.descriptor);
    cache_invalidate_range(descriptor, sizeof(*descriptor));
    if (descriptor->self_address != candidate.descriptor ||
        descriptor->context_generation != control_probe::kContextGeneration ||
        descriptor->worker_count != control_probe::kWorkers)
        return 1;
    prepared = candidate;
    publish_result();
    return 0;
}

// Only run with the entire device exclusively held and both branches complete.
extern "C" __attribute__((visibility("default"))) int control_probe_seed_registers(void *) {
    if (prepared.descriptor == 0 || failed) return 1;
    const auto *registers = reinterpret_cast<const uint64_t *>(prepared.registers);
    const uint32_t count = platform_get_physical_cores_count();
    result.seeded_registers = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (registers[i] == 0) continue;
        write_reg(registers[i], RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        if (read_reg(registers[i], RegId::DATA_MAIN_BASE) != AICORE_EXIT_SIGNAL) return 1;
        ++result.seeded_registers;
    }
    rmb();
    publish_result();
    return result.seeded_registers != 0 ? 0 : 1;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *args) {
    using namespace simpler::tmr;
    if (prepared.descriptor == 0 || failed || args == nullptr) return 1;
    control_probe::Invocation invocation{};
    std::memcpy(&invocation, args, sizeof(invocation));
    const auto &descriptor = *reinterpret_cast<const TmrKernelContextDescriptor *>(prepared.descriptor);
    KernelCoreGroup cores;
    if (!cores.attach(
            {reinterpret_cast<TmrLaunchControl *>(descriptor.control_address),
             reinterpret_cast<TmrCoreReport *>(descriptor.reports_address), descriptor.worker_count, ++result.epoch}
        ))
        return 1;
    result.opened = 0;
    int32_t status = cores.collect_reports(
        reinterpret_cast<const uint64_t *>(prepared.registers), platform_get_physical_cores_count()
    );
    if (status == 0 && invocation.mode == control_probe::Mode::Reject) status = control_probe::kAdmissionRejected;
    if (status == 0 && (invocation.reserved != 0 || invocation.mode == control_probe::Mode::HostCancel)) status = -1;
    if (status == 0) {
        for (int32_t i = 0; i < descriptor.worker_count; ++i) {
            cores.open(i);
            ++result.opened;
        }
        const uint64_t deadline = platform_aicore_exit_deadline();
        for (int32_t i = 0; i < descriptor.worker_count; ++i) {
            while (read_reg(cores.register_address(i), RegId::COND) != AICORE_IDLE_VALUE) {
                if (get_sys_cnt_aicpu() > deadline) {
                    status = -1;
                    break;
                }
            }
        }
    }
    result.cleanup_status = cores.finish();
    result.runtime_status = status;
    ++result.completed_rounds;
    cores.publish_status(status, result.cleanup_status);
    publish_result();
    failed = result.cleanup_status != 0;
    // Logical rejection is reported in control; a transport error poisons CANN.
    return failed ? 1 : 0;
}
