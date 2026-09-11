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

#pragma once

#include "kernel_execution.h"

namespace simpler::tmr {

// The two architecture executors supply the same cold-path operations. This
// coordinator owns publication/retirement, not scheduling or resource ownership.
template <typename Executor>
int32_t execute_kernel_round_impl(
    Executor &executor, const KernelExecutionRequest &request, int32_t cpu, KernelFinalStatus *out = nullptr
) noexcept {
    // These are trusted prepare-time topology values, checked before launch.
    if (request.allowed_cpus == nullptr || request.execution_threads <= 0 ||
        request.execution_threads > request.launched_threads || request.launched_threads > MAX_GATE_THREADS)
        return -1;
    for (int32_t i = 0; i < request.execution_threads; ++i) {
        if (request.allowed_cpus[i] < 0) return -1;
        for (int32_t j = 0; j < i; ++j)
            if (request.allowed_cpus[i] == request.allowed_cpus[j]) return -1;
    }
    auto &gate = executor.kernel_gate_;
    KernelRoundTicket ticket;
    if (!gate.join(request.launched_threads, cpu, &ticket)) return -1;
    if (ticket.launch_index == 0) {
        auto handshake = request.handshake;
        handshake.epoch = ticket.epoch;
        executor.kernel_control_attached_ = executor.kernel_cores_.attach(handshake);
        int32_t status = request.admission_status;
        if (!executor.kernel_control_attached_) status = -1;
        if (status == 0) {
            try {
                status = executor.prepare_kernel_round(request);
            } catch (...) {
                status = -1;
            }
        }
        if (!gate.publish_admission(ticket, request.allowed_cpus, request.execution_threads, status)) return -1;
    }
    KernelRoundAdmission admission;
    if (!gate.wait_admission(ticket, &admission)) return -1;
    int32_t status = admission.status;
    const KernelThreadView thread{admission.execution_index, request.execution_threads};
    if (status == 0) {
        if (thread.execution_index >= 0) {
            int32_t initialized;
            try {
                initialized = executor.initialize_kernel_thread(thread);
            } catch (...) {
                initialized = -1;
            }
            if (!gate.report_init(ticket, initialized)) return -1;
        }
        if (ticket.launch_index == 0 && !gate.publish_init_verdict(ticket, [&]() noexcept {
                try {
                    return executor.complete_kernel_init();
                } catch (...) {
                    return int32_t{-1};
                }
            }))
            return -1;
        if (!gate.wait_init_verdict(ticket, &status)) return -1;
        if (status == 0 && thread.execution_index >= 0) {
            const auto &invocation = executor.kernel_invocation_;
            try {
                status = executor.run(invocation.resident(), invocation.inputs(), &thread);
            } catch (...) {
                status = -1;
            }
            if (status != 0) executor.cancel_kernel_round();
        }
    }
    const auto arrival = gate.arrive(ticket, status);
    if (arrival == RoundArrival::Invalid) return -1;
    KernelFinalStatus result;
    if (arrival == RoundArrival::Finalizer) {
        // The SM error is saved before Runtime destruction changes its views.
        const int32_t sm_status = executor.kernel_status();
        int32_t cleanup = executor.kernel_control_attached_ ? executor.kernel_cores_.finish() : -1;
        if (cleanup == 0) {
            try {
                cleanup = executor.finalize_kernel_round();
            } catch (...) {
                cleanup = -1;
            }
        }
        if (!gate.publish_final_status(
                ticket, sm_status, cleanup,
                [&](const KernelFinalStatus &final) noexcept {
                    if (executor.kernel_control_attached_)
                        executor.kernel_cores_.publish_status(final.runtime_status, final.cleanup_status);
                }
            ) ||
            !gate.read_final_status(ticket, &result))
            return -1;
    } else if (!gate.read_final_status(ticket, &result)) {
        return -1;
    }
    if (out != nullptr) *out = result;
    const auto departure = gate.depart(ticket);
    if (departure == RoundDeparture::Invalid) return -1;
    if (departure == RoundDeparture::Last && result.cleanup_status == 0) {
        executor.clear_kernel_round();
        if (!gate.complete_departure(ticket)) return -1;
    }
    // Cleanup failure retains the retiring gate and every borrowed argument;
    // a timeout is not proof that an AICore stopped reading the storage.
    return result.cleanup_status != 0 ? result.cleanup_status : result.runtime_status;
}

}  // namespace simpler::tmr
