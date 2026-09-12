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

#include <cstddef>
#include <cstdint>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/core_type.h"
#include "common/memory_barrier.h"
#include "tmr_kernel_control.h"

namespace simpler::tmr {

struct KernelHandshakeView {
    TmrLaunchControl *control{nullptr};
    TmrCoreReport *reports{nullptr};
    int32_t worker_count{0};
    uint64_t epoch{0};
};

// One run's borrowed control region and register mappings. The prepare provider
// owns/pins every address. Attach, report collection and finish have one owner;
// open is partitioned by worker index and finishes before cancel or dispatch.
// This class never clears a Host-owned region or infers readiness from an SPR.
class KernelCoreGroup {
public:
    bool attach(KernelHandshakeView view) noexcept {
        if (view.control == nullptr || view.reports == nullptr || view.epoch == 0 || view.worker_count <= 0 ||
            view.worker_count > PLATFORM_MAX_CORES ||
            reinterpret_cast<uintptr_t>(view.control) % alignof(TmrLaunchControl) != 0 ||
            reinterpret_cast<uintptr_t>(view.reports) % alignof(TmrCoreReport) != 0)
            return false;
        const uintptr_t control = reinterpret_cast<uintptr_t>(view.control);
        const uintptr_t reports = reinterpret_cast<uintptr_t>(view.reports);
        const size_t reports_bytes = static_cast<size_t>(view.worker_count) * sizeof(TmrCoreReport);
        if (sizeof(TmrLaunchControl) > UINTPTR_MAX - control || reports_bytes > UINTPTR_MAX - reports ||
            !(control + sizeof(TmrLaunchControl) <= reports || reports + reports_bytes <= control))
            return false;
        view_ = view;
        for (int32_t i = 0; i < view.worker_count; ++i) {
            regs_[i] = 0;
            physical_ids_[i] = 0;
            types_[i] = CoreType::AIC;
            opened_[i] = false;
        }
        return true;
    }

    // Caller supplies a prepare-validated platform register table. Validate all
    // reports (including uniqueness) before opening any window: an invalid or
    // duplicate physical ID must not cause MMIO to an unowned core.
    int32_t collect_reports(const uint64_t *registers, uint32_t physical_count) noexcept {
        if (registers == nullptr || physical_count == 0) return -1;
        auto &control = *view_.control;
        cache_invalidate_range(&control, sizeof(control));
        if (load(control.host_cancel) != 0 || load(control.completion) != 0 || control.runtime_status != 0 ||
            control.cleanup_status != 0 || control.round_epoch != 0)
            return -1;
        const uint64_t deadline = platform_aicore_exit_deadline();
        int32_t aic_count = 0;
        for (int32_t i = 0; i < view_.worker_count; ++i) {
            auto &report = view_.reports[i];
            for (;;) {
                cache_invalidate_range(&report, sizeof(report));
                if (load(report.command) != 0 || load(report.release) != 0 || report.round_epoch != 0 ||
                    load(report.exited) != 0 || load(control.host_cancel) != 0)
                    return -1;
                if (load(report.ready) == static_cast<uint32_t>(i + 1)) break;
                if (get_sys_cnt_aicpu() > deadline) return -1;
            }
            rmb();
            const uint32_t pcid = report.physical_core_id;
            const uint32_t type = report.core_type;
            if (pcid >= physical_count || registers[pcid] == 0 ||
                (type != static_cast<uint32_t>(CoreType::AIC) && type != static_cast<uint32_t>(CoreType::AIV)))
                return -1;
            for (int32_t j = 0; j < i; ++j)
                if (physical_ids_[j] == pcid) return -1;
            physical_ids_[i] = pcid;
            types_[i] = static_cast<CoreType>(type);
            if (types_[i] == CoreType::AIC) ++aic_count;
            regs_[i] = registers[pcid];
        }
        // Scheduler cluster assignment requires one AIC and two AIV workers.
        return view_.worker_count % 3 == 0 && aic_count == view_.worker_count / 3 ? 0 : -1;
    }

    uint64_t register_address(int32_t index) const noexcept { return regs_[index]; }
    uint32_t physical_id(int32_t index) const noexcept { return physical_ids_[index]; }
    CoreType core_type(int32_t index) const noexcept { return types_[index]; }

    void open(int32_t index) noexcept {
        platform_init_aicore_regs(regs_[index]);
        (void)read_reg(regs_[index], RegId::DATA_MAIN_BASE);
        rmb();
        auto &report = view_.reports[index];
        // A nonzero epoch witnesses completed window-open even if the core
        // observes a later CANCEL without ever observing OPEN itself.
        report.round_epoch = view_.epoch;
        store(report.command, 1);
        cache_flush_range(&report.command, sizeof(report) - offsetof(TmrCoreReport, command));
        opened_[index] = true;
    }

    void cancel() noexcept {
        for (int32_t i = 0; i < view_.worker_count; ++i) {
            auto &report = view_.reports[i];
            if (!opened_[i]) __atomic_store_n(&report.round_epoch, uint64_t{0}, __ATOMIC_RELAXED);
            store(report.command, 2);
            cache_flush_range(&report.command, sizeof(report) - offsetof(TmrCoreReport, command));
        }
    }

    // Called after all CPU consumers stop, including all failed initializers.
    // Unopened cores observe CANCEL without accessing registers. Opened cores
    // acknowledge via report and COND, then wait until their window is closed.
    // A timeout retains the unreleased window/storage for fatal recovery; it
    // must not be converted into a successful reusable round.
    int32_t finish() noexcept {
        cancel();
        for (int32_t i = 0; i < view_.worker_count; ++i)
            if (opened_[i]) write_reg(regs_[i], RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        wmb();
        const uint64_t deadline = platform_aicore_exit_deadline();
        bool acknowledged[PLATFORM_MAX_CORES]{};
        int32_t remaining = view_.worker_count;
        while (remaining != 0) {
            for (int32_t i = 0; i < view_.worker_count; ++i) {
                if (acknowledged[i]) continue;
                auto &report = view_.reports[i];
                cache_invalidate_range(&report, offsetof(TmrCoreReport, command));
                if (load(report.exited) != static_cast<uint32_t>(i + 1)) continue;
                if (opened_[i] && read_reg(regs_[i], RegId::COND) != AICORE_EXITED_VALUE) continue;
                acknowledged[i] = true;
                --remaining;
            }
            if (remaining == 0 || get_sys_cnt_aicpu() > deadline) break;
        }
        for (int32_t i = 0; i < view_.worker_count; ++i)
            if (opened_[i] && acknowledged[i]) platform_close_aicore_window(regs_[i]);
        rmb();
        for (int32_t i = 0; i < view_.worker_count; ++i) {
            if (!opened_[i] || !acknowledged[i]) continue;
            auto &report = view_.reports[i];
            store(report.release, 1);
            cache_flush_range(&report.command, sizeof(report) - offsetof(TmrCoreReport, command));
        }
        return remaining == 0 ? 0 : -1;
    }

    void publish_status(int32_t runtime, int32_t cleanup) noexcept {
        auto &control = *view_.control;
        control.runtime_status = runtime;
        control.cleanup_status = cleanup;
        control.round_epoch = view_.epoch;
        store(control.completion, 1);
        cache_flush_range(&control.runtime_status, sizeof(control) - offsetof(TmrLaunchControl, runtime_status));
    }

private:
    static uint32_t load(const uint32_t &word) noexcept { return __atomic_load_n(&word, __ATOMIC_ACQUIRE); }
    static void store(uint32_t &word, uint32_t value) noexcept { __atomic_store_n(&word, value, __ATOMIC_RELEASE); }

    KernelHandshakeView view_{};
    uint64_t regs_[PLATFORM_MAX_CORES]{};
    uint32_t physical_ids_[PLATFORM_MAX_CORES]{};
    CoreType types_[PLATFORM_MAX_CORES]{};
    bool opened_[PLATFORM_MAX_CORES]{};
};

}  // namespace simpler::tmr
