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
/**
 * @file device_run_result_aicpu.h
 * @brief AICPU-side publication of one run's terminal record.
 *
 * See `common/device_run_result.h` for why the region exists. The publisher's
 * obligations, all of which this helper carries:
 *
 *   - Run **once per run, from the run's sole cleanup owner**, after the last
 *     work that could still change the run's outcome, so the record is the
 *     run's final state rather than one thread's view of it.
 *   - Run **before the kernel returns**, so the write precedes the fence that
 *     releases any successor to reset the shared state it was folded from.
 *   - Write back the record on A2/A3 for the host's D2H read. A5 is coherent;
 *     the host reads only after the run's kernel completion boundaries, so
 *     the epoch is an identity check, not a flag polled during publication.
 *
 * A record is refused rather than clamped or repaired. A payload that outgrew
 * the capacity, a success carrying a code, and a failure carrying no code or no
 * known code source are all producer defects; publishing a corrected version of
 * one would hide it, and refusing leaves the region holding some earlier run's
 * epoch, which the host reads as undecided. The accepted shapes are the ones
 * `device_run_result_terminal` will decide, so nothing this publishes can read
 * back as undecided.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/device_run_result.h"
#include "common/run_terminal_accumulator.h"

// Call only after the prior writers have physically exited and their required
// evidence has been retained outside this region. Replayed bindings can carry
// the same epoch: invalidate before any new producer can publish under it.
inline bool aicpu_invalidate_run_terminal(uint64_t region_base, uint64_t run_epoch) {
    if (region_base == 0 && run_epoch == 0) return true;
    if (region_base == 0 || run_epoch == 0) return false;
    auto *region = reinterpret_cast<DeviceRunResultRegion *>(region_base);
    region->published = 0;
    cache_flush_range(&region->published, sizeof(region->published));
    return true;
}

/**
 * Publish this run's terminal record into the region at `region_base` under
 * `run_epoch`.
 *
 * `payload` is optional diagnostic detail for a failure; success publishes none.
 * Returns whether the record was published, so a caller can log a refusal
 * rather than leave the host to infer one from an undecided read.
 */
inline bool aicpu_publish_run_terminal(
    uint64_t region_base, uint64_t run_epoch, DeviceRunVerdict verdict, int32_t completion_code,
    DeviceRunCodeSource code_source, const void *payload, size_t payload_bytes
) {
    if (region_base == 0 || run_epoch == 0) return false;
    if (payload_bytes > DEVICE_RUN_RESULT_PAYLOAD_BYTES) return false;
    if (payload == nullptr) payload_bytes = 0;

    if (verdict == DeviceRunVerdict::Ok) {
        if (completion_code != 0 || code_source != DeviceRunCodeSource::None || payload_bytes != 0) return false;
    } else if (verdict == DeviceRunVerdict::Error) {
        if (completion_code == 0 || !device_run_code_source_is_failure(code_source)) return false;
    } else {
        return false;
    }

    auto *region = reinterpret_cast<DeviceRunResultRegion *>(region_base);
    if (payload_bytes != 0) {
        std::memcpy(region->payload, payload, payload_bytes);
    }
    region->payload_bytes = static_cast<uint32_t>(payload_bytes);
    region->verdict = static_cast<uint32_t>(verdict);
    region->completion_code = completion_code;
    region->code_source = static_cast<uint32_t>(code_source);
    // The host reads after kernel completion, with the slot still reserved.
    // A2/A3 needs writeback; A5 needs no additional publication barrier here.
    // The header words are contiguous from `payload_bytes`.
    if (payload_bytes != 0) {
        cache_flush_range(region->payload, payload_bytes);
    }
    cache_flush_range(&region->payload_bytes, offsetof(DeviceRunResultRegion, payload) - sizeof(region->published));
    region->published = run_epoch;
    cache_flush_range(&region->published, sizeof(region->published));
    return true;
}

/**
 * Carries a run's terminal record across the gap the four executors share:
 * the record is decided where the runtime's shared state is still valid, and
 * published where the run's teardown is already finished.
 *
 * `take` snapshots the runtime result before teardown. The runtime cleanup
 * owner calls `handoff` after teardown; the native completion owner publishes
 * only after every callback's work and local destructors finish. The runtime
 * rendezvous makes this snapshot visible to its cleanup owner.
 *
 * Both handoff and the standalone `publish` helper consume the snapshot.
 */
class RunTerminalPublisher {
public:
    void take(RunTerminalSelection selection, const void *payload, size_t payload_bytes) {
        selection_ = selection;
        payload_bytes_ = 0;
        if (selection.verdict != DeviceRunVerdict::Error || payload == nullptr) return;
        if (payload_bytes == 0 || payload_bytes > DEVICE_RUN_RESULT_PAYLOAD_BYTES) return;
        std::memcpy(payload_, payload, payload_bytes);
        payload_bytes_ = static_cast<uint32_t>(payload_bytes);
    }

    bool snapshot_is_ok() const { return selection_.verdict == DeviceRunVerdict::Ok; }

    bool handoff() {
        const auto selection = selection_;
        const auto bytes = payload_bytes_;
        selection_ = {};
        payload_bytes_ = 0;
        return aicpu_accept_runtime_terminal(selection, bytes != 0 ? payload_ : nullptr, bytes);
    }

    bool publish(uint64_t region_base, uint64_t run_epoch) {
        const RunTerminalSelection selection = selection_;
        const uint32_t payload_bytes = payload_bytes_;
        selection_ = RunTerminalSelection{};
        payload_bytes_ = 0;
        if (selection.verdict == DeviceRunVerdict::None) return false;
        return aicpu_publish_run_terminal(
            region_base, run_epoch, selection.verdict, selection.code, selection.source,
            payload_bytes != 0 ? payload_ : nullptr, payload_bytes
        );
    }

private:
    RunTerminalSelection selection_{};
    uint32_t payload_bytes_{0};
    uint8_t payload_[DEVICE_RUN_RESULT_PAYLOAD_BYTES]{};
};
