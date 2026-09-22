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
 *   - Make the record visible to the host **before** the epoch that marks it
 *     valid. A thread-ordering fence does not do this: the host reads through a
 *     D2H copy of device memory, so the bytes have to leave this core's cache.
 *     Hence an explicit flush of payload and header words, then of the epoch.
 *
 * A record is refused rather than clamped or repaired. A payload that outgrew
 * the capacity, a success carrying a code, and a failure carrying no code or no
 * known code source are all producer defects; publishing a corrected version of
 * one would hide it, and refusing leaves the region holding some earlier run's
 * epoch, which the host reads as undecided. The accepted shapes are the ones
 * `device_run_result_terminal` will decide, so nothing this publishes can read
 * back as undecided.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_RUN_RESULT_AICPU_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_RUN_RESULT_AICPU_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "common/device_run_result.h"
#include "common/run_terminal_accumulator.h"

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
    // Record out of cache first; only then the epoch that says it is readable.
    // A host that sees the epoch must already be able to see what it vouches
    // for. The header words are contiguous from `payload_bytes`, so one flush
    // covers them.
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
 * `take` runs in the run's finalizer, before anything resets the state it read
 * from; `publish` runs on the sole cleanup owner, after teardown and before
 * that thread returns from the kernel. The rendezvous between them
 * (`ThreadCompletionGate`) is what makes the snapshot visible to the owner, so
 * this holds no synchronization of its own.
 *
 * Publishing clears the snapshot, so a run whose owner never publishes cannot
 * hand a successor a predecessor's verdict.
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

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_RUN_RESULT_AICPU_H_
