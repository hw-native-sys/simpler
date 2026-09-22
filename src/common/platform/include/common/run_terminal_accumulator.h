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
 * @file run_terminal_accumulator.h
 * @brief Folding one run's per-participant errors into a single terminal code.
 *
 * Four runtimes publish the same terminal record, so the rule that picks its
 * code lives here once rather than in four executors. Two properties it has to
 * hold, both of which a pair of separate stores would lose:
 *
 *   - **The code and the state that selected it are one value.** They are
 *     chosen by a single compare-exchange over a packed word, so the pairing
 *     holds for any reader, including one that reads while a participant is
 *     recording — rather than only for a reader the rendezvous already orders
 *     after every writer.
 *   - **Accumulation is monotonic.** The first failure recorded wins and no
 *     later store — including a participant that succeeded — can clear it.
 *
 * A runtime's shared error state is deliberately not accumulated here: it is
 * read once by the cleanup owner, when every participant's writes to it are
 * visible, and it outranks a participant's own return. See
 * `run_terminal_select`.
 */

#pragma once

#include <atomic>
#include <cstdint>

#include "common/device_run_result.h"

/**
 * The first failing participant's code and the state that selected it.
 *
 * Reset between runs by the executor's deinit, like the rest of its per-run
 * state.
 */
class RunTerminalAccumulator {
public:
    /** Record `code` under `source`. A zero code, and any later call once a
     *  failure is held, are both no-ops. */
    void record(int32_t code, DeviceRunCodeSource source) {
        if (code == 0 || source == DeviceRunCodeSource::None) return;
        uint64_t expected = 0;
        packed_.compare_exchange_strong(
            expected, pack(code, source), std::memory_order_acq_rel, std::memory_order_relaxed
        );
    }

    /**
     * Record one participant's outcome: its execution return, or its teardown
     * return when execution had nothing to report. A participant that reaches
     * its final arrival must have called this first, so the run's last arrival
     * observes every contribution.
     */
    void record_participant(int32_t run_rc, int32_t shutdown_rc) {
        if (run_rc != 0) {
            record(run_rc, DeviceRunCodeSource::ThreadRc);
        } else {
            record(shutdown_rc, DeviceRunCodeSource::ShutdownRc);
        }
    }

    bool has_failure() const { return packed_.load(std::memory_order_acquire) != 0; }

    int32_t code() const {
        return static_cast<int32_t>(static_cast<uint32_t>(packed_.load(std::memory_order_acquire)));
    }

    DeviceRunCodeSource source() const {
        return static_cast<DeviceRunCodeSource>(static_cast<uint32_t>(packed_.load(std::memory_order_acquire) >> 32));
    }

    void reset() { packed_.store(0, std::memory_order_release); }

private:
    static uint64_t pack(int32_t code, DeviceRunCodeSource source) {
        // A recorded code is never 0, so the packed word is never 0 whatever
        // the source is — which is what makes "0 means no failure" exact.
        return (static_cast<uint64_t>(source) << 32) | static_cast<uint32_t>(code);
    }

    std::atomic<uint64_t> packed_{0};
};

/** What a producer should publish, or None when there is nothing to publish. */
struct RunTerminalSelection {
    DeviceRunVerdict verdict{DeviceRunVerdict::None};
    int32_t code{0};
    DeviceRunCodeSource source{DeviceRunCodeSource::None};
};

/**
 * Select this run's terminal record.
 *
 * `header_status` is the runtime's shared error state, already mapped to its
 * signed runtime status, read by the cleanup owner. It outranks a
 * participant's return because a thread can finish with a zero return while
 * the state it wrote — a scheduler timeout latched next to a non-negative
 * completed count — is the run's actual failure.
 *
 * `normal_path_completed` is the producer's own assertion that the run reached
 * the end of its audited normal path. Without it a run with nothing to report
 * is not a success: an early return that bypassed the rendezvous and a run that
 * genuinely did nothing are the same absence of evidence, so the selection is
 * None and the producer publishes nothing, leaving the host undecided.
 */
inline RunTerminalSelection
run_terminal_select(bool normal_path_completed, int32_t header_status, const RunTerminalAccumulator &participants) {
    if (header_status != 0) {
        return {DeviceRunVerdict::Error, header_status, DeviceRunCodeSource::Header};
    }
    if (participants.has_failure()) {
        return {DeviceRunVerdict::Error, participants.code(), participants.source()};
    }
    if (!normal_path_completed) {
        return {};
    }
    return {DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None};
}
