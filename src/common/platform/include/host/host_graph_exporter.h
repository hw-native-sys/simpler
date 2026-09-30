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
 * @file host_graph_exporter.h
 * @brief Owns the host-built dependency graphs a run hands over, and publishes
 *        them off the submit path.
 *
 * One per runner / device context, like every other collector: the pools,
 * budget, error record and writer thread of one context are nobody else's.
 *
 * What it is for. A host-orchestrating runtime finishes its graph inside
 * `prepare`, then serializes and writes it there, on the submitting thread. The
 * graph is complete at that point — the device contributes nothing to it — so
 * the only thing keeping the write on that thread is that the graph lives in
 * thread-local state. Moving it into an owned export lifts that constraint.
 *
 * The three rules that make it safe:
 *
 * 1. **A lease before anything.** `seal` takes one before it touches the
 *    capture, the error record, the budget or a slot. `finish` closes admission
 *    and waits for outstanding leases *in the same critical section that closes
 *    it*, so no operation can appear after the close observed none.
 * 2. **One publication policy.** Queued and inline writes both reserve an
 *    exclusive temporary, check the stream after flushing and closing it, and
 *    publish with `link`, which never replaces a name. Atomicity does not depend
 *    on whether the queue had room.
 * 3. **Verdict before deregistration.** Every route records its result under the
 *    mutex before it stops being counted, so a flush that observes no
 *    outstanding work has already observed every verdict.
 *
 * Retention bounds what is *kept*, not what may run: with both slots full, or a
 * graph larger than the budget can take, the calling thread publishes it
 * immediately instead of failing the run. That costs the submit path the write
 * it was going to cost anyway before this existed.
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "host/chip_swimlane_runs.h"  // Verdict, ErrorSummary, HostBudget, k* limits
#include "host/host_graph_runs.h"

namespace simpler::dfx::host_graph {

/** Counters a test reads to tell the routes apart. */
struct ExporterStats {
    uint64_t published{0};      // graphs linked into place
    uint64_t inline_writes{0};  // published by the sealing thread itself
    uint64_t failed{0};         // sealed but produced no file
    uint64_t not_captured{0};   // nothing on the sealing thread to publish
    uint64_t lease_refused{0};  // arrived after admission closed
    uint64_t open_slots{0};     // unpublished exports held right now
    size_t charged_bytes{0};
    /**
     * Sealing threads inside their own write right now.
     *
     * `inline_writes` counts completed ones, so it is the wrong thing to wait
     * on: the inline route is declared before any I/O and the count only moves
     * once the write returns. A case that holds publication to observe the
     * route waits on this instead.
     */
    uint32_t inline_in_flight{0};
    /**
     * Admission has been closed by a terminal close.
     *
     * Published so a case can wait on the transition instead of a sleep: the
     * close sets this and then waits for the leases already taken, so seeing
     * it means the close is inside that wait.
     */
    bool admission_closed{false};
};

class HostGraphExporter {
public:
    HostGraphExporter() = default;
    ~HostGraphExporter();
    HostGraphExporter(const HostGraphExporter &) = delete;
    HostGraphExporter &operator=(const HostGraphExporter &) = delete;

    /**
     * Latch retention and this exporter's own host byte budget.
     *
     * Before any run, once, like every other collector's. With retention off
     * every entry point below is inert and the synchronous path is unchanged.
     */
    void configure_retained_runs(bool retain_across_runs, size_t budget_bytes);
    bool retains_runs() const { return retain_across_runs_; }

    /**
     * Take this thread's finished graph and publish it, in the background when
     * there is room for it and on this thread when there is not.
     *
     * `take` is the hand-off the capture side provides; it is passed in rather
     * than called directly so the platform layer names no runtime symbol and a
     * test can drive the exporter without a capture.
     *
     * Returns false when nothing was published and something should have been —
     * the caller logs, and the failure is sticky until a flush reports it.
     * Never throws: it runs inside a `prepare` whose own failure would
     * otherwise be replaced by a diagnostic's.
     */
    using TakeFn = int (*)(HostGraphExport *);
    bool seal(uint64_t run_epoch, const std::string &output_dir, TakeFn take) noexcept;

    /**
     * Wait for every publication already under way, then report.
     *
     * Waits for the queue, the writer and any inline write that has registered.
     * It does not close admission and does not wait for a seal that has not yet
     * declared a write: such a seal has no verdict to miss, and if it publishes
     * later that is after this call's linearization point. A timeout claims
     * nothing about the artifacts.
     */
    bool flush_retained_runs(int timeout_ms, std::string *error);

    /**
     * Close admission, drain what is outstanding, join the writer.
     *
     * Two phases, because the consumer has to outlive every producer that may
     * still enqueue: admission closes and this waits for the leases already
     * taken — with the writer still available to publish whatever they go on to
     * queue — and only once no lease can exist is the writer asked to stop and
     * joined. Stopping it in the same breath as closing admission would let it
     * leave on an empty queue an in-flight lease had not reached yet.
     *
     * No deadline: teardown already waits for the other collectors, and a slow
     * disk lengthens it. A seal that arrives after this publishes nothing and
     * touches nothing. Nothing already accepted is dropped.
     */
    void finish_retained_runs();

    ExporterStats stats_for_test() const;
    /**
     * Hold every publication — the writer's and an inline one alike — just
     * before it touches the filesystem.
     *
     * A case needs this to observe the states a busy publication produces: both
     * slots held, a seal taking the inline route, a flush with something to wait
     * for. It is one hook rather than two because the two routes share the
     * publication, which is the point of §6.4. A terminal close overrides it, so
     * a case that forgets to release it still tears down.
     */
    void pause_publication_for_test(bool paused);

private:
    struct Slot {
        bool occupied{false};
        uint64_t run_epoch{0};
        size_t charged{0};
    };

    /** The two device-facing steps a publication takes, for the tests' benefit. */
    void writer_loop();
    void ensure_writer_started();
    bool publish(const HostGraphExport &graph) noexcept;
    /** Record a verdict and its detail under `mu_`. Allocates nothing. */
    void note_locked(runs::Verdict verdict, uint64_t run_epoch, const char *detail);
    bool idle_locked() const;

    bool retain_across_runs_{false};
    size_t budget_bytes_{runs::kDefaultBudgetBytes};
    bool ready_{false};

    mutable std::mutex mu_;
    std::condition_variable cv_;
    Slot slots_[runs::kMaxOpenEpochs];
    std::deque<std::unique_ptr<HostGraphExport>> queue_;
    bool writer_busy_{false};
    /** Seals between lease acquisition and release, inline writes included. */
    uint32_t leases_{0};
    /** The subset of `leases_` currently writing on their own thread. */
    uint32_t inline_writes_{0};
    bool admission_closed_{false};
    bool writer_running_{false};
    bool writer_stop_{false};
    bool publication_paused_{false};
    std::thread writer_;
    runs::ErrorSummary errors_;
    runs::HostBudget budget_;
    ExporterStats stats_;
};

}  // namespace simpler::dfx::host_graph
