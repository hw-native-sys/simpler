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

#include "host/host_graph_exporter.h"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

#include "common/unified_log.h"
#include "host/raii_scope_guard.h"

namespace simpler::dfx::host_graph {

namespace {

/**
 * The fixed cost an exporter reserves before it accepts anything.
 *
 * One serialization stream buffer for the writer, plus a destination for each
 * unpublished export it may hold. `HostGraphExport::output_dir` is
 * fixed-capacity and `seal` refuses a longer one, so this reservation is what
 * those destinations actually cost — not an estimate of them.
 */
constexpr size_t kFixedOverheadBytes = runs::kWriterScratchBytes + runs::kMaxOpenEpochs * kMaxOutputDirBytes;

/** The name a published graph takes, and the name it is staged under. */
constexpr const char *kArtifactName = "deps.json";
constexpr const char *kTempName = "deps.json.tmp";

}  // namespace

HostGraphExporter::~HostGraphExporter() {
    // A context can be destroyed without a teardown ever calling `finish`: an
    // init that fails after the runner exists drops the context directly, and
    // only the platform `finalize` runs. The writer must not outlive this
    // object, so the same close-wait-join happens here.
    try {
        finish_retained_runs();
    } catch (...) {
        // A destructor reports nothing. The thread is joined below either way.
    }
    if (writer_.joinable()) {
        try {
            writer_.join();
        } catch (...) {}
    }
}

void HostGraphExporter::configure_retained_runs(bool retain_across_runs, size_t budget_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    retain_across_runs_ = retain_across_runs;
    budget_bytes_ = budget_bytes;
}

bool HostGraphExporter::idle_locked() const { return queue_.empty() && !writer_busy_ && inline_writes_ == 0; }

void HostGraphExporter::note_locked(runs::Verdict verdict, uint64_t run_epoch, const char *detail) {
    // `ErrorSummary` takes its own lock and allocates nothing, so recording
    // while holding `mu_` adds no ordering beyond what the caller already needs:
    // the verdict must be visible before the operation stops being counted.
    errors_.record(run_epoch, verdict, detail);
}

bool HostGraphExporter::publish(const HostGraphExport &graph) noexcept {
    {
        // Ahead of every filesystem step, and of nothing else: a paused
        // publication still holds whatever registered it, which is what lets a
        // case observe a busy writer or an inline write in flight. A terminal
        // close releases the wait, so a forgotten pause cannot wedge teardown.
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] {
            return !publication_paused_ || writer_stop_;
        });
    }
    try {
        std::error_code ec;
        const std::filesystem::path dir(graph.output_dir);
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            LOG_ERROR("host graph: could not create %s: %s", graph.output_dir, ec.message().c_str());
            return false;
        }
        const std::string final_path = (dir / kArtifactName).string();
        const std::string temp_path = (dir / kTempName).string();

        // O_EXCL on the temporary, `link` for the publication: `link` never
        // replaces an existing name, so a destination this run does not own
        // fails it with the file already there untouched, and a partly written
        // graph is never visible under the real name. A temporary left by
        // something else is named rather than removed — it cannot be proved to
        // be ours.
        const int fd = ::open(temp_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd < 0) {
            LOG_ERROR("host graph: could not reserve %s exclusively", temp_path.c_str());
            return false;
        }
        ::close(fd);
        // Armed only once the O_EXCL create above has succeeded, which is what
        // proves this temporary is this publication's own: a temporary left by
        // something else is reported by that branch and never reaches here, so
        // it is still never unlinked. From here every exit removes ours, the
        // throwing ones included — the stream's construction and the body's
        // serialization both allocate, and a temporary that outlived its
        // publication would make O_EXCL refuse this destination for good.
        auto temp_guard = RAIIScopeGuard([&temp_path] {
            ::unlink(temp_path.c_str());
        });

        bool wrote = false;
        {
            std::ofstream out(temp_path, std::ios::out | std::ios::trunc);
            if (out) {
                write_host_graph_body(out, graph);
                out.flush();
                out.close();
                // After the flush and the close, not before: a graph held
                // entirely in the userspace buffer would otherwise report
                // success and lose its bytes at close.
                wrote = static_cast<bool>(out);
            }
        }
        if (!wrote) {
            LOG_ERROR("host graph: writing %s did not complete", temp_path.c_str());
            return false;
        }
        if (::link(temp_path.c_str(), final_path.c_str()) != 0) {
            LOG_ERROR("host graph: %s is already occupied; the existing file is kept", final_path.c_str());
            return false;
        }
        // The temporary is removed by the guard, whichever way this returns.
        LOG_INFO(
            "host graph: wrote %s (tasks=%zu, tensors=%zu, edges=%zu)", final_path.c_str(), graph.tasks.size(),
            graph.tensors.size(), graph.edges.size()
        );
        return true;
    } catch (...) {
        // Path composition and the stream both allocate. A publication that
        // could not run is a failed publication, not an exception the caller has
        // to absorb: `seal` runs inside a prepare and the writer has no
        // boundary of its own.
        LOG_ERROR("host graph: publishing a graph failed unexpectedly");
        return false;
    }
}

bool HostGraphExporter::seal(uint64_t run_epoch, const std::string &output_dir, TakeFn take) noexcept {
    if (!retain_across_runs_ || take == nullptr) return true;

    // --- the lease, before anything is read or touched -----------------------
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (admission_closed_) {
            // Torn down, or being torn down. Nothing may be published and no
            // state here may be read: there is no longer anyone to report to.
            stats_.lease_refused++;
            LOG_WARN(
                "host graph: run %llu arrived after diagnostics closed; no graph is written",
                static_cast<unsigned long long>(run_epoch)
            );
            return false;
        }
        if (!ready_) {
            if (!budget_.open(budget_bytes_, kFixedOverheadBytes)) {
                errors_.record_fatal("the host-graph budget could not reserve its fixed overhead");
                return false;
            }
            ready_ = true;
        }
        leases_++;
    }

    // From here every exit must release the lease and, where a write was
    // declared, record its verdict first. One guard rather than a return at
    // each branch: `take` and `publish` both run under it.
    bool published = false;
    bool declared_inline = false;
    struct LeaseGuard {
        HostGraphExporter *self;
        bool *declared;
        ~LeaseGuard() {
            std::lock_guard<std::mutex> lk(self->mu_);
            if (*declared) self->inline_writes_--;
            self->leases_--;
            self->cv_.notify_all();
        }
    } guard{this, &declared_inline};

    auto graph = std::unique_ptr<HostGraphExport>();
    try {
        graph = std::make_unique<HostGraphExport>();
    } catch (...) {
        std::lock_guard<std::mutex> lk(mu_);
        errors_.record_fatal("a host graph could not be taken: out of memory");
        return false;
    }

    if (!output_dir_fits(output_dir.size())) {
        std::lock_guard<std::mutex> lk(mu_);
        stats_.failed++;
        note_locked(runs::Verdict::WriteFailed, run_epoch, "the output path exceeds the destination allowance");
        LOG_ERROR(
            "host graph: output prefix of %zu bytes exceeds the %zu byte allowance", output_dir.size(),
            kMaxOutputDirBytes
        );
        return false;
    }
    std::memcpy(graph->output_dir, output_dir.c_str(), output_dir.size() + 1);
    graph->output_dir_len = static_cast<uint32_t>(output_dir.size());
    graph->run_epoch = run_epoch;

    const int outcome = take(graph.get());
    if (outcome != static_cast<int>(TakeOutcome::Complete)) {
        std::lock_guard<std::mutex> lk(mu_);
        stats_.not_captured++;
        note_locked(
            runs::Verdict::WriteFailed, run_epoch,
            outcome == static_cast<int>(TakeOutcome::Incomplete) ? "a task was left open, so the graph is incomplete" :
                                                                   "no host graph was captured on the sealing thread"
        );
        LOG_ERROR(
            "host graph: run %llu has no complete capture on this thread; no graph is written",
            static_cast<unsigned long long>(run_epoch)
        );
        return false;
    }

    // --- retain it if there is room, otherwise write it here -----------------
    const size_t bytes = graph->payload_bytes();
    {
        std::lock_guard<std::mutex> lk(mu_);
        size_t free_slot = runs::kMaxOpenEpochs;
        for (size_t i = 0; i < runs::kMaxOpenEpochs; i++) {
            if (!slots_[i].occupied) {
                free_slot = i;
                break;
            }
        }
        if (free_slot != runs::kMaxOpenEpochs && budget_.charge(bytes)) {
            bool queued = false;
            try {
                // The node first, the graph after: `push_back` allocates and can
                // throw, and the assignment that follows cannot, so ownership
                // never ends up in a container the caller believes it failed to
                // reach.
                queue_.push_back(nullptr);
                queued = true;
            } catch (...) {
                budget_.credit(bytes);
            }
            if (queued) {
                slots_[free_slot].occupied = true;
                slots_[free_slot].run_epoch = run_epoch;
                slots_[free_slot].charged = bytes;
                stats_.open_slots++;
                stats_.charged_bytes = budget_.charged();
                // Handed over and done with, in one place: the only statement
                // after the graph leaves this owner is the return, so nothing
                // downstream can be reading an owner that gave its graph away.
                queue_.back() = std::move(graph);
                ensure_writer_started();
                cv_.notify_all();
                return true;
            }
        }
        // No room, the budget refused, or the queue could not take it. This
        // thread publishes now — the cost this work exists to remove, paid
        // rather than dropping a graph or failing a run. Declared before any
        // I/O so a concurrent flush waits for it.
        inline_writes_++;
        declared_inline = true;
    }

    published = publish(*graph);
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (published) {
            stats_.published++;
            stats_.inline_writes++;
            note_locked(runs::Verdict::Published, run_epoch, nullptr);
        } else {
            stats_.failed++;
            note_locked(runs::Verdict::WriteFailed, run_epoch, "a host graph could not be published");
        }
    }
    return published;
}

void HostGraphExporter::ensure_writer_started() {
    // A running writer is a *live* writer here: the loop cannot exit while any
    // lease is outstanding, and every caller of this holds one. So the early
    // return can never hand a queued graph to a consumer that has already left.
    if (writer_running_) return;
    // Started under `mu_`, so two seals cannot both spawn one. A thread that
    // cannot start leaves the queue non-empty, which a flush reports rather than
    // waiting on forever — the fatal is what breaks that wait.
    try {
        writer_ = std::thread([this] {
            writer_loop();
        });
        writer_running_ = true;
    } catch (...) {
        errors_.record_fatal("the host-graph writer thread could not be started");
        // Drain what is queued here rather than leaving it unpublished with no
        // writer: the charge is released and the flush sees an empty queue with
        // the fatal recorded.
        while (!queue_.empty()) {
            queue_.front()->release();
            queue_.pop_front();
        }
        for (Slot &slot : slots_) {
            if (!slot.occupied) continue;
            budget_.credit(slot.charged);
            slot = Slot{};
            if (stats_.open_slots > 0) stats_.open_slots--;
        }
        stats_.charged_bytes = budget_.charged();
    }
}

void HostGraphExporter::writer_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        // The consumer outlives every producer that may still enqueue. A stop
        // alone is not enough to leave: an admitted seal holding a lease has not
        // reached its `push_back` yet, and exiting on the empty queue it happens
        // to see would leave that graph with nobody to publish it and the close
        // waiting on a queue no thread owns.
        cv_.wait(lk, [this] {
            return !queue_.empty() || (writer_stop_ && leases_ == 0);
        });
        if (queue_.empty()) return;
        std::unique_ptr<HostGraphExport> graph = std::move(queue_.front());
        queue_.pop_front();
        writer_busy_ = true;
        lk.unlock();

        const bool ok = publish(*graph);
        const uint64_t epoch = graph->run_epoch;
        graph->release();
        graph.reset();

        lk.lock();
        // Verdict first, then stop being counted: a flush that sees the writer
        // idle has already seen this result.
        if (ok) {
            stats_.published++;
            note_locked(runs::Verdict::Published, epoch, nullptr);
        } else {
            stats_.failed++;
            note_locked(runs::Verdict::WriteFailed, epoch, "a host graph could not be published");
        }
        for (Slot &slot : slots_) {
            if (!slot.occupied || slot.run_epoch != epoch) continue;
            budget_.credit(slot.charged);
            slot = Slot{};
            if (stats_.open_slots > 0) stats_.open_slots--;
            break;
        }
        stats_.charged_bytes = budget_.charged();
        writer_busy_ = false;
        cv_.notify_all();
    }
}

bool HostGraphExporter::flush_retained_runs(int timeout_ms, std::string *error) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!ready_ && idle_locked() && !errors_.has_error()) return true;
    // Every declared publication, not every future one: a seal holding a lease
    // that has not yet declared a write has no verdict for this call to miss,
    // and if it publishes later that is after this call's linearization point.
    if (timeout_ms < 0) {
        cv_.wait(lk, [this] {
            return idle_locked();
        });
    } else if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
                   return idle_locked();
               })) {
        if (error != nullptr) {
            *error = "the host-graph writer did not finish within the flush budget; no artifact is claimed";
        }
        return false;
    }
    if (!errors_.has_error()) return true;
    if (error != nullptr) *error = "host graph: " + errors_.report();
    return false;
}

void HostGraphExporter::finish_retained_runs() {
    std::thread writer;
    {
        std::unique_lock<std::mutex> lk(mu_);
        // Two phases, and the order is the whole point.
        //
        // First: close admission and wait for the operations already admitted.
        // The writer is *not* asked to stop yet, so it stays available to
        // publish whatever a lease still in flight goes on to enqueue. Asking it
        // to stop here instead would let it leave on the empty queue it happens
        // to see, and the graph that lease enqueues afterwards would have no
        // consumer — the wait below would never end. A held publication is
        // released too, or a forgotten one would hold this phase open.
        admission_closed_ = true;
        publication_paused_ = false;
        cv_.notify_all();
        cv_.wait(lk, [this] {
            return leases_ == 0 && idle_locked();
        });
        // No lease can be taken and none is outstanding, so nothing can enqueue
        // again. Only now is the consumer redundant.
        writer_stop_ = true;
        cv_.notify_all();
        writer = std::move(writer_);
        writer_running_ = false;
    }
    if (writer.joinable()) writer.join();
    std::lock_guard<std::mutex> lk(mu_);
    // The queue is empty by the phase-one predicate and nothing can add to it,
    // so this is an assertion about the exit rather than a drain. A graph left
    // here would be one accepted and never published, which is the thing this
    // ordering exists to make impossible.
    if (!queue_.empty()) {
        errors_.record_fatal("the host-graph writer exited with graphs still queued");
        while (!queue_.empty()) {
            queue_.front()->release();
            queue_.pop_front();
        }
    }
    if (ready_) {
        budget_.close();
        ready_ = false;
    }
}

ExporterStats HostGraphExporter::stats_for_test() const {
    std::lock_guard<std::mutex> lk(mu_);
    ExporterStats out = stats_;
    out.charged_bytes = budget_.charged();
    out.inline_in_flight = inline_writes_;
    out.admission_closed = admission_closed_;
    return out;
}

void HostGraphExporter::pause_publication_for_test(bool paused) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        publication_paused_ = paused;
    }
    cv_.notify_all();
}

}  // namespace simpler::dfx::host_graph
