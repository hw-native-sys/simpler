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
 * @file dep_gen_retained_runs.cpp
 * @brief DepGen's cross-run retention: admission, the run boundary's seal, the
 *        background writer, and the quarantine a run with no completion proof
 *        takes instead of an artifact.
 *
 * The boundary keeps every step that touches the device — the receive drain and
 * the terminal read — and hands the replay, the serialization and the file
 * write to the writer. What moves off the boundary is host work on host-owned
 * data; what stays is everything the device is party to.
 *
 * The writer asks for its replay working storage only once it has taken an
 * export, so however many exports are sealed there is one working set at a
 * time and the budget never has to hold two.
 */

#include "host/dep_gen_collector.h"

#include <chrono>
#include <cstring>
#include <new>
#include <fcntl.h>
#include <filesystem>
#include <system_error>
#include <unistd.h>
#include <vector>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"
#include "tensormap_and_ringbuffer/host/dep_gen_replay.h"

// The strong symbol lives in the tensormap_and_ringbuffer runtime, which is
// the only runtime that captures its graph on the device and therefore the
// only one that replays records. A host_build_graph build links no replay, so
// this weak fallback keeps the .so loadable; retention is only ever configured
// on the device-orchestrating path, so the fallback is not reachable when
// dep_gen is on. LOG_DEBUG rather than WARN for the same reason the runners'
// own fallbacks use it.
extern "C" __attribute__((weak, visibility("hidden"))) int dep_gen_replay_emit_deps_json_budgeted(
    const struct DepGenRecord * /*records*/, size_t /*num_records*/, const char * /*deps_json_path*/,
    const struct DepGenReplayBudget * /*budget*/
) {
    LOG_DEBUG("dep_gen replay not implemented for this runtime — deps.json skipped");
    return -1;
}

namespace runs = simpler::dfx::runs;
namespace dg_runs = simpler::dfx::dep_gen_runs;

namespace {

/** Bounded fixed state this collector reserves before it admits anything. */
size_t retained_fixed_overhead() {
    // Two export slots' metadata and their path reservations, the permanent
    // error record, and the publication path's own scratch. Reserved rather
    // than charged per run, so an admitted run can never be refused the
    // storage its own publication needs.
    return runs::kMaxOpenEpochs * (sizeof(dg_runs::RunExport) + runs::kPathAllowanceBytes) +
           sizeof(runs::ErrorSummary) + runs::kWriterScratchBytes;
}

/** Charge/credit trampolines the replay allocates through. */
bool replay_charge(void *ctx, size_t bytes) { return static_cast<runs::HostBudget *>(ctx)->charge(bytes); }
void replay_credit(void *ctx, size_t bytes) { static_cast<runs::HostBudget *>(ctx)->credit(bytes); }

}  // namespace

void DepGenCollector::note_host_failure(const char *detail) noexcept {
    // Sticky for the runner's whole life, and reported by every later public
    // flush and by close. Allocates nothing: these paths are reached because
    // an allocation has just failed.
    run_errors_.record_fatal(detail);
    std::lock_guard<std::mutex> lk(retained_mu_);
    stats_.host_failures++;
}

void DepGenCollector::configure_retained_runs(bool retain_across_runs, size_t budget_bytes) {
    retain_across_runs_ = retain_across_runs;
    retained_budget_bytes_ = budget_bytes;
}

bool DepGenCollector::charge_record_block(size_t bytes) {
    // The default path keeps today's unbounded accumulation: only a retained
    // run is charged, and only against this collector's own budget.
    if (!retain_across_runs_ || !retained_ready_) return true;
    return host_budget_.charge(bytes);
}

void DepGenCollector::credit_record_block(size_t bytes) {
    if (!retain_across_runs_ || !retained_ready_) return;
    host_budget_.credit(bytes);
}

bool DepGenCollector::epoch_admitted_locked(uint64_t run_epoch) const {
    for (const Slot &slot : slots_) {
        if (slot.state != SlotState::Free && slot.run_epoch == run_epoch) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Admission and close
// ---------------------------------------------------------------------------

bool DepGenCollector::run_begin(uint64_t run_epoch, const std::string &output_prefix) {
    if (!retain_across_runs_) return false;
    if (shm_host_ == nullptr) {
        LOG_ERROR("dep_gen: the collector is not initialized, so it can retain no runs");
        return false;
    }
    if (output_prefix.size() + 64 > runs::kPathAllowanceBytes) {
        LOG_ERROR(
            "dep_gen: output prefix of %zu bytes exceeds the %zu byte path allowance", output_prefix.size(),
            runs::kPathAllowanceBytes
        );
        return false;
    }
    size_t admitted = runs::kMaxOpenEpochs;
    {
        std::unique_lock<std::mutex> lk(retained_mu_);
        if (!retained_ready_) {
            if (!host_budget_.open(retained_budget_bytes_, retained_fixed_overhead())) return false;
            retained_ready_ = true;
        }
        // A quarantine holds this collector's live record store, and the
        // `begin_run()` below would release it. Refusing here fails the run
        // before anything reaches the device, which is the only way to keep
        // those copies until the reader threads are joined.
        if (quarantine_held_) {
            LOG_ERROR(
                "dep_gen: run %llu is refused, a run whose completion could not be proved still holds its records",
                static_cast<unsigned long long>(run_epoch)
            );
            return false;
        }
        size_t free_slot = runs::kMaxOpenEpochs;
        for (size_t i = 0; i < runs::kMaxOpenEpochs; i++) {
            if (slots_[i].state == SlotState::Free) {
                free_slot = i;
                break;
            }
        }
        if (free_slot == runs::kMaxOpenEpochs) {
            LOG_ERROR(
                "dep_gen: run %llu is refused, both unpublished export slots are in use",
                static_cast<unsigned long long>(run_epoch)
            );
            return false;
        }
        try {
            slots_[free_slot].output_dir = output_prefix;
        } catch (...) {
            // The path copy is the only allocation here. A failed admission
            // must leave the slot free, or it is occupied and targetless for
            // the runner's life.
            slots_[free_slot] = Slot{};
            lk.unlock();
            note_host_failure("admitting a run could not allocate its output path");
            return false;
        }
        slots_[free_slot].state = SlotState::Open;
        slots_[free_slot].run_epoch = run_epoch;
        admitted = free_slot;
        // The receive path reads these under `records_mutex_`, so that is what
        // owns them. Taken inside `retained_mu_` here, which is the one order
        // any path needing both uses.
        std::scoped_lock records(records_mutex_);
        retained_epoch_ = run_epoch;
        retained_epoch_open_ = true;
    }
    try {
        ensure_writer_started();
    } catch (...) {
        // Starting the writer is the one step after the slot is taken that can
        // fail, so the slot is given back rather than held by a run that will
        // never close.
        {
            std::lock_guard<std::mutex> lk(retained_mu_);
            slots_[admitted] = Slot{};
            // The flag is the receive path's, so it is written under its lock,
            // taken inside `retained_mu_` as everywhere else.
            std::scoped_lock records(records_mutex_);
            retained_epoch_open_ = false;
        }
        retained_cv_.notify_all();
        note_host_failure("the background writer could not be started");
        return false;
    }
    if (!begin_run()) {
        // The device would reconcile this run against the previous run's
        // totals, so the run is refused here — before any kernel is submitted
        // — rather than publishing a graph whose completeness nothing
        // established.
        // Before the slot is released, for the reason in `writer_loop`.
        run_errors_.record(run_epoch, runs::Verdict::WriteFailed, "this run's device counter reset was not published");
        {
            std::lock_guard<std::mutex> lk(retained_mu_);
            slots_[admitted] = Slot{};
            stats_.refused++;
            // Same ownership as above: the receive path's flag under the
            // receive path's lock.
            std::scoped_lock records(records_mutex_);
            retained_epoch_open_ = false;
        }
        retained_cv_.notify_all();
        return false;
    }
    return true;
}

void DepGenCollector::abandon_run(uint64_t run_epoch) {
    size_t charged = 0;
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        bool held_by_this_run = false;
        for (Slot &slot : slots_) {
            if (slot.state != SlotState::Open || slot.run_epoch != run_epoch) continue;
            slot = Slot{};
            held_by_this_run = true;
            break;
        }
        if (!held_by_this_run) {
            // Nothing of this epoch's to give back. Releasing the record store
            // unconditionally would discard whatever run *is* collecting —
            // this is reached from a rollback path, so the epoch it names is
            // the only thing it may touch.
            return;
        }
        std::scoped_lock records(records_mutex_);
        if (retained_epoch_open_ && retained_epoch_ == run_epoch) {
            retained_epoch_open_ = false;
            charged = retained_records_.release();
        }
    }
    if (charged > 0) host_budget_.credit(charged);
    retained_cv_.notify_all();
}

void DepGenCollector::quarantine_locked(uint64_t run_epoch, const char *detail) {
    for (Slot &slot : slots_) {
        if (slot.state != SlotState::Open || slot.run_epoch != run_epoch) continue;
        slot.state = SlotState::Quarantined;
        break;
    }
    quarantine_held_ = true;
    stats_.quarantined++;
    run_errors_.record(run_epoch, runs::Verdict::Quarantined, detail);
}

void DepGenCollector::run_close(uint64_t run_epoch, bool device_execution_complete) {
    // The boundary is `void` and is reached from the runner's teardown, so the
    // run's own result stays primary: a host failure here becomes a persistent
    // diagnostic error, never an exception the teardown has to absorb.
    try {
        run_close_locked_path(run_epoch, device_execution_complete);
    } catch (...) {
        note_host_failure("closing a run's diagnostics boundary failed");
    }
}

void DepGenCollector::run_close_locked_path(uint64_t run_epoch, bool device_execution_complete) {
    if (!retain_across_runs_ || !retained_ready_) return;

    // No completion proof: read nothing shared, recover nothing, publish
    // nothing. The host copies stay exactly where the collector threads may
    // still be appending to them, and are discarded only once those threads
    // have been joined by the `stop()` teardown already performs.
    if (!device_execution_complete) {
        std::lock_guard<std::mutex> lk(retained_mu_);
        quarantine_locked(run_epoch, "device completion was not proved; no graph is written");
        // Closing the window is all this does: the collector threads may still
        // be appending, and the records they are appending to stay exactly
        // where they are until the `stop()` teardown joins those threads. The
        // flag is the receive path's, so it is written under its lock.
        std::scoped_lock records(records_mutex_);
        retained_epoch_open_ = false;
        return;
    }

    // The receive drain and the terminal read stay here, under the claim.
    quiesce();
    const dg_runs::ReconcileReport report = reconcile_report();

    dg_runs::RunExport data;
    data.run_epoch = run_epoch;
    data.report = report;
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        for (const Slot &slot : slots_) {
            if (slot.state == SlotState::Open && slot.run_epoch == run_epoch) {
                data.output_dir = slot.output_dir;
                break;
            }
        }
        std::scoped_lock records(records_mutex_);
        retained_epoch_open_ = false;
    }
    if (data.output_dir.empty()) {
        // No admission record names this epoch, so there is no destination and
        // no slot that belongs to it. Settling anything here would settle
        // another run's state.
        run_errors_.record(
            run_epoch, runs::Verdict::WriteFailed, "a run closed without an admission record; no graph is written"
        );
        std::lock_guard<std::mutex> lk(retained_mu_);
        stats_.refused++;
        return;
    }

    if (!dg_runs::publishable(report)) {
        // Every reason is a reason this graph is not this run's graph, and
        // `deps.json` has nowhere to say so — one whole graph or no file.
        const char *detail = dg_runs::first_refusal(report);
        LOG_ERROR("dep_gen: run %llu produces no graph: %s", static_cast<unsigned long long>(run_epoch), detail);
        run_errors_.record(run_epoch, runs::Verdict::WriteFailed, detail);
        {
            std::lock_guard<std::mutex> lk(retained_mu_);
            size_t charged = 0;
            {
                std::scoped_lock records(records_mutex_);
                charged = retained_records_.release();
            }
            if (charged > 0) host_budget_.credit(charged);
            for (Slot &slot : slots_) {
                if (slot.state == SlotState::Open && slot.run_epoch == run_epoch) slot = Slot{};
            }
            stats_.refused++;
        }
        // Outside the lock, and unconditional: a caller blocked in
        // `flush_retained_runs` reads the slot table this just settled, so a
        // refusal that did not notify would make it wait out its deadline and
        // report a timeout instead of the cause.
        retained_cv_.notify_all();
        return;
    }

    {
        // The records are the receive path's until this moves them out, so the
        // move happens under that path's lock.
        std::scoped_lock records(records_mutex_);
        data.records = std::move(retained_records_);
        retained_records_ = dg_runs::RecordBlocks{};
    }

    // From here the records belong to `data`. Every exit below has to settle
    // them, or their bytes are stranded in the budget and this run's slot holds
    // an export that never reaches the writer.
    bool handed_over = false;
    struct HandoffGuard {
        DepGenCollector *self;
        dg_runs::RunExport *data;
        const bool *handed_over;
        ~HandoffGuard() {
            if (*handed_over) return;
            const size_t charged = data->records.release();
            // Recorded before the slot is given back, for the reason in
            // `writer_loop`: the release is what a blocked flush observes, so
            // it must not become visible ahead of the verdict that explains it.
            self->run_errors_.record(
                data->run_epoch, runs::Verdict::WriteFailed, "handing this run's graph to the writer failed"
            );
            {
                std::lock_guard<std::mutex> lk(self->retained_mu_);
                if (charged > 0) self->host_budget_.credit(charged);
                // Reclaimed by run identity, in either state a handoff can
                // throw in: matching only `Publishing` would strand the slot
                // when the throw came from the queue push, which still reads
                // `Open`. A quarantined slot is not this run's to take back.
                for (DepGenCollector::Slot &slot : self->slots_) {
                    const bool held_by_this_run = (slot.state == DepGenCollector::SlotState::Open ||
                                                   slot.state == DepGenCollector::SlotState::Publishing) &&
                                                  slot.run_epoch == data->run_epoch;
                    if (held_by_this_run) slot = DepGenCollector::Slot{};
                }
                self->stats_.refused++;
            }
            self->retained_cv_.notify_all();
        }
    } guard{this, &data, &handed_over};

    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        for (Slot &slot : slots_) {
            if (slot.state == SlotState::Open && slot.run_epoch == run_epoch) slot.state = SlotState::Publishing;
        }
        queue_.push_back(std::move(data));
        handed_over = true;
    }
    retained_cv_.notify_all();
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

int DepGenCollector::publish_export(dg_runs::RunExport &data) {
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::path(data.output_dir);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_ERROR("dep_gen: could not create %s: %s", dir.c_str(), ec.message().c_str());
        return -1;
    }
    const std::string final_path = (dir / "deps.json").string();
    const std::string temp_path = final_path + ".tmp";

    // O_EXCL on the temporary, `link` for the publication: `link` never
    // replaces an existing name, so a destination this run does not own fails
    // it with the file already there untouched, and a partly written graph is
    // never visible under the real name.
    const int fd = ::open(temp_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("dep_gen: could not reserve %s exclusively", temp_path.c_str());
        return -1;
    }
    ::close(fd);

    // Contiguous, because the replay indexes its input and scans forward
    // through overflow chains. Charged before it is allocated, and released
    // together with the blocks it was copied from.
    const size_t count = data.records.size();
    std::vector<DepGenRecord> flat;
    size_t flat_bytes = 0;
    int rc = -1;
    if (!dg_runs::checked_mul(count, sizeof(DepGenRecord), &flat_bytes)) {
        LOG_ERROR(
            "dep_gen: run %llu holds more records than can be addressed",
            static_cast<unsigned long long>(data.run_epoch)
        );
        ::unlink(temp_path.c_str());
        return -1;
    }
    if (!host_budget_.charge(flat_bytes)) {
        LOG_ERROR(
            "dep_gen: run %llu needs %zu B to lay its records out contiguously and the budget cannot pay",
            static_cast<unsigned long long>(data.run_epoch), flat_bytes
        );
        ::unlink(temp_path.c_str());
        return -1;
    }
    try {
        flat.resize(count);
        for (size_t i = 0; i < count; i++)
            flat[i] = data.records[i];
        // The blocks have served their purpose and the copy is now the sole
        // holder, so they are freed here and credited from the figure the
        // release itself reports — a second release returns 0, so no credit
        // can happen twice.
        const size_t charged = data.records.release();
        if (charged > 0) host_budget_.credit(charged);

        const DepGenReplayBudget budget{&host_budget_, &replay_charge, &replay_credit};
        rc = dep_gen_replay_emit_deps_json_budgeted(flat.data(), count, temp_path.c_str(), &budget);
    } catch (...) {
        host_budget_.credit(flat_bytes);
        ::unlink(temp_path.c_str());
        throw;
    }
    std::vector<DepGenRecord>{}.swap(flat);
    host_budget_.credit(flat_bytes);

    if (rc != 0) {
        LOG_ERROR(
            "dep_gen: replaying run %llu failed (%d) — deps.json not produced",
            static_cast<unsigned long long>(data.run_epoch), rc
        );
        ::unlink(temp_path.c_str());
        return rc;
    }
    if (::link(temp_path.c_str(), final_path.c_str()) != 0) {
        LOG_ERROR(
            "dep_gen: %s is already occupied; run %llu's graph was not published and the file there is untouched",
            final_path.c_str(), static_cast<unsigned long long>(data.run_epoch)
        );
        ::unlink(temp_path.c_str());
        return -1;
    }
    ::unlink(temp_path.c_str());
    LOG_INFO(
        "dep_gen: published run %llu's graph (%zu records) to %s", static_cast<unsigned long long>(data.run_epoch),
        count, final_path.c_str()
    );
    return 0;
}

void DepGenCollector::writer_loop() {
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        writer_running_ = true;
    }
    retained_cv_.notify_all();
    while (true) {
        dg_runs::RunExport data;
        {
            std::unique_lock<std::mutex> lk(retained_mu_);
            retained_cv_.wait(lk, [this] {
                return writer_stop_ || (!queue_.empty() && !writer_paused_);
            });
            if (queue_.empty()) {
                if (writer_stop_) break;
                continue;
            }
            data = std::move(queue_.front());
            queue_.pop_front();
            writer_busy_ = true;
        }
        int rc = -1;
        try {
            rc = publish_export(data);
        } catch (...) {
            rc = -1;
            note_host_failure("publishing a run's graph failed");
        }
        const size_t charged = data.records.release();
        // The verdict is recorded *before* the drain state it belongs to
        // becomes observable. `flush_retained_runs` waits on the queue, the
        // busy flag and the Publishing slots, and then reads the error record:
        // clearing those first would let a flush that arrives in between see a
        // drained collector with no error and report success for a run whose
        // graph was never written. Notifying afterwards does not close that
        // window — the waiter may be checking the predicate already, or wake
        // spuriously.
        //
        // `ErrorSummary` takes its own lock, and this call holds none, so the
        // existing `retained_mu_` -> error-record order (see
        // `quarantine_locked`) is unchanged.
        if (rc == 0) {
            run_errors_.record(data.run_epoch, runs::Verdict::Published, "published");
        } else {
            run_errors_.record(data.run_epoch, runs::Verdict::WriteFailed, "this run's graph could not be published");
        }
        {
            std::lock_guard<std::mutex> lk(retained_mu_);
            if (charged > 0) host_budget_.credit(charged);
            for (Slot &slot : slots_) {
                if (slot.state == SlotState::Publishing && slot.run_epoch == data.run_epoch) slot = Slot{};
            }
            if (rc == 0) {
                stats_.published++;
            } else {
                stats_.refused++;
            }
            writer_busy_ = false;
        }
        // Outside the lock and on both outcomes: the slot release and the
        // error are what a blocked flush is waiting to observe.
        retained_cv_.notify_all();
    }
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        writer_running_ = false;
    }
    retained_cv_.notify_all();
}

void DepGenCollector::pause_writer_for_test(bool paused) {
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        writer_paused_ = paused;
    }
    retained_cv_.notify_all();
}

void DepGenCollector::ensure_writer_started() {
    std::thread started;
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        if (writer_running_ || writer_.joinable()) return;
        writer_stop_ = false;
    }
    // Constructed before the flag is published: setting `writer_running_`
    // first would leave it true when the construction throws, and the next
    // admission would queue an export nothing drains.
    started = std::thread([this] {
        writer_loop();
    });
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        writer_ = std::move(started);
    }
}

void DepGenCollector::stop_writer() {
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        writer_stop_ = true;
        writer_paused_ = false;
    }
    retained_cv_.notify_all();
    if (writer_.joinable()) writer_.join();
}

// ---------------------------------------------------------------------------
// Public completion
// ---------------------------------------------------------------------------

bool DepGenCollector::flush_retained_runs(int timeout_ms, std::string *error) {
    if (!retained_ready_ && !run_errors_.has_error()) return true;
    {
        std::unique_lock<std::mutex> lk(retained_mu_);
        const bool drained = retained_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
            if (!queue_.empty() || writer_busy_) return false;
            for (const Slot &slot : slots_) {
                if (slot.state == SlotState::Publishing) return false;
            }
            return true;
        });
        if (!drained) {
            if (error != nullptr)
                *error = "dep_gen: a retained run was still unpublished when the flush deadline ran out";
            return false;
        }
    }
    if (run_errors_.has_error()) {
        if (error != nullptr) *error = std::string("dep_gen: ") + run_errors_.report();
        return false;
    }
    return true;
}

void DepGenCollector::finish_retained_runs() {
    if (!retained_ready_) return;
    stop_writer();
}

void DepGenCollector::discard_quarantined_runs() {
    // Only legal once the collector threads are joined: until then they may
    // still be appending to the very storage this frees. `finalize()` calls it
    // immediately after `stop()`, which is that join.
    //
    // This disposes host copies and nothing else. It does not clear the sticky
    // diagnostic error the quarantine recorded — that stays for the runner's
    // life and is still reported by every later flush and by close — and it
    // grants no claim about the device: an unproved run stays unproved, and
    // the pooled device buffers keep their own lifetime proof in `finalize()`.
    size_t charged = 0;
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        for (dg_runs::RunExport &held : quarantined_) {
            charged += held.records.release();
        }
        quarantined_.clear();
        {
            std::scoped_lock records(records_mutex_);
            charged += retained_records_.release();
        }
        for (Slot &slot : slots_) {
            if (slot.state == SlotState::Quarantined) slot = Slot{};
        }
        quarantine_held_ = false;
    }
    if (charged > 0) host_budget_.credit(charged);
}

DepGenCollector::RetainedRunStats DepGenCollector::retained_run_stats_for_test() const {
    std::lock_guard<std::mutex> lk(retained_mu_);
    RetainedRunStats out;
    out.published = stats_.published;
    out.refused = stats_.refused;
    out.quarantined = stats_.quarantined;
    out.host_failures = stats_.host_failures;
    out.refused_records = refused_records_;
    out.foreign_epoch_records = foreign_epoch_records_;
    out.charged_bytes = host_budget_.charged();
    out.has_error = run_errors_.has_error();
    for (const Slot &slot : slots_) {
        if (slot.state != SlotState::Free) out.open_slots++;
    }
    return out;
}
