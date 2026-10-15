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
 * @file scope_stats_retained_runs.cpp
 * @brief ScopeStats' cross-run retention: admission, the run boundary's
 *        snapshot, the background writer, and the quarantine that a run with
 *        no completion proof takes instead of an artifact.
 *
 * The boundary keeps every ownership step it has today — the receive drain and
 * the terminal read — and hands only the rendering and the file write to the
 * writer. What moves off the boundary is host work on host-owned data; what
 * stays is everything that touches the device.
 */

#include "host/scope_stats_collector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <new>
#include <system_error>
#include <utility>
#include <fcntl.h>
#include <filesystem>
#include <system_error>
#include <unistd.h>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"
#include "../../../worker/runtime_c_api.h"

namespace runs = simpler::dfx::runs;
namespace scope_runs = simpler::dfx::scope_stats_runs;
namespace credit = simpler::dfx::credit;

namespace {

/** Bounded fixed state this collector reserves before it admits anything. */
size_t retained_fixed_overhead() {
    // Two export slots' metadata and their path reservations, the permanent
    // error record, and the writer's staging block. Reserved rather than
    // charged per run, so an admitted run can never be refused the storage its
    // own artifact needs.
    return runs::kMaxOpenEpochs *
               (sizeof(scope_runs::RunExport) + sizeof(scope_runs::CapacityMetadata) + runs::kPathAllowanceBytes) +
           sizeof(runs::ErrorSummary) + runs::kWriterScratchBytes;
}

}  // namespace

void ScopeStatsCollector::note_host_failure(const char *detail) noexcept {
    // Sticky for the runner's whole life, and reported by every later public
    // flush and by close. Allocates nothing: these paths are reached because
    // an allocation has just failed.
    run_errors_.record_fatal(detail);
    std::lock_guard<std::mutex> lk(retained_mu_);
    stats_.host_failures++;
}

void ScopeStatsCollector::report_foreign_epoch_records() noexcept {
    uint64_t lost = 0;
    {
        std::scoped_lock records(records_mutex_);
        if (!foreign_epoch_pending_report_) return;
        foreign_epoch_pending_report_ = false;
        lost = foreign_epoch_records_;
    }
    // Outside the receive path's lock, because the error channel takes
    // `retained_mu_`. Sticky for the runner's life: records that name no
    // admitted run belong to no artifact, so there is no run whose verdict
    // could carry the loss instead.
    (void)lost;
    note_host_failure("records arrived naming no admitted run; they belong to no artifact");
}

void ScopeStatsCollector::configure_retained_runs(bool retain_across_runs, size_t budget_bytes) {
    retain_across_runs_ = retain_across_runs;
    retained_budget_bytes_ = budget_bytes;
}

bool ScopeStatsCollector::charge_record_block(size_t bytes) {
    // The default path keeps today's unbounded in-memory accumulation: only a
    // retained run is charged, and only against this collector's own budget.
    if (!retain_across_runs_ || !retained_ready_) return true;
    return host_budget_.charge(bytes);
}

// ---------------------------------------------------------------------------
// Checked device reads
// ---------------------------------------------------------------------------

bool ScopeStatsCollector::read_terminal_checked(scope_runs::DeviceSnapshot *out) {
    if (out == nullptr || shm_host_ == nullptr) return false;
    *out = scope_runs::DeviceSnapshot{};
    if (fail_terminal_copy_) return false;
    // Every device copy this path needs is checked. A failed transfer leaves
    // whatever the previous one wrote in the host region, and publishing those
    // bytes would report a stale run's accounting as this one's — so a failure
    // is the end of the path, not a warning on the way through it.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        if (profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_) != 0) {
            LOG_ERROR("scope_stats: the terminal read of the shared region failed; this run's counts are untrusted");
            return false;
        }
    }
    rmb();
    *out = snapshot_unchecked();
    return out->valid;
}

bool ScopeStatsCollector::recover_unpublished_buffer_checked() {
    ScopeStatsBufferState *state = scope_stats_state(0);
    const uint64_t buf_dev = state->current_buf_ptr;
    // Publication clears this pointer immediately after a successful
    // `enqueue_ready`, and the end-of-run flush zeroes the buffer's count when
    // its own enqueue fails. So a non-zero pointer naming a non-empty buffer is
    // the producer's own evidence that the buffer was never handed over, and no
    // de-duplication bookkeeping is needed to know it has not been counted.
    if (buf_dev == 0) return true;
    void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
    if (host_ptr == nullptr) {
        LOG_ERROR(
            "scope_stats: the producer's last buffer 0x%lx has no host mapping; this run's counts are untrusted",
            static_cast<unsigned long>(buf_dev)
        );
        return false;
    }
    if (profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_dev), sizeof(ScopeStatsBuffer)) != 0) {
        LOG_ERROR("scope_stats: reading the producer's last buffer failed; this run's counts are untrusted");
        return false;
    }
    if (reinterpret_cast<const ScopeStatsBuffer *>(host_ptr)->count == 0) return true;
    // Takes `records_mutex_` itself; locking here as well would deadlock.
    append_buffer_records(host_ptr);
    return true;
}

// ---------------------------------------------------------------------------
// Admission and close
// ---------------------------------------------------------------------------

bool ScopeStatsCollector::run_begin(
    uint64_t run_epoch, const std::string &output_prefix, simpler::dfx::runs::AdmissionRefusal *why,
    const scope_runs::CapacityMetadata *capacities
) {
    if (why != nullptr) *why = simpler::dfx::runs::AdmissionRefusal::Fatal;
    if (!retain_across_runs_) return false;
    if (terminal_banks_.opened() && (capacities == nullptr || !capacities->complete())) {
        LOG_ERROR("scope_stats: run %llu has no bound capacity metadata", static_cast<unsigned long long>(run_epoch));
        return false;
    }
    if (shm_host_ == nullptr) {
        LOG_ERROR("scope_stats: the collector is not initialized, so it can retain no runs");
        return false;
    }
    if (output_prefix.size() + 64 > runs::kPathAllowanceBytes) {
        LOG_ERROR(
            "scope_stats: output prefix of %zu bytes exceeds the %zu byte path allowance", output_prefix.size(),
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
        // A quarantine holds the collector's live record store, and `begin_run`
        // below would clear it. Refusing here fails the run before anything
        // reaches the device, which is the only way to keep those copies.
        if (quarantine_held_) {
            LOG_ERROR(
                "scope_stats: run %llu is refused, a run whose completion could not be proved still holds its records",
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
            if (why != nullptr) *why = simpler::dfx::runs::AdmissionRefusal::Capacity;
            LOG_ERROR(
                "scope_stats: run %llu is refused, both unpublished export slots are in use",
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
        if (capacities != nullptr) slots_[free_slot].capacities = *capacities;
        // The bank this run's producer publishes into. Refusing here fails the
        // run before anything reaches the device, which is the same rule the
        // output slot above follows: a run that cannot be accounted for must
        // not run with diagnostics rather than run unaccounted.
        if (terminal_banks_.opened()) {
            slots_[free_slot].terminal_bank = terminal_banks_.reserve(run_epoch);
            if (!slots_[free_slot].terminal_bank.held()) {
                slots_[free_slot] = Slot{};
                lk.unlock();
                if (why != nullptr) *why = simpler::dfx::runs::AdmissionRefusal::Capacity;
                LOG_ERROR(
                    "scope_stats: run %llu is refused, every per-run terminal bank is in use",
                    static_cast<unsigned long long>(run_epoch)
                );
                return false;
            }
        }
        admitted = free_slot;
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
        }
        retained_cv_.notify_all();
        if (credit_notifier_) credit_notifier_();
        note_host_failure("the background writer could not be started");
        return false;
    }
    // With a bank mapped the producer resets its own device counters and adopts
    // its own carried-over buffer at this run's device init, so the host must do
    // neither: its write would land either side of a producer transition, which
    // is the hazard the move to the device removes. The host half of the
    // admission — this run's records, receipts and completion observation — is
    // not the producer's and runs either way.
    //
    // The host half is this run's own receipts, not a reset of the collector's:
    // a predecessor may still be open, and `begin_run_host_state` would release
    // the records it has not sealed. Each admitted run gets its own entry and
    // retires it at its own close.
    if (terminal_banks_.opened()) {
        if (!open_run_receipts(run_epoch)) {
            {
                std::lock_guard<std::mutex> lk(retained_mu_);
                slots_[admitted] = Slot{};
            }
            retained_cv_.notify_all();
            if (credit_notifier_) credit_notifier_();
            if (why != nullptr) *why = simpler::dfx::runs::AdmissionRefusal::Capacity;
            LOG_ERROR(
                "scope_stats: run %llu is refused, every per-run receipt store is in use",
                static_cast<unsigned long long>(run_epoch)
            );
            return false;
        }
        execution_complete_.store(false, std::memory_order_release);
        return true;
    }
    if (!begin_run(run_epoch)) {
        // The device would attribute this run's records to the previous one, so
        // the run is refused here — before any kernel is submitted — rather
        // than publishing an artifact whose records are not all its own. Same
        // rollback as the writer's: the slot is given back, not held by a run
        // that will never close.
        {
            std::lock_guard<std::mutex> lk(retained_mu_);
            slots_[admitted] = Slot{};
        }
        retained_cv_.notify_all();
        if (credit_notifier_) credit_notifier_();
        note_host_failure("the producer's carried-over buffer could not be re-stamped for the admitted run");
        return false;
    }
    return true;
}

void ScopeStatsCollector::abandon_run(uint64_t run_epoch) {
    // Outside the lock: the receipts are the receive path's and are released
    // under its own lock, which no path holds together with this one.
    drop_run_receipts(run_epoch);
    std::lock_guard<std::mutex> lk(retained_mu_);
    for (Slot &slot : slots_) {
        if (slot.state != SlotState::Open || slot.run_epoch != run_epoch) continue;
        slot = Slot{};
        retained_cv_.notify_all();
        if (credit_notifier_) credit_notifier_();
        return;
    }
}

void ScopeStatsCollector::quarantine_locked(uint64_t run_epoch, const char *detail) {
    for (Slot &slot : slots_) {
        if (slot.state != SlotState::Open || slot.run_epoch != run_epoch) continue;
        slot.state = SlotState::Quarantined;
        break;
    }
    quarantine_held_ = true;
    stats_.quarantined++;
    run_errors_.record(run_epoch, runs::Verdict::Quarantined, detail);
}

void ScopeStatsCollector::run_close(uint64_t run_epoch, bool device_execution_complete) {
    // The boundary is `void` and is reached from the runner's teardown, so the
    // run's own result stays primary: a host failure here becomes a persistent
    // diagnostic error, never an exception the teardown has to absorb.
    try {
        run_close_locked_path(run_epoch, device_execution_complete);
    } catch (...) {
        note_host_failure("closing a run's diagnostics boundary failed");
    }
}

void ScopeStatsCollector::run_close_locked_path(uint64_t run_epoch, bool device_execution_complete) {
    if (!retain_across_runs_ || !retained_ready_) return;

    // Before anything this run decides: a record naming no admitted run is the
    // collector's loss, not this run's, so it is reported on its own channel
    // rather than folded into whichever run happens to close next.
    report_foreign_epoch_records();

    // No completion proof: read nothing shared, recover nothing, free nothing,
    // and publish nothing. The host copies stay exactly where the collector
    // threads may still be appending to them, and are discarded only once
    // those threads have been joined by the `stop()` teardown already performs.
    if (!device_execution_complete) {
        std::lock_guard<std::mutex> lk(retained_mu_);
        quarantine_locked(run_epoch, "device completion was not proved; no artifact is written");
        return;
    }

    // The receive drain and the terminal read stay here, under the claim.
    quiesce();

    // The live shared header belongs to whichever run is executing, so a mapped
    // run reads none of it: its evidence is the bank its own producer published
    // and the receipts it owns. Recovering the producer's held buffer is
    // likewise withdrawn — a successor adopts that buffer at its own device
    // init, and the run's own `live_at_close` already reports what it held.
    scope_runs::DeviceSnapshot device;
    if (!terminal_banks_.opened()) {
        if (!read_terminal_checked(&device) || !recover_unpublished_buffer_checked()) {
            std::lock_guard<std::mutex> lk(retained_mu_);
            quarantine_locked(run_epoch, "the terminal read failed; no artifact is written");
            return;
        }
    }

    // Taken before the records lock, never under it: every other path that
    // holds both takes the records lock first.
    credit::Reservation bank;
    if (terminal_banks_.opened()) {
        std::lock_guard<std::mutex> lk(retained_mu_);
        for (Slot &slot : slots_) {
            if (slot.run_epoch != run_epoch || slot.state != SlotState::Open) continue;
            if (!slot.capacities.complete()) {
                quarantine_locked(run_epoch, "this run has no bound capacity metadata; no artifact is written");
                return;
            }
            if (slot.capacities.values) {
                const auto &caps = *slot.capacities.values;
                std::copy(caps.task_window.begin(), caps.task_window.end(), device.task_window_cap);
                std::copy(caps.heap.begin(), caps.heap.end(), device.heap_cap);
                std::copy(caps.dep_pool.begin(), caps.dep_pool.end(), device.dep_pool_cap);
                device.tensormap_cap = caps.tensormap;
            }
            bank = std::move(slot.terminal_bank);
            break;
        }
    }

    scope_runs::RunExport data;
    data.run_epoch = run_epoch;
    data.device = device;
    if (terminal_banks_.opened()) {
        // This run's own records and receipts, taken from the entry it has
        // owned since admission. A predecessor's or a successor's are in their
        // own entries and are not touched here, which is what lets both be open
        // at once.
        if (!take_run_receipts(run_epoch, &data)) {
            std::lock_guard<std::mutex> lk(retained_mu_);
            quarantine_locked(run_epoch, "this run holds no receipts to publish; no artifact is written");
            return;
        }
        // Its producer accounting, settled out of the bank it has held since
        // admission — the one reading that cannot be straddling a producer
        // transition. Settling consumes the reservation, which frees the bank
        // for the next run while this one's artifact is still unwritten.
        const uint64_t received = data.collection.received;
        const uint64_t retained = data.collection.retained;
        const credit::RunReconciliation settled = terminal_banks_.settle(bank, run_epoch, received);
        data.collection = scope_runs::classify_reconciled(settled, retained);
        // Counters name this bank's completed run; capacities name the immutable
        // binding admitted into the same epoch's slot.
        data.device.valid = settled.terminal_seen;
        data.device.total_records = static_cast<uint32_t>(settled.produced);
        data.device.dropped_records = static_cast<uint32_t>(settled.dropped);
        data.device.fatal_latched = settled.fatal_latched ? 1u : 0u;
    } else {
        std::scoped_lock lock(records_mutex_);
        const uint64_t received = total_collected_;
        const uint64_t retained = total_collected_ - refused_records_;
        data.collection = scope_runs::classify(device, received, retained);
        data.records = std::move(records_);
        records_ = scope_runs::RecordBlocks{};
    }
    // From here the records belong to `data` and no longer to the collector, so
    // every exit below has to settle them. A throw while handing them over
    // would otherwise strand their bytes in the budget and leave this run's
    // slot holding an export that never reaches the writer.
    bool handed_over = false;
    struct HandoffGuard {
        ScopeStatsCollector *self;
        scope_runs::RunExport *data;
        const bool *handed_over;
        ~HandoffGuard() {
            if (*handed_over) return;
            const size_t charged = data->records.release();
            {
                std::lock_guard<std::mutex> lk(self->retained_mu_);
                if (charged > 0) self->host_budget_.credit(charged);
                // Reclaimed by *run identity*, in either state a handoff can
                // throw in. Matching only `Publishing` would strand the slot
                // when the throw came from the path copy below, which still
                // reads `Open` — and two such failures would exhaust
                // admission. A quarantined slot is not this run's to take
                // back, so it is left alone.
                for (ScopeStatsCollector::Slot &slot : self->slots_) {
                    const bool held_by_this_run = (slot.state == ScopeStatsCollector::SlotState::Open ||
                                                   slot.state == ScopeStatsCollector::SlotState::Publishing) &&
                                                  slot.run_epoch == data->run_epoch;
                    if (held_by_this_run) slot = ScopeStatsCollector::Slot{};
                }
                self->stats_.write_failed++;
                self->run_errors_.record(
                    data->run_epoch, runs::Verdict::WriteFailed, "handing this run's records to the writer failed"
                );
            }
            // Outside the lock: a flush or an admission waiting on this
            // collector has to learn that the slot came back.
            self->retained_cv_.notify_all();
            if (self->credit_notifier_) self->credit_notifier_();
        }
    } handoff{this, &data, &handed_over};

    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        // The slot is *not* released here: it stays held until the artifact is
        // published, which is what makes "two unpublished exports" a bound on
        // what this collector owns rather than on what is merely still filling.
        for (size_t i = 0; i < runs::kMaxOpenEpochs; i++) {
            if (slots_[i].state != SlotState::Open || slots_[i].run_epoch != run_epoch) continue;
            // Index and state first, path copy second: the copy is the only
            // allocation here, so recording what this run holds before it runs
            // leaves the guard above nothing to guess at.
            data.slot = i;
            slots_[i].state = SlotState::Publishing;
            if (fail_handoff_) throw std::bad_alloc();
            data.output_dir = slots_[i].output_dir;
            break;
        }
        stats_.refused_records += refused_records_;
        if (data.collection.counts_unknown) {
            stats_.counts_unknown++;
        } else if (data.collection.verdict == runs::Verdict::PartialSafe) {
            stats_.partial++;
        }
        write_queue_.push_back(std::move(data));
        handed_over = true;
    }
    retained_cv_.notify_all();
    if (credit_notifier_) credit_notifier_();
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

int ScopeStatsCollector::publish_export(const scope_runs::RunExport &data) {
    if (throw_in_writer_) throw std::bad_alloc();
    std::filesystem::path dir = std::filesystem::path(data.output_dir) / "scope_stats";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_ERROR("scope_stats: cannot create %s: %s", dir.c_str(), ec.message().c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const std::string path = (dir / "scope_stats.jsonl").string();
    const std::string tmp = path + ".tmp";

    // Exclusive create: a stale temp file from an earlier crash is an error
    // rather than a silent reuse.
    int fd = ::open(tmp.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("scope_stats: cannot create %s: %s", tmp.c_str(), std::strerror(errno));
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    std::FILE *fp = ::fdopen(fd, "w");
    if (fp == nullptr) {
        ::close(fd);
        ::unlink(tmp.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // The stream and the temp file are owned by scopes from here on, so a
    // throw out of the rendering below still closes the one and removes the
    // other. A leaked FILE would strand a descriptor for the runner's life,
    // and a leaked temp file would refuse the next run's exclusive create.
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    {
        struct StreamOwner {
            std::FILE *fp;
            int close_rc{0};
            ~StreamOwner() {
                if (fp != nullptr) close_rc = std::fclose(fp);
            }
        } stream{fp};
        struct TempOwner {
            const std::string &path;
            bool keep{false};
            ~TempOwner() {
                if (!keep) ::unlink(path.c_str());
            }
        } temp{tmp};

        rc = render_jsonl_to(fp, data.device, data.records, &data.collection);
        // A close failure is a write failure: the bytes may never have reached
        // the file, so it must not be published under the name that means
        // complete. Closed here rather than by the guard so its status is
        // readable before the publication decision.
        stream.fp = nullptr;
        if (std::fclose(fp) != 0) rc = PTO_RUNTIME_ERR_INTERNAL;
        if (rc == 0) temp.keep = true;
    }
    if (rc != 0) return rc;
    // `link` never replaces an existing name, so a destination this run does
    // not own is a failure and the file already there is left untouched. The
    // temp file is removed either way, so a failure leaves no debris.
    if (::link(tmp.c_str(), path.c_str()) != 0) {
        LOG_ERROR("scope_stats: cannot publish %s: %s", path.c_str(), std::strerror(errno));
        ::unlink(tmp.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    ::unlink(tmp.c_str());
    return 0;
}

void ScopeStatsCollector::writer_loop() {
    std::unique_lock<std::mutex> lk(retained_mu_);
    while (true) {
        retained_cv_.wait(lk, [this] {
            return !write_queue_.empty() || !writer_running_;
        });
        if (write_queue_.empty()) {
            if (!writer_running_) return;
            continue;
        }
        if (writer_paused_) {
            retained_cv_.wait(lk, [this] {
                return !writer_paused_ || !writer_running_;
            });
            if (!writer_running_ && write_queue_.empty()) return;
            if (writer_paused_) continue;
        }
        scope_runs::RunExport data = std::move(write_queue_.front());
        write_queue_.pop_front();
        writing_ = true;
        lk.unlock();

        // Nothing may escape the writer thread: path and staging allocations
        // can fail, and an escape here terminates the chip subprocess instead
        // of failing the flush. A throw is the same outcome as a write error —
        // no file under the final name — so it settles as one.
        int rc = PTO_RUNTIME_ERR_INTERNAL;
        bool threw = false;
        try {
            rc = publish_export(data);
        } catch (...) {
            threw = true;
        }
        const size_t charged = data.records.release();

        lk.lock();
        // Publication is the second, independent axis: a write or link failure
        // fails this run whatever its collection verdict said, so a complete
        // collection can never report success through a file that does not
        // exist under its final name.
        if (rc != 0) {
            stats_.write_failed++;
            run_errors_.record(
                data.run_epoch, runs::Verdict::WriteFailed,
                threw ? "publishing the artifact failed to allocate" : "the artifact could not be published"
            );
            if (threw) stats_.host_failures++;
        } else {
            run_errors_.record(data.run_epoch, data.collection.verdict, "collection did not settle clean");
            if (data.collection.verdict == runs::Verdict::Published) stats_.published++;
        }
        if (charged > 0) host_budget_.credit(charged);
        if (data.slot < runs::kMaxOpenEpochs && slots_[data.slot].state == SlotState::Publishing) {
            slots_[data.slot] = Slot{};
        }
        writing_ = false;
        retained_cv_.notify_all();
        if (credit_notifier_) credit_notifier_();
    }
}

void ScopeStatsCollector::pause_writer_for_test(bool paused) {
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        writer_paused_ = paused;
    }
    retained_cv_.notify_all();
    if (credit_notifier_) credit_notifier_();
}

void ScopeStatsCollector::ensure_writer_started() {
    std::lock_guard<std::mutex> lk(retained_mu_);
    if (writer_running_) return;
    if (fail_writer_start_) throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
    // The flag is published only once the thread really exists. Setting it
    // first and constructing after would leave it true when construction
    // throws, and the next admission would then believe a consumer is running:
    // its export would sit in a queue nothing drains, so the flush would time
    // out and `finish_retained_runs` would wait forever. The new thread's first
    // act is to take this same lock, so it cannot observe either the flag or
    // `writer_` before both are set.
    std::thread started([this] {
        writer_loop();
    });
    writer_running_ = true;
    writer_ = std::move(started);
}

void ScopeStatsCollector::stop_writer() {
    {
        std::lock_guard<std::mutex> lk(retained_mu_);
        if (!writer_running_) return;
        writer_running_ = false;
        writer_paused_ = false;
    }
    retained_cv_.notify_all();
    if (credit_notifier_) credit_notifier_();
    if (writer_.joinable()) writer_.join();
}

// ---------------------------------------------------------------------------
// Flush, finish, discard
// ---------------------------------------------------------------------------

bool ScopeStatsCollector::flush_retained_runs(int timeout_ms, std::string *error) {
    // Before the lock: a foreign-record observation belongs to no run, so a
    // caller that flushes without closing another run is the only one who would
    // otherwise never hear about it.
    report_foreign_epoch_records();
    // Deliberately not gated on `retains_runs()`: an error recorded before a
    // collector rebuild turned retention off must still be reported here.
    std::unique_lock<std::mutex> lk(retained_mu_);
    if (quarantine_held_) {
        // No timeout can produce the collector-thread join this needs, so the
        // wait is skipped rather than served.
        lk.unlock();
        if (error != nullptr) *error = "scope_stats: " + run_errors_.report();
        return false;
    }
    const bool drained = retained_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
        return write_queue_.empty() && !writing_;
    });
    // Read while the writer cannot be mid-update: a verdict is recorded and
    // its counter bumped under this same lock.
    const bool clean =
        stats_.counts_unknown == 0 && stats_.partial == 0 && stats_.write_failed == 0 && stats_.quarantined == 0;
    lk.unlock();
    if (!drained) {
        if (error != nullptr) *error = "scope_stats: publication did not finish within the flush budget";
        return false;
    }
    // A published artifact is not by itself a success. The shared error record
    // counts `PartialSafe` and `CutUnknown` as publishing verdicts, which is
    // swimlane's rule; ScopeStats' contract is stricter — loss, a device fatal
    // and unknown counts each fail this call even though the file exists, so
    // the verdict counters are consulted rather than the error flag alone.
    if (clean && !run_errors_.has_error()) return true;
    if (error != nullptr) *error = "scope_stats: " + run_errors_.report();
    return false;
}

void ScopeStatsCollector::finish_retained_runs() {
    std::unique_lock<std::mutex> lk(retained_mu_);
    if (!writer_running_) return;
    retained_cv_.wait(lk, [this] {
        return write_queue_.empty() && !writing_;
    });
}

void ScopeStatsCollector::discard_quarantined_runs() {
    // Only legal once the collector threads are joined: until then a shard may
    // still be appending into this very storage. Freeing host copies grants no
    // right to release device memory — the pooled buffers keep their own
    // lifetime proof and are released by finalize().
    std::lock_guard<std::mutex> lk(retained_mu_);
    if (!quarantine_held_) return;
    size_t charged = quarantined_records_.release();
    {
        std::scoped_lock records_lock(records_mutex_);
        charged += records_.release();
    }
    if (charged > 0) host_budget_.credit(charged);
    for (Slot &slot : slots_) {
        if (slot.state == SlotState::Quarantined) slot = Slot{};
    }
    quarantine_held_ = false;
}

ScopeStatsCollector::RetainedRunStats ScopeStatsCollector::retained_run_stats_for_test() const {
    std::lock_guard<std::mutex> lk(retained_mu_);
    RetainedRunStats out = stats_;
    out.open_slots = 0;
    for (const Slot &slot : slots_) {
        if (slot.state != SlotState::Free) out.open_slots++;
    }
    out.charged_bytes = host_budget_.charged();
    out.has_error = run_errors_.has_error();
    return out;
}

uint64_t ScopeStatsCollector::run_terminal_bank_address(uint64_t run_epoch) const {
    if (run_epoch == 0) return 0;
    std::lock_guard<std::mutex> lk(retained_mu_);
    for (const Slot &slot : slots_) {
        if (slot.run_epoch != run_epoch || !slot.terminal_bank.held()) continue;
        return terminal_banks_.bank_address(slot.terminal_bank.index());
    }
    return 0;
}

void ScopeStatsCollector::discard_run_terminal_bank(uint64_t run_epoch) {
    if (run_epoch == 0) return;
    std::lock_guard<std::mutex> lk(retained_mu_);
    for (Slot &slot : slots_) {
        if (slot.run_epoch != run_epoch || !slot.terminal_bank.held()) continue;
        // Only the device storage. Whatever this run quarantined on the host
        // stays quarantined: its readers are this collector's own shards, and
        // the producer-stop proof says nothing about them.
        slot.terminal_bank.release();
        LOG_INFO(
            "scope_stats terminal: run %llu returns its bank unread — it failed with every producer proved stopped, "
            "so its entries are discarded rather than withdrawn",
            static_cast<unsigned long long>(run_epoch)
        );
        return;
    }
}

bool ScopeStatsCollector::can_admit_retained_run() const {
    if (!retain_across_runs_) return false;
    std::lock_guard<std::mutex> lk(retained_mu_);
    if (!retained_ready_) return false;
    if (run_errors_.has_error()) return false;
    for (const Slot &slot : slots_) {
        if (slot.state == SlotState::Free) return true;
    }
    return false;
}

bool ScopeStatsCollector::retained_runs_idle() const {
    if (!retain_across_runs_) return true;
    std::lock_guard<std::mutex> lk(retained_mu_);
    for (const Slot &slot : slots_) {
        if (slot.state != SlotState::Free) return false;
    }
    return true;
}
