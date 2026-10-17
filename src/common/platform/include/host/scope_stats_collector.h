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
 * @file scope_stats_collector.h
 * @brief Host-side scope_stats streaming collector + NDJSON export.
 *
 * Architecture mirrors PmuCollector: BufferPoolManager<ScopeStatsModule> runs
 * split mgmt threads (drain polls per-thread ready queues and refills the
 * single instance's free_queue from recycled lanes; replenish returns done
 * buffers to recycled lanes). ScopeStatsCollector's collector thread shards
 * append each full buffer's ScopeStatsRecords to an in-memory vector. After
 * stop(), write_jsonl() renders them to
 * <output_dir>/scope_stats/scope_stats.jsonl.
 *
 * Memory mirroring is handled by the framework via the MemoryOps installed
 * at set_memory_context time:
 *   - SVM platforms (a2a3): no copy_* callbacks installed; mirror_/copy_*
 *     short-circuit to no-ops, host writes go directly to device memory.
 *   - Non-SVM platforms (a5): profiling_copy_* installed; the framework's
 *     mgmt loop mirrors the shm region per tick; per-buffer payloads
 *     (ScopeStatsBuffer) are pulled on demand inside ProfilerAlgorithms.
 *
 * Lifecycle:
 *   init()               — Allocate header + 1 BufferState + N ScopeStatsBuffers
 *                          (pre-fills free_queue; surplus → recycled pool).
 *   start(tf)            — Inherited: launches mgmt + collector threads.
 *   [device execution]
 *   stop()               — Inherited: drain queues, join threads.
 *   reconcile_counters() — Recover any un-flushed current buffer left by an
 *                          abnormal exit, then cross-check collected ==
 *                          total - dropped.
 *   write_jsonl()        — Emit scope_stats/scope_stats.jsonl
 *                          (meta line + one record/line).
 *   finalize()           — Free all device memory, unregister.
 *
 * Output (scope_stats/scope_stats.jsonl), NDJSON:
 *   line 1: {"fatal":bool,"dropped":uint,"total":uint,
 *            "task_window_max":[...],"heap_max":[...],
 *            "dep_pool_max":[...],"tensormap_max":uint}
 *   line k: {"site":"file:line","phase":"begin|end","depth":int,
 *            "ring":int,"task_window_start":int,"task_window_end":int,
 *            "heap_start":uint,"heap_end":uint,
 *            "dep_pool_start":int,"dep_pool_end":int,
 *            "tensormap":int,"run_epoch":uint,"buf_seq":uint}
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/platform_config.h"
#include "common/scope_stats.h"
#include "common/unified_log.h"
#include "host/collected_record.h"
#include "host/diagnostic_run_banks.h"
#include "host/profiler_base.h"
#include "host/scope_stats_runs.h"
#include "host/scope_stats_capacity.h"

// ---------------------------------------------------------------------------
// scope_stats Module (drives BufferPoolManager<ScopeStatsModule>)
// ---------------------------------------------------------------------------

struct ScopeStatsReadyBufferInfo {
    uint32_t instance_index;  // Always 0 (single instance)
    uint32_t thread_index;    // AICPU thread queue index this entry came from
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

/**
 * A collected scope_stats record with its run. Alias over the shared wrapper —
 * see host/collected_record.h for why identity is copied rather than referenced.
 */
using CollectedScopeStatsRecord = CollectedRecord<ScopeStatsRecord>;

struct ScopeStatsModule {
    using DataHeader = ScopeStatsDataHeader;
    using ReadyEntry = ScopeStatsReadyQueueEntry;
    using ReadyBufferInfo = ::ScopeStatsReadyBufferInfo;
    using FreeQueue = ScopeStatsFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_SCOPE_STATS_READYQUEUE_SIZE;
    static constexpr uint32_t kHostPoolQueueSize =
        PLATFORM_MAX_AICPU_THREADS * PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE;
    static constexpr uint32_t kSlotCount = PLATFORM_SCOPE_STATS_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "ScopeStatsModule";
    // The orchestrator is the sole device-side producer (scope_stats_collector_aicpu
    // enqueues into queues[s_orch_thread_idx]), so one drain thread scanning
    // every AICPU ready queue covers it; further shards would only ever be empty.
    static constexpr int kMaxCollectorThreads = 1;

    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE - PLATFORM_SCOPE_STATS_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_scope_stats_header(shm); }

    static std::optional<profiling_common::EntrySite<ScopeStatsModule>>
    resolve_entry(void *shm, DataHeader *header, int q, const ReadyEntry &entry) {
        if (shm == nullptr || header == nullptr) {
            LOG_ERROR("ScopeStatsModule: invalid shared memory/header while resolving ready entry");
            return std::nullopt;
        }
        if (header->num_instances != 1 || entry.instance_index >= header->num_instances) {
            LOG_ERROR(
                "ScopeStatsModule: invalid ready entry instance=%u (num_instances=%u)", entry.instance_index,
                header->num_instances
            );
            return std::nullopt;
        }
        ScopeStatsBufferState *state = get_scope_stats_buffer_state(shm, static_cast<int>(entry.instance_index));
        profiling_common::EntrySite<ScopeStatsModule> site;
        site.kind = 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = sizeof(ScopeStatsBuffer);
        site.info.instance_index = entry.instance_index;
        site.info.thread_index = static_cast<uint32_t>(q);
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *shm, DataHeader *header, Cb &&cb) {
        const int n = static_cast<int>(header->num_instances);
        for (int i = 0; i < n; i++) {
            ScopeStatsBufferState *state = get_scope_stats_buffer_state(shm, i);
            cb(/*kind=*/0, &state->free_queue, sizeof(ScopeStatsBuffer));
        }
    }
};

using ScopeStatsAllocCallback = profiling_common::ProfAllocCallback;
using ScopeStatsRegisterCallback = profiling_common::ProfRegisterCallback;
using ScopeStatsUnregisterCallback = profiling_common::ProfUnregisterCallback;
using ScopeStatsFreeCallback = profiling_common::ProfFreeCallback;

// ---------------------------------------------------------------------------
// ScopeStatsCollector
// ---------------------------------------------------------------------------

class ScopeStatsCollector : public profiling_common::ProfilerBase<ScopeStatsCollector, ScopeStatsModule> {
public:
    ScopeStatsCollector() = default;
    ~ScopeStatsCollector();

    ScopeStatsCollector(const ScopeStatsCollector &) = delete;
    ScopeStatsCollector &operator=(const ScopeStatsCollector &) = delete;

    static constexpr int kIdleTimeoutSec = PLATFORM_SCOPE_STATS_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "ScopeStats";

    int init(
        int num_threads, const ScopeStatsAllocCallback &alloc_cb, ScopeStatsRegisterCallback register_cb,
        const ScopeStatsFreeCallback &free_cb, int device_id
    );

    // Start a run's collection window: drop the previous run's records, its
    // counter, and the recovered-buffer bookkeeping reconcile_counters() leaves
    // behind. execution_complete_ is re-armed because it is what tells the
    // collector loop a run is still producing.
    //
    // The collector initializes once and serves every run, so this is the only
    // point at which they are cleared; init() clears none of them, and left
    // alone they accumulate across runs. The device's producer state is part of
    // that: `run_epoch` is the identity a buffer the previous run left in
    // `current_buf_ptr` is re-stamped with, so this run's records cannot inherit
    // the previous run's.
    //
    // False iff such a buffer could not be made this run's, which leaves the
    // device able to attribute this run's records to the previous one. Callers
    // that keep per-run artifacts must refuse the run.
    bool begin_run(uint64_t run_epoch);

    // Device pointer to the ScopeStatsDataHeader. Set
    // kernel_args.scope_stats_data_base to this after init().
    void *get_scope_stats_shm_device_ptr() const { return shm_dev_; }

    // Poll-thread hook: append the buffer's records to the in-memory vector.
    void on_buffer_collected(const ScopeStatsReadyBufferInfo &info);

    // After stop(): recover a non-empty current buffer left by abnormal exit,
    // warn on drops, and cross-check collected == total - dropped. Returns
    // true iff the run is clean.
    bool reconcile_counters();

    // Render the collected records to
    // <output_dir>/scope_stats/scope_stats.jsonl. Reads the static capacity
    // metadata + fatal latch from the shared header (constant after
    // orchestrator init). Must be called after stop().
    int write_jsonl(const std::string &output_dir);

    void finalize(ScopeStatsUnregisterCallback unregister_cb, const ScopeStatsFreeCallback &free_cb);

    bool is_initialized() const { return initialized_; }

    /**
     * How many records this collector has received, across every run.
     *
     * Read under `records_mutex_`, the lock the collector thread increments it
     * with, so an observer on another thread is ordered against that thread's
     * appends rather than racing them. Collector-wide telemetry: a run's
     * verdict is built from its own receipts, which this deliberately does not
     * separate.
     */
    uint64_t total_collected() const {
        std::scoped_lock lock(records_mutex_);
        return total_collected_;
    }

    /**
     * Collected records, each carrying the run that produced it.
     *
     * Returned by value because the collector thread appends concurrently. A
     * record's `run_epoch` is the one stamped on the device buffer it was
     * copied from, so grouping by it attributes records to runs without relying
     * on the collector having been cleared between them.
     */
    std::vector<CollectedScopeStatsRecord> collected_records() const;

    /** How many collected records belong to `run_epoch`. */
    size_t collected_for_run(uint64_t run_epoch) const;

    // --- Cross-run retention -------------------------------------------------
    //
    // Off unless `configure_retained_runs(true, ...)` is called before init, in
    // which case the run boundary keeps its ownership steps — the receive drain
    // and the terminal read — and hands only the rendering and the file write
    // to a writer thread. Every entry point below is inert with retention off.

    /** Latch retention and this collector's own host byte budget. */
    void configure_retained_runs(bool retain_across_runs, size_t budget_bytes);
    bool retains_runs() const { return retain_across_runs_; }

    /**
     * Map the device bank array this collector's producer publishes its per-run
     * accounting into, for `expected_producers` participants.
     *
     * Until this is called the collector reconciles a run exactly as it does
     * today, from the live shared header under the run's execution claim. That
     * is correct only while runs are serialized, which is what the caller is
     * asserting by *not* calling this: the address only reaches the device
     * through the launching run's kernel arguments, so a mapped array with no
     * one publishing into it would report every participant missing.
     *
     * One producer — the orchestrator thread is this collector's only writer —
     * so `expected_producers` is 1 and the stride is the same.
     */
    bool open_run_terminal_banks(int expected_producers);

    /** True while a run's own producer accounting is what settles its verdict. */
    bool device_publishes_run_terminal() const { return terminal_banks_.opened(); }
    bool terminal_backing_retained() const { return terminal_banks_.backing_retained(); }

    /** The address this run's producer publishes into, for its kernel arguments. */
    uint64_t run_terminal_device_address() const { return terminal_banks_.device_address(); }

    /**
     * Terminal banks held out of this collector's pool right now — one per live
     * run that reserved one, plus any a run that ended with neither a
     * completion nor a producer-stop proof still owes. The pool's own
     * occupancy, which is the only figure that still answers once a run's slot
     * has been retired.
     */
    size_t run_terminal_banks_in_use() const { return terminal_banks_.credits().banks_in_use(); }

    /**
     * The device address this run's producer publishes its accounting into.
     *
     * Resolved from the bank the run's own admission reserved, so two runs open
     * at once get two addresses. 0 when the run holds no reservation — it was
     * never admitted, its admission was refused, or retention is off — which is
     * what the device reads as "publish no accounting".
     */
    uint64_t run_terminal_bank_address(uint64_t run_epoch) const;

    /**
     * Whether this run could take a terminal bank right now.
     *
     * True when the run already holds one — the question is idempotent across
     * launch rounds — or when the array has one free. A run ordered behind the
     * open ones with no bank left waits rather than launching without the
     * accounting its diagnostic exists to produce; the banks come back as the
     * runs ahead of it report.
     */
    bool can_take_run_terminal_bank(uint64_t run_epoch) const {
        if (run_epoch == 0) return false;
        if (run_terminal_bank_address(run_epoch) != 0) return true;
        return terminal_banks_.can_reserve();
    }

    /**
     * Give back the bank `run_epoch` holds without reading it, on proof that
     * every producer that could write into it has terminated.
     *
     * For a run whose completion was never observed: its accounting is settled
     * by `run_close` on the completion path alone, so here the bank holds
     * whatever its producers managed before the failure. None of it is read —
     * no consumer waits on it, the host never copies it, and an entry that
     * outlives this epoch is counted as foreign by the next occupant, whose
     * reconciliation compares each entry against its own run identity. So the
     * storage is free and the host copies this run quarantined are untouched:
     * they are the reader threads' to release at the join, which this proof
     * says nothing about.
     *
     * The proof is the caller's and it is about producers, not success: a
     * simulated device's producers are host threads and joining them
     * establishes that none remains, while a real device's do not become
     * observable that way. Without it the quarantine keeps the bank, because a
     * producer may still be writing into it.
     */
    void discard_run_terminal_bank(uint64_t run_epoch);

    /**
     * Whether a further run could be admitted to the retained path.
     *
     * Retention configured, no sticky failure, and an output slot actually
     * free — read as one answer, because a free slot on a collector that is not
     * retaining proves nothing and a failed collector refuses admission
     * whatever its capacity.
     */
    bool can_admit_retained_run() const;

    /**
     * Whether this collector holds no unpublished retained run.
     *
     * True with retention off, and true once every admitted run's writer has
     * published. What it answers is whether a writer may still be reading
     * this collector's pools — which outlives a run's native retirement, so
     * it is not the same question as whether a run is executing.
     */
    bool retained_runs_idle() const;

    /**
     * Tell the owning context that this collector's capacity may have moved.
     *
     * Set once by the runner after init. Over-notifying is free — a woken
     * admission re-attempts and parks again — while under-notifying costs a
     * blocked run its whole deadline, so every path that signals this
     * collector's own waiters signals this too rather than reasoning about
     * whether it had to. Invoked with no collector lock held.
     */
    void set_credit_release_notifier(std::function<void()> notifier) { credit_notifier_ = std::move(notifier); }

    /**
     * Admit one run and open its export slot. False refuses the run, and the
     * caller must fail it before anything reaches the device.
     *
     * Refused when quarantined host copies from a run whose completion could
     * not be proved are still held — `begin_run()` would clear the very records
     * the quarantine protects — and when both unpublished export slots are in
     * use.
     *
     * Mapped banks require the binding's capacity metadata. TMR must supply its
     * resolved values; a runtime without TMR rings marks them not required. The
     * slot takes a copy before any producer starts, and withdrawal releases it.
     *
     * `why` is meaningful only when this returns false, and says whether the
     * refusal is a capacity shortage the caller should re-attempt when credit
     * moves or a condition re-attempting cannot change. Any refusal this does
     * not classify reports `Fatal`: an over-strict answer fails the run early,
     * where an over-lenient one would retry until its deadline.
     */
    bool run_begin(
        uint64_t run_epoch, const std::string &output_prefix, simpler::dfx::runs::AdmissionRefusal *why = nullptr,
        const simpler::dfx::scope_stats_runs::CapacityMetadata *capacities = nullptr
    );

    /**
     * Close one run's boundary, under its execution claim.
     *
     * `device_execution_complete` is the caller's own fence observation and is
     * the whole of this collector's completion proof. Without it, and without a
     * terminal read every device copy reported success for, the run produces no
     * artifact at all: nothing shared is read, the host copies are quarantined
     * until the existing collector-thread join, and a sticky error is recorded.
     */
    void run_close(uint64_t run_epoch, bool device_execution_complete);

    /** Give an admitted run's slot back when its launch submitted nothing. */
    void abandon_run(uint64_t run_epoch);

    /** Wait for every closed run to be published. False when any run failed. */
    bool flush_retained_runs(int timeout_ms, std::string *error);

    /** Publish everything still queued. Called before the collector threads stop. */
    void finish_retained_runs();

    /** Free quarantined host copies. Only legal once the reader threads are joined. */
    void discard_quarantined_runs();

    /** Counters a test reads instead of parsing files. */
    struct RetainedRunStats {
        uint64_t published{0};
        uint64_t partial{0};
        uint64_t counts_unknown{0};
        uint64_t write_failed{0};
        uint64_t quarantined{0};
        uint64_t refused_records{0};
        uint64_t host_failures{0};
        size_t open_slots{0};
        size_t charged_bytes{0};
        bool has_error{false};
    };
    RetainedRunStats retained_run_stats_for_test() const;
    /** Make the next terminal device copy report failure. */
    void fail_terminal_copy_for_test(bool fail) { fail_terminal_copy_ = fail; }
    /** Hold the writer before it publishes, so a test can occupy export slots. */
    void pause_writer_for_test(bool paused);
    /** Throw from the collector thread's append, where nothing may escape. */
    void throw_in_collector_for_test(bool fail) { throw_in_collector_ = fail; }
    /** Throw from the writer thread's publish, where nothing may escape. */
    void throw_in_writer_for_test(bool fail) { throw_in_writer_ = fail; }
    /** Fail the writer-thread construction, as a real `std::thread` can. */
    void fail_writer_start_for_test(bool fail) { fail_writer_start_ = fail; }
    /** Throw at the handoff, where a run's records have already left the store. */
    void fail_handoff_for_test(bool fail) { fail_handoff_ = fail; }

private:
    bool initialized_ = false;

    // Shared memory region (ScopeStatsDataHeader + ScopeStatsBufferState).
    // shm_host_ / shm_size_ / device_id_ live on ProfilerBase (set via
    // set_memory_context in init()).
    void *shm_dev_ = nullptr;

    simpler::dfx::scope_stats_runs::RecordBlocks records_;
    mutable std::mutex records_mutex_;
    uint64_t total_collected_ = 0;
    uint64_t refused_records_ = 0;

    /**
     * One admitted run's own receipts, so two runs can be open at once.
     *
     * The single accumulator above cannot serve a joined pair: admitting a
     * successor would release records its predecessor has not yet sealed, and
     * every later arrival would be counted against whichever run happened to be
     * admitted last. Each entry is claimed by `run_begin`, routed to by
     * `append_buffer_records` using the identity the producer stamped on the
     * buffer, and retired by that run's own close.
     *
     * Guarded by `records_mutex_`, the lock the receive path already holds, so
     * routing adds no lock interaction. The accumulator above stays in use on
     * the unmapped path, where no successor can be admitted at all and the
     * artifact's bytes must not change.
     */
    struct RunReceipts {
        uint64_t run_epoch{0};
        bool open{false};
        simpler::dfx::scope_stats_runs::RecordBlocks records;
        uint64_t received{0};
        uint64_t refused{0};
    };
    std::array<RunReceipts, simpler::dfx::runs::kMaxOpenEpochs> run_receipts_{};
    // Records that arrived stamped with a run this collector holds no open
    // receipts for — a run already sealed, or one it never admitted. Reported,
    // never folded into another run's totals.
    uint64_t foreign_epoch_records_ = 0;
    // One-shot, so the receive path logs once rather than per record.
    bool foreign_epoch_reported_ = false;
    // Set by the receive path under `records_mutex_` and drained by the
    // boundary, which is where the sticky error channel can be reached:
    // `note_host_failure` takes `retained_mu_`, and the receive path must not
    // hold both. Counting alone is not a reported loss, so this is what turns
    // it into one.
    bool foreign_epoch_pending_report_ = false;

    /**
     * Turn a pending foreign-record observation into this collector's sticky
     * error, outside the receive path's lock.
     *
     * Safe to call with no observation pending; it is a no-op then.
     */
    void report_foreign_epoch_records() noexcept;

    /** Claim `run_epoch` its own receipts. False when every entry is in use. */
    bool open_run_receipts(uint64_t run_epoch);
    /** Hand one run its receipts and close its entry. */
    bool take_run_receipts(uint64_t run_epoch, simpler::dfx::scope_stats_runs::RunExport *into);
    /** Release one run's receipts without publishing them. */
    void drop_run_receipts(uint64_t run_epoch);
    uint64_t recovered_current_buf_ = 0;
    uint64_t recovered_current_total_ = 0;

    ScopeStatsDataHeader *scope_stats_header() const { return get_scope_stats_header(shm_host_); }
    ScopeStatsBufferState *scope_stats_state(int idx = 0) const { return get_scope_stats_buffer_state(shm_host_, idx); }

    void append_buffer_records(const void *buf_host_ptr);

    /**
     * Make the buffer the producer still names in `current_buf_ptr` belong to
     * `run_epoch`, by rewriting its 64-byte header and nothing below it.
     *
     * A producer that reuses a non-zero `current_buf_ptr` never pops, so the
     * engine's `claim_free` — the only writer of a buffer's `count` reset and,
     * via `on_pop_success`, of its identity stamp — does not run. Left alone,
     * this run would append after the previous run's records and carry its
     * `run_epoch` and `local_seq`, which is a duplicate in one artifact and a
     * misattribution in the other.
     *
     * The buffer is pool memory outside the manager's shm window, so
     * `publish_field` cannot reach it; this publishes the header range alone
     * rather than the whole buffer, so the records below it are never written
     * back from the host shadow.
     *
     * The pointer is left in place. Clearing it would orphan a pool buffer, and
     * pushing it back onto the free queue would make this a second runtime
     * writer of `free_queue.tail` — the drain shard that serves the buffer's
     * ready queue is its sole runtime writer (see `profiler_base.h`).
     *
     * Only ever called between runs, where the previous run's completion is
     * proved and this run has submitted nothing, so the producer is not running.
     */
    bool adopt_current_buffer(uint64_t run_epoch);

    /**
     * Render one artifact's bytes. `extra` adds the background-mode metadata
     * keys; a null `extra` reproduces today's metadata line exactly, which is
     * what keeps the default path's file unchanged.
     */
    static int render_jsonl_to(
        std::FILE *fp, const simpler::dfx::scope_stats_runs::DeviceSnapshot &device,
        const simpler::dfx::scope_stats_runs::RecordBlocks &records,
        const simpler::dfx::scope_stats_runs::Collection *extra
    );

    /** Today's unchecked read of the shared header, for the default path only. */
    simpler::dfx::scope_stats_runs::DeviceSnapshot snapshot_unchecked() const;

    /**
     * Charge one record block, or refuse it.
     *
     * Always true with retention off: the default path keeps today's unbounded
     * in-memory accumulation, and only a retained run is charged.
     */
    bool charge_record_block(size_t bytes);

    /**
     * Record a host-side failure that belongs to the collector rather than to
     * one run, and keep it until the runner is destroyed.
     *
     * Allocates nothing, so it is safe on the paths that call it precisely
     * because an allocation has just failed.
     */
    void note_host_failure(const char *detail) noexcept;

    // --- Cross-run retention state ------------------------------------------

    // A slot is held from admission until its artifact is published, so
    // "two unpublished exports" is a bound on what the collector is holding,
    // not merely on what is still filling.
    enum class SlotState : int { Free = 0, Open = 1, Publishing = 2, Quarantined = 3 };

    struct Slot {
        SlotState state{SlotState::Free};
        uint64_t run_epoch{0};
        std::string output_dir;
        simpler::dfx::scope_stats_runs::CapacityMetadata capacities;
        // The device bank this run's producer publishes into, held from
        // admission until the host has copied the accounting out of it. Every
        // path that gives a slot back does so by assigning a fresh `Slot`,
        // which releases this with it — so a refused admission, a failed
        // handoff and a reclaimed quarantine all return the bank without
        // naming it.
        simpler::dfx::credit::Reservation terminal_bank;
    };

    /**
     * Copy the shared region and this run's terminal into `out`, reporting
     * whether every device copy it needed succeeded.
     *
     * A false return leaves `out.valid` false. The fields a failed copy may
     * have left behind are never published: the caller quarantines instead.
     */
    bool read_terminal_checked(simpler::dfx::scope_stats_runs::DeviceSnapshot *out);

    /**
     * Reset the host state one run owns: its records, its receipt counters and
     * its completion observation.
     *
     * Separate from `begin_run` because the device half of that call — the
     * counter reset and the held-buffer adoption — belongs to the producer once
     * a terminal bank is mapped, while this half never does.
     */
    void begin_run_host_state();

    /** Device allocation trampolines for the bank array, bound to `this`. */
    static void *bank_alloc(void *owner, size_t bytes, void **host_out);
    static int bank_initialize(void *owner, void *device_dst, void *host_dst, const void *initial, size_t bytes);
    static int bank_unmap(void *owner, void *device_ptr);
    static int bank_free(void *owner, void *device_ptr);
    static int bank_copy(void *owner, void *host_dst, const void *device_src, const void *host_src, size_t bytes);

    /**
     * Take the producer's unpublished current buffer, if it left one.
     *
     * `current_buf_ptr != 0 && count != 0` is device-written evidence that the
     * buffer was never enqueued: publication clears the pointer immediately
     * after a successful enqueue, and the end-of-run flush zeroes the count
     * when its own enqueue fails. So no de-duplication bookkeeping is needed,
     * and a published buffer is never recovered twice.
     */
    bool recover_unpublished_buffer_checked();

    /** `run_close`'s body, wrapped by the boundary's exception guard. */
    void run_close_locked_path(uint64_t run_epoch, bool device_execution_complete);

    void quarantine_locked(uint64_t run_epoch, const char *detail);
    void writer_loop();
    void ensure_writer_started();
    void stop_writer();
    int publish_export(const simpler::dfx::scope_stats_runs::RunExport &data);

    std::function<void()> credit_notifier_;
    simpler::dfx::credit::RunTerminalBanks terminal_banks_{};
    // Kept from `initialize` so an allocation this collector makes later — the
    // terminal bank array, which only exists once retention maps one — uses the
    // same device allocator as its buffers. `initialize` takes them by
    // reference and `finalize` takes them again, so neither call is the owner
    // of a lifetime that spans both.
    ScopeStatsAllocCallback device_alloc_cb_{nullptr};
    ScopeStatsFreeCallback device_free_cb_{nullptr};
    /**
     * The callback the terminal bank array is released through.
     *
     * Separate from `device_free_cb_` because the two are not the same function
     * at the only moment the array is released: `finalize` is handed its own
     * free callback, and a teardown that abandons device resources hands one
     * that frees nothing. The array is this collector's last device allocation
     * and must follow the same policy as the pooled buffers released beside it,
     * so `finalize` rebinds this to its own argument before releasing — using
     * the `initialize`-time allocator there would hand memory back to an
     * allocator the caller has deliberately stopped using.
     */
    ScopeStatsFreeCallback bank_free_cb_{nullptr};
    /**
     * The teardown's unregister callback, stashed beside the free one and for
     * the same reason: the bank array's host mapping has to come down before
     * its memory does, and `finalize` is the only place that callback exists.
     */
    ScopeStatsUnregisterCallback bank_unregister_cb_{nullptr};

    bool retain_across_runs_{false};
    size_t retained_budget_bytes_{0};
    bool retained_ready_{false};
    bool fail_terminal_copy_{false};
    bool throw_in_collector_{false};
    bool throw_in_writer_{false};
    bool fail_writer_start_{false};
    bool fail_handoff_{false};
    simpler::dfx::runs::HostBudget host_budget_;
    simpler::dfx::runs::ErrorSummary run_errors_;

    mutable std::mutex retained_mu_;
    std::condition_variable retained_cv_;
    Slot slots_[simpler::dfx::runs::kMaxOpenEpochs];
    std::deque<simpler::dfx::scope_stats_runs::RunExport> write_queue_;
    simpler::dfx::scope_stats_runs::RecordBlocks quarantined_records_;
    bool quarantine_held_{false};
    bool writing_{false};
    bool writer_running_{false};
    bool writer_paused_{false};
    std::thread writer_;
    RetainedRunStats stats_{};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_
