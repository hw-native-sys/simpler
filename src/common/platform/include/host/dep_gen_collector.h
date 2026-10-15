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
 * @file dep_gen_collector.h
 * @brief Host-side dep_gen (SubmitTrace) buffer allocation and streaming
 *        collection for in-memory replay.
 *
 * Architecture:
 * - BufferPoolManager<DepGenModule>: shared mgmt-thread infrastructure that
 *   polls per-thread ready queues, drains done-queue shards, and replenishes
 *   the single instance's free_queue from shard-local recycled lanes.
 * - DepGenCollector: collector thread shards pop full DepGenBuffers from the
 *   manager and append their DepGenRecords to in-memory storage consumed by
 *   host replay after device execution completes, grouped by the run that
 *   produced them.
 *
 * Lifecycle:
 *   init()                       — Allocate header + 1 BufferState + N DepGenBuffers
 *                                  (pre-fills free_queue; surplus → recycled pool).
 *                                  Calls set_memory_context() on the base.
 *   start(tf)                    — Inherited: launches mgmt + collector threads.
 *   [device execution]
 *   stop()                       — Inherited: drain queues, join threads.
 *   reconcile_counters()         — Sanity-check that no buffer the device still
 *                                  holds has records in it, run the
 *                                  collected+dropped==total cross-check. A
 *                                  buffer AICPU could not hand over stays the
 *                                  pool's with count 0, which is not a failure;
 *                                  records left in one are. If
 *                                  dropped_record_count > 0, the host caller
 *                                  skips deps.json emission (incomplete graph;
 *                                  user gets a warning).
 *   finalize()                   — Free all device memory, unregister.
 *
 * Output contract: per run, a contiguous in-memory stream of DepGenRecord
 * values. Host replay consumes one run's stream directly; no submit_trace.bin
 * intermediary is written by the collector.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <array>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "common/dep_gen.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/dep_gen_runs.h"
#include "host/diagnostic_run_banks.h"
#include "host/profiler_base.h"

// ---------------------------------------------------------------------------
// dep_gen Module (drives BufferPoolManager<DepGenModule>)
// ---------------------------------------------------------------------------

/**
 * Internal hand-off struct delivered from a drain thread to a collector shard.
 * thread_index identifies the AICPU thread queue the entry was popped from
 * (always equal to the orchestrator thread index, since dep_gen is single-
 * instance — exposed for symmetry with PmuReadyBufferInfo).
 */
struct DepGenReadyBufferInfo {
    uint32_t instance_index;  // Always 0 (single instance)
    uint32_t thread_index;    // AICPU thread queue index this entry came from
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

struct DepGenModule {
    using DataHeader = DepGenDataHeader;
    using ReadyEntry = DepGenReadyQueueEntry;
    using ReadyBufferInfo = ::DepGenReadyBufferInfo;
    using FreeQueue = DepGenFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_DEP_GEN_READYQUEUE_SIZE;
    static constexpr uint32_t kHostPoolQueueSize = PLATFORM_MAX_AICPU_THREADS * PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE;
    static constexpr uint32_t kSlotCount = PLATFORM_DEP_GEN_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "DepGenModule";
    // The orchestrator is the sole device-side producer (dep_gen_collector_aicpu
    // enqueues into queues[s_orch_thread_idx]), so one drain thread scanning
    // every AICPU ready queue covers it; further shards would only ever be empty.
    static constexpr int kMaxCollectorThreads = 1;

    /**
     * Startup-only batch allocation size for proactive_replenish when the
     * recycled lanes need additional buffers before drain threads start.
     */
    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE - PLATFORM_DEP_GEN_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_dep_gen_header(shm); }

    /**
     * `count` is intentionally NOT reset here — AICPU is the sole writer and
     * resets it itself on flush/drop/pop.
     */
    static std::optional<profiling_common::EntrySite<DepGenModule>>
    resolve_entry(void *shm, DataHeader *header, int q, const ReadyEntry &entry) {
        if (shm == nullptr || header == nullptr) {
            LOG_ERROR("DepGenModule: invalid shared memory/header while resolving ready entry");
            return std::nullopt;
        }
        if (header->num_instances != 1 || entry.instance_index >= header->num_instances) {
            LOG_ERROR(
                "DepGenModule: invalid ready entry instance=%u (num_instances=%u)", entry.instance_index,
                header->num_instances
            );
            return std::nullopt;
        }
        DepGenBufferState *state = get_dep_gen_buffer_state(shm, static_cast<int>(entry.instance_index));
        profiling_common::EntrySite<DepGenModule> site;
        site.kind = 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = sizeof(DepGenBuffer);
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
            DepGenBufferState *state = get_dep_gen_buffer_state(shm, i);
            cb(/*kind=*/0, &state->free_queue, sizeof(DepGenBuffer));
        }
    }
};

// ---------------------------------------------------------------------------
// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals (halHost*).
// ---------------------------------------------------------------------------

using DepGenAllocCallback = profiling_common::ProfAllocCallback;
using DepGenRegisterCallback = profiling_common::ProfRegisterCallback;
using DepGenUnregisterCallback = profiling_common::ProfUnregisterCallback;
using DepGenFreeCallback = profiling_common::ProfFreeCallback;

// ---------------------------------------------------------------------------
// DepGenCollector
// ---------------------------------------------------------------------------

class DepGenCollector : public profiling_common::ProfilerBase<DepGenCollector, DepGenModule> {
public:
    DepGenCollector() = default;
    ~DepGenCollector();

    DepGenCollector(const DepGenCollector &) = delete;
    DepGenCollector &operator=(const DepGenCollector &) = delete;

    static constexpr int kIdleTimeoutSec = PLATFORM_DEP_GEN_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "DepGen";

    /**
     * Allocate dep_gen shared memory and pre-populate the free_queue.
     *
     * Allocates a DepGenDataHeader + 1 DepGenBufferState, plus
     * PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE DepGenBuffers. The first
     * PLATFORM_DEP_GEN_SLOT_COUNT buffers go directly into the free_queue;
     * the surplus go into BufferPoolManager's shard-local recycled lanes.
     *
     * @param num_threads     Number of AICPU scheduling threads (so the
     *                        DataHeader sizes its per-thread ready queues)
     * @param alloc_cb        Memory allocation callback
     * @param register_cb     halHostRegister callback (nullptr on non-SVM platforms)
     * @param free_cb         Memory free callback
     * @param device_id       Device ID
     * @return 0 on success, non-zero on failure
     */
    int init(
        int num_threads, const DepGenAllocCallback &alloc_cb, DepGenRegisterCallback register_cb,
        const DepGenFreeCallback &free_cb, int device_id, DepGenUnregisterCallback unregister_cb = nullptr
    );

    /**
     * Start a run's collection window. Clears what the previous run left in the
     * in-memory record set and its counter, and zeroes the device-side record
     * counters reconcile compares against.
     *
     * The collector initializes once and serves every run, so this is the only
     * point at which they are cleared; init() clears none of them, and nothing on
     * the device clears the record counters — they are documented as monotonic.
     * Skip it and the second run's deps.json carries the first run's edges, and
     * reconcile compares one run's collected count against a device total
     * accumulated over both, which suppresses the export.
     *
     * Called with the device quiesced, so the AICPU is not writing these and the
     * collector threads are idle.
     */
    bool begin_run();

    /**
     * Reset the host state one run owns: its records, its receipt counters and
     * its refusal counts.
     *
     * Separate from `begin_run` because the device half of that call — the
     * counter reset — belongs to the producer once a terminal bank is mapped,
     * while this half never does.
     */
    void begin_run_host_state();

    /**
     * Map the device bank array this collector's producer publishes its per-run
     * accounting into, for `expected_producers` participants.
     *
     * Until this is called the collector reconciles a run from the live shared
     * header under its execution claim, which is correct only while runs are
     * serialized. The address only reaches the device through the launching
     * run's kernel arguments, so a mapped array with nobody publishing into it
     * would report every participant missing.
     *
     * One producer — the orchestrator thread is this collector's only writer.
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
     * The same settlement ScopeStats and the swimlane make: a run whose
     * completion was never observed has its accounting read by nobody, and an
     * entry that outlives this epoch is counted as foreign by the next
     * occupant, whose reconciliation compares each entry against its own run
     * identity. Only the device storage comes back; this run's quarantined host
     * records stay held for the reader join, which this proof does not address.
     *
     * The proof is the caller's and it is about producers, not success. Without
     * it the quarantine keeps the bank, because a producer may still be writing
     * into it.
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

    /** Device allocation trampolines for the bank array, bound to `this`. */
    static void *bank_alloc(void *owner, size_t bytes, void **host_out);
    static int bank_initialize(void *owner, void *device_dst, void *host_dst, const void *initial, size_t bytes);
    static int bank_unmap(void *owner, void *device_ptr);
    static int bank_free(void *owner, void *device_ptr);
    static int bank_copy(void *owner, void *host_dst, const void *device_src, const void *host_src, size_t bytes);

    /**
     * Report what this run's boundary actually established about its transport,
     * separately from whether the counts happened to balance.
     *
     * `reconcile_counters()` answers one `bool` that conflates "the comparison
     * balanced" with "the comparison was made": a failed region copy leaves the
     * host shadow holding the zeroes `begin_run()` wrote, which satisfies the
     * identity for a run that collected nothing. Every distinguishable reason a
     * graph is untrustworthy is a field here, and both modes refuse on the same
     * set — they differ only in which channel reports it.
     *
     * Call after the receive drain, with the device quiesced.
     */
    simpler::dfx::dep_gen_runs::ReconcileReport reconcile_report();

    /**
     * This run's report, built from the bank its producer published into.
     *
     * Consumes the reservation: the figures are in host storage before the bank
     * is handed on, so the next run's reuse cannot overwrite evidence nobody
     * has read. The host's own observations — a clamped buffer, a foreign
     * record — are read here because the device cannot see either.
     */
    simpler::dfx::dep_gen_runs::ReconcileReport
    reconcile_report_from_terminal(uint64_t run_epoch, simpler::dfx::credit::Reservation &bank);

    /**
     * Device pointer to the DepGenDataHeader. Set kernel_args.dep_gen_data_base
     * to this after init() so AICPU can find the shared memory via
     * set_platform_dep_gen_base().
     */
    void *get_dep_gen_shm_device_ptr() const { return shm_dev_; }

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Appends the
     * buffer's DepGenRecord entries to in-memory storage, grouped by the run
     * that produced them (no disk I/O — the host replay consumes one run's
     * records directly via ``window_records()`` once the device run
     * completes).
     */
    void on_buffer_collected(const DepGenReadyBufferInfo &info);

    /**
     * After stop(): cross-check collected + dropped == total. If dropped > 0,
     * the host caller skips deps.json emission so users get an incomplete-
     * graph warning rather than partial data they might mistake for complete.
     *
     * @return true iff the run captured a complete trace (no drops, no leftovers).
     */
    bool reconcile_counters();

    /**
     * Free all device memory and release the in-memory record buffer. Idempotent.
     */
    void finalize(DepGenUnregisterCallback unregister_cb, const DepGenFreeCallback &free_cb);

    /**
     * @return true if init() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Total DepGenRecords drained from the device-side ring buffer so far.
     */
    uint64_t total_collected() const { return total_collected_; }

    /**
     * Every run whose records this collector holds, keyed by run epoch, each
     * run's records contiguous and in arrival order.
     *
     * Records are grouped rather than flattened because a buffer's identity is
     * the only thing that says which graph its records belong to, and the pool
     * hands the same storage to a later run. Today the collection window holds
     * exactly one run — ``begin_run()`` clears this — so there is exactly one
     * entry; the grouping is what lets that stop being true without silently
     * merging two graphs.
     */
    const std::map<uint64_t, std::vector<DepGenRecord>> &runs() const { return records_by_run_; }

    /**
     * The records this collection window should emit as its graph.
     *
     * Three cases, and the distinction matters because a window with no records
     * is not the same as a window whose contents are ambiguous:
     *
     *   - one run  — that run's records, and its epoch through @p run_epoch_out.
     *   - no run   — an **empty** span, epoch 0. A run that submitted nothing
     *                has an empty graph, not a missing one, and the replay
     *                writer accepts `num_records == 0` to emit exactly that.
     *   - several  — nullptr. One path names one graph, so the caller cannot
     *                emit; silently picking one would produce a deps.json that
     *                claims to be the whole graph.
     *
     * A run present in ``runs()`` always has at least one record, because
     * ``append_buffer_records`` never creates an entry for an empty buffer — so
     * the one-run case can never masquerade as the no-run one.
     *
     * Valid between init() and finalize(); pointer/size stay stable after
     * stop() returns, which is when the caller hands them to
     * ``dep_gen_replay_emit_deps_json``.
     */
    // --- Cross-run retention -------------------------------------------------
    //
    // Off unless `configure_retained_runs(true, ...)` runs before init, in
    // which case the boundary keeps every step that touches the device — the
    // receive drain and the terminal read — and hands the replay, the
    // serialization and the file write to one background writer. Every entry
    // point below is inert with retention off.

    /** Latch retention and this collector's own host byte budget. */
    void configure_retained_runs(bool retain_across_runs, size_t budget_bytes);
    bool retains_runs() const { return retain_across_runs_; }

    /**
     * Admit one run, taking an export slot and recording the identity and the
     * destination the writer will publish under.
     *
     * False refuses the run, and the caller must fail it before anything
     * reaches the device: both unpublished slots are in use, the prefix does
     * not fit the path allowance, the budget cannot open, or this run's device
     * counter reset was not published. A refusal leaves no slot taken.
     *
     * `why` is meaningful only when this returns false, and says whether the
     * refusal is a capacity shortage the caller should re-attempt when credit
     * moves or a condition re-attempting cannot change. Any refusal this does
     * not classify reports `Fatal`: an over-strict answer fails the run early,
     * where an over-lenient one would retry until its deadline.
     */
    bool run_begin(
        uint64_t run_epoch, const std::string &output_prefix, simpler::dfx::runs::AdmissionRefusal *why = nullptr
    );

    /**
     * Close one run's boundary, under its execution claim.
     *
     * `device_execution_complete` is the caller's own fence observation and the
     * whole of this collector's completion proof. Without it the run reads
     * nothing shared, seals nothing and publishes nothing: the host copies are
     * quarantined until the existing collector-thread join, and a sticky error
     * is recorded.
     */
    void run_close(uint64_t run_epoch, bool device_execution_complete);

    /** Give an admitted run's slot back when its launch submitted nothing. */
    void abandon_run(uint64_t run_epoch);

    /** Wait for every closed run to be published. False when any run failed. */
    bool flush_retained_runs(int timeout_ms, std::string *error);

    /** Publish everything still queued. Runs before the collector threads stop. */
    void finish_retained_runs();

    /** Free quarantined host copies. Only legal once the reader threads are joined. */
    void discard_quarantined_runs();

    /** Counters a test reads instead of parsing files. */
    struct RetainedRunStats {
        uint64_t published{0};
        uint64_t refused{0};
        uint64_t quarantined{0};
        uint64_t host_failures{0};
        uint64_t refused_records{0};
        uint64_t foreign_epoch_records{0};
        size_t open_slots{0};
        size_t charged_bytes{0};
        bool has_error{false};
    };
    RetainedRunStats retained_run_stats_for_test() const;
    /** Hold the writer before it publishes, so a test can occupy export slots. */
    void pause_writer_for_test(bool paused);
    /** Make the next shared-region read report failure. */
    void fail_region_read_for_test(bool fail) { fail_region_read_ = fail; }
    /** Make this run's counter-reset publication report failure. */
    void fail_counter_reset_for_test(bool fail) { fail_counter_reset_ = fail; }
    /** Shrink the budget to the figure a test needs a charge to be refused at. */
    void shrink_budget_for_test(size_t budget_bytes) { retained_budget_bytes_ = budget_bytes; }

    const std::vector<DepGenRecord> *window_records(uint64_t *run_epoch_out = nullptr) const {
        if (records_by_run_.size() > 1) return nullptr;
        if (records_by_run_.empty()) {
            if (run_epoch_out != nullptr) *run_epoch_out = 0;
            return &kNoRecords;
        }
        const auto &entry = *records_by_run_.begin();
        if (run_epoch_out != nullptr) *run_epoch_out = entry.first;
        return &entry.second;
    }

private:
    bool initialized_ = false;
    int num_threads_ = 0;

    // Shared memory region (DepGenDataHeader + DepGenBufferState[1]).
    // shm_host_ / device_id_ live on ProfilerBase (set via set_memory_context
    // in init()).
    void *shm_dev_ = nullptr;
    size_t shm_size_ = 0;

    // In-memory records — drained from the device ring on
    // on_buffer_collected() and consumed by the host replay directly (no disk
    // hop), grouped by the run that produced them. Mutex serializes the mgmt
    // thread's appends against the (rare) reader on the same collector
    // instance.
    std::map<uint64_t, std::vector<DepGenRecord>> records_by_run_;
    std::mutex records_mutex_;

    // Returned by window_records() when no run produced anything, so the caller
    // gets an empty span rather than having to special-case a null.
    static inline const std::vector<DepGenRecord> kNoRecords{};

    // Running total of records appended across every run held here. Equal to
    // the summed sizes in ``records_by_run_`` after every append; kept
    // separately for the reconcile_counters cross-check even when the records
    // may be inspected concurrently.
    uint64_t total_collected_ = 0;

    DepGenDataHeader *dep_gen_header() const { return get_dep_gen_header(shm_host_); }
    DepGenBufferState *dep_gen_state(int idx = 0) const { return get_dep_gen_buffer_state(shm_host_, idx); }

    void append_buffer_records(const void *buf_host_ptr);

    // --- Cross-run retention state -------------------------------------------

    // A slot is held from admission until its artifact exists, so "two
    // unpublished exports" bounds what the collector owns rather than what is
    // merely still filling. The run being collected holds one of the two.
    enum class SlotState : int { Free = 0, Open = 1, Publishing = 2, Quarantined = 3 };

    struct Slot {
        SlotState state{SlotState::Free};
        uint64_t run_epoch{0};
        std::string output_dir;
        // The device bank this run's producer publishes into, held from
        // admission until the host has copied the accounting out of it. Every
        // path that gives a slot back assigns a fresh `Slot`, which releases
        // this with it.
        simpler::dfx::credit::Reservation terminal_bank;
    };

    /** Charge one record block, or refuse it. Always true with retention off. */
    bool charge_record_block(size_t bytes);
    /** Give a charged figure back. A no-op with retention off. */
    void credit_record_block(size_t bytes);

    /**
     * Record a host failure that belongs to the collector rather than to one
     * run, and keep it until the runner is destroyed.
     *
     * Allocates nothing, so it is safe on the paths that reach it precisely
     * because an allocation has just failed.
     */
    void note_host_failure(const char *detail) noexcept;

    void run_close_locked_path(uint64_t run_epoch, bool device_execution_complete);
    void quarantine_locked(uint64_t run_epoch, const char *detail);
    /** Publish one export: materialize, replay, then link the result into place. */
    int publish_export(simpler::dfx::dep_gen_runs::RunExport &data);
    void writer_loop();
    void ensure_writer_started();
    void stop_writer();
    /** True while this epoch may still stamp records the collector accepts. */
    bool epoch_admitted_locked(uint64_t run_epoch) const;

    std::function<void()> credit_notifier_;
    simpler::dfx::credit::RunTerminalBanks terminal_banks_{};
    // Kept from `init` so an allocation this collector makes later — the
    // terminal bank array, which only exists once retention maps one — uses the
    // same device allocator as its buffers.
    DepGenAllocCallback device_alloc_cb_{nullptr};
    DepGenFreeCallback device_free_cb_{nullptr};
    /**
     * The callback the terminal bank array is released through.
     *
     * Separate from `device_free_cb_` because the two are not the same
     * function at the only moment the array is released: `finalize` is handed
     * its own free callback, and a teardown that abandons device resources
     * hands one that frees nothing. The array is this collector's last device
     * allocation and must follow the same policy as the pooled buffers
     * released beside it, so `finalize` rebinds this to its own argument
     * before releasing — using the init-time allocator there would hand memory
     * back to an allocator the caller has deliberately stopped using.
     */
    DepGenFreeCallback bank_free_cb_{nullptr};
    /**
     * The teardown's unregister callback, stashed beside the free one and for
     * the same reason: the bank array's host mapping has to come down before
     * its memory does, and `finalize` is the only place that callback exists.
     */
    DepGenUnregisterCallback bank_unregister_cb_{nullptr};

    bool retain_across_runs_ = false;
    size_t retained_budget_bytes_ = simpler::dfx::runs::kDefaultBudgetBytes;
    bool retained_ready_ = false;
    bool quarantine_held_ = false;
    bool fail_region_read_ = false;
    bool fail_counter_reset_ = false;

    // Two locks, one order. `records_mutex_` owns everything the collector
    // threads write — the retained record store, the open epoch and the
    // receive-side counters — because the receive path is the hot one and must
    // never wait on the writer. `retained_mu_` owns the slots, the export
    // queue, the writer state and the stats. A path needing both takes
    // `retained_mu_` first, and `append_buffer_records` needs only the first.
    simpler::dfx::dep_gen_runs::RecordBlocks retained_records_;

    /**
     * One admitted run's own receipts, so two runs can be open at once.
     *
     * The retained path above recognises a single admitted epoch and files
     * everything else as foreign, which cannot serve a joined pair: the
     * successor's records would be refused as foreign while the predecessor is
     * sealing, and both runs would share one record store. Each entry is
     * claimed by `run_begin`, routed to by `append_buffer_records` using the
     * identity the producer stamped, and retired by that run's own close.
     *
     * Guarded by `records_mutex_`, the lock the receive path already holds. The
     * single store above stays in use on the unmapped path, where only one run
     * can be admitted at a time and the graph's bytes must not change.
     */
    struct RunReceipts {
        uint64_t run_epoch{0};
        bool open{false};
        simpler::dfx::dep_gen_runs::RecordBlocks records;
        uint64_t received{0};
        uint64_t refused{0};
        bool clamped{false};
    };
    std::array<RunReceipts, simpler::dfx::runs::kMaxOpenEpochs> run_receipts_{};

    /** Claim `run_epoch` its own receipts. False when every entry is in use. */
    bool open_run_receipts(uint64_t run_epoch);
    /** Release one run's receipts without publishing them. */
    void drop_run_receipts(uint64_t run_epoch);
    uint64_t retained_epoch_ = 0;
    bool retained_epoch_open_ = false;
    uint64_t refused_records_ = 0;
    uint64_t foreign_epoch_records_ = 0;
    bool host_clamped_ = false;
    bool reset_unpublished_ = false;

    mutable std::mutex retained_mu_;
    std::condition_variable retained_cv_;
    Slot slots_[simpler::dfx::runs::kMaxOpenEpochs];
    std::deque<simpler::dfx::dep_gen_runs::RunExport> queue_;
    std::vector<simpler::dfx::dep_gen_runs::RunExport> quarantined_;
    simpler::dfx::runs::HostBudget host_budget_;
    simpler::dfx::runs::ErrorSummary run_errors_;
    std::thread writer_;
    bool writer_running_ = false;
    bool writer_stop_ = false;
    bool writer_paused_ = false;
    bool writer_busy_ = false;

    struct RetainedStats {
        uint64_t published{0};
        uint64_t refused{0};
        uint64_t quarantined{0};
        uint64_t host_failures{0};
    };
    RetainedStats stats_;
};

/**
 * Build the ``deps.json`` output path under the caller-provided per-task
 * directory. Filename is fixed (no timestamp) — the directory is the
 * per-task uniqueness boundary, mirroring make_pmu_csv_path().
 */
inline std::string make_deps_json_path(const std::string &output_dir) {
    std::filesystem::path dir(output_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARN("Failed to create dep_gen output directory %s: %s", output_dir.c_str(), ec.message().c_str());
    }
    return (dir / "deps.json").string();
}

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_
