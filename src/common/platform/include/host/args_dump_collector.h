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
 * @file args_dump_collector.h
 * @brief Host-side args dump collector with independent shared memory.
 *
 * Architecture:
 * - BufferPoolManager<DumpModule>: shared split-mgmt infrastructure that
 *   polls per-thread ready queues, replenishes free_queues, and hands
 *   full DumpMetaBuffers off to collector thread shards.
 * - ArgsDumpCollector: copies tensor metadata + arena bytes into host
 *   vectors and writes the result to disk (.bin + JSON).
 *
 * a5 specifics: device↔host transfers use rtMemcpy / memcpy via
 * profiling_copy.h. The framework's mgmt loop mirrors the shm region per
 * tick; per-buffer payloads (metadata buffers) are pulled on demand inside
 * ProfilerAlgorithms. The collector additionally pulls arena bytes inside
 * on_buffer_collected, since arenas live outside the shm region and only
 * the part needed for the buffer's records is worth copying.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_ARGS_DUMP_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_ARGS_DUMP_COLLECTOR_H_

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/args_dump.h"
#include "common/unified_log.h"
#include "data_type.h"
#include "host/args_dump_runs.h"
#include "host/profiler_base.h"
#include "host/profiling_copy.h"

// ---------------------------------------------------------------------------
// Args Dump profiling Module (drives BufferPoolManager<DumpModule>)
// ---------------------------------------------------------------------------

/**
 * One buffer kind (DumpMetaBuffer); one ready_queue per AICPU thread.
 * Per-thread arena buffers are owned by the collector itself, not the
 * framework. Runtime refill uses the owning drain shard's local
 * recycled/done lanes; proactive_replenish may batch-allocate before drain
 * and collector threads start.
 */

/**
 * Information about a ready (full) dump metadata buffer.
 */
struct DumpReadyBufferInfo {
    uint32_t thread_index;
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

struct DumpModule {
    using DataHeader = DumpDataHeader;
    using ReadyEntry = DumpReadyQueueEntry;
    using ReadyBufferInfo = ::DumpReadyBufferInfo;
    using FreeQueue = DumpFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_DUMP_READYQUEUE_SIZE;
    static constexpr uint32_t kHostPoolQueueSize = PLATFORM_MAX_AICPU_THREADS * PLATFORM_DUMP_BUFFERS_PER_THREAD;
    static constexpr uint32_t kSlotCount = PLATFORM_DUMP_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "DumpModule";
    // Producers are the scheduler threads, one per AICPU thread.
    static constexpr int kMaxCollectorThreads = PLATFORM_MAX_AICPU_THREADS;

    /**
     * Args-dump bursts can be very large; this is the startup-only batch
     * size used when proactive_replenish needs to grow recycled lanes.
     */
    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_DUMP_BUFFERS_PER_THREAD - PLATFORM_DUMP_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_dump_header(shm); }

    static std::optional<profiling_common::EntrySite<DumpModule>>
    resolve_entry(void *shm, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        if (shm == nullptr || header == nullptr) {
            LOG_ERROR("DumpModule: invalid shared memory/header while resolving ready entry");
            return std::nullopt;
        }
        if (entry.thread_index >= header->num_dump_threads ||
            entry.thread_index >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
            LOG_ERROR(
                "DumpModule: invalid ready entry thread=%u (num_dump_threads=%u, max=%u)", entry.thread_index,
                header->num_dump_threads, static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)
            );
            return std::nullopt;
        }
        DumpBufferState *state = get_dump_buffer_state(shm, static_cast<int>(entry.thread_index));
        profiling_common::EntrySite<DumpModule> site;
        site.kind = 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = sizeof(DumpMetaBuffer);
        site.info.thread_index = entry.thread_index;
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *shm, DataHeader *header, Cb &&cb) {
        const int n_threads = static_cast<int>(header->num_dump_threads);
        for (int t = 0; t < n_threads; t++) {
            DumpBufferState *state = get_dump_buffer_state(shm, t);
            cb(/*kind=*/0, &state->free_queue, sizeof(DumpMetaBuffer));
        }
    }
};

// ---------------------------------------------------------------------------
// Memory operation callbacks (injected by DeviceRunner)
// ---------------------------------------------------------------------------

// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals. On a5 onboard the runner passes
// register_cb=nullptr and the framework installs a malloc-shadow + DMA
// fallback inline in ProfilerBase::set_memory_context().
using DumpAllocCallback = profiling_common::ProfAllocCallback;
using DumpRegisterCallback = profiling_common::ProfRegisterCallback;
using DumpUnregisterCallback = profiling_common::ProfUnregisterCallback;
using DumpFreeCallback = profiling_common::ProfFreeCallback;

// =============================================================================
// ArgsDumpCollector
// =============================================================================

/**
 * Collected arg metadata + payload bytes
 */
struct DumpedArg {
    // Which run produced this arg. Copied from the device buffer's stamp at
    // collection time, never read back from it: the pool reuses that storage and
    // a later run re-stamps it in place. 0 means the producer had no run
    // identity to stamp.
    uint64_t run_epoch;
    uint32_t local_seq;  // Producing buffer's position within its own run
    uint64_t task_id;
    int32_t func_ids[ARGS_DUMP_MAX_FUNC_IDS];  // task's active-subtask set (mix membership); -1 unknown
    int32_t func_count;                        // number of valid entries in func_ids
    uint32_t arg_index;
    ArgsDumpRole role;
    ArgsDumpStage stage;
    uint8_t dtype;
    uint8_t ndims;
    uint8_t flags;
    ArgsDumpKind kind;
    uint64_t scalar_value;
    uint64_t start_offset;                     // 1D element offset of the view origin
    uint32_t shapes[PLATFORM_DUMP_MAX_DIMS];   // Current view shape
    uint32_t strides[PLATFORM_DUMP_MAX_DIMS];  // Element stride per dim (> 0, type-enforced)
    bool is_contiguous;
    bool truncated;
    // True when this record's payload could not be taken on the host — a
    // refused charge or a failed allocation. The metadata is kept and the
    // manifest says so per arg; the run's verdict says so for the run.
    bool host_discarded{false};
    uint64_t payload_size;
    uint64_t bin_offset;
    std::vector<uint8_t> bytes;
};

/**
 * The retained-run budget quotes this size per collected record, so it is
 * pinned rather than estimated: a field added without revisiting the charge
 * would make the accounting silently optimistic.
 */
static_assert(sizeof(DumpedArg) <= 256, "DumpedArg grew past the size the retained-run budget charges");

class ArgsDumpCollector : public profiling_common::ProfilerBase<ArgsDumpCollector, DumpModule> {
public:
    ArgsDumpCollector() = default;
    ~ArgsDumpCollector();

    ArgsDumpCollector(const ArgsDumpCollector &) = delete;
    ArgsDumpCollector &operator=(const ArgsDumpCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_DUMP_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "ArgsDump";

    /**
     * Initialize args dump shared memory.
     *
     * Allocates the DumpDataHeader + per-thread DumpBufferState array, the
     * per-thread arenas (single contiguous payload region per thread), and
     * the initial DumpMetaBuffers. The first PLATFORM_DUMP_SLOT_COUNT meta
     * buffers are pushed into each thread's free_queue; the rest go into
     * the BufferPoolManager's recycled pool.
     *
     * `output_prefix` is the per-task directory under which `args_dump/`
     * lands. Required (non-empty); CallConfig::validate() enforces this
     * upstream. Stored on the collector so the lazily-started writer thread
     * (kicked off inside on_buffer_collected) can derive its run_dir
     * without threading the prefix through the buffer-pool callback path.
     *
     * @param num_dump_threads  Number of AICPU scheduling threads
     * @param device_id         Device ID
     * @param alloc_cb          Memory allocation callback
     * @param register_cb       Host-visibility callback (nullptr on a5)
     * @param free_cb           Memory free callback
     * @param user_data         Opaque pointer forwarded to callbacks
     * @param output_prefix     Per-task directory; args_dump/ subdir lands here
     * @param dump_args_level OFF / PARTIAL (only Arg::dump()-marked args) /
     *                          FULL / HYBRID (every task's metadata,
     *                          with Arg::dump()-marked tensor payload). Written
     *                          to DumpDataHeader so the AICPU latches the mode
     *                          before any dispatch.
     * @return 0 on success, error code on failure
     */
    // Allocates the device-side resources: header, per-thread DumpBufferStates,
    // DumpMetaBuffers and payload arenas.
    //
    // The level is taken here because it is written into DumpDataHeader with the
    // rest of the layout. The prefix is bound by begin_run(), which runs once per
    // run and may run either side of this.
    int initialize(
        int num_dump_threads, int device_id, DumpArgsLevel dump_args_level, const DumpAllocCallback &alloc_cb,
        DumpRegisterCallback register_cb, const DumpFreeCallback &free_cb
    );

    // Start a run's collection window: bind its artifact configuration, drop the
    // previous run's shard state and counters, and — once the region exists —
    // republish the level the device reads. The prefix is read when the writer
    // thread starts lazily on the first collected buffer.
    //
    // The collector initializes once and serves every run, so this is the only
    // point at which a run's collected records, dropped/truncated counts and
    // device level are established; initialize() establishes none of them.
    void begin_run(const std::string &output_prefix, DumpArgsLevel dump_args_level);

    void start(const profiling_common::ThreadFactory &thread_factory);

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Pulls the
     * relevant portion of the originating thread's arena from device, copies
     * tensor metadata + arena bytes into host-side DumpedArg records, and
     * queues payloads to the writer thread. The writer thread is started
     * lazily on the first invocation per run.
     */
    void on_buffer_collected(const DumpReadyBufferInfo &info, int collector_shard);

    /**
     * Write collected dumps to <output_prefix>/args_dump/{*.bin, *.json}.
     * Sorts args by (task_id, stage, arg_index, role).
     */
    int export_dump_files();

    /**
     * After stop():
     *   - Recover records from any non-empty DumpBufferState::current_buf_ptr
     *     left behind by abnormal exit before device-side flush ran.
     *   - Accumulate device-side dropped_record_count into
     *     total_dropped_record_count_ for the final anomaly report.
     * Must be called after stop().
     */
    void reconcile_counters();

    /**
     * Free all device memory and unregister mappings (per-thread arenas,
     * DumpMetaBuffers held by the framework or still in per-pool free
     * queues). Idempotent on a collector that was never initialized.
     *
     * On the retained path this is also where the last host sealing happens,
     * and its result is returned rather than logged: an error discovered here
     * is later than the caller's own diagnostic flush, so it can only reach the
     * caller through this return value.
     */
    int finalize(DumpUnregisterCallback unregister_cb, const DumpFreeCallback &free_cb);

    // -----------------------------------------------------------------------
    // Cross-run retention
    // -----------------------------------------------------------------------

    /**
     * Whether this collector may hold a run's payload file and manifest past
     * that run's boundary.
     *
     * Latched by the runner at device init from `collect_across_runs`. Default
     * false, and with it false every path below is unreachable: the collector
     * keeps its single-run window, its per-run counter resets, its `args.bin`
     * and its boundary join, byte for byte.
     */
    void configure_retained_runs(bool enabled, size_t budget_bytes);
    bool retains_runs() const { return retain_across_runs_; }

    /**
     * Admit one run, freezing everything its records and its files depend on.
     *
     * Returns false — before any kernel is submitted — when both epoch slots
     * are occupied, when the destination is owned by an open epoch or still
     * holds a failed run's preserved temporary manifest, when no exclusive
     * output token is available, when the path exceeds the allowance, when the
     * fixed charge is refused, when the device did not take this run's dump
     * level, or when the collector is fatal. A refusal must fail the run:
     * falling back to `begin_run` would reset counters a predecessor's writer
     * is still acknowledging against.
     */
    bool run_begin(uint64_t run_epoch, const std::string &output_prefix, DumpArgsLevel dump_args_level);

    /**
     * Close one run's collection window while it still holds the execution
     * claim and the successor cannot start: read this run's terminal lane
     * state, complete the finite transport cut — including the proof that
     * every buffer it published has been processed into host-owned storage —
     * decide each lane's leftover buffer, and hand the run to the background
     * writer. Does not drain and does not merge.
     *
     * Returns 0 when this run's content is proved owned by the host. Non-zero
     * means the proof did not land inside the claim: nothing unproved was read,
     * no payload still in the arena was acknowledged, and the caller owes its
     * own error rather than a run that merely publishes an incomplete file.
     */
    int run_close(uint64_t run_epoch, bool device_execution_complete);

    /**
     * Give back the slot of a run that was admitted and never launched.
     *
     * Returns false when the reference release could not be proved; the epoch
     * is then quarantined rather than released, exactly as a failed close is.
     */
    bool abandon_run(uint64_t run_epoch);

    /**
     * Wait for every run closed up to now to be published, then report.
     *
     * Always both halves: the wait is what a caller about to read this run's
     * files needs, and the sticky record is what a caller needs after a
     * rebuild turned retention off. A disk that is not draining makes this
     * time out — the wait is bounded by the caller's budget, not by a promise
     * that the writer finishes.
     */
    bool flush_retained_runs(int timeout_ms, std::string *error);

    /** Stop admitting and publish what is still retained. Reports nothing. */
    void finish_retained_runs();

    /** ProfilerBase hook: adopt the epoch table, then acknowledge it. */
    void refresh_retained_run_view(int collector_shard);

    /** ProfilerBase hook: transport progress the background writer waits on. */
    void note_transport_progress();

    /** Counts a test can assert on without reaching into collector internals. */
    struct RetainedRunStats {
        bool retaining{false};
        bool ready{false};
        bool fatal{false};
        size_t open_epochs{0};
        size_t quarantined_epochs{0};
        size_t charged_bytes{0};
        uint64_t budget_refusals{0};
        // Records the still-open runs have received, so a case can wait for a
        // receipt it needs to have landed before it closes the run.
        uint64_t open_collected_records{0};
        uint64_t open_discarded_args{0};
        simpler::dfx::args_dump::ErrorSummary::Counts errors{};
    };
    RetainedRunStats retained_run_stats_for_test() const;

    /**
     * Hold the background writer just before it would take anything, so a case
     * can prove that a payload was host-owned while no byte of it had reached
     * the disk, or that a manifest was written while a named run was still
     * open. Held means this thread does nothing at all: no payload write and no
     * seal, while the shards keep receiving. Bounded and self-releasing — the
     * writer stop clears it before joining. Production never sets it.
     */
    void hold_retained_writer_for_test(bool held);

    /**
     * Make the retained path's arena reads report failure, so a case can prove
     * that a failed device-to-host copy is recorded as loss instead of
     * exporting whatever the host shadow still held. Affects only the retained
     * receive path; production never sets it.
     */
    void fail_arena_copy_for_test(bool fail) { retained_fail_arena_copy_.store(fail, std::memory_order_release); }

    /**
     * The lifetime transport credit one lane has accumulated: the payloads this
     * host has taken from that lane's arena plus the ones it wrote off, which
     * is the left-hand side of the acknowledgement equation. A case asserts on
     * it to prove that a recovered payload — which the device never published —
     * contributed nothing to it.
     */
    uint64_t retained_lane_transport_credit_for_test(int lane) const {
        if (lane < 0 || static_cast<size_t>(lane) >= received_payload_counts_.size()) return 0;
        return received_payload_counts_[static_cast<size_t>(lane)].load(std::memory_order_acquire) +
               discarded_payload_counts_[static_cast<size_t>(lane)].load(std::memory_order_acquire);
    }

    /**
     * Deliver one already-filled buffer through the real routing path, on the
     * caller's thread, so a case can choose the instant of a receipt that the
     * drain and collector threads would otherwise choose for it.
     */
    void deliver_buffer_for_test(const DumpReadyBufferInfo &info, int collector_shard);

    /**
     * @return true if initialize() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return shm_host_ != nullptr; }

    /**
     * Device pointer to the DumpDataHeader. Set kernel_args.dump_data_base
     * to this after initialize() succeeds so the AICPU side can find the
     * shared memory.
     */
    void *get_dump_shm_device_ptr() const { return dump_shared_mem_dev_; }

    /**
     * Publish, per AICPU thread, how many of that thread's payloads have reached
     * args.bin. The device blocks on this watermark before overwriting arena
     * bytes. Called once per replenish tick; a no-op before initialize().
     */
    void publish_arena_acks();

private:
    // Declared here so the receive path can take one by pointer; defined with
    // the rest of the retention state below.
    struct RetainedEpoch;

    struct alignas(64) CollectorShardCounters {
        uint64_t total_collected{0};
    };
    static_assert(
        sizeof(CollectorShardCounters) % 64 == 0, "CollectorShardCounters must not share cache lines across shards"
    );

    void *dump_shared_mem_dev_{nullptr};
    int num_dump_threads_{0};

    // Per-task output directory captured at initialize() time. The writer
    // thread builds run_dir_ = output_prefix_ / "args_dump" lazily on the
    // first on_buffer_collected.
    std::string output_prefix_;

    // Per-thread arena pointers (device + host shadow)
    struct ArenaInfo {
        void *dev_ptr{nullptr};
        void *host_ptr{nullptr};
        uint64_t size{0};
    };
    std::vector<ArenaInfo> arenas_;

    // Merged dump args (metadata only; payloads live in args.bin).
    // Collector shards append to collected_by_collector_ on the hot path, then
    // export folds those shards into collected_ before sorting/writing JSON.
    std::vector<DumpedArg> collected_;
    std::vector<std::vector<DumpedArg>> collected_by_collector_;
    std::vector<CollectorShardCounters> collector_counters_;
    bool collector_shards_merged_{false};
    std::atomic<uint64_t> total_metadata_collected_{0};

    // Stats
    std::atomic<uint32_t> total_dropped_record_count_{0};
    std::atomic<uint32_t> total_truncated_count_{0};
    std::array<std::atomic<uint64_t>, PLATFORM_MAX_AICPU_THREADS> written_payload_counts_{};

    // Run-scoped state for the writer thread (lazily started on first
    // on_buffer_collected and joined by export_dump_files).
    std::chrono::steady_clock::time_point run_start_time_;
    std::atomic<int64_t> last_progress_ms_{0};
    bool writer_started_{false};

    size_t normalize_collector_shard(int collector_shard) const;
    void reset_collector_shards();
    void merge_collector_shards();
    /**
     * Copy one delivered buffer's records — and the payload bytes they name —
     * into host-owned storage.
     *
     * `forced_epoch` is the retained close boundary's route for a buffer the
     * device never published: it names the epoch and the bucket directly
     * because that buffer reached no ready queue, so no shard view resolves it
     * and no receipt describes it. Everything else — the charges, the arena
     * copy and the single offset-and-enqueue commit — is the same path a
     * delivered buffer takes.
     */
    void process_dump_buffer(
        const DumpReadyBufferInfo &info, int collector_shard, RetainedEpoch *forced_epoch = nullptr,
        size_t forced_bucket = 0
    );
    void start_writer_thread_once();

    // Writer thread: streams arg payloads to a single args.bin
    struct PayloadWriteRequest {
        uint32_t thread_index;
        // Which retained epoch's payload file these bytes belong to, and the
        // identity that slot must still carry when the writer reaches them.
        // -1 on the default path, where there is one file and one run.
        int epoch_slot{-1};
        uint64_t epoch{0};
        std::vector<uint8_t> bytes;
    };
    std::thread writer_thread_;
    std::mutex writer_start_mutex_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    std::queue<PayloadWriteRequest> write_queue_;
    std::atomic<bool> writer_done_{false};

    // What one queued payload request costs beyond its own bytes: the request
    // and a bound on its share of the deque block it lives in. Charged with the
    // payload so an admitted payload always has a node to travel in, and
    // credited when the writer is done with it.
    static constexpr size_t kRetainedQueueNodeBytes = sizeof(PayloadWriteRequest) + 64;

    // Resolved dump level; HYBRID creates .bin lazily when an
    // Arg::dump()-selected tensor contributes payload.
    DumpArgsLevel dump_args_level_{DumpArgsLevel::OFF};

    // Output directory and single binary file
    std::filesystem::path run_dir_;
    std::ofstream bin_file_;
    uint64_t next_bin_offset_{0};

    // Writer stats
    std::atomic<uint64_t> bytes_written_{0};

    void writer_loop();

    /**
     * Ask the writer thread to finish, in the one order that cannot lose the
     * wakeup: set the stop flag under `write_mutex_`, then notify. Both stop
     * sites (export and finalize) go through here so neither can regress to a
     * bare atomic store — see the definition for why the mutex is required even
     * though the flag is atomic.
     */
    void request_writer_stop();

    // -----------------------------------------------------------------------
    // Cross-run retention state
    // -----------------------------------------------------------------------

    /**
     * A retained epoch's lifecycle.
     *
     * `Admitting` is the only state a collector shard may route a buffer into;
     * `Closing` is the window in which the reference release is being proved;
     * `Quarantined` is terminal — its files and its slot are held until the
     * collector threads are joined, so the slot never returns to `Free` by any
     * other path.
     */
    enum class EpochState : int { Free = 0, Admitting, Closing, Quarantined };

    /** One retained run's frozen configuration, host records and proofs. */
    struct RetainedEpoch {
        std::atomic<int> state{static_cast<int>(EpochState::Free)};
        std::atomic<uint64_t> epoch{0};

        // Frozen at admission, before any callback can receive this run's data.
        std::filesystem::path run_dir;
        simpler::dfx::args_dump::OutputToken token;
        DumpArgsLevel level{DumpArgsLevel::OFF};

        // One metadata bucket per collector shard plus one the close boundary
        // owns for what it recovers, so a recovered buffer never shares a
        // vector with a live drain shard. Each bucket's charged bytes are its
        // capacity, so a growth's transient peak is charged before it happens.
        std::vector<std::vector<DumpedArg>> buckets;
        std::vector<size_t> bucket_charged;

        // Per lane, written by the drain shard that serves that lane while the
        // epoch admits, read by the close boundary afterwards.
        std::vector<simpler::dfx::args_dump::LaneReceipt> receipts;

        // Per lane at admission. The device's published and dropped counters
        // are not reset per run on this path, so a run's own figures are its
        // close reading minus these.
        std::vector<uint64_t> admit_published;
        std::vector<uint32_t> admit_dropped;

        // The payload cursor and the write queue's order are committed under
        // `write_mutex_` together, so the manifest's offsets and the file's
        // append order cannot disagree.
        uint64_t next_bin_offset{0};
        // This run's requests still in the shared write queue. Incremented in
        // the same commit as the offset and decremented when the writer is
        // done with one, so a seal can drain exactly its own run rather than
        // waiting for a queue a successor keeps refilling.
        std::atomic<uint64_t> queued_payloads{0};
        std::ofstream payload_file;
        bool payload_opened{false};
        std::atomic<uint64_t> payload_bytes_written{0};

        std::atomic<uint64_t> collected_records{0};
        std::atomic<uint64_t> truncated_records{0};
        std::atomic<uint64_t> discarded_args{0};
        std::atomic<uint64_t> discarded_metadata_records{0};
        std::atomic<bool> io_failed{false};

        simpler::dfx::args_dump::RecordProofs records{};
        simpler::dfx::args_dump::CutProofs cut{};
        int cut_slot{-1};
        uint64_t cut_request{0};
        bool target_installed{false};
        // Set by the close once it has taken this run's cut proof under the
        // execution claim. The writer seals on what the close recorded and
        // never re-evaluates it: a later, more favourable reading of the same
        // cut would turn a run whose leftover was deliberately not read into a
        // run that reads as complete.
        bool cut_settled_at_close{false};
        std::chrono::steady_clock::time_point closed_at{};
        size_t fixed_charge{0};

        void reset_run_state(size_t bucket_count, int lane_count);
    };

    /** Which epochs one collector shard may route into. */
    struct ShardEpochView {
        struct Entry {
            uint64_t epoch{0};
            int slot{-1};
        };
        std::array<Entry, simpler::dfx::runs::kMaxOpenEpochs> entries{};
        size_t count{0};

        int slot_for(uint64_t epoch) const {
            for (size_t i = 0; i < count; i++) {
                if (entries[i].epoch == epoch && entries[i].slot >= 0) return entries[i].slot;
            }
            return -1;
        }
    };

    bool retain_across_runs_{false};
    size_t retained_budget_bytes_{simpler::dfx::runs::kDefaultBudgetBytes};
    std::atomic<bool> retained_ready_{false};
    std::atomic<bool> retained_fatal_{false};
    std::string retained_fatal_reason_;
    std::array<RetainedEpoch, simpler::dfx::runs::kMaxOpenEpochs> retained_epochs_{};
    std::array<ShardEpochView, DumpModule::kMaxCollectorThreads> shard_views_{};
    simpler::dfx::runs::HostBudget retained_budget_;
    // Sticky to this collector's destruction: no reset, no clear, no ack API,
    // and removing the evidence files on disk does not clear it either.
    simpler::dfx::args_dump::ErrorSummary retained_errors_;

    mutable std::mutex retained_mu_;
    std::condition_variable retained_cv_;
    uint64_t retained_progress_{0};
    std::atomic<uint64_t> retained_close_watermark_{0};
    std::atomic<bool> retained_release_deferred_{false};
    std::atomic<bool> retained_writer_held_{false};
    // Test seam only; see `fail_arena_copy_for_test`. Production never sets it,
    // so the retained receive path pays one relaxed load per buffer.
    std::atomic<bool> retained_fail_arena_copy_{false};
    // Per lane, host dispositions that did not write a payload byte. Counted
    // beside the writer's own count so the arena acknowledgement can settle a
    // lane whose payload was deliberately written off.
    std::array<std::atomic<uint64_t>, PLATFORM_MAX_AICPU_THREADS> discarded_payload_counts_{};
    // Per lane, payloads this host has taken out of the arena into storage it
    // owns. On the retained path this — not the disk count — is what releases
    // the producer's arena barrier: once the bytes are host-owned the arena is
    // free whatever the disk is doing, and disk completion is proved by the
    // flush instead. Monotonic for the collector's life, like the lane counters
    // it is compared against.
    std::array<std::atomic<uint64_t>, PLATFORM_MAX_AICPU_THREADS> received_payload_counts_{};

    bool retained_ensure_ready();
    void retained_set_fatal(const char *reason);
    int retained_find_slot(uint64_t run_epoch) const;
    void retained_release_slot(size_t slot);
    void retained_credit_epoch(size_t slot);
    bool retained_bucket_reserve_one(RetainedEpoch &epoch, size_t bucket);
    void retained_service();
    void retained_seal(size_t slot);
    bool retained_publish_manifest(RetainedEpoch &epoch, simpler::dfx::args_dump::Verdict verdict);
    void retained_finish_epoch(size_t slot, simpler::dfx::args_dump::Verdict verdict, const char *detail);
    bool retained_read_terminal_state(RetainedEpoch &epoch);
    void retained_decide_leftovers(RetainedEpoch &epoch);
    bool retained_publish_level(DumpArgsLevel level);
    /**
     * Append one received record to its run's bucket, committing a payload's
     * offset and its queue node together.
     *
     * `transport_published` is false for a forced-recovery buffer: its
     * payloads never entered a ready queue, so the device never counted them
     * and neither may this lane's acknowledgement counters. The record itself
     * is still this run's output, and its loss is still this run's loss.
     */
    bool retained_append_record(
        RetainedEpoch &epoch, size_t bucket, DumpedArg &&arg, uint32_t lane, bool has_payload, bool transport_published
    );
    void retained_withdraw_unpublished_slot(size_t slot);
    void retained_writer_loop();
    bool retained_drain_payload_batch(int max_requests);
    bool retained_wait_for_processing(RetainedEpoch &epoch, uint64_t run_epoch);
    void retained_bump_progress();
    void retained_start_writer();
    void retained_stop_writer();
    void retained_finish();
    void retained_release_resources();
    void retained_release_quarantined();
    int retained_recover_leftover_records(RetainedEpoch &epoch, int lane, uint64_t dev_ptr, uint32_t expected_seq);

    std::thread retained_writer_thread_;
    std::atomic<bool> retained_writer_running_{false};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_ARGS_DUMP_COLLECTOR_H_
