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
 * @file args_dump_collector.cpp
 * @brief Host-side args dump collector implementation. The mgmt-thread +
 *        buffer-pool machinery lives in profiling_common::BufferPoolManager
 *        parameterized by DumpModule (host/args_dump_collector.h); the
 *        poll loop lives in profiling_common::ProfilerBase. This file owns
 *        the per-buffer on_buffer_collected callback, arena reads, and disk
 *        export.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. The
 * framework pulls queue fields and each popped DumpMetaBuffer on demand.
 * on_buffer_collected separately refreshes the originating thread's arena
 * write cursor and payload bytes because arenas live outside the shm region.
 */

#include "host/args_dump_collector.h"

#include "data_type.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/args_dump_manifest.h"
#include "../../../worker/runtime_c_api.h"

// =============================================================================
// ArgsDumpCollector
// =============================================================================

ArgsDumpCollector::~ArgsDumpCollector() {
    // Both threads this collector can own, in the order that cannot leave one
    // joinable: a `std::thread` destroyed while joinable terminates the
    // process, and the retained writer is started by an admission rather than
    // by `start()`, so a collector destroyed without a finalize — a test that
    // unwound, a runner that failed before teardown — must still join it here.
    retained_stop_writer();
    if (writer_thread_.joinable()) {
        request_writer_stop();
        writer_thread_.join();
    }
    stop();
}

static int64_t steady_clock_ms(std::chrono::steady_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
}

size_t ArgsDumpCollector::normalize_collector_shard(int collector_shard) const {
    const size_t shard_count = collected_by_collector_.size();
    const bool valid_shard = collector_shard >= 0 && static_cast<size_t>(collector_shard) < shard_count;
    if (!valid_shard) {
        assert(false && "collector_shard out of range");
        return shard_count;
    }
    return static_cast<size_t>(collector_shard);
}

void ArgsDumpCollector::begin_run(const std::string &output_prefix, DumpArgsLevel dump_args_level) {
    output_prefix_ = output_prefix;
    dump_args_level_ = dump_args_level;
    reset_collector_shards();
    total_dropped_record_count_.store(0, std::memory_order_relaxed);
    total_truncated_count_.store(0, std::memory_order_relaxed);
    last_progress_ms_.store(0, std::memory_order_relaxed);
    // The lane counters are the one piece of per-run state a retained run must
    // not reset: a predecessor's writer is still acknowledging payloads against
    // them, and zeroing them here would make its residual progress land in this
    // run's generation. `run_begin` is the retained path's admission and this
    // function is not on it, so the guard only documents which path owns them.
    if (retain_across_runs_) {
        LOG_ERROR("Args dump: begin_run reached on a collector that retains runs; run_begin owns admission");
        return;
    }
    for (auto &count : written_payload_counts_) {
        count.store(0, std::memory_order_relaxed);
    }
    for (auto &count : discarded_payload_counts_) {
        count.store(0, std::memory_order_relaxed);
    }
    for (auto &count : received_payload_counts_) {
        count.store(0, std::memory_order_relaxed);
    }

    // Before the first initialize() there is no region; initialize() writes the
    // level from the member just set. Afterwards the device needs the new value
    // by another route, and it is one narrow field rather than a bulk write-back
    // so it cannot race the AICPU's own header fields.
    if (shm_host_ != nullptr) {
        DumpDataHeader *header = get_dump_header(shm_host_);
        header->dump_args_level = static_cast<uint32_t>(dump_args_level_);
        wmb();
        publish_field(&header->dump_args_level, sizeof(header->dump_args_level), "dump_args_level");

        // The per-thread payload counters are what reconcile compares against,
        // and nothing on the device resets them. published/completed/dropped are
        // contiguous, so one write-back per thread covers them.
        //
        // arena_write_offset is deliberately NOT reset: it is a monotonic cursor
        // the host reads modulo arena_size, so it stays correct across runs.
        static_assert(
            offsetof(DumpBufferState, dropped_record_count) ==
                offsetof(DumpBufferState, published_payload_count) + 2 * sizeof(uint64_t),
            "the payload counters must stay contiguous for this single write-back to cover them"
        );
        constexpr size_t kCounterSpan = 2 * sizeof(uint64_t) + sizeof(uint32_t);
        // The region holds num_dump_threads_ states (calc_dump_data_size), so
        // that is the bound — a wider loop writes past its end. The runner
        // rebuilds this collector when a run's thread count changes, so a
        // resident one is never asked to reset a state it does not own.
        for (int t = 0; t < num_dump_threads_; t++) {
            DumpBufferState *state = get_dump_buffer_state(shm_host_, t);
            state->published_payload_count = 0;
            state->completed_payload_count = 0;
            state->dropped_record_count = 0;
            wmb();
            publish_field(&state->published_payload_count, kCounterSpan, "payload counters");
        }
    }
}

void ArgsDumpCollector::reset_collector_shards() {
    const size_t shard_count = static_cast<size_t>(manager_.shard_count());
    collected_.clear();
    collected_by_collector_.assign(shard_count, {});
    collector_counters_.assign(shard_count, {});
    collector_shards_merged_ = false;
    total_metadata_collected_.store(0, std::memory_order_relaxed);
}

void ArgsDumpCollector::merge_collector_shards() {
    if (collector_shards_merged_) {
        return;
    }

    size_t total_records = 0;
    for (const auto &shard_records : collected_by_collector_) {
        total_records += shard_records.size();
    }

    collected_.clear();
    collected_.reserve(total_records);
    for (const auto &shard_records : collected_by_collector_) {
        collected_.insert(collected_.end(), shard_records.begin(), shard_records.end());
    }
    collector_shards_merged_ = true;
}

int ArgsDumpCollector::initialize(
    int num_dump_threads, int device_id, DumpArgsLevel dump_args_level, const DumpAllocCallback &alloc_cb,
    DumpRegisterCallback register_cb, const DumpFreeCallback &free_cb
) {
    if (shm_host_ != nullptr) {
        // Already holding this run's device resources. They are not per-run:
        // configuration arrives via begin_run() and the layout is fixed at
        // compile time, so there is nothing here left to re-apply.
        return 0;
    }
    dump_args_level_ = dump_args_level;
    if (num_dump_threads <= 0 || num_dump_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "ArgsDumpCollector::initialize: invalid num_dump_threads=%d (valid range: 1-%d)", num_dump_threads,
            PLATFORM_MAX_AICPU_THREADS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Must precede the recycled-lane seeding below: push_recycled() folds its
    // shard argument modulo the manager's shard count.
    set_aicpu_thread_num(num_dump_threads);

    num_dump_threads_ = num_dump_threads;
    reset_collector_shards();
    total_dropped_record_count_.store(0, std::memory_order_relaxed);
    total_truncated_count_.store(0, std::memory_order_relaxed);
    last_progress_ms_.store(0, std::memory_order_relaxed);
    for (auto &count : written_payload_counts_) {
        count.store(0, std::memory_order_relaxed);
    }

    // Stash the memory context on the base up-front so alloc_paired_buffer
    // (which reads alloc_cb_/register_cb_/free_cb_/device_id_)
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — that nullptr guard makes a post-failure
    // start(tf) a no-op without further bookkeeping.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // RAII rollback: any early return after this point releases the shm
    // region + per-thread arenas + DumpMetaBuffers through the framework's
    // dev→host map. `guard.commit()` runs on the success path before the
    // trailing return 0.
    profiling_common::InitRollbackGuard<decltype(manager_)> guard(manager_, free_cb);

    // Allocate dump shared memory (header + buffer states)
    size_t shm_size = calc_dump_data_size(num_dump_threads);
    void *shm_host_local = nullptr;
    void *shm_dev_local = alloc_paired_buffer(shm_size, &shm_host_local);
    if (shm_dev_local == nullptr) {
        LOG_ERROR("Failed to allocate dump shared memory (%zu bytes)", shm_size);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Initialize header on host shadow
    std::memset(shm_host_local, 0, shm_size);
    DumpDataHeader *header = get_dump_header(shm_host_local);
    header->magic = ARGS_DUMP_MAGIC;
    header->num_dump_threads = static_cast<uint32_t>(num_dump_threads);
    header->records_per_buffer = PLATFORM_DUMP_RECORDS_PER_BUFFER;
    header->dump_args_level = static_cast<uint32_t>(dump_args_level_);

    uint64_t arena_size = calc_dump_arena_size();
    header->arena_size_per_thread = arena_size;

    // Allocate per-thread arenas (device + host shadow). Track the dev↔host
    // mapping so on_buffer_collected can pull arena bytes via the framework.
    arenas_.resize(num_dump_threads);
    for (int t = 0; t < num_dump_threads; t++) {
        ArenaInfo &ai = arenas_[t];
        ai.size = arena_size;
        ai.dev_ptr = alloc_paired_buffer(arena_size, &ai.host_ptr);
        if (ai.dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate dump arena for thread %d (%lu bytes)", t, arena_size);
            return PTO_RUNTIME_ERR_INTERNAL;
        }

        DumpBufferState *state = get_dump_buffer_state(shm_host_local, t);
        state->arena_base = reinterpret_cast<uint64_t>(ai.dev_ptr);
        state->arena_size = arena_size;
        state->arena_write_offset = 0;
        state->published_payload_count = 0;
        state->completed_payload_count = 0;
        state->dropped_record_count = 0;

        LOG_INFO(
            "Thread %d: dump arena allocated (dev=%p, host=%p, size=%lu MB)", t, ai.dev_ptr, ai.host_ptr,
            arena_size / (1024 * 1024)
        );
    }

    // Allocate initial DumpMetaBuffers and push into free_queues
    for (int t = 0; t < num_dump_threads; t++) {
        DumpBufferState *state = get_dump_buffer_state(shm_host_local, t);

        for (int b = 0; b < PLATFORM_DUMP_BUFFERS_PER_THREAD; b++) {
            void *host_ptr = nullptr;
            void *dev_ptr = alloc_paired_buffer(sizeof(DumpMetaBuffer), &host_ptr);
            if (dev_ptr == nullptr) {
                LOG_ERROR("Failed to allocate dump meta buffer %d for thread %d", b, t);
                return PTO_RUNTIME_ERR_INTERNAL;
            }
            // alloc_paired_buffer already registered dev→host via the manager.

            if (b < PLATFORM_DUMP_SLOT_COUNT) {
                uint32_t tail = state->free_queue.tail;
                state->free_queue.buffer_ptrs[tail % PLATFORM_DUMP_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
                state->free_queue.tail = tail + 1;
            } else {
                if (!manager_.push_recycled(0, dev_ptr, t)) {
                    (void)manager_.retire_unqueued_buffer(0, dev_ptr, t);
                }
            }
        }
    }

    // Push the entire initialized shm region (header + BufferStates +
    // free_queue contents) to device.
    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    // Publish shm pointers on the base now that the region is ready. start(tf)
    // gates on shm_host_ being non-null, so this re-set_memory_context call
    // is the moment the collector becomes startable.
    dump_shared_mem_dev_ = shm_dev_local;
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        shm_dev_local, shm_host_local, shm_size, device_id
    );

    LOG_INFO(
        "Args dump initialized: %d threads, arena=%lu MB/thread, %d buffers/thread", num_dump_threads,
        arena_size / (1024 * 1024), PLATFORM_DUMP_BUFFERS_PER_THREAD
    );

    guard.commit();
    return 0;
}

void ArgsDumpCollector::start(const profiling_common::ThreadFactory &thread_factory) {
    if (shm_host_ == nullptr) return;
    reset_collector_shards();
    profiling_common::ProfilerBase<ArgsDumpCollector, DumpModule>::start(thread_factory);
}

void ArgsDumpCollector::start_writer_thread_once() {
    std::scoped_lock<std::mutex> lock(writer_start_mutex_);
    if (writer_started_) return;
    writer_started_ = true;

    // `output_prefix_` is bound by begin_run() and is the per-task uniqueness
    // boundary; the dump dir name is fixed (`<prefix>/args_dump`).
    std::string run_dir_name = "args_dump";
    run_dir_ = std::filesystem::path(output_prefix_) / run_dir_name;
    std::filesystem::create_directories(run_dir_);
    // Hybrid Level 3 opens args.bin lazily if an Arg::dump()-selected tensor
    // emits payload; an unmarked run remains metadata-only.
    if (dump_args_level_ != DumpArgsLevel::HYBRID) {
        bin_file_.open(run_dir_ / "args.bin", std::ios::binary);
    }
    next_bin_offset_ = 0;

    writer_done_.store(false);
    bytes_written_.store(0);
    run_start_time_ = std::chrono::steady_clock::now();
    last_progress_ms_.store(steady_clock_ms(run_start_time_), std::memory_order_relaxed);

    writer_thread_ = std::thread(&ArgsDumpCollector::writer_loop, this);
}

void ArgsDumpCollector::process_dump_buffer(
    const DumpReadyBufferInfo &info, int collector_shard, RetainedEpoch *forced_epoch, size_t forced_bucket
) {
    DumpMetaBuffer *buf = reinterpret_cast<DumpMetaBuffer *>(info.host_buffer_ptr);
    uint32_t count = buf->count;

    // Read the identity before the loop: the device buffer goes back to the pool
    // after this and a later run re-stamps it, so it may not be consulted again.
    const uint64_t run_epoch = buf->run_epoch;
    const uint32_t local_seq = buf->local_seq;

    // On the retained path the epoch is resolved from the buffer's own stamp,
    // and the receipt is folded in **before** any record is read: the ledger
    // must see every delivery this lane made, including one that carried no
    // record, or the sequence it expects next would drift and the close could
    // not tell a handed-over buffer from an unpublished one.
    RetainedEpoch *epoch = forced_epoch;
    size_t shard = forced_bucket;
    if (forced_epoch == nullptr) {
        shard = normalize_collector_shard(collector_shard);
        if (retain_across_runs_) {
            if (shard >= shard_views_.size()) {
                LOG_ERROR("Args dump: collected buffer carried shard index %d", collector_shard);
                return;
            }
            const int slot = shard_views_[shard].slot_for(run_epoch);
            if (slot < 0) {
                // Sealed or never admitted: these records cannot be attributed
                // to a run whose verdict is already published, and a bounded
                // collector-scoped error is what replaces rewriting one.
                retained_errors_.record_unknown_epoch(run_epoch, count);
                return;
            }
            epoch = &retained_epochs_[static_cast<size_t>(slot)];
            const int lane = static_cast<int>(info.thread_index);
            if (lane >= 0 && static_cast<size_t>(lane) < epoch->receipts.size()) {
                epoch->receipts[static_cast<size_t>(lane)].observe(info.buffer_seq, local_seq);
            }
        } else if (shard >= collected_by_collector_.size()) {
            return;
        }
    }

    if (count == 0) return;

    if (count > PLATFORM_DUMP_RECORDS_PER_BUFFER) {
        LOG_ERROR(
            "Dump collector: invalid record count %u in buffer (thread=%u, seq=%u, max=%d), skipping", count,
            info.thread_index, info.buffer_seq, PLATFORM_DUMP_RECORDS_PER_BUFFER
        );
        if (epoch != nullptr) {
            // Unreachable records, which the equation cannot see: recorded as a
            // loss rather than left to look balanced.
            epoch->discarded_metadata_records.fetch_add(count, std::memory_order_relaxed);
        }
        return;
    }

    uint64_t records_appended = 0;

    // Whether this buffer's payloads are part of the lifetime acknowledgement
    // equation at all.
    //
    // `published_payload_count` is advanced by the device only in
    // `write_ready_entry`, i.e. only for a payload it actually placed in a
    // ready queue. A forced-recovery buffer was proved **never** enqueued, so
    // none of its payloads is counted there — and crediting one to this lane's
    // receipt or discard count would hand the equation a payload the device
    // never asked about. Because these counters are monotonic for the
    // collector's life, that credit does not expire: a later run's genuinely
    // published payload could then be acknowledged before anyone had copied
    // it, and the producer would recycle an arena still holding it.
    //
    // A recovered record is output evidence, not a transport acknowledgement.
    // It is counted in this run's own record and loss accounting below and in
    // nothing else.
    const bool transport_published = (forced_epoch == nullptr);
    // Charge one payload's disposition to this lane's transport credit, for a
    // payload that has one. Every failure path in the loop settles through
    // here rather than touching the counter directly, so none of them can
    // credit a recovered payload by omission.
    auto settle_transport_discard = [this, &info, transport_published](bool had_payload) {
        if (!transport_published || !had_payload) return;
        if (info.thread_index < discarded_payload_counts_.size()) {
            discarded_payload_counts_[info.thread_index].fetch_add(1, std::memory_order_release);
        }
    };

    // a5: pull the relevant portion of the originating thread's arena from
    // device. The arena lives outside the shared-memory region, so refresh
    // its write cursor explicitly before copying the payload bytes.
    //
    // On the retained path both copies are checked, because the bytes they
    // land in are this host's shadow of a *previous* transfer: using them after
    // a failed copy would export another run's content as this one's and call
    // it a success. A failure makes this buffer's payloads unavailable rather
    // than stale, and every record that names one is marked and settled below.
    // The single-run path keeps its existing behaviour.
    bool arena_readable = true;
    int thread_idx = static_cast<int>(info.thread_index);
    if (thread_idx >= 0 && thread_idx < static_cast<int>(arenas_.size())) {
        ArenaInfo &ai = arenas_[thread_idx];
        DumpBufferState *state = get_dump_buffer_state(shm_host_, thread_idx);
        DumpBufferState *device_state = get_dump_buffer_state(dump_shared_mem_dev_, thread_idx);
        const bool inject_failure = epoch != nullptr && retained_fail_arena_copy_.load(std::memory_order_acquire);
        const int offset_rc = inject_failure ? -1 :
                                               profiling_copy_from_device(
                                                   &state->arena_write_offset, &device_state->arena_write_offset,
                                                   sizeof(state->arena_write_offset)
                                               );
        if (offset_rc != 0 && epoch != nullptr) {
            LOG_ERROR(
                "Args dump: lane %d arena cursor was not readable (%d); this buffer's payloads are lost", thread_idx,
                offset_rc
            );
            arena_readable = false;
        }
        uint64_t write_offset = state->arena_write_offset;
        uint64_t bytes_to_copy = (write_offset < ai.size) ? write_offset : ai.size;
        if (arena_readable && bytes_to_copy > 0) {
            const int arena_rc = profiling_copy_from_device(ai.host_ptr, ai.dev_ptr, bytes_to_copy);
            if (arena_rc != 0 && epoch != nullptr) {
                LOG_ERROR(
                    "Args dump: lane %d arena bytes were not readable (%d); this buffer's payloads are lost",
                    thread_idx, arena_rc
                );
                arena_readable = false;
            }
        }
    }

    for (uint32_t i = 0; i < count; i++) {
        const ArgsDumpRecord &rec = buf->records[i];
        DumpedArg dt{};
        dt.run_epoch = run_epoch;
        dt.local_seq = local_seq;
        dt.task_id = rec.task_id;
        // rec is read from device shared memory (untrusted): clamp func_count so a
        // corrupt oversized value can't drive an out-of-bounds read of the
        // fixed-size dt.func_ids[] when the record is serialized later.
        uint8_t func_count = rec.func_count;
        if (func_count > ARGS_DUMP_MAX_FUNC_IDS) {
            LOG_WARN(
                "Dump collector: func_count %u exceeds max %d (corrupt record?), clamping", func_count,
                ARGS_DUMP_MAX_FUNC_IDS
            );
            func_count = ARGS_DUMP_MAX_FUNC_IDS;
        }
        dt.func_count = func_count;
        for (uint8_t f = 0; f < func_count; f++) {
            dt.func_ids[f] = (rec.func_ids[f] == 0xFFFF) ? -1 : static_cast<int32_t>(rec.func_ids[f]);
        }
        dt.arg_index = rec.arg_index;
        dt.role = static_cast<ArgsDumpRole>(rec.role);
        dt.stage = static_cast<ArgsDumpStage>(rec.stage);
        dt.dtype = rec.dtype;
        uint8_t ndims = rec.ndims;
        if (ndims > PLATFORM_DUMP_MAX_DIMS) {
            LOG_ERROR(
                "Dump collector: ndims %u exceeds max %u (corrupt record?), clamping", static_cast<unsigned>(ndims),
                static_cast<unsigned>(PLATFORM_DUMP_MAX_DIMS)
            );
            ndims = PLATFORM_DUMP_MAX_DIMS;
        }
        dt.ndims = ndims;
        dt.flags = rec.flags;
        dt.kind = static_cast<ArgsDumpKind>(rec.kind);
        dt.scalar_value = rec.scalar_value;
        dt.is_contiguous = (rec.is_contiguous != 0);
        dt.truncated = (rec.truncated != 0);
        dt.start_offset = rec.start_offset;
        std::memcpy(dt.shapes, rec.shapes, sizeof(dt.shapes));
        std::memcpy(dt.strides, rec.strides, sizeof(dt.strides));

        if (dt.truncated) {
            if (epoch != nullptr) {
                epoch->truncated_records.fetch_add(1, std::memory_order_relaxed);
            }
            if (total_truncated_count_.fetch_add(1, std::memory_order_relaxed) == 0) {
                LOG_WARN("Args dump truncation detected. Increase PLATFORM_DUMP_AVG_TENSOR_BYTES.");
            }
        }

        // The metadata slot is charged and reserved before anything else this
        // record needs, so every failure below can record the loss without
        // allocating: the append it lands in cannot fail afterwards.
        if (epoch != nullptr && !retained_bucket_reserve_one(*epoch, shard)) {
            epoch->discarded_metadata_records.fetch_add(1, std::memory_order_relaxed);
            // A published payload is settled here or its arena barrier would
            // never clear; a recovered one was never in the equation.
            settle_transport_discard(dt.kind == ArgsDumpKind::TENSOR && rec.payload_size > 0);
            continue;
        }

        uint64_t payload_size = 0;
        if (dt.kind == ArgsDumpKind::TENSOR && thread_idx >= 0 && thread_idx < static_cast<int>(arenas_.size())) {
            payload_size = rec.payload_size;
        }
        // A failed arena copy leaves the shadow holding an earlier transfer, so
        // this record's bytes are lost rather than stale. The metadata stays and
        // says so, and the lane is settled because nothing will read those arena
        // bytes again.
        if (!arena_readable && payload_size > 0 && epoch != nullptr) {
            payload_size = 0;
            dt.host_discarded = true;
            epoch->discarded_args.fetch_add(1, std::memory_order_relaxed);
            settle_transport_discard(true);
        }
        bool payload_charged = false;
        if (epoch != nullptr && payload_size > 0) {
            // The owned copy and the queue node it will travel in are charged
            // together, so a payload that is admitted can always be handed to
            // the writer.
            if (retained_budget_.charge(static_cast<size_t>(payload_size) + kRetainedQueueNodeBytes)) {
                payload_charged = true;
            } else {
                payload_size = 0;
                dt.host_discarded = true;
                epoch->discarded_args.fetch_add(1, std::memory_order_relaxed);
                settle_transport_discard(true);
            }
        }

        if (payload_size > 0) {
            ArenaInfo &ai = arenas_[thread_idx];
            char *arena_host = reinterpret_cast<char *>(ai.host_ptr);
            uint64_t arena_sz = ai.size;
            try {
                dt.bytes.resize(payload_size);
            } catch (const std::bad_alloc &) {
                // Nothing further is allocated on this path: the metadata slot
                // is already reserved and no queue node is needed.
                dt.bytes.clear();
                payload_size = 0;
                dt.host_discarded = true;
                if (payload_charged) {
                    retained_budget_.credit(static_cast<size_t>(rec.payload_size) + kRetainedQueueNodeBytes);
                    payload_charged = false;
                }
                if (epoch != nullptr) {
                    epoch->discarded_args.fetch_add(1, std::memory_order_relaxed);
                    settle_transport_discard(true);
                }
            }
            if (payload_size > 0) {
                uint64_t pos = rec.payload_offset % arena_sz;
                if (pos + payload_size <= arena_sz) {
                    std::memcpy(dt.bytes.data(), arena_host + pos, payload_size);
                } else {
                    uint64_t first = arena_sz - pos;
                    std::memcpy(dt.bytes.data(), arena_host + pos, first);
                    std::memcpy(dt.bytes.data() + first, arena_host, payload_size - first);
                }
            }
        }

        dt.payload_size = dt.bytes.size();
        bool has_payload = dt.kind == ArgsDumpKind::TENSOR && !dt.bytes.empty();
        if (epoch != nullptr) {
            if (!retained_append_record(
                    *epoch, shard, std::move(dt), info.thread_index, has_payload, transport_published
                ) &&
                payload_charged) {
                // The commit released the bytes and its own charge.
                payload_charged = false;
            }
            records_appended++;
            continue;
        }
        if (has_payload) {
            PayloadWriteRequest writer_item{info.thread_index, -1, 0, std::move(dt.bytes)};
            {
                std::scoped_lock<std::mutex> lock(write_mutex_);
                dt.bin_offset = next_bin_offset_;
                next_bin_offset_ += dt.payload_size;
                write_queue_.push(std::move(writer_item));
            }
            collected_by_collector_[shard].push_back(std::move(dt));
            write_cv_.notify_one();
        } else {
            dt.bin_offset = 0;
            dt.bytes.clear();
            collected_by_collector_[shard].push_back(std::move(dt));
        }
        records_appended++;
    }

    if (records_appended > 0) {
        if (epoch != nullptr) {
            epoch->collected_records.fetch_add(records_appended, std::memory_order_relaxed);
        } else {
            collector_counters_[shard].total_collected += records_appended;
        }
        total_metadata_collected_.fetch_add(records_appended, std::memory_order_relaxed);
    }
}

void ArgsDumpCollector::on_buffer_collected(const DumpReadyBufferInfo &info, int collector_shard) {
    // The retained path opens each run's own payload file at admission, so the
    // one lazily-opened `args.bin` and its shared cursor belong to the default
    // path alone.
    if (!retain_across_runs_) start_writer_thread_once();
    process_dump_buffer(info, collector_shard);

    auto now = std::chrono::steady_clock::now();
    int64_t now_ms = steady_clock_ms(now);
    int64_t last_ms = last_progress_ms_.load(std::memory_order_relaxed);
    if (now_ms - last_ms >= 5000 &&
        last_progress_ms_.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed)) {
        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - run_start_time_).count();
        LOG_INFO(
            "Collecting: %lu args, %.1f GB written (%lds)",
            static_cast<unsigned long>(total_metadata_collected_.load(std::memory_order_relaxed)),
            bytes_written_.load() / 1e9, elapsed_s
        );
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters: recover un-flushed current buffers + dropped accounting
// ---------------------------------------------------------------------------

void ArgsDumpCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return;
    // The retained path decides a leftover buffer at its own close, under that
    // run's execution claim and against that run's receipt ledger. Reaching
    // this bulk read afterwards would copy the whole shared region back over
    // the free-queue cursors the drain and replenish threads own, and would
    // read a buffer whose identity nothing here can check.
    if (retain_across_runs_) return;
    report_drain_drops();

    // Pull the latest BufferStates (current_buf_ptr, dropped_record_count)
    // before the per-thread loop so leftovers reflect post-stop() device
    // state.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();

    uint32_t dropped_total = 0;
    int recovered_threads = 0;
    // After stop(), a non-zero current_buf_ptr with records means the device
    // never ran dump_args_flush for that thread. The common cause is a hang:
    // the AICPU op is reaped by the hardware op-execution timeout (507xxx)
    // before its graceful scheduler-timeout shutdown can flush. The host still
    // holds the buffer and the originating arena, so recover the records here
    // (the same path the poll thread uses for a ready buffer) instead of
    // dropping them — export_dump_files() then writes them like any normally
    // collected buffer, so a hung run still yields its dumped inputs/outputs.
    for (int t = 0; t < num_dump_threads_; t++) {
        DumpBufferState *state = get_dump_buffer_state(shm_host_, t);

        total_dropped_record_count_.fetch_add(state->dropped_record_count, std::memory_order_relaxed);
        dropped_total += state->dropped_record_count;

        uint64_t cur_ptr = state->current_buf_ptr;
        if (cur_ptr == 0) continue;

        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(cur_ptr));
        if (host_ptr == nullptr) continue;

        profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(cur_ptr), sizeof(DumpMetaBuffer));
        uint32_t count = reinterpret_cast<DumpMetaBuffer *>(host_ptr)->count;
        if (count == 0) continue;

        DumpReadyBufferInfo info;
        info.thread_index = static_cast<uint32_t>(t);
        info.dev_buffer_ptr = reinterpret_cast<void *>(cur_ptr);
        info.host_buffer_ptr = host_ptr;
        info.buffer_seq = state->current_buf_seq;
        on_buffer_collected(info, static_cast<int>(t));
        recovered_threads++;
        LOG_WARN(
            "Dump reconcile: thread %d had an un-flushed buffer (count=%u) — device flush did not run "
            "(AICPU likely reaped on a hang); recovered the records host-side",
            t, count
        );
    }

    if (dropped_total > 0) {
        LOG_WARN(
            "Dump reconcile: %u records dropped on device side. "
            "Increase PLATFORM_DUMP_BUFFERS_PER_THREAD or PLATFORM_DUMP_READYQUEUE_SIZE.",
            dropped_total
        );
    }
    if (recovered_threads > 0) {
        LOG_WARN(
            "Dump reconcile: recovered un-flushed buffers from %d thread(s) (device-side flush was skipped, "
            "typically an AICPU hang reap)",
            recovered_threads
        );
    }
}

// ---------------------------------------------------------------------------
// Writer thread + export
// ---------------------------------------------------------------------------

void ArgsDumpCollector::request_writer_stop() {
    // The stop flag must change under `write_mutex_`, not merely be atomic.
    //
    // `writer_loop` evaluates its predicate while holding that mutex and only
    // then blocks, releasing the mutex as it registers on the condition
    // variable. A stop that sets the flag without the mutex can land in the
    // window between those two steps: the waiter has already read
    // `writer_done_ == false`, is not yet registered, so `notify_one()` reaches
    // nobody and the waiter blocks on a condition that is already true. The
    // subsequent `join()` then never returns.
    //
    // Setting it under the mutex closes the window, because the waiter holds the
    // mutex across its own check-then-block. This is the same rule
    // `BufferPoolManager::notify_ready_waiters()` follows, and the reason the
    // producer side at the payload-enqueue site is already correct: it pushes
    // under the mutex and notifies afterwards.
    {
        std::scoped_lock<std::mutex> lock(write_mutex_);
        writer_done_.store(true);
    }
    write_cv_.notify_one();
}

void ArgsDumpCollector::writer_loop() {
    while (true) {
        PayloadWriteRequest request;
        {
            std::unique_lock<std::mutex> lock(write_mutex_);
            write_cv_.wait(lock, [this] {
                return !write_queue_.empty() || writer_done_.load();
            });
            if (write_queue_.empty() && writer_done_.load()) {
                break;
            }
            request = std::move(write_queue_.front());
            write_queue_.pop();
        }

        if (!request.bytes.empty()) {
            if (!bin_file_.is_open()) {
                bin_file_.open(run_dir_ / "args.bin", std::ios::binary);
            }
            bin_file_.write(
                reinterpret_cast<const char *>(request.bytes.data()), static_cast<std::streamsize>(request.bytes.size())
            );
            written_payload_counts_[request.thread_index].fetch_add(1, std::memory_order_release);
        }

        bytes_written_ += request.bytes.size();
    }
}

void ArgsDumpCollector::publish_arena_acks() {
    if (shm_host_ == nullptr || dump_shared_mem_dev_ == nullptr) {
        return;
    }
    // Per lane, and independently of every other lane: each AICPU thread owns its
    // own arena, so thread t may reuse its arena bytes as soon as thread t's own
    // payloads are accounted for. Holding t behind a sibling's progress would
    // serialize unrelated arenas for no safety gain.
    for (int t = 0; t < num_dump_threads_; t++) {
        DumpBufferState *host_state = get_dump_buffer_state(shm_host_, t);
        DumpBufferState *device_state = get_dump_buffer_state(dump_shared_mem_dev_, t);
        if (profiling_copy_from_device(
                &host_state->published_payload_count, &device_state->published_payload_count,
                sizeof(host_state->published_payload_count)
            ) != 0) {
            continue;
        }
        const uint64_t published = host_state->published_payload_count;
        // What releases this lane's arena differs by path, and the difference is
        // the whole point of the retained one.
        //
        // Default path, unchanged: the writer bumps `written_payload_counts_[t]`
        // once `args.bin` accepted the bytes, and that equality is the proof.
        // One run owns the file for its whole boundary, so tying reuse to disk
        // costs that run nothing — and `begin_run` zeroes these counters per
        // run, which is what keeps that path's own recovery of an un-flushed
        // buffer from crediting a payload the device never published.
        //
        // `written_payload_counts_[t]` is deliberately **not** part of the
        // retained equation: the writer advances it for recovered payloads too,
        // which the device never counted, so using it there would be the same
        // over-credit this comment exists to prevent.
        //
        // Retained path: reuse is released by **host ownership**, not by disk.
        // `received_payload_counts_[t]` counts the payloads copied out of the
        // arena into storage this host owns, and
        // `discarded_payload_counts_[t]` the ones deliberately written off;
        // their sum reaching `published` means the first `published` payloads of
        // this lane are no longer in the arena, which is exactly what the
        // producer's barrier waits for. Receipt is FIFO per lane, so the sum
        // cannot run ahead of the bytes it describes. A slow or failing disk
        // therefore no longer holds a producer behind bytes the host already
        // has — the write failure stays a sticky error and fails that run's
        // flush, and disk completion is proved by `flush_diagnostics` alone.
        uint64_t taken = 0;
        if (retain_across_runs_) {
            taken = received_payload_counts_[t].load(std::memory_order_acquire) +
                    discarded_payload_counts_[t].load(std::memory_order_acquire);
        } else {
            taken = written_payload_counts_[t].load(std::memory_order_acquire);
        }
        if (taken < published) {
            continue;
        }
        if (host_state->completed_payload_count >= published) {
            // Already acknowledged at or past this watermark. The counters are
            // monotonic for the collector's life on the retained path, so the
            // acknowledgement must never move backwards.
            continue;
        }
        if (profiling_copy_to_device(&device_state->completed_payload_count, &published, sizeof(published)) != 0) {
            continue;
        }
        host_state->completed_payload_count = published;
        wmb();
    }
}

int ArgsDumpCollector::export_dump_files() {
    // The retained path publishes each run from the background writer, against
    // that run's own exclusively owned file pair. Running this here would join
    // the wrong writer and write a second manifest over the published one.
    if (retain_across_runs_) return 0;
    // Stop the writer thread (started lazily in on_buffer_collected). Safe
    // to skip when writer_started_ is false (collector ran but produced no
    // buffers, or never started at all).
    if (writer_started_) {
        request_writer_stop();
        while (writer_thread_.joinable()) {
            size_t remaining = 0;
            {
                std::scoped_lock<std::mutex> lock(write_mutex_);
                remaining = write_queue_.size();
            }
            if (remaining == 0) {
                writer_thread_.join();
                break;
            }
            auto elapsed_s =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - run_start_time_)
                    .count();
            LOG_INFO(
                "Writing to disk: %.1f GB written, %zu args remaining (%lds)", bytes_written_.load() / 1e9, remaining,
                elapsed_s
            );
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (bin_file_.is_open()) {
            bin_file_.close();
        }

        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - run_start_time_)
                .count();
        LOG_INFO(
            "Collected %lu args, wrote %.1f GB to disk (%.1fs)",
            static_cast<unsigned long>(total_metadata_collected_.load(std::memory_order_relaxed)),
            bytes_written_.load() / 1e9, elapsed_ms / 1000.0
        );
    }

    merge_collector_shards();
    if (collected_.empty()) {
        LOG_WARN("No args dump data to export");
        reset_collector_shards();
        total_dropped_record_count_.store(0, std::memory_order_relaxed);
        total_truncated_count_.store(0, std::memory_order_relaxed);
        writer_started_ = false;
        return 0;
    }
    auto export_start = std::chrono::steady_clock::now();

    std::sort(collected_.begin(), collected_.end(), [](const DumpedArg &a, const DumpedArg &b) {
        if (a.task_id != b.task_id) return a.task_id < b.task_id;
        if (a.stage != b.stage) return static_cast<uint8_t>(a.stage) < static_cast<uint8_t>(b.stage);
        if (a.arg_index != b.arg_index) return a.arg_index < b.arg_index;
        return static_cast<uint8_t>(a.role) < static_cast<uint8_t>(b.role);
    });

    LOG_INFO("Writing JSON manifest for %zu args...", collected_.size());

    uint32_t num_before_dispatch = 0;
    uint32_t num_after_completion = 0;
    uint32_t num_input_args = 0;
    uint32_t num_output_args = 0;
    uint32_t num_inout_args = 0;
    for (const auto &dt : collected_) {
        if (dt.stage == ArgsDumpStage::BEFORE_DISPATCH) {
            num_before_dispatch++;
        } else {
            num_after_completion++;
        }
        switch (dt.role) {
        case ArgsDumpRole::INPUT:
            num_input_args++;
            break;
        case ArgsDumpRole::OUTPUT:
            num_output_args++;
            break;
        case ArgsDumpRole::INOUT:
            num_inout_args++;
            break;
        }
    }

    std::string run_dir_name = run_dir_.filename().string();
    simpler::dfx::args_dump::ManifestMeta meta;
    meta.run_dir_name = run_dir_name;
    meta.dump_args_level = static_cast<uint32_t>(dump_args_level_);
    meta.total_args = collected_.size();
    meta.before_dispatch = num_before_dispatch;
    meta.after_completion = num_after_completion;
    meta.input_args = num_input_args;
    meta.output_args = num_output_args;
    meta.inout_args = num_inout_args;
    meta.truncated_args = total_truncated_count_.load(std::memory_order_relaxed);
    meta.dropped_records = total_dropped_record_count_.load(std::memory_order_relaxed);
    if (dump_args_level_ != DumpArgsLevel::HYBRID || bytes_written_.load() != 0) {
        meta.bin_file = "args.bin";
    }

    std::ofstream json(run_dir_ / "args_dump.json");
    simpler::dfx::args_dump::write_manifest_prologue(json, meta);

    bool first_entry = true;

    for (size_t i = 0; i < collected_.size(); i++) {
        if (!first_entry) json << ",\n";
        first_entry = false;
        simpler::dfx::args_dump::write_arg_json(json, collected_[i]);
    }

    simpler::dfx::args_dump::write_manifest_epilogue(json);
    json.close();

    auto export_end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(export_end - export_start).count();
    LOG_INFO("Wrote JSON manifest (%zu args) to %s (%ldms)", collected_.size(), run_dir_.c_str(), total_ms);

    uint32_t truncated = total_truncated_count_.load(std::memory_order_relaxed);
    uint32_t dropped = total_dropped_record_count_.load(std::memory_order_relaxed);
    if (truncated > 0 || dropped > 0) {
        LOG_WARN("Args dump anomalies: truncated=%u, dropped_records=%u", truncated, dropped);
    }

    // Clear state so subsequent runs don't accumulate data from previous runs
    collected_.clear();
    collected_by_collector_.clear();
    collector_counters_.clear();
    collector_shards_merged_ = false;
    total_metadata_collected_.store(0, std::memory_order_relaxed);
    total_dropped_record_count_.store(0, std::memory_order_relaxed);
    total_truncated_count_.store(0, std::memory_order_relaxed);
    writer_started_ = false;
    return 0;
}

int ArgsDumpCollector::finalize(DumpUnregisterCallback unregister_cb, const DumpFreeCallback &free_cb) {
    if (shm_host_ == nullptr) return 0;

    int retained_rc = 0;
    if (retain_across_runs_) {
        // Publish whatever the writer can still publish, for a caller that
        // never reached `finish_retained_runs()`. It runs **before** the
        // collector threads are joined because the reference release a seal
        // needs can only be proved while those shards are still there to
        // acknowledge it.
        retained_finish();
        // Then the writer, before the threads whose references it asks about.
        retained_stop_writer();
    }

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    if (retain_across_runs_) {
        // Only now: the shards whose references a quarantined run could not
        // prove released are joined, so its host records are unreachable by any
        // reader. The files stay on disk as evidence.
        retained_release_resources();
        retained_release_quarantined();
        // An error found here is later than the caller's own diagnostic flush,
        // which has already run and returned, so this return value is the only
        // way it can reach the caller. The caller keeps its first device error
        // ahead of this one.
        if (retained_errors_.has_error() || retained_fatal_.load(std::memory_order_acquire)) {
            LOG_ERROR("Args dump: retained runs ended with failures: %s", retained_errors_.report().c_str());
            retained_rc = PTO_RUNTIME_ERR_INTERNAL;
        }
    }

    // ProfilerBase::stop() only joins the mgmt + poll threads. The writer
    // thread is otherwise torn down solely by export_dump_files(), so any path
    // that skips export — e.g. drain bailing on a device error before its
    // collector-teardown block — would leak it: left blocked on write_cv_ with
    // writer_done_ == false while writer_thread_ stays joinable, which trips
    // std::terminate when the collector is destroyed or re-run. finalize() is
    // reached via device-runner active-run cleanup on every exit path, so join
    // the writer here too. Idempotent: export_dump_files() clears writer_started_
    // on the success path, making this a no-op.
    if (writer_started_ && writer_thread_.joinable()) {
        request_writer_stop();
        writer_thread_.join();
    }

    // The writer thread opens bin_file_ in start_writer_thread_once() and it is
    // otherwise closed only by export_dump_files(). Close it here too so an
    // export-skipping path does not leave it open — a stale-open stream makes
    // the next run's bin_file_.open() set failbit. Guarded for idempotency.
    if (bin_file_.is_open()) {
        bin_file_.close();
    }

    auto release_dev = [&](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    };

    // Free DumpMetaBuffers still in per-thread free_queues / current_buf_ptr.
    // These are owned by AICPU at runtime; the framework tracks them via
    // dev_to_host_ but doesn't enumerate them in release_owned_buffers.
    // Release the device pointer only — the paired host shadow stays in
    // dev_to_host_ and is freed by clear_mappings() below.
    if (shm_host_ != nullptr) {
        for (int t = 0; t < num_dump_threads_; t++) {
            DumpBufferState *state = get_dump_buffer_state(shm_host_, t);

            release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
            state->current_buf_ptr = 0;

            rmb();
            uint32_t head = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            uint32_t queued = tail - head;
            if (queued > PLATFORM_DUMP_SLOT_COUNT) {
                queued = PLATFORM_DUMP_SLOT_COUNT;
            }
            for (uint32_t i = 0; i < queued; i++) {
                uint32_t slot = (head + i) % PLATFORM_DUMP_SLOT_COUNT;
                release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
                state->free_queue.buffer_ptrs[slot] = 0;
            }
            state->free_queue.head = tail;
        }
    }

    // Release framework-owned device allocations (recycled pools,
    // ready_queue, done_queue). Host shadows are freed by clear_mappings().
    manager_.release_owned_buffers([&](void *p) {
        release_dev(p);
    });

    // Free arenas (device only — shadows tracked in dev_to_host_).
    for (auto &ai : arenas_) {
        if (ai.dev_ptr != nullptr) {
            release_dev(ai.dev_ptr);
            ai.dev_ptr = nullptr;
            ai.host_ptr = nullptr;
        }
    }
    arenas_.clear();

    // Free shared memory region (device only — shadow stays in
    // dev_to_host_ until clear_mappings).
    if (dump_shared_mem_dev_ != nullptr) {
        release_dev(dump_shared_mem_dev_);
        dump_shared_mem_dev_ = nullptr;
    }

    // Free remaining host shadows: per-state buffers + arenas + shm region.
    manager_.clear_mappings();

    // Reset state
    num_dump_threads_ = 0;
    collected_.clear();
    collected_by_collector_.clear();
    collector_counters_.clear();
    collector_shards_merged_ = false;
    total_metadata_collected_.store(0, std::memory_order_relaxed);
    total_dropped_record_count_.store(0, std::memory_order_relaxed);
    total_truncated_count_.store(0, std::memory_order_relaxed);
    writer_started_ = false;
    clear_memory_context();
    for (auto &count : written_payload_counts_) {
        count.store(0, std::memory_order_relaxed);
    }
    for (auto &count : discarded_payload_counts_) {
        count.store(0, std::memory_order_relaxed);
    }
    for (auto &count : received_payload_counts_) {
        count.store(0, std::memory_order_relaxed);
    }

    return retained_rc;
}
