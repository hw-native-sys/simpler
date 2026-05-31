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
 * @file l2_swimlane_collector.cpp
 * @brief Performance data collector implementation. The mgmt-thread + buffer-pool
 *        machinery lives in profiling_common::BufferPoolManager parameterized by
 *        L2SwimlaneModule (host/l2_swimlane_collector.h); the poll loop lives in
 *        profiling_common::ProfilerBase. This file owns the per-buffer
 *        on_buffer_collected callback and the export logic.
 */

#include "host/l2_swimlane_collector.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

// =============================================================================
// L2SwimlaneCollector Implementation
// =============================================================================

/**
 * Check if a phase ID belongs to a scheduler phase (vs orchestrator phase).
 * Scheduler phases: SCHED_COMPLETE(0), SCHED_DISPATCH(1).
 * Orchestrator phases: ORCH_SUBMIT(25) — one record per submit_task() call,
 * folded from 6 historical sub-step phases (ORCH_SYNC..ORCH_FANIN). Old
 * captures may carry the per-sub-step ids (16-24) — they fall through the
 * orch branch and the JSON writer labels them "unknown"; downstream tools
 * drop "unknown" records.
 *
 * The boundary is the historical orch id base (16), not ORCH_SUBMIT itself:
 * legacy ids 16-24 must be routed orch-side so they don't accidentally
 * masquerade as scheduler-side phases when decoding old captures.
 *
 * Legacy IDs 2 (SCHED_SCAN, never emitted) and 3 (SCHED_IDLE_WAIT, dropped
 * by PR #869) classify as scheduler-side; the host parser then drops them
 * because idle is reconstructed from record gaps.
 */
static constexpr uint32_t kAicpuOrchPhaseIdBase = 16;
static bool is_scheduler_phase(L2SwimlaneAicpuPhaseId id) { return static_cast<uint32_t>(id) < kAicpuOrchPhaseIdBase; }

L2SwimlaneCollector::~L2SwimlaneCollector() {
    stop();
    if (shm_host_ != nullptr) {
        LOG_WARN("L2SwimlaneCollector destroyed without finalize()");
    }
}

void *L2SwimlaneCollector::alloc_single_buffer(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size);
    if (dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate buffer (%zu bytes)", size);
        *host_ptr_out = nullptr;
        return nullptr;
    }

    if (register_cb_ != nullptr) {
        void *host_ptr = nullptr;
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("Buffer registration failed: %d", rc);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else {
        *host_ptr_out = dev_ptr;
    }

    // Register mapping so the BufferPoolManager can resolve dev→host
    manager_.register_mapping(dev_ptr, *host_ptr_out);
    return dev_ptr;
}

int L2SwimlaneCollector::initialize(
    int num_aicore, int device_id, L2SwimlaneLevel l2_swimlane_level, const L2SwimlaneAllocCallback &alloc_cb,
    L2SwimlaneRegisterCallback register_cb, const L2SwimlaneFreeCallback &free_cb, const std::string &output_prefix
) {
    if (shm_host_ != nullptr) {
        LOG_ERROR("L2SwimlaneCollector already initialized");
        return -1;
    }

    LOG_INFO_V0("Initializing performance profiling");

    if (num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("Invalid number of AICores: %d (max=%d)", num_aicore, PLATFORM_MAX_CORES);
        return -1;
    }

    num_aicore_ = num_aicore;
    l2_swimlane_level_ = l2_swimlane_level;
    output_prefix_ = output_prefix;
    total_perf_collected_ = 0;
    total_phase_collected_ = 0;

    // Stash the memory context on the base up-front so alloc_single_buffer
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — the nullptr guard makes a post-failure
    // start(tf) a no-op.
    set_memory_context(
        alloc_cb, register_cb, free_cb, /*copy_to=*/nullptr, /*copy_from=*/nullptr, /*shm_dev=*/nullptr,
        /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // Step 1: Calculate shared memory size (slot arrays only, no actual buffers)
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    size_t total_size = calc_perf_data_size_with_phases(num_aicore, num_phase_threads);

    LOG_DEBUG("Shared memory allocation plan:");
    LOG_DEBUG("  Number of cores:      %d", num_aicore);
    LOG_DEBUG("  Header size:          %zu bytes", sizeof(L2SwimlaneDataHeader));
    LOG_DEBUG("  L2SwimlaneAicpuTaskPool size: %zu bytes each", sizeof(L2SwimlaneAicpuTaskPool));
    LOG_DEBUG("  L2SwimlaneAicpuPhasePool size:%zu bytes each", sizeof(L2SwimlaneAicpuPhasePool));
    LOG_DEBUG("  Total shared memory:  %zu bytes (%zu KB)", total_size, total_size / 1024);

    // Step 2: Allocate shared memory for slot arrays
    void *perf_dev_ptr = alloc_cb(total_size);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate shared memory (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated shared memory: %p", perf_dev_ptr);

    // Step 3: Register to host mapping (optional)
    void *perf_host_ptr = nullptr;
    if (register_cb != nullptr) {
        int rc = register_cb(perf_dev_ptr, total_size, device_id, &perf_host_ptr);
        if (rc != 0) {
            LOG_ERROR("Memory registration failed: %d", rc);
            return rc;
        }
        if (perf_host_ptr == nullptr) {
            LOG_ERROR("register_cb succeeded but returned null host_ptr");
            return -1;
        }
        LOG_DEBUG("Mapped to host memory: %p", perf_host_ptr);
    } else {
        perf_host_ptr = perf_dev_ptr;
        LOG_DEBUG("Simulation mode: host_ptr = dev_ptr = %p", perf_host_ptr);
    }

    // Step 4: Initialize header
    L2SwimlaneDataHeader *header = get_l2_swimlane_header(perf_host_ptr);

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        memset(header->queues[t], 0, sizeof(header->queues[t]));
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }

    header->num_cores = num_aicore;
    header->l2_swimlane_level = static_cast<uint32_t>(l2_swimlane_level_);

    LOG_DEBUG("Initialized L2SwimlaneDataHeader:");
    LOG_DEBUG("  num_cores:              %d", header->num_cores);
    LOG_DEBUG("  l2_swimlane_level: %u", header->l2_swimlane_level);
    LOG_DEBUG("  buffer_capacity:        %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:         %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Step 5: Initialize L2SwimlaneAicpuTaskPools — 1 buffer per core in free_queue, rest to recycled pool
    for (int i = 0; i < num_aicore; i++) {
        L2SwimlaneAicpuTaskPool *state = get_perf_buffer_state(perf_host_ptr, i);
        memset(state, 0, sizeof(L2SwimlaneAicpuTaskPool));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(L2SwimlaneAicpuTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2SwimlaneAicpuTaskBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2SwimlaneAicpuTaskBuffer));
            buf->count = 0;

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::AICPU_TASK), dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = 1;
        wmb();
    }

    // Step 5b: Initialize L2SwimlaneAicoreTaskPools — per-core AICore rotation
    // channel + buffer pool. Same SPSC pattern as the AICPU pool above.
    for (int i = 0; i < num_aicore; i++) {
        L2SwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(perf_host_ptr, num_aicore, i);
        memset(ac_state, 0, sizeof(L2SwimlaneAicoreTaskPool));

        for (int s = 0; s < PLATFORM_AICORE_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(L2SwimlaneAicoreTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2SwimlaneAicoreTaskBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            L2SwimlaneAicoreTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2SwimlaneAicoreTaskBuffer));
            buf->count = 0;

            if (s == 0) {
                ac_state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::AICORE_TASK), dev_buf_ptr);
            }
        }
        wmb();
        ac_state->free_queue.tail = 1;
        wmb();
    }
    LOG_DEBUG(
        "Initialized buffer pools: %d L2SwimlaneAicpuTaskBuffers/core + %d L2SwimlaneAicoreTaskBuffers/core (1 in "
        "free_queue, "
        "rest in recycled pool)",
        PLATFORM_PROF_BUFFERS_PER_CORE, PLATFORM_AICORE_BUFFERS_PER_CORE
    );

    // Step 5c: Standalone uint64_t[num_aicore] table that will hold per-core
    // L2SwimlaneAicoreRotation device addresses. Host only allocates the bytes and
    // hands the device pointer to AICPU via KernelArgs::l2_swimlane_aicore_rotation_table;
    // AICPU itself fills the entries inside `l2_swimlane_aicpu_init` (it has
    // direct access to `&ac_state->rotation` device addresses, no
    // host-to-device translation needed). AICore reads
    // rotation_table[block_idx] at kernel entry.
    {
        size_t table_bytes = static_cast<size_t>(num_aicore) * sizeof(uint64_t);
        void *rotation_table_host = nullptr;
        void *rotation_table_dev = alloc_single_buffer(table_bytes, &rotation_table_host);
        if (rotation_table_dev == nullptr) {
            LOG_ERROR("Failed to allocate l2_swimlane_aicore_rotation_table (rotation) table (%zu bytes)", table_bytes);
            return -1;
        }
        aicore_ring_addr_table_dev_ = rotation_table_dev;
    }

    // Step 6: Initialize PhaseBufferStates — 1 buffer per thread in free_queue, rest to recycled pool
    for (int t = 0; t < num_phase_threads; t++) {
        L2SwimlaneAicpuPhasePool *state = get_phase_buffer_state(perf_host_ptr, num_aicore, t);
        memset(state, 0, sizeof(L2SwimlaneAicpuPhasePool));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_THREAD; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(L2SwimlaneAicpuPhaseBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2SwimlaneAicpuPhaseBuffer for thread %d, buffer %d", t, s);
                return -1;
            }
            L2SwimlaneAicpuPhaseBuffer *buf = reinterpret_cast<L2SwimlaneAicpuPhaseBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2SwimlaneAicpuPhaseBuffer));
            buf->count = 0;

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::AICPU_PHASE), dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = 1;
        wmb();
    }
    LOG_DEBUG(
        "Initialized %d PhaseBufferStates: 1 buffer/thread, %d in recycled pool", num_phase_threads,
        num_phase_threads * (PLATFORM_PROF_BUFFERS_PER_THREAD - 1)
    );

    wmb();

    // Step 7: Stash device pointer for the caller to publish via
    // kernel_args.l2_swimlane_data_base (read back via get_l2_swimlane_setup_device_ptr()).
    LOG_DEBUG("L2 swimlane device base = 0x%lx", reinterpret_cast<uint64_t>(perf_dev_ptr));

    perf_shared_mem_dev_ = perf_dev_ptr;
    // Refresh memory context with the now-known SHM tuple. start(tf) (inherited)
    // gates on shm_host_, so this is the moment the collector becomes startable.
    set_memory_context(
        alloc_cb, register_cb, free_cb, /*copy_to=*/nullptr, /*copy_from=*/nullptr, perf_dev_ptr, perf_host_ptr,
        total_size, device_id
    );

    collected_perf_records_.assign(num_aicore_, {});
    collected_aicore_records_.assign(num_aicore_, {});
    collected_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});

    LOG_INFO_V0("Performance profiling initialized (dynamic buffer mode)");
    return 0;
}

// ---------------------------------------------------------------------------
// ProfilerBase callbacks
// ---------------------------------------------------------------------------

void L2SwimlaneCollector::copy_perf_buffer(const ReadyBufferInfo &info) {
    L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > PLATFORM_PROF_BUFFER_SIZE) {
        count = PLATFORM_PROF_BUFFER_SIZE;
    }
    uint32_t core_index = info.index;
    if (core_index < static_cast<uint32_t>(num_aicore_)) {
        for (uint32_t i = 0; i < count; i++) {
            collected_perf_records_[core_index].push_back(buf->records[i]);
        }
        total_perf_collected_ += count;
    }
}

void L2SwimlaneCollector::copy_phase_buffer(const ReadyBufferInfo &info) {
    L2SwimlaneAicpuPhaseBuffer *buf = reinterpret_cast<L2SwimlaneAicpuPhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    if (tidx < collected_phase_records_.size()) {
        for (uint32_t i = 0; i < count; i++) {
            collected_phase_records_[tidx].push_back(buf->records[i]);
        }
        total_phase_collected_ += count;
        if (count > 0) {
            has_phase_data_ = true;
        }
    }
}

// AICore record buffers arrive on the ready queue in per-core rotation order
// (AICPU enqueues them at PLATFORM_AICORE_BUFFER_SIZE dispatch boundaries +
// once at flush). Within a single buffer, AICore wrote records[0..buf->count)
// in the order tasks ran on that core (completion-before-dispatch invariant
// + AICPU stamps buf->count just before enqueue). Flattening in arrival
// order gives us the per-core task stream that join_aicore_records()
// indexes by reg_task_id.
//
// Defensive filter: skip records whose `start_time == 0`. AICore writes
// `get_sys_cnt_aicore()` (a free-running cycle counter, always non-zero in
// practice) at task end, so a zero start_time means the slot was never
// written by AICore for this session. This handles two edge cases without
// special-casing them:
//   - Recycled buffer where AICore wrote fewer records than the count stamp
//     (e.g., the rare dispatch-boundary race for sub-microsecond kernels
//     where AICore's next record_task fires before AICPU's rotation has
//     propagated). The "missing" slot's previous contents are zero because
//     allocate_single_buffer memsets at allocation.
//   - Flush-path partial buffer whose tail wasn't reached.
void L2SwimlaneCollector::copy_aicore_buffer(const ReadyBufferInfo &info) {
    L2SwimlaneAicoreTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t core_index = info.index;
    if (core_index >= static_cast<uint32_t>(num_aicore_)) {
        return;
    }
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE)) {
        count = PLATFORM_AICORE_BUFFER_SIZE;
    }
    auto &dst = collected_aicore_records_[core_index];
    dst.reserve(dst.size() + count);
    uint32_t skipped = 0;
    for (uint32_t i = 0; i < count; i++) {
        const L2SwimlaneAicoreTaskRecord &r = buf->records[i];
        if (r.start_time == 0) {
            skipped++;
            continue;
        }
        dst.push_back(r);
    }
    if (skipped > 0) {
        LOG_WARN(
            "Core %u: skipped %u AICore record slot(s) with start_time=0 (race-window write or "
            "recycled-buffer tail). buf seq=%u count=%u",
            core_index, skipped, info.buffer_seq, count
        );
    }
}

void L2SwimlaneCollector::on_buffer_collected(const ReadyBufferInfo &info) {
    if (info.type == ProfBufferType::AICPU_TASK) {
        copy_perf_buffer(info);
    } else if (info.type == ProfBufferType::AICPU_PHASE) {
        copy_phase_buffer(info);
    } else {
        copy_aicore_buffer(info);
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters / read_phase_header_metadata
// ---------------------------------------------------------------------------
//
// Host never recovers records from device-side current_buf_ptr. Device flush
// is the only data path: a flush failure must bump dropped_record_count and
// clear current_buf_ptr on the device side. Host's job here is purely
// accounting + sanity check.

void L2SwimlaneCollector::reconcile_counters() {
    if (shm_host_ == nullptr) {
        return;
    }

    rmb();

    // Two-bucket invariant (post-AICore-as-producer): every commit attempt
    // bumps total_record_count; capacity-driven drops (no free buffer /
    // queue full / flush failure) bump dropped_record_count.
    //   silent_loss = device_total - (collected + dropped)
    // and any non-zero silent loss flags an unaccounted gap on top of the
    // already-classified dropped losses.
    //
    // Sanity sub-check: after stop(), any active buffer with records must
    // have been flushed by AICPU (success → current_buf_ptr=0; failure →
    // bump dropped, clear count + current_buf_ptr). A non-zero pointer with
    // non-zero count means records AICPU neither delivered nor accounted
    // for — i.e. a device-side flush bug. Empty buffers (count=0, never
    // written) are fine; AICPU's flush legitimately skips them.
    auto reconcile_one = [&](const char *kind, const char *unit_name, int unit_count, auto get_state,
                             auto read_buf_count, uint64_t collected, bool optional) {
        int leftover_active = 0;
        for (int i = 0; i < unit_count; i++) {
            L2SwimlaneAicpuTaskPool *state = get_state(i);
            uint64_t buf_ptr = state->current_buf_ptr;
            if (buf_ptr == 0) continue;
            void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
            if (host_ptr == nullptr) continue;
            uint32_t count = read_buf_count(host_ptr);
            if (count == 0) continue;
            LOG_ERROR(
                "L2Swimlane reconcile: %s %d has un-flushed %s buffer (current_buf_ptr=0x%lx, count=%u) "
                "after stop() — device flush failed",
                unit_name, i, kind, static_cast<unsigned long>(buf_ptr), count
            );
            leftover_active++;
        }

        uint64_t total_device = 0;
        uint64_t dropped_device = 0;
        for (int i = 0; i < unit_count; i++) {
            L2SwimlaneAicpuTaskPool *state = get_state(i);
            total_device += state->total_record_count;
            dropped_device += state->dropped_record_count;
        }

        // PHASE counters are populated only by runtimes that actually emit
        // phase records; skip the comparison entirely when nothing happened.
        if (optional && total_device == 0 && collected == 0 && dropped_device == 0) {
            return;
        }

        if (dropped_device > 0) {
            LOG_WARN(
                "L2Swimlane reconcile: %lu %s records dropped on device side (buffer full / "
                "ready_queue full).",
                static_cast<unsigned long>(dropped_device), kind
            );
        }
        uint64_t accounted = collected + dropped_device;
        if (accounted != total_device) {
            LOG_WARN(
                "L2Swimlane reconcile: %s count mismatch (collected=%lu + dropped=%lu != "
                "device_total=%lu, silent_loss=%ld)",
                kind, static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device), static_cast<long>(total_device) - static_cast<long>(accounted)
            );
        } else {
            LOG_INFO_V0(
                "L2Swimlane reconcile: %s counts match (collected=%lu, dropped=%lu, device_total=%lu)", kind,
                static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device)
            );
        }

        if (leftover_active > 0) {
            LOG_ERROR(
                "L2Swimlane reconcile: %d %s(s) had un-cleared %s current_buf_ptr — see prior errors", leftover_active,
                unit_name, kind
            );
        }
    };

    reconcile_one(
        "PERF", "core", num_aicore_,
        [this](int core_index) {
            return get_perf_buffer_state(shm_host_, core_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(host_ptr)->count;
        },
        total_perf_collected_, /*optional=*/false
    );

    reconcile_one(
        "PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_phase_buffer_state(shm_host_, num_aicore_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2SwimlaneAicpuPhaseBuffer *>(host_ptr)->count;
        },
        total_phase_collected_, /*optional=*/true
    );
}

void L2SwimlaneCollector::read_phase_header_metadata() {
    if (shm_host_ == nullptr) {
        return;
    }

    rmb();

    L2SwimlaneAicpuPhaseHeader *phase_header = get_phase_header(shm_host_, num_aicore_);

    if (phase_header->magic != L2_SWIMLANE_AICPU_PHASE_MAGIC) {
        LOG_INFO_V0(
            "No phase profiling data found (magic mismatch: 0x%x vs 0x%x)", phase_header->magic,
            L2_SWIMLANE_AICPU_PHASE_MAGIC
        );
        return;
    }

    int num_sched_threads = phase_header->num_sched_threads;
    if (num_sched_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Invalid num_sched_threads %d from shared memory (max=%d)", num_sched_threads, PLATFORM_MAX_AICPU_THREADS
        );
        return;
    }
    LOG_INFO_V0("Collecting phase metadata: %d scheduler threads", num_sched_threads);

    // Per-thread breakdown of records already collected via the buffer pipeline.
    for (size_t t = 0; t < collected_phase_records_.size(); t++) {
        if (!collected_phase_records_[t].empty()) {
            size_t sched_count = 0, orch_count = 0;
            for (const auto &r : collected_phase_records_[t]) {
                if (is_scheduler_phase(r.phase_id)) sched_count++;
                else orch_count++;
            }
            LOG_INFO_V0(
                "  Thread %zu: %zu records (sched=%zu, orch=%zu)", t, collected_phase_records_[t].size(), sched_count,
                orch_count
            );
        }
    }

    // has_phase_data_ is set by copy_phase_buffer() during the drain — every
    // push into collected_phase_records_ goes through that single call site
    // and toggles the flag. No re-scan needed here.

    // Core-to-thread mapping (header-resident; not buffered).
    int num_cores = static_cast<int>(phase_header->num_cores);
    if (num_cores > 0 && num_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(phase_header->core_to_thread, phase_header->core_to_thread + num_cores);
        LOG_INFO_V0("  Core-to-thread mapping: %d cores", num_cores);
    }

    LOG_INFO_V0("Phase metadata collection complete: has_phase_data=%s", has_phase_data_ ? "yes" : "no");
}

// AICore-as-producer post-processing: walk each L2SwimlaneAicpuTaskRecord we collected
// and patch start/end/duration from the per-core stream of AICore records
// that arrived through the ready queue. AICore rotation guarantees each
// per-core stream is a complete prefix of "all dispatched tasks on this
// core" with no wrap loss (the AICore buffer pool is recycled via
// free_queue while the session runs, so an arbitrarily long session works).
//
// We build a small `reg_task_id → (start, end)` map per core (size on the
// order of N_tasks_per_core) and patch each L2SwimlaneAicpuTaskRecord by its
// reg_task_id field. Using a map instead of direct indexing tolerates
// AICPU-side L2SwimlaneAicpuTaskBuffer drops (a missing L2SwimlaneAicpuTaskRecord doesn't break
// alignment) and lets the same code work for both runtimes.
void L2SwimlaneCollector::join_aicore_records() {
    if (shm_host_ == nullptr) {
        return;
    }
    rmb();

    uint64_t total_patched = 0;
    uint64_t total_unmatched = 0;

    // reg_task_id is per-core monotonic. For sessions that don't run long
    // enough to wrap the 31-bit `dispatch_seq & TASK_ID_MASK`, a direct
    // vector index beats a hashmap on both build and lookup. Cap the vector
    // length to keep memory bounded; if a core ever produces an outlier
    // reg_task_id (recycled session, manual reset), fall back to the
    // hashmap so we don't allocate gigabytes.
    constexpr uint32_t kDirectIndexCap = 1u << 24;  // 16 M slots = 256 MB / core max

    for (int core_idx = 0; core_idx < num_aicore_; core_idx++) {
        const auto &ac_stream = collected_aicore_records_[core_idx];
        if (collected_perf_records_[core_idx].empty()) {
            continue;
        }

        uint32_t max_reg = 0;
        for (const auto &r : ac_stream) {
            if (r.task_id > max_reg) max_reg = r.task_id;
        }
        for (const auto &lr : collected_perf_records_[core_idx]) {
            if (lr.reg_task_id > max_reg) max_reg = lr.reg_task_id;
        }

        uint64_t patched = 0;
        uint64_t unmatched = 0;

        if (max_reg < kDirectIndexCap) {
            std::vector<std::pair<uint64_t, uint64_t>> ts_by_task(static_cast<size_t>(max_reg) + 1, {0, 0});
            for (const auto &r : ac_stream) {
                ts_by_task[r.task_id] = {r.start_time, r.end_time};
            }
            for (auto &lr : collected_perf_records_[core_idx]) {
                const auto &entry = ts_by_task[lr.reg_task_id];
                if (entry.first == 0 && entry.second == 0) {
                    unmatched++;
                    continue;
                }
                lr.start_time = entry.first;
                lr.end_time = entry.second;
                lr.duration = (lr.end_time > lr.start_time) ? (lr.end_time - lr.start_time) : 0;
                patched++;
            }
        } else {
            std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> ts_by_task;
            ts_by_task.reserve(ac_stream.size() * 2);
            for (const auto &r : ac_stream) {
                ts_by_task[r.task_id] = {r.start_time, r.end_time};
            }
            for (auto &lr : collected_perf_records_[core_idx]) {
                auto it = ts_by_task.find(lr.reg_task_id);
                if (it == ts_by_task.end()) {
                    unmatched++;
                    continue;
                }
                lr.start_time = it->second.first;
                lr.end_time = it->second.second;
                lr.duration = (lr.end_time > lr.start_time) ? (lr.end_time - lr.start_time) : 0;
                patched++;
            }
        }

        total_patched += patched;
        total_unmatched += unmatched;
        if (unmatched > 0) {
            LOG_WARN(
                "Core %d: %lu L2SwimlaneAicpuTaskRecord(s) had no matching AICore entry (AICore buffer drops on "
                "rotation? "
                "PLATFORM_AICORE_BUFFERS_PER_CORE=%d may be undersized for host drain rate)",
                core_idx, static_cast<unsigned long>(unmatched), PLATFORM_AICORE_BUFFERS_PER_CORE
            );
        }
    }

    LOG_INFO_V0(
        "AICore-as-producer join: patched=%lu, unmatched=%lu", static_cast<unsigned long>(total_patched),
        static_cast<unsigned long>(total_unmatched)
    );
}

int L2SwimlaneCollector::export_swimlane_json() {
    // Step 0: Join AICore-emitted start/end/task_id records into the AICPU
    // record stream (AICore-as-producer design).
    join_aicore_records();

    // Step 1: Validate collected data
    bool has_any_records = false;
    for (const auto &core_records : collected_perf_records_) {
        if (!core_records.empty()) {
            has_any_records = true;
            break;
        }
    }
    if (!has_any_records) {
        LOG_WARN("Warning: No performance data to export.");
        return -1;
    }

    // Step 2: Create output directory (recursively — parent `outputs/` may not
    // yet exist on a clean checkout / standalone run). `output_prefix_` was
    // captured at initialize() time.
    std::error_code ec;
    std::filesystem::create_directories(output_prefix_, ec);
    if (ec) {
        LOG_ERROR("Error: Failed to create output directory %s: %s", output_prefix_.c_str(), ec.message().c_str());
        return -1;
    }

    // Step 3: Flatten per-core vectors into tagged records with core_id derived from index
    struct TaggedRecord {
        const L2SwimlaneAicpuTaskRecord *record;
        uint32_t core_id;
    };
    std::vector<TaggedRecord> tagged_records;
    size_t total_records = 0;
    for (const auto &core_records : collected_perf_records_) {
        total_records += core_records.size();
    }
    tagged_records.reserve(total_records);
    for (size_t core_idx = 0; core_idx < collected_perf_records_.size(); core_idx++) {
        for (const auto &record : collected_perf_records_[core_idx]) {
            tagged_records.push_back({&record, static_cast<uint32_t>(core_idx)});
        }
    }

    // Sort by canonical task_id (64-bit PTO2 raw)
    std::sort(tagged_records.begin(), tagged_records.end(), [](const TaggedRecord &a, const TaggedRecord &b) {
        return a.record->task_id < b.record->task_id;
    });

    // Step 4: Calculate base time (minimum timestamp across all records).
    // Records whose AICore timing was never filled in by join_aicore_records()
    // (e.g. AICore buffer wrap) leave start_time = 0 and would otherwise
    // anchor the entire swimlane at cycle 0 — gate them out alongside the
    // dispatch_time check.
    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto &tagged : tagged_records) {
        if (tagged.record->start_time > 0 && tagged.record->start_time < base_time_cycles) {
            base_time_cycles = tagged.record->start_time;
        }
        if (tagged.record->dispatch_time > 0 && tagged.record->dispatch_time < base_time_cycles) {
            base_time_cycles = tagged.record->dispatch_time;
        }
    }

    // Include phase record timestamps in base_time calculation
    if (has_phase_data_) {
        for (const auto &thread_records : collected_phase_records_) {
            for (const auto &pr : thread_records) {
                if (pr.start_time > 0 && pr.start_time < base_time_cycles) {
                    base_time_cycles = pr.start_time;
                }
            }
        }
    }

    // Step 5: Compose output path. Filename is fixed (no timestamp) — the
    // caller-provided directory is the per-task uniqueness boundary.
    std::string filepath = output_prefix_ + "/l2_swimlane_records.json";

    // Step 6: Open JSON file for writing
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    // Step 7: Write JSON data
    // Fanout fields are emitted as empty/zero — the device-side hot path no
    // longer carries them. Downstream (swimlane_converter.py) joins fanout
    // from the sibling deps.json (dep_gen output).
    int l2_swimlane_level = static_cast<int>(l2_swimlane_level_);
    outfile << "{\n";
    outfile << "  \"l2_swimlane_level\": " << l2_swimlane_level << ",\n";
    outfile << "  \"tasks\": [\n";

    // First pass: filter unmatched records (start_time == 0) so we emit a
    // valid JSON without trailing-comma fix-ups. Unmatched records arise when
    // the AICore-side rotation dropped a buffer (free queue empty) and that
    // task's AICore record never made it to the host, leaving the AICPU-side
    // L2SwimlaneAicpuTaskRecord with `start_time == 0`. Subtracting base_time_cycles from
    // 0 would underflow to a huge double timestamp, painting an off-the-chart
    // bar in the swimlane viewer; safer to drop the record. The drop count is
    // already surfaced via `dropped_record_count` and the join warning logged
    // in join_aicore_records().
    std::vector<size_t> emit_indices;
    emit_indices.reserve(tagged_records.size());
    size_t unmatched_dropped = 0;
    for (size_t i = 0; i < tagged_records.size(); ++i) {
        if (tagged_records[i].record->start_time == 0) {
            unmatched_dropped++;
            continue;
        }
        emit_indices.push_back(i);
    }
    if (unmatched_dropped > 0) {
        LOG_WARN("Dropped %zu task record(s) with unmatched AICore timing from swimlane export", unmatched_dropped);
    }

    for (size_t e = 0; e < emit_indices.size(); ++e) {
        const auto &tagged = tagged_records[emit_indices[e]];
        const auto &record = *tagged.record;

        // Convert times to microseconds
        double start_us = cycles_to_us(record.start_time - base_time_cycles);
        double end_us = cycles_to_us(record.end_time - base_time_cycles);
        double duration_us = end_us - start_us;
        double dispatch_us = (record.dispatch_time > 0) ? cycles_to_us(record.dispatch_time - base_time_cycles) : 0.0;
        double finish_us = (record.finish_time > 0) ? cycles_to_us(record.finish_time - base_time_cycles) : 0.0;

        const char *core_type_str = (record.core_type == CoreType::AIC) ? "aic" : "aiv";

        outfile << "    {\n";
        outfile << "      \"task_id\": " << record.task_id << ",\n";
        outfile << "      \"func_id\": " << record.func_id << ",\n";
        outfile << "      \"core_id\": " << tagged.core_id << ",\n";
        outfile << "      \"core_type\": \"" << core_type_str << "\",\n";
        outfile << "      \"ring_id\": " << static_cast<int>(record.task_id >> 32) << ",\n";
        outfile << "      \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us << ",\n";
        outfile << "      \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ",\n";
        outfile << "      \"duration_us\": " << std::fixed << std::setprecision(3) << duration_us << ",\n";
        outfile << "      \"dispatch_time_us\": " << std::fixed << std::setprecision(3) << dispatch_us << ",\n";
        outfile << "      \"finish_time_us\": " << std::fixed << std::setprecision(3) << finish_us << "\n";
        // Fanout is no longer carried on the device hot path — dep_gen replay
        // (deps.json) is the sole source of truth, joined in by tooling.
        outfile << "    }";
        if (e + 1 < emit_indices.size()) {
            outfile << ",";
        }
        outfile << "\n";
    }
    outfile << "  ]";

    // Step 8: Write phase profiling data (level >= 3)
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        auto sched_phase_name = [](L2SwimlaneAicpuPhaseId id) -> const char * {
            switch (id) {
            case L2SwimlaneAicpuPhaseId::SCHED_COMPLETE:
                return "complete";
            case L2SwimlaneAicpuPhaseId::SCHED_DISPATCH:
                return "dispatch";
            default:
                // Legacy SCHED_IDLE_WAIT (3) and SCHED_SCAN (2) land here on
                // old captures; host tools skip "unknown" sched records and
                // rebuild idle from gaps between known records on the
                // same thread.
                return "unknown";
            }
        };

        auto orch_phase_name = [](L2SwimlaneAicpuPhaseId id) -> const char * {
            switch (id) {
            case L2SwimlaneAicpuPhaseId::ORCH_SUBMIT:
                return "orch_submit";
            default:
                // Legacy per-sub-step orch ids 17-24 land here on old captures;
                // host tools drop "unknown" records.
                return "unknown";
            }
        };

        // AICPU scheduler phases (filtered from unified collected_phase_records_)
        outfile << ",\n  \"aicpu_scheduler_phases\": [\n";
        for (size_t t = 0; t < collected_phase_records_.size(); t++) {
            outfile << "    [\n";
            bool first = true;
            for (const auto &pr : collected_phase_records_[t]) {
                if (!is_scheduler_phase(pr.phase_id)) continue;
                double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                if (!first) outfile << ",\n";
                outfile << "      {\"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                        << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ", \"phase\": \""
                        << sched_phase_name(pr.phase_id) << "\""
                        << ", \"loop_iter\": " << pr.loop_iter << ", \"tasks_processed\": " << pr.tasks_processed;
                // Phase-specific deltas (currently only SCHED_DISPATCH carries
                // pop_hit / pop_miss). Other phases pass zero extras; omitting
                // them keeps the JSON terse per record.
                if (pr.phase_id == L2SwimlaneAicpuPhaseId::SCHED_DISPATCH) {
                    outfile << ", \"pop_hit\": " << pr.extra1 << ", \"pop_miss\": " << pr.extra2;
                }
                outfile << "}";
                first = false;
            }
            if (!first) outfile << "\n";
            outfile << "    ]";
            if (t < collected_phase_records_.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "  ]";

        // Orchestrator timing is no longer emitted as a separate aggregate
        // block. Per-event L2SwimlaneAicpuPhaseRecord[] entries (emitted as
        // aicpu_orchestrator_phases below) are the single source of truth;
        // the run-window envelope is still visible in the device-side
        // LOG_INFO_V9 "Thread N: orch_start=… orch_end=… orch_cost=…" line.

        // Per-task orchestrator phase records (level >= 4, filtered from unified collected_phase_records_)
        bool has_orch_phases = false;
        if (l2_swimlane_level_ >= L2SwimlaneLevel::ORCH_PHASES) {
            for (const auto &v : collected_phase_records_) {
                for (const auto &r : v) {
                    if (!is_scheduler_phase(r.phase_id)) {
                        has_orch_phases = true;
                        break;
                    }
                }
                if (has_orch_phases) break;
            }
        }
        if (has_orch_phases) {
            outfile << ",\n  \"aicpu_orchestrator_phases\": [\n";
            for (size_t t = 0; t < collected_phase_records_.size(); t++) {
                outfile << "    [\n";
                bool first = true;
                for (const auto &pr : collected_phase_records_[t]) {
                    if (is_scheduler_phase(pr.phase_id)) continue;
                    double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                    double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                    if (!first) outfile << ",\n";
                    outfile << "      {\"phase\": \"" << orch_phase_name(pr.phase_id) << "\""
                            << ", \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                            << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us
                            << ", \"submit_idx\": " << pr.loop_iter << ", \"task_id\": " << pr.task_id << "}";
                    first = false;
                }
                if (!first) outfile << "\n";
                outfile << "    ]";
                if (t < collected_phase_records_.size() - 1) outfile << ",";
                outfile << "\n";
            }
            outfile << "  ]";
        }
    }

    // Core-to-thread mapping
    if (!core_to_thread_.empty()) {
        outfile << ",\n  \"core_to_thread\": [";
        for (size_t i = 0; i < core_to_thread_.size(); i++) {
            outfile << static_cast<int>(core_to_thread_[i]);
            if (i < core_to_thread_.size() - 1) outfile << ", ";
        }
        outfile << "]";
    }

    outfile << "\n}\n";

    // Step 9: Close file
    outfile.close();

    uint32_t record_count = static_cast<uint32_t>(tagged_records.size());
    LOG_INFO_V0("=== JSON Export Complete ===");
    LOG_INFO_V0("File: %s", filepath.c_str());
    LOG_INFO_V0("Records: %u", record_count);

    return 0;
}

int L2SwimlaneCollector::finalize(L2SwimlaneUnregisterCallback unregister_cb, const L2SwimlaneFreeCallback &free_cb) {
    if (shm_host_ == nullptr) {
        return 0;
    }

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    LOG_DEBUG("Cleaning up performance profiling resources");

    // Every release site below goes through release_one_buffer so the
    // unregister and free are an inseparable pair — each dev_ptr that
    // alloc_single_buffer installed via halHostRegister is unregistered
    // before its device memory is freed. Without this the Ascend HAL's
    // per-device registration table accumulates leaked entries across
    // init_l2_swimlane() invocations and back-to-back l2_swimlane tests on
    // a reused Worker fail at rc=8 from halHostRegister.

    // Free standalone l2_swimlane_aicore_rotation_table table
    release_one_buffer(aicore_ring_addr_table_dev_, unregister_cb, free_cb);
    aicore_ring_addr_table_dev_ = nullptr;

    // Release framework-owned buffers (recycled pools, done_queue, ready_queue).
    manager_.release_owned_buffers([this, unregister_cb, free_cb](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    });

    // Per-core: current buffer + free_queue slots — these were owned by
    // the AICPU side, not the framework. Same drain pattern for both the
    // L2SwimlaneAicpuTaskBuffer pool and the L2SwimlaneAicoreTaskBuffer pool.
    auto drain_free_queue = [&](L2SwimlaneFreeQueue &fq) {
        rmb();
        uint32_t head = fq.head;
        uint32_t tail = fq.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            release_one_buffer(reinterpret_cast<void *>(fq.buffer_ptrs[slot]), unregister_cb, free_cb);
            fq.buffer_ptrs[slot] = 0;
        }
        fq.head = tail;
    };

    for (int i = 0; i < num_aicore_; i++) {
        L2SwimlaneAicpuTaskPool *state = get_perf_buffer_state(shm_host_, i);
        release_one_buffer(reinterpret_cast<void *>(state->current_buf_ptr), unregister_cb, free_cb);
        state->current_buf_ptr = 0;
        drain_free_queue(state->free_queue);

        L2SwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(shm_host_, num_aicore_, i);
        release_one_buffer(reinterpret_cast<void *>(ac_state->rotation.current_buf_ptr), unregister_cb, free_cb);
        ac_state->rotation.current_buf_ptr = 0;
        drain_free_queue(ac_state->free_queue);
    }

    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    for (int t = 0; t < num_phase_threads; t++) {
        L2SwimlaneAicpuPhasePool *state = get_phase_buffer_state(shm_host_, num_aicore_, t);

        release_one_buffer(reinterpret_cast<void *>(state->current_buf_ptr), unregister_cb, free_cb);
        state->current_buf_ptr = 0;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            release_one_buffer(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]), unregister_cb, free_cb);
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    // Main shm: unregister + free as a pair, same as every other buffer.
    // ProfilerBase's set_memory_context handed register_cb == nullptr iff the
    // caller doesn't intend to register, so checking unregister_cb inside
    // release_one_buffer is sufficient — no separate ``was_registered_`` flag.
    release_one_buffer(perf_shared_mem_dev_, unregister_cb, free_cb);
    LOG_DEBUG("Main shm released");

    perf_shared_mem_dev_ = nullptr;
    // shm_host_ aliases freed device/host memory now; null it so is_initialized()
    // reports false, the dtor's "destroyed without finalize()" warning stays
    // quiet, and a re-entrant finalize() / re-init hits the early-out instead of
    // walking freed buffer state. Mirrors PMU/DepGen/TensorDump collectors.
    shm_host_ = nullptr;
    collected_perf_records_.clear();
    collected_phase_records_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    total_perf_collected_ = 0;
    total_phase_collected_ = 0;
    clear_memory_context();

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
