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
 * Distributed runtime — shared scheduling types.
 *
 * Every level in the hierarchy (L3 HostWorker, L4, L5, …) runs the same
 * scheduling engine.  This header defines:
 *   - WorkerType / TaskState enumerations
 *   - TaskSlotState: per-task scheduling bookkeeping (stores TaskArgs
 *                        directly — no separate dispatch carrier struct)
 *   - ReadyQueue: Orch→Scheduler notification channel
 *
 * Dispatch encodes (callable hash digest, CallConfig, TaskArgs) into the
 * per-WorkerThread shm mailbox with inline std::memcpy of
 * [hash digest][int32 T][int32 S][ContinuousTensor × T][uint64 × S]; the
 * forked child decodes the same layout to rebuild a TaskArgsView and resolves
 * the digest to a target-private execution slot.
 */

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/task_args.h"

// =============================================================================
// TensorKey — compound key for TensorMap dependency tracking
// =============================================================================

enum class TensorAddressKind : int32_t {
    LOCAL_HOST = 0,
    LOCAL_CHILD = 1,
    REMOTE_BUFFER = 2,
    HOST_INLINE = 3,
};

struct TensorKey {
    uint64_t ptr;
    int32_t worker_id;  // -1 = host; non-negative values are NEXT_LEVEL worker ids
    TensorAddressKind address_kind{TensorAddressKind::LOCAL_HOST};
    int32_t owner_worker_id{-1};
    uint64_t buffer_id{0};
    uint64_t generation{0};
    uint64_t offset_begin{0};

    static TensorKey local_host(uint64_t ptr) { return TensorKey{ptr, -1, TensorAddressKind::LOCAL_HOST}; }
    static TensorKey local_child(uint64_t ptr, int32_t worker_id) {
        return TensorKey{ptr, worker_id, TensorAddressKind::LOCAL_CHILD};
    }
    static TensorKey remote_buffer(
        TensorAddressKind address_kind, int32_t owner_worker_id, uint64_t buffer_id, uint64_t generation,
        uint64_t offset_begin
    ) {
        TensorKey key{};
        key.ptr = 0;
        key.worker_id = -1;
        key.address_kind = address_kind;
        key.owner_worker_id = owner_worker_id;
        key.buffer_id = buffer_id;
        key.generation = generation;
        key.offset_begin = offset_begin;
        return key;
    }

    bool operator==(const TensorKey &o) const {
        return ptr == o.ptr && worker_id == o.worker_id && address_kind == o.address_kind &&
               owner_worker_id == o.owner_worker_id && buffer_id == o.buffer_id && generation == o.generation &&
               offset_begin == o.offset_begin;
    }
};

struct TensorKeyHash {
    size_t operator()(const TensorKey &k) const {
        size_t h = std::hash<uint64_t>{}(k.ptr);
        auto mix = [&h](size_t v) {
            h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        mix(std::hash<int>{}(k.worker_id));
        mix(std::hash<int>{}(static_cast<int>(k.address_kind)));
        mix(std::hash<int>{}(k.owner_worker_id));
        mix(std::hash<uint64_t>{}(k.buffer_id));
        mix(std::hash<uint64_t>{}(k.generation));
        mix(std::hash<uint64_t>{}(k.offset_begin));
        return h;
    }
};

// =============================================================================
// Constants
// =============================================================================

// User-visible scope-nesting cap. Matches L2 PTO2_MAX_SCOPE_DEPTH.
static constexpr int32_t MAX_SCOPE_DEPTH = 64;

// Number of independent HeapRing layers inside Ring. Scope depth maps
// to ring index via `min(depth, MAX_RING_DEPTH - 1)` (L2-style);
// scopes deeper than MAX_RING_DEPTH share the innermost ring.
// Matches L2's PTO2_MAX_RING_DEPTH (Strict-1).
static constexpr int32_t MAX_RING_DEPTH = 4;

static constexpr int32_t INVALID_SLOT = -1;

// =============================================================================
// Task slot index type
// =============================================================================

using TaskSlot = int32_t;

static constexpr size_t CALLABLE_HASH_DIGEST_SIZE = 32;

enum class CallableKind : int32_t {
    CHIP_CALLABLE = 1,
    PYTHON_SERIALIZED = 2,
    PYTHON_IMPORT = 3,
};

enum class TargetNamespace : int32_t {
    LOCAL_CHIP = 1,
    LOCAL_PYTHON = 2,
    REMOTE_TASK_DISPATCHER = 3,
};

struct CallableIdentity {
    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest{};
    CallableKind kind{CallableKind::CHIP_CALLABLE};
    TargetNamespace target_namespace{TargetNamespace::LOCAL_CHIP};
};

// =============================================================================
// WorkerType
// =============================================================================

enum class WorkerType : int32_t {
    NEXT_LEVEL = 0,  // Next-level Worker (L3→ChipWorker, L4→Worker(L3), …)
    SUB = 1,         // SubWorker: fork/shm Python function
};

// =============================================================================
// TaskState
// =============================================================================

enum class TaskState : int32_t {
    FREE = 0,       // slot not in use
    PENDING = 1,    // waiting for fanin dependencies
    READY = 2,      // all fanins satisfied, in ready queue
    RUNNING = 3,    // dispatched to a worker
    COMPLETED = 4,  // worker finished, outputs may still be referenced
    FAILED = 5,     // worker failed or a producer poisoned this slot
    CONSUMED = 6,   // all references released, slot may be reused
};

enum class EndpointOutcome : int32_t {
    SUCCESS = 0,
    TASK_FAILURE = 1,
    ENDPOINT_FAILURE = 2,
    SKIPPED = 3,
};

enum class RemoteAddressSpace : int32_t {
    HOST_INLINE = 1,
    REMOTE_DEVICE = 2,
    REMOTE_WINDOW = 3,
    UB_LDST = 4,
};

// RemoteBufferHandle combines wire-visible identity/mapping metadata
// (`worker_id`, `owner_worker_id`, `buffer_id`, `generation`, `import_id`,
// `address_space`, `nbytes`, `offset`, `remote_addr`, `rkey_or_token`,
// `ub_ldst_va`, `access_flags`) with parent-local lifecycle state (`released`,
// `live_slot_refs`). Keep wire formats in remote_wire.cpp explicit; do not
// serialize this struct by raw POD copy.
struct RemoteBufferHandle {
    int32_t worker_id{-1};
    int32_t owner_worker_id{-1};
    uint64_t buffer_id{0};
    uint64_t generation{0};
    uint64_t import_id{0};
    RemoteAddressSpace address_space{RemoteAddressSpace::REMOTE_DEVICE};
    uint64_t nbytes{0};
    uint64_t offset{0};
    uint64_t remote_addr{0};
    uint64_t rkey_or_token{0};
    uint64_t ub_ldst_va{0};
    uint32_t access_flags{0};
    bool released{false};
    int32_t live_slot_refs{0};
};

struct RemoteBufferExport {
    int32_t owner_worker_id{-1};
    uint64_t buffer_id{0};
    uint64_t generation{0};
    RemoteAddressSpace address_space{RemoteAddressSpace::REMOTE_WINDOW};
    uint64_t offset{0};
    uint64_t nbytes{0};
    uint64_t export_id{0};
    uint64_t remote_addr{0};
    uint64_t rkey_or_token{0};
    uint64_t ub_ldst_va{0};
    uint32_t access_flags{0};
    std::string transport_profile;
    std::vector<uint8_t> transport_descriptor;
};

struct RemoteTensorDesc {
    RemoteAddressSpace address_space{RemoteAddressSpace::REMOTE_DEVICE};
    int32_t owner_worker_id{-1};
    uint64_t buffer_id{0};
    uint64_t offset{0};
    uint64_t nbytes{0};
    uint64_t remote_addr{0};
    uint64_t rkey_or_token{0};
    uint64_t generation{0};
    uint64_t inline_payload_offset{0};
    uint64_t inline_payload_len{0};
    uint64_t flags{0};
};

struct RemoteTensorRef {
    RemoteBufferHandle handle{};
    uint64_t offset{0};
    std::vector<uint32_t> shape;
    DataType dtype{DataType::FLOAT32};
};

struct RemoteTensorSidecar {
    bool present{false};
    RemoteTensorDesc desc{};
};

struct RemoteTaskArgsSidecar {
    std::vector<RemoteTensorSidecar> tensors;
    std::vector<uint8_t> inline_payload;

    bool empty() const {
        if (!inline_payload.empty()) return false;
        for (const auto &tensor : tensors) {
            if (tensor.present) return false;
        }
        return true;
    }

    void clear() {
        tensors.clear();
        inline_payload.clear();
    }
};

enum class GroupMemberState : int32_t {
    NOT_DISPATCHED = 0,
    RUNNING = 1,
    SUCCESS = 2,
    FAILED = 3,
    SKIPPED = 4,
};

struct WorkerCompletion {
    TaskSlot task_slot{INVALID_SLOT};
    int32_t group_index{0};
    EndpointOutcome outcome{EndpointOutcome::SUCCESS};
    std::string error_message;
};

// =============================================================================
// TaskSlotState — per-task scheduling bookkeeping
// =============================================================================
//
// Stores the submitted TaskArgs directly. Dispatch builds a TaskArgsView on
// demand via `args_view(i)` and encodes it into the mailbox blob via
// write_blob; the child decodes with read_blob. There is no separate
// dispatch carrier struct — the slot itself is the dispatch state.

struct TaskSlotState {
    std::atomic<TaskState> state{TaskState::FREE};

    // --- Fanin (orch writes once; scheduler reads atomically) ---
    int32_t fanin_count{0};
    std::atomic<int32_t> fanin_released{0};  // incremented by each completing producer

    // --- Fanout (protected by fanout_mu) ---
    // orch adds consumers; scheduler traverses on completion
    std::mutex fanout_mu;
    std::vector<TaskSlot> fanout_consumers;
    int32_t fanout_total{0};                  // 1 (scope ref) + fanout_consumers.size()
    std::atomic<int32_t> fanout_released{0};  // incremented as each ref is released

    // --- TensorMap keys registered by this task (for cleanup on CONSUMED) ---
    std::vector<TensorKey> output_keys;

    // Empty outer vector means legacy/unconstrained dispatch. When present,
    // each group member's vector is the final callable/data worker-id
    // intersection and must be non-empty.
    std::vector<std::vector<int32_t>> eligible_worker_ids;

    // --- Submit affinity (set by submit_next_level with worker= parameter) ---
    // Empty = unconstrained. For NEXT_LEVEL, non-negative values are stable
    // worker ids. SUB has no public affinity surface and keeps index
    // semantics for any internal callers that populate this vector.
    std::vector<int32_t> affinities;

    int32_t get_affinity(int i) const {
        if (affinities.empty()) return -1;
        return affinities[static_cast<size_t>(i)];
    }

    // --- Producer tasks this task depends on (for deferred release) ---
    // When this task reaches COMPLETED, the Scheduler releases one fanout ref
    // on each producer — mirroring L2's "deferred release: walk fanin" step.
    std::vector<TaskSlot> fanin_producers;

    // --- Failure state ---
    // Root worker failures and downstream poison both land here. The
    // WorkerManager still owns first-error-wins reporting for drain().
    std::string failure_message;

    // --- Task data (stored on parent heap, lives until slot CONSUMED) ---
    WorkerType worker_type{WorkerType::NEXT_LEVEL};
    // Stable callable identity submitted by the parent. Child-local integer
    // execution slots stay private to the target process.
    CallableIdentity callable{};
    CallConfig config{};  // NEXT_LEVEL config (block_dim, aicpu_thread_num, diagnostics sub-features)

    // Unified task-args storage: `task_args` is the single-task builder;
    // when `is_group_` is true, `task_args_list` carries one TaskArgs per
    // worker (N-SPMD group, L3-flavoured — each member has its own distinct
    // tensors/scalars, unlike L2's SPMD single-payload). `task_args` stays
    // empty for groups.
    TaskArgs task_args;
    std::vector<TaskArgs> task_args_list;
    bool is_group_{false};
    RemoteTaskArgsSidecar remote_sidecar;
    std::vector<RemoteTaskArgsSidecar> remote_sidecars;

    // Runtime-owned OUTPUT slabs live in the Worker's HeapRing and are
    // reclaimed implicitly by Ring::release(slot) — no per-slot
    // munmap is needed. See docs/orchestrator.md §8b.

    // --- HeapRing layer membership (Strict-1 per-scope rings) ---
    // Set by Ring::alloc from the caller's scope depth. ring_idx picks
    // which of the MAX_RING_DEPTH heaps holds this slot's slab;
    // ring_slot_idx is the slot's position within that ring's FIFO order
    // and indexes the ring's per-slot released/heap_end vectors.
    int32_t ring_idx{0};
    int32_t ring_slot_idx{0};

    // --- Group bookkeeping ---
    std::atomic<int32_t> sub_complete_count{0};
    std::mutex group_mu;
    std::vector<GroupMemberState> group_member_states;
    std::vector<EndpointOutcome> group_member_outcomes;
    std::atomic<int32_t> group_terminal_count{0};
    std::atomic<int32_t> group_dispatched_count{0};
    bool group_failed{false};
    int32_t group_first_failure_index{-1};
    std::string group_first_failure_message;

    bool is_group() const { return is_group_; }
    int32_t group_size() const { return is_group_ ? static_cast<int32_t>(task_args_list.size()) : 1; }
    bool has_worker_constraints() const { return !eligible_worker_ids.empty(); }

    const std::vector<int32_t> &eligible_workers_for(int32_t i) const {
        static const std::vector<int32_t> empty;
        if (eligible_worker_ids.empty()) return empty;
        if (i < 0 || static_cast<size_t>(i) >= eligible_worker_ids.size()) return empty;
        return eligible_worker_ids[static_cast<size_t>(i)];
    }

    bool worker_allowed(int32_t i, int32_t worker_id) const {
        if (eligible_worker_ids.empty()) return true;
        const auto &eligible = eligible_workers_for(i);
        for (int32_t id : eligible) {
            if (id == worker_id) return true;
        }
        return false;
    }

    const RemoteTaskArgsSidecar &remote_sidecar_for(int32_t i) const {
        static const RemoteTaskArgsSidecar empty;
        if (is_group_) {
            if (i < 0 || static_cast<size_t>(i) >= remote_sidecars.size()) return empty;
            return remote_sidecars[static_cast<size_t>(i)];
        }
        return remote_sidecar;
    }

    // Zero-copy view over the i-th worker's args (THREAD-mode dispatch).
    // `i` must be 0 for non-group slots; 0..group_size()-1 for groups.
    TaskArgsView args_view(int32_t i) const {
        return is_group_ ? make_view(task_args_list[static_cast<size_t>(i)]) : make_view(task_args);
    }

    TaskSlotState() = default;
    TaskSlotState(const TaskSlotState &) = delete;
    TaskSlotState &operator=(const TaskSlotState &) = delete;

    void reset();
};

// =============================================================================
// ReadyQueue — Orch pushes, Scheduler pops
// =============================================================================

class ReadyQueue {
public:
    void push(TaskSlot slot);

    // Non-blocking: returns false immediately if empty.
    bool try_pop(TaskSlot &out);

    // Blocking: waits until a slot is available or shutdown() is called.
    // Returns false only when shutdown and queue is empty.
    bool wait_pop(TaskSlot &out);

    void shutdown();

private:
    std::queue<TaskSlot> q_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_{false};
};

// =============================================================================
// RunTiming — wall-clock breakdown returned by ChipWorker::run
// =============================================================================

// host_wall_ns is the steady_clock delta wrapping the dispatch; device_wall_ns
// is on-NPU wall captured by the platform AICPU entry (see KernelArgs::
// device_wall_ns). Mirrors PtoRunTiming in src/common/worker/pto_runtime_c_api.h
// so the value flows through unchanged from the dlsym ABI up to the Python
// binding.
struct RunTiming {
    uint64_t host_wall_ns = 0;
    uint64_t device_wall_ns = 0;
};
