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
 * [hash digest][int32 T][int32 S][ChipTensor × T][uint64 × S]; the
 * forked child decodes the same layout to rebuild a TaskArgsView and resolves
 * the digest to a target-private execution slot.
 */

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/task_args_wire.h"
#include "../worker/runtime_c_api.h"

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

// User-visible scope-nesting cap. Matches L2 CHIP_MAX_SCOPE_DEPTH.
static constexpr int32_t MAX_SCOPE_DEPTH = 64;

// Number of independent HeapRing layers inside Ring. Scope depth maps
// to ring index via `min(depth, MAX_RING_DEPTH - 1)` (L2-style);
// scopes deeper than MAX_RING_DEPTH share the innermost ring.
// Matches L2's CHIP_MAX_RING_DEPTH (Strict-1).
static constexpr int32_t MAX_RING_DEPTH = 4;

static constexpr int32_t INVALID_SLOT = -1;

using RunId = uint64_t;
static constexpr RunId INVALID_RUN_ID = 0;
using TaskSlot = int32_t;

// Admission reserves a generation-safe pipeline slot before graph construction.
// Closing the graph publishes PREPARED; only the whole-run FIFO head may enter
// EXECUTING and reach a device endpoint.
enum class RunPhase : int32_t {
    RESERVED = 0,
    BUILDING = 1,
    PREPARED = 2,
    EXECUTING = 3,
    COMPLETED = 4,
    FAILED = 5,
};

struct RunState {
    RunState(RunId run_id, PipelineSlotLease slot_lease) :
        id(run_id),
        lease(slot_lease) {}

    RunId id{INVALID_RUN_ID};
    // Written only while holding both the Orchestrator's `runs_mu_` and this
    // run's `completion_mu`, so a reader holding either sees a whole value.
    // Generation zero means no native pipeline slot is held.
    PipelineSlotLease lease{};
    std::atomic<RunPhase> phase{RunPhase::RESERVED};
    std::atomic<int32_t> active_tasks{0};
    std::atomic<int32_t> pending_accepts{0};
    mutable std::mutex completion_mu;
    std::condition_variable completion_cv;
    std::exception_ptr first_error;
    std::vector<TaskSlot> task_slots;
    bool submission_closed{false};
    bool submission_failed{false};
    bool lease_released{false};
    // When this run reached a terminal phase. `node.post_fence_retirement`
    // measures from here to the end of release_run, the only work this process
    // does outside every other span.
    int64_t trace_terminal_ns{0};
};

// =============================================================================
// Task slot index type
// =============================================================================

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
    // Submit owns the slot. It is already reachable from its producers' fanout
    // lists — wiring has to happen under each producer's fanout_mu, so the slot
    // cannot be kept private until it is finished — but fanin_count,
    // fanout_total and the ref counters are not final yet. No other thread may
    // advance a BUILDING slot: a producer that completes leaves its fanin
    // release for submit to observe, and a producer that fails parks the slot
    // at FAILED for submit to propagate. See claim_task_failure().
    BUILDING = 7,
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
    RunId run_id{INVALID_RUN_ID};
    PipelineSlotLease pipeline_lease{};

    // --- Fanin ---
    // `fanin_count` is written once by submit's publication and read by every
    // completing producer. The two are concurrent — a producer can complete
    // while the slot is still BUILDING — so both the write and every read live
    // under `fanout_mu`, which also carries the state transition they decide
    // on. See try_mark_ready() for why they cannot be separated.
    std::atomic<int32_t> fanin_count{0};
    std::atomic<int32_t> fanin_released{0};  // incremented by each completing producer

    // Set when a BUILDING failure claim leaves propagation to submit, and by
    // cancellation when it claims a PENDING or READY slot itself. Fallible
    // preparation happens while it remains set, so cancellation can retry its
    // own FAILED slot without allowing another active owner to take it over.
    std::atomic<bool> failure_propagation_pending{false};

    // --- Fanout, plus the slot's pre-dispatch state transitions ---
    // orch adds wait consumers; scheduler traverses on completion. The same mutex
    // covers every BUILDING/PENDING -> {READY, FAILED} transition and the
    // fields each of those decisions reads.
    std::mutex fanout_mu;
    std::vector<TaskSlot> fanout_consumers;
    int32_t fanout_total{0};                  // scope ref + retained consumer refs
    std::atomic<int32_t> fanout_released{0};  // incremented as each ref is released

    // --- TensorMap keys registered by this task (for cleanup on CONSUMED) ---
    std::vector<TensorKey> output_keys;

    // Exact stable NEXT_LEVEL worker id for each task/group member. SUB slots
    // leave this empty because SUB has no worker-selection API.
    std::vector<int32_t> target_worker_ids;

    int32_t target_worker_id(int32_t i) const { return target_worker_ids.at(static_cast<size_t>(i)); }

    // --- Producer tasks this task depends on (for deferred release) ---
    // When this task reaches COMPLETED, the Scheduler releases one fanout ref
    // on each producer — mirroring L2's "deferred release: walk fanin" step.
    std::vector<TaskSlot> fanin_producers;

    // --- Failure state ---
    // Root worker failures and downstream poison both land here. The
    // originating RunState owns first-error-wins reporting.
    //
    // Guarded by fanout_mu, the same lock that guards the fanout snapshot a
    // failure propagates over: whoever claims the failure writes the message
    // and reads the consumer list under one acquisition, so a consumer that
    // sees FAILED while wiring itself in also sees the reason.
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
    std::mutex group_mu;
    std::vector<GroupMemberState> group_member_states;
    std::vector<EndpointOutcome> group_member_outcomes;
    std::atomic<int32_t> group_terminal_count{0};
    bool group_failed{false};
    int32_t group_first_failure_index{-1};
    std::string group_first_failure_message;

    bool is_group() const { return is_group_; }
    int32_t group_size() const { return is_group_ ? static_cast<int32_t>(task_args_list.size()) : 1; }
    const RemoteTaskArgsSidecar &remote_sidecar_for(int32_t i) const {
        static const RemoteTaskArgsSidecar empty;
        if (is_group_) {
            if (i < 0 || static_cast<size_t>(i) >= remote_sidecars.size()) return empty;
            return remote_sidecars[static_cast<size_t>(i)];
        }
        return remote_sidecar;
    }

    // The i-th worker's Tensor args (the L3→L2 wire element).
    // `i` must be 0 for non-group slots; 0..group_size()-1 for groups.
    const TaskArgs &args(int32_t i) const { return is_group_ ? task_args_list[static_cast<size_t>(i)] : task_args; }

    TaskSlotState() = default;
    TaskSlotState(const TaskSlotState &) = delete;
    TaskSlotState &operator=(const TaskSlotState &) = delete;

    void reset();
};

// The one claim every failure path uses to move a task to FAILED: a device
// completion poisoning its consumers, a run cancellation, and a submit that
// wires onto a producer that has already failed. Returns the state the slot
// held when the claim was won, or std::nullopt when the slot was not claimable
// — already terminal, or RUNNING and therefore owned by the device until its
// own completion arrives.
//
// The claim and failure_message are published under fanout_mu, the lock a
// propagation snapshots the consumer list under. That is what makes the
// snapshot exact: a task wiring itself onto this producer either registers
// before the claim and is poisoned by it, or observes FAILED — with its
// message — and poisons itself.
//
// A claim won from BUILDING stops there. The submitting thread still owns that
// slot's fanin/fanout bookkeeping, so it completes the propagation itself; a
// caller that touched the reference counters would race with the wiring that is
// still in flight. Claims from PENDING and READY are exclusively owned by the
// claimant and may propagate immediately; they do not advertise takeover debt.
std::optional<TaskState> claim_task_failure(TaskSlotState &s, const std::string &message);

// Settle a fully prepared propagation debt and claim its reference releases.
// Callers finish every fallible local snapshot first. Only BUILDING handoff and
// cancellation-owned claims use debt, so no active propagation contender may
// be preparing the same slot concurrently.
bool commit_failure_propagation(TaskSlotState &s) noexcept;

// Records every not-yet-terminal member of a group slot as SKIPPED. A no-op on
// a single-task slot. Called by whoever owns a failure's propagation, so a
// group that never dispatched still reports one terminal outcome per member.
void mark_group_members_skipped(TaskSlotState &s, const std::string &message);

// The single PENDING -> READY transition, taken by submit's publication and by
// every completing producer. Returns true for the one caller that made it, who
// then owns enqueuing the slot.
//
// "Every live producer has released" is not two independently readable facts.
// `fanin_count` is published by submit and `fanin_released` is advanced by
// producers, so a producer that reads a count submit has not published yet —
// zero, which any release passes — and acts on the state afterwards would
// launch a task whose remaining producers are still running. Comparing the
// pair and changing the state under one acquisition is what ties the count a
// caller judges to the state it changes.
//
// Neither side can lose the other's half: submit takes the same lock to
// publish the count together with the transition out of BUILDING, so a
// producer that arrives first is re-evaluated by submit, and one that arrives
// after sees a published count.
bool try_mark_ready(TaskSlotState &s);

// =============================================================================
// ReadyQueue — Orch pushes, Scheduler pops
// =============================================================================

class ReadyQueue {
public:
    void push(TaskSlot slot);
    void push(RunId run_id, TaskSlot slot);

    // Non-blocking: returns false immediately if empty.
    bool try_pop(TaskSlot &out);
    bool try_pop(RunId run_id, TaskSlot &out);

    bool empty() const;
    bool empty(RunId run_id) const;

    // Non-blocking: copies the front without removing it.
    bool try_front(TaskSlot &out);
    bool try_front(RunId run_id, TaskSlot &out);

    void erase_run(RunId run_id);

private:
    std::unordered_map<RunId, std::queue<TaskSlot>> queues_;
    std::deque<RunId> run_order_;
    mutable std::mutex mu_;
};

// Directed NEXT_LEVEL queues. Worker registration is complete before reset()
// and the worker-id mapping is immutable while scheduling is active.
class NextLevelReadyQueues {
public:
    void reset(const std::vector<int32_t> &worker_ids);
    void push_single(int32_t worker_id, TaskSlot slot);
    void push_single(int32_t worker_id, RunId run_id, TaskSlot slot);
    bool try_pop_single(int32_t worker_id, TaskSlot &out);
    bool try_pop_single(int32_t worker_id, RunId run_id, TaskSlot &out);
    bool try_front_single(int32_t worker_id, TaskSlot &out);
    bool try_front_single(int32_t worker_id, RunId run_id, TaskSlot &out);
    void push_group(TaskSlot slot);
    void push_group(RunId run_id, TaskSlot slot);
    bool try_front_group(TaskSlot &out);
    bool try_front_group(RunId run_id, TaskSlot &out);
    bool try_pop_group(TaskSlot &out);
    bool try_pop_group(RunId run_id, TaskSlot &out);
    bool groups_empty() const;
    bool groups_empty(RunId run_id) const;
    bool single_empty(int32_t worker_id, RunId run_id) const;
    bool singles_empty(RunId run_id) const;
    void erase_run(RunId run_id);
    const std::vector<int32_t> &worker_ids() const { return worker_ids_; }

private:
    size_t index_for(int32_t worker_id) const;

    std::vector<int32_t> worker_ids_;
    std::vector<std::unique_ptr<ReadyQueue>> queues_;
    ReadyQueue group_queue_;
};
