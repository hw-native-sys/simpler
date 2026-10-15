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

#include "worker_manager.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/host_span_names.h"
#include "common/host_span_scope.h"
#include "ring.h"

namespace {

// Read the child-written error message from the mailbox, guaranteeing
// NUL-termination even if the child wrote exactly MAILBOX_ERROR_MSG_SIZE
// bytes without a terminator.
std::string read_error_msg(const char *mbox) {
    char buf[MAILBOX_ERROR_MSG_SIZE + 1] = {};
    std::memcpy(buf, mbox + MAILBOX_OFF_ERROR_MSG, MAILBOX_ERROR_MSG_SIZE);
    buf[MAILBOX_ERROR_MSG_SIZE] = '\0';
    return std::string(buf);
}

std::string format_digest(const uint8_t *digest) {
    if (digest == nullptr) return "sha256:<null>";
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out = "sha256:";
    out.reserve(71);
    for (size_t i = 0; i < CALLABLE_HASH_DIGEST_SIZE; ++i) {
        uint8_t v = digest[i];
        out.push_back(kHex[v >> 4]);
        out.push_back(kHex[v & 0x0F]);
    }
    return out;
}

#if SIMPLER_HOST_STRACE
const char *endpoint_kind_name(WorkerEndpointKind kind) {
    switch (kind) {
    case WorkerEndpointKind::LOCAL_MAILBOX:
        return "local_mailbox";
    case WorkerEndpointKind::REMOTE_L3:
        return "remote_l3";
    }
    return "unknown";
}

RunId trace_run_id(Ring *ring, TaskSlot task_slot) {
    if (ring == nullptr) return INVALID_RUN_ID;
    TaskSlotState *state = ring->slot_state(task_slot);
    return state == nullptr ? INVALID_RUN_ID : state->run_id;
}

uint64_t trace_callable_hash(Ring *ring, TaskSlot task_slot) {
    if (ring == nullptr) return 0;
    TaskSlotState *state = ring->slot_state(task_slot);
    if (state == nullptr) return 0;
    uint64_t hash = 0;
    std::memcpy(&hash, state->callable.digest.data(), sizeof(hash));
    return hash;
}

std::string
trace_dispatch_attrs(RunId run_id, const WorkerDispatch &dispatch, const WorkerEndpointCaps &caps, const char *role) {
    std::ostringstream attrs;
    attrs << "run_id=" << run_id << " task_slot=" << dispatch.task_slot << " group_index=" << dispatch.group_index
          << " worker_id=" << caps.worker_id << " dispatch_id=" << dispatch.dispatch_id
          << " endpoint_kind=" << endpoint_kind_name(caps.kind)
          << " prepare_only=" << static_cast<int>(dispatch.prepare_only) << " role=" << role;
    return attrs.str();
}

// The lease the dispatch's slot holds, read from the ring. Callers read this
// before publishing the frame: past that point the endpoint may retire the slot.
std::string trace_lease_attrs(Ring *ring, TaskSlot task_slot) {
    if (ring == nullptr) return "";
    const TaskSlotState *state = ring->slot_state(task_slot);
    if (state == nullptr) return "";
    std::ostringstream attrs;
    attrs << " slot_id=" << state->pipeline_lease.slot_id << " generation=" << state->pipeline_lease.generation;
    return attrs.str();
}
#endif

// Wall-clock period between child liveness samples. Every mailbox wait spins,
// so an iteration count would not map to a bounded wall time.
constexpr std::chrono::milliseconds kChildLivenessPollPeriod{10};

// Held back from a flush budget so the child has time to pack its report and
// this side to observe CONTROL_DONE. It shortens the child's wait relative to
// the parent's; it is not an ordering guarantee — the child's scheduling and
// reply can exceed it, and a request smaller than the margin leaves the child
// 1 ms. The parent's own timeout, poisoning and late-response ownership are
// what remain load-bearing when that happens.
constexpr uint64_t kChildFlushReplyMarginMs = 250;

// The child's stand-in for an unbounded wait. Its own budget is a millisecond
// count in an `int`, so "no deadline" is the largest count that fits rather
// than a sentinel the child would have to special-case.
constexpr uint64_t kUnboundedChildFlushBudgetMs = static_cast<uint64_t>(std::numeric_limits<int>::max());

std::string child_status_message(int child_pid, int status) {
    std::string msg = "child process pid=" + std::to_string(child_pid) + " exited before mailbox completion";
    if (WIFEXITED(status)) {
        msg += " (exit_status=" + std::to_string(WEXITSTATUS(status)) + ")";
    } else if (WIFSIGNALED(status)) {
        msg += " (signal=" + std::to_string(WTERMSIG(status)) + ")";
    } else {
        msg += " (status=" + std::to_string(status) + ")";
    }
    return msg;
}

}  // namespace

namespace {

[[noreturn]] void throw_unsupported_control(const char *op_name) {
    throw std::runtime_error(std::string(op_name) + " is not supported by this WorkerEndpoint");
}

}  // namespace

uint64_t WorkerEndpoint::control_malloc(size_t) { throw_unsupported_control("control_malloc"); }
uint64_t WorkerEndpoint::control_committed_device_memory() {
    throw_unsupported_control("control_committed_device_memory");
}
DeviceMemoryInfo WorkerEndpoint::control_device_memory_info() {
    throw_unsupported_control("control_device_memory_info");
}
DfxFlushReport WorkerEndpoint::control_dfx_flush(double) { throw_unsupported_control("control_dfx_flush"); }
void WorkerEndpoint::control_free(uint64_t) { throw_unsupported_control("control_free"); }
void WorkerEndpoint::control_copy_to(const BufferDescriptor &, const BufferDescriptor &, const CopySpan &) {
    throw_unsupported_control("control_copy_to");
}
void WorkerEndpoint::control_copy_from(const BufferDescriptor &, const BufferDescriptor &, const CopySpan &) {
    throw_unsupported_control("control_copy_from");
}
void WorkerEndpoint::control_prepare(const uint8_t *) { throw_unsupported_control("control_prepare"); }
void WorkerEndpoint::control_register(const char *, size_t, const uint8_t *) {
    throw_unsupported_control("control_register");
}
void WorkerEndpoint::control_unregister(const uint8_t *) { throw_unsupported_control("control_unregister"); }
void WorkerEndpoint::control_remote_prepare_register(
    remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *, const void *, size_t
) {
    throw_unsupported_control("control_remote_prepare_register");
}
void WorkerEndpoint::control_remote_commit_register(remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *) {
    throw_unsupported_control("control_remote_commit_register");
}
void WorkerEndpoint::control_remote_abort_register(remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *) {
    throw_unsupported_control("control_remote_abort_register");
}
void WorkerEndpoint::control_remote_unregister(remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *) {
    throw_unsupported_control("control_remote_unregister");
}
RemoteBufferHandle WorkerEndpoint::control_remote_malloc(size_t) { throw_unsupported_control("control_remote_malloc"); }
void WorkerEndpoint::control_remote_free(const RemoteBufferHandle &) {
    throw_unsupported_control("control_remote_free");
}
void WorkerEndpoint::control_remote_copy_to(const RemoteBufferHandle &, uint64_t, const void *, size_t) {
    throw_unsupported_control("control_remote_copy_to");
}
void WorkerEndpoint::control_remote_copy_from(void *, const RemoteBufferHandle &, uint64_t, size_t) {
    throw_unsupported_control("control_remote_copy_from");
}
RemoteBufferExport
WorkerEndpoint::control_remote_export(const RemoteBufferHandle &, uint64_t, uint64_t, uint32_t, const std::string &) {
    throw_unsupported_control("control_remote_export");
}
RemoteBufferHandle WorkerEndpoint::control_remote_import(int32_t, const RemoteBufferExport &, uint32_t) {
    throw_unsupported_control("control_remote_import");
}
void WorkerEndpoint::control_remote_release_import(const RemoteBufferHandle &) {
    throw_unsupported_control("control_remote_release_import");
}
std::vector<uint8_t> WorkerEndpoint::control_remote_domain(remote_l3::ControlName, const std::vector<uint8_t> &, bool) {
    throw_unsupported_control("control_remote_domain");
}
void WorkerEndpoint::control_generic(uint64_t, const char *, size_t, double, const uint8_t *) {
    throw_unsupported_control("control_generic");
}
void WorkerEndpoint::control_alloc_domain(const char *, const char *) {
    throw_unsupported_control("control_alloc_domain");
}
void WorkerEndpoint::control_release_domain(const char *) { throw_unsupported_control("control_release_domain"); }
void WorkerEndpoint::control_comm_init(const char *) { throw_unsupported_control("control_comm_init"); }

void WorkerEndpoint::submit_progress(Ring *, const WorkerDispatch &) {
    throw std::runtime_error("progress submission is not supported by this WorkerEndpoint");
}
bool WorkerEndpoint::poll_progress(WorkerEndpointProgress &) { return false; }
size_t WorkerEndpoint::available_request_transport() { return std::numeric_limits<size_t>::max(); }
bool WorkerEndpoint::activate_progress(RunId) { return false; }
void WorkerEndpoint::request_progress_stop() noexcept {}
void WorkerEndpoint::report_progress_error(const std::string &) { request_progress_stop(); }
bool WorkerEndpoint::report_submission_error(const WorkerDispatch &, const std::string &reason) {
    report_progress_error(reason);
    return false;
}

// =============================================================================
// LocalMailboxEndpoint — mailbox helpers
// =============================================================================

namespace {

// Refuse a registration whose region cannot hold the table its grant implies.
//
// Called twice for every endpoint, on purpose: once where the registration is
// recorded, so a malformed one installs no entry and reaches no thread, and
// once in the constructor, which is public and reachable without going through
// a manager at all. The second is the one that matters for the memory — it runs
// before the first write — and the first is what keeps a rejected worker from
// occupying a worker_id.
void validate_mailbox_region(const char *who, const void *mailbox, size_t mailbox_bytes, uint32_t granted_sets) {
    const std::string context(who);
    if (mailbox == nullptr) throw std::invalid_argument(context + ": null mailbox");
    if (granted_sets == 0) throw std::invalid_argument(context + ": invalid run-resource grant");
    const size_t required = mailbox_size_for_grant(granted_sets);
    if (required == 0) {
        throw std::invalid_argument(
            context + ": a run-resource grant of " + std::to_string(granted_sets) +
            " is outside the admitted range [1, " + std::to_string(MAILBOX_MAX_GRANTED_SETS) + "]"
        );
    }
    if (mailbox_bytes < required) {
        throw std::invalid_argument(
            context + ": a grant of " + std::to_string(granted_sets) + " run-resource sets needs " +
            std::to_string(required) + " mailbox bytes, and this registration mapped " + std::to_string(mailbox_bytes)
        );
    }
}

}  // namespace

LocalMailboxEndpoint::LocalMailboxEndpoint(
    int32_t worker_id, void *mailbox, size_t mailbox_bytes, int child_pid, uint32_t granted_sets
) :
    mailbox_(mailbox),
    granted_sets_(granted_sets),
    child_pid_(child_pid) {
    // Before the resize, and before the memset below reaches the last entry:
    // every one of those addresses comes from the grant, so the region has to be
    // known to contain them first.
    validate_mailbox_region("LocalMailboxEndpoint", mailbox, mailbox_bytes, granted_sets);
    // `M = granted + K`. The extra entries are what let a request be handed over
    // before a set is free: the entry exists from the moment the parent
    // publishes, and the run it names waits for a set without holding a frame.
    entries_.resize(mailbox_run_entry_count(granted_sets));
    request_frames_.resize(MAILBOX_REQUEST_FRAMES);
    // Every entry starts free and at an even generation, so a reference from a
    // previous mapping of this region is refused rather than matched.
    for (size_t index = 0; index < entries_.size(); ++index) {
        MailboxRunEntry *entry = run_entry(index);
        std::memset(entry, 0, sizeof(*entry));
    }
    caps_.worker_id = worker_id;
    // What the scheduler may have in flight is the entry count, and staging
    // needs somewhere for a successor to go while a predecessor runs.
    caps_.max_inflight_tasks = static_cast<uint32_t>(entries_.size());
    caps_.supports_frame_staging = granted_sets > 1;
    next_liveness_check_ = std::chrono::steady_clock::now() + kChildLivenessPollPeriod;
}

std::string LocalMailboxEndpoint::check_child_death() {
    std::lock_guard<std::mutex> child_lk(child_mu_);
    if (child_dead_) return child_death_reason_;
    if (child_pid_ <= 0) return {};

    int status = 0;
    pid_t r = 0;
    do {
        r = waitpid(static_cast<pid_t>(child_pid_), &status, WNOHANG);
    } while (r < 0 && errno == EINTR);

    if (r == 0) return {};

    if (r < 0 && errno != ECHILD) {
        // Any other waitpid() failure says nothing about the child, so the
        // caller keeps polling rather than tearing down a live worker.
        return {};
    }

    child_dead_ = true;
    if (r < 0) {
        child_death_reason_ = "child process pid=" + std::to_string(child_pid_) +
                              " is no longer waitable (reaped elsewhere) before mailbox completion";
    } else {
        child_death_reason_ = child_status_message(child_pid_, status);
    }
    return child_death_reason_;
}

MailboxState LocalMailboxEndpoint::read_mailbox_state(const char *frame) const {
    const char *base = frame == nullptr ? mbox() : frame;
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(const_cast<char *>(base) + MAILBOX_OFF_STATE);
    int32_t v;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(v) : "r"(ptr) : "memory");
#elif defined(__x86_64__)
    v = *ptr;
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
#endif
    return static_cast<MailboxState>(v);
}

void LocalMailboxEndpoint::write_mailbox_state(MailboxState s, char *frame) {
    char *base = frame == nullptr ? mbox() : frame;
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(base + MAILBOX_OFF_STATE);
    int32_t v = static_cast<int32_t>(s);
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" : : "r"(v), "r"(ptr) : "memory");
#elif defined(__x86_64__)
    __asm__ volatile("" ::: "memory");
    *ptr = v;
#else
    __atomic_store(ptr, &v, __ATOMIC_RELEASE);
#endif
}

bool mailbox_compare_exchange_state(char *frame, MailboxState expected, MailboxState desired) noexcept {
    auto *ptr = reinterpret_cast<int32_t *>(frame + MAILBOX_OFF_STATE);
    int32_t expected_value = static_cast<int32_t>(expected);
    return __atomic_compare_exchange_n(
        ptr, &expected_value, static_cast<int32_t>(desired), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
    );
}

namespace {

int32_t load_entry_i32(const int32_t *ptr) {
    int32_t v = 0;
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
    return v;
}

uint64_t load_entry_u64(const uint64_t *ptr) {
    uint64_t v = 0;
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
    return v;
}

void store_entry_i32(int32_t *ptr, int32_t v) { __atomic_store(ptr, &v, __ATOMIC_RELEASE); }

void store_entry_u64(uint64_t *ptr, uint64_t v) { __atomic_store(ptr, &v, __ATOMIC_RELEASE); }

}  // namespace

void LocalMailboxEndpoint::shutdown_child() {
    // Sticky word first: a child that samples it between the two stores leaves
    // by the shutdown path even though the state word still reads IDLE.
    int32_t *ptr = reinterpret_cast<int32_t *>(mbox() + MAILBOX_OFF_SHUTDOWN);
    int32_t requested = MAILBOX_SHUTDOWN_REQUESTED;
    __atomic_store(ptr, &requested, __ATOMIC_RELEASE);
    write_mailbox_state(MailboxState::SHUTDOWN);
}

char *LocalMailboxEndpoint::task_frame(size_t index) const {
    return mbox() +
           static_cast<ptrdiff_t>(MAILBOX_FIRST_TASK_FRAME + index) * static_cast<ptrdiff_t>(MAILBOX_FRAME_SIZE);
}

MailboxRunEntry *LocalMailboxEndpoint::run_entry(size_t index) const {
    return reinterpret_cast<MailboxRunEntry *>(
        mbox() + MAILBOX_RUN_ENTRY_TABLE_OFFSET + MAILBOX_RUN_ENTRY_STRIDE * index
    );
}

size_t LocalMailboxEndpoint::free_entry_locked() const {
    for (size_t index = 0; index < entries_.size(); ++index) {
        if (!entries_[index].occupied) return index;
    }
    return entries_.size();
}

size_t LocalMailboxEndpoint::free_request_frame_locked() const {
    for (size_t index = 0; index < request_frames_.size(); ++index) {
        if (!request_frames_[index].published) return index;
    }
    return request_frames_.size();
}

void LocalMailboxEndpoint::reclaim_consumed_request_frames_locked() {
    for (size_t index = 0; index < request_frames_.size(); ++index) {
        RequestFrameRecord &frame = request_frames_[index];
        if (!frame.published) continue;
        if (frame.entry >= entries_.size()) {
            // The entry this frame was published for is gone, so nothing will
            // ever echo its generation; the frame is this side's again.
            frame.published = false;
            continue;
        }
        const EntryRecord &record = entries_[frame.entry];
        if (!record.occupied || record.request_frame != index || record.request_frame_generation != frame.generation) {
            // Same conclusion from the record rather than the index: the
            // occupant this frame was published for has been released, so the
            // child refuses the frame on identity and acknowledges nothing. The
            // frame comes back on this side's own evidence instead of waiting
            // for a level that cannot arrive.
            frame.published = false;
            frame.entry = kNoRequestFrame;
            continue;
        }
        const MailboxRunEntry *entry = run_entry(frame.entry);
        // A level, not a pulse: the child stores the frame's own generation, so
        // an echo this poll did not see is seen by the next one.
        if (load_entry_u64(&entry->frame_consumed_generation) != frame.generation) continue;
        frame.published = false;
        frame.entry = kNoRequestFrame;
    }
}

// =============================================================================
// WorkerThread — lifecycle
// =============================================================================

void WorkerThread::start(
    Ring *ring, const std::function<void(WorkerCompletion)> &on_complete,
    const std::function<void(WorkerDispatch)> &on_accept, const std::function<void(WorkerDispatch)> &on_staged,
    std::unique_ptr<WorkerEndpoint> endpoint
) {
    if (!endpoint) throw std::invalid_argument("WorkerThread::start: null endpoint");
    ring_ = ring;
    on_complete_ = on_complete;
    on_accept_ = on_accept;
    on_staged_ = on_staged;
    endpoint_ = std::move(endpoint);
    shutdown_ = false;
    if (endpoint_->caps().max_inflight_tasks == 0) {
        throw std::invalid_argument("WorkerThread::start: endpoint capacity is zero");
    }
    inflight_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        // One active lane plus one staged lane per further run the endpoint may
        // hold, which is exactly the capacity it just published.
        lanes_.assign(endpoint_->caps().max_inflight_tasks, LaneState{});
        activation_scratch_.assign(endpoint_->caps().max_inflight_tasks, INVALID_RUN_ID);
    }
}

void WorkerThread::dispatch(WorkerDispatch d) {
    d.prepare_only = false;
    bool lane_occupied = false;
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        LaneState &active = lane(LaneKind::ACTIVE);
        if (active.occupied) {
            lane_occupied = true;
        } else {
            active.occupied = true;
        }
    }
    if (lane_occupied) {
        complete_unpublished(d, "WorkerThread::dispatch: active lane is occupied");
        return;
    }
    SubmitDispatchResult result;
    try {
        result = submit_dispatch(d, kActiveLane);
    } catch (const std::exception &e) {
        release_lane_unconditional(kActiveLane);
        complete_unpublished(d, std::string("WorkerThread::dispatch: submit failed: ") + e.what());
        return;
    } catch (...) {
        release_lane_unconditional(kActiveLane);
        complete_unpublished(d, "WorkerThread::dispatch: submit failed");
        return;
    }
    // PUBLICATION_REPORTED included: this dispatch's terminal has already been
    // delivered, by the endpoint or by fail_submission, so reporting again here
    // would be a second completion for one task.
    if (result == SubmitDispatchResult::SUBMITTED || result == SubmitDispatchResult::PUBLICATION_REPORTED) return;
    release_lane_unconditional(kActiveLane);
    if (result == SubmitDispatchResult::STOPPING) {
        complete_unpublished(d, "WorkerThread::dispatch: worker is stopping");
    } else {
        complete_unpublished(d, "WorkerThread::dispatch: endpoint capacity exceeded");
    }
}

void WorkerThread::dispatch_prepared(WorkerDispatch d) {
    d.prepare_only = true;
    if (!caps().supports_frame_staging) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: endpoint does not support frame staging");
        return;
    }
    if (ring_ == nullptr) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: null ring");
        return;
    }
    TaskSlotState *slot = ring_->slot_state(d.task_slot);
    if (slot == nullptr || slot->run_id == INVALID_RUN_ID) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: dispatch has no run identity");
        return;
    }
    size_t staged_index = 0;
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        staged_index = free_staged_lane_locked();
        if (staged_index < lanes_.size()) {
            LaneState &staged = lane_at(staged_index);
            staged.occupied = true;
            staged.run_id = slot->run_id;
        }
    }
    if (staged_index >= lanes_.size()) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: worker has no free staged lane");
        return;
    }
    SubmitDispatchResult result;
    try {
        result = submit_dispatch(d, staged_index, slot->run_id);
    } catch (const std::exception &e) {
        release_lane_unconditional(staged_index);
        complete_unpublished(d, std::string("WorkerThread::dispatch_prepared: submit failed: ") + e.what());
        return;
    } catch (...) {
        release_lane_unconditional(staged_index);
        complete_unpublished(d, "WorkerThread::dispatch_prepared: submit failed");
        return;
    }
    // See the ordinary lane on PUBLICATION_REPORTED.
    if (result == SubmitDispatchResult::SUBMITTED || result == SubmitDispatchResult::PUBLICATION_REPORTED) return;
    release_lane_unconditional(staged_index);
    if (result == SubmitDispatchResult::STOPPING) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: worker is stopping");
    } else if (result == SubmitDispatchResult::CAPACITY_EXCEEDED) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: endpoint capacity exceeded");
    } else {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: staged lane identity changed before submission");
    }
}

WorkerThread::SubmitDispatchResult
WorkerThread::submit_dispatch(WorkerDispatch d, size_t lane_index, RunId expected_run_id) {
#if SIMPLER_HOST_STRACE
    const bool trace_enabled = simpler::host_trace::enabled();
    const int64_t trace_start_ns = trace_enabled ? simpler::host_trace::now_ns() : 0;
#endif
    std::unique_lock<std::mutex> admission_lk(admission_mu_);
    if (shutdown_.load(std::memory_order_acquire)) {
        return SubmitDispatchResult::STOPPING;
    }
    if (inflight_.load(std::memory_order_acquire) >= endpoint_->caps().max_inflight_tasks) {
        return SubmitDispatchResult::CAPACITY_EXCEEDED;
    }
    d.dispatch_id = next_dispatch_id_;
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        LaneState &dispatch_lane = lane_at(lane_index);
        if (!dispatch_lane.occupied || dispatch_lane.dispatch_id != 0 ||
            (expected_run_id != INVALID_RUN_ID && dispatch_lane.run_id != expected_run_id)) {
            return SubmitDispatchResult::STAGED_IDENTITY_CHANGED;
        }
        dispatch_lane.dispatch_id = d.dispatch_id;
    }
    ++next_dispatch_id_;
    inflight_.fetch_add(1, std::memory_order_release);
#if SIMPLER_HOST_STRACE
    RunId trace_run = INVALID_RUN_ID;
    uint64_t trace_hash = 0;
    std::string trace_lease;
    if (trace_enabled) {
        trace_run = trace_run_id(ring_, d.task_slot);
        trace_hash = trace_callable_hash(ring_, d.task_slot);
        trace_lease = trace_lease_attrs(ring_, d.task_slot);
    }
#endif
    bool published = true;
    try {
        endpoint_->submit_progress(ring_, d);
    } catch (const std::exception &e) {
        published = false;
        fail_submission(d, std::string("submit_progress failed: ") + e.what());
    } catch (...) {
        published = false;
        fail_submission(d, "submit_progress failed with unknown exception");
    }
    admission_lk.unlock();
#if SIMPLER_HOST_STRACE
    if (trace_enabled) {
        const int64_t trace_end_ns = simpler::host_trace::now_ns();
        const std::string trace_attrs =
            trace_dispatch_attrs(trace_run, d, endpoint_->caps(), "scheduler") + trace_lease;
        simpler::host_trace::emit(
            simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Dispatch), trace_run, trace_hash, 0,
            trace_start_ns, trace_end_ns - trace_start_ns, trace_attrs.c_str()
        );
    }
#endif
    // fail_submission has already delivered this dispatch's terminal and stopped
    // the worker, so the caller must not report it again — but it is not a
    // submission either, and saying so is what lets the caller distinguish a run
    // that is live from one that is already failed.
    return published ? SubmitDispatchResult::SUBMITTED : SubmitDispatchResult::PUBLICATION_REPORTED;
}

bool WorkerThread::activate_prepared(RunId run_id) {
    if (run_id == INVALID_RUN_ID) return false;
    std::lock_guard<std::mutex> admission_lk(admission_mu_);
    if (shutdown_.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    LaneState &active = lane(LaneKind::ACTIVE);
    const size_t staged_index = staged_lane_of_locked(run_id);
    if (staged_index >= lanes_.size() || active.occupied) return false;
    LaneState &staged = lane_at(staged_index);
    if (staged.dispatch_id == 0) return false;
    active = staged;
    active.activation_requested = true;
    staged = {};
    return true;
}

bool WorkerThread::authorize_staged_launch(RunId run_id) {
    if (run_id == INVALID_RUN_ID) return false;
    std::lock_guard<std::mutex> admission_lk(admission_mu_);
    if (shutdown_.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    const size_t staged_index = staged_lane_of_locked(run_id);
    if (staged_index >= lanes_.size()) return false;
    LaneState &staged = lane_at(staged_index);
    if (staged.dispatch_id == 0) return false;
    // Only a natively prepared frame holds work that can be ordered behind
    // another run. A validated-only one fell back to depth one at the child, and
    // a frame whose staging has not been observed yet has said nothing at all;
    // both take the ordinary path when they reach the front.
    if (staged.preparation_disposition != MailboxPreparationDisposition::NATIVE_PREPARED) return false;
    if (staged.activation_requested) return false;
    staged.activation_requested = true;
    return true;
}

bool WorkerThread::has_staged_run(RunId run_id) const {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    return staged_lane_of_locked(run_id) < lanes_.size();
}

bool WorkerThread::can_stage() const {
    const WorkerEndpointCaps &endpoint_caps = caps();
    if (!endpoint_caps.supports_frame_staging) return false;
    if (inflight_.load(std::memory_order_acquire) >= endpoint_caps.max_inflight_tasks) return false;
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        if (free_staged_lane_locked() >= lanes_.size()) return false;
    }
    // Transport is bounded independently of the grant, so a staging that found
    // no frame is ordinary saturation rather than a fault, and this is where it
    // is answered: before a task is claimed, by the admitter, which then simply
    // asks again next round.
    //
    // What makes this answer still true at the publication it gates — see
    // `can_dispatch` for the same argument — is that `Scheduler::dispatch_claimed`
    // is the only caller of `dispatch` and `dispatch_prepared`, so every
    // publication into every endpoint happens on the scheduler's own thread. The
    // only other party that touches frame ownership is the child, whose
    // acknowledgement can only free a frame. Availability therefore cannot fall
    // between this gate and its own publication, and a second publication in the
    // same round re-reads it after the first has marked its frame published.
    //
    // Queried outside lane_mu_: progress() takes the endpoint's locks and then
    // the lane's, so holding the lane lock across this call would invert that.
    return endpoint_->available_request_transport() > 0;
}

bool WorkerThread::can_dispatch() const {
    // The ordinary lane's own gate, and the reason it exists: `idle()` answers
    // only for the lane, so the publication caller that does not consult
    // `can_stage` had no transport term at all and was the one that could reach
    // `submit_progress` with nowhere to write. Both gates now ask the same
    // question before anything is claimed; see `can_stage` for why the answer
    // survives to the publication.
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        if (lane(LaneKind::ACTIVE).occupied) return false;
    }
    return endpoint_->available_request_transport() > 0;
}

bool WorkerThread::idle() const {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    return !lane(LaneKind::ACTIVE).occupied;
}

void WorkerThread::release_lane(size_t lane_index, uint64_t dispatch_id) {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    LaneState &dispatch_lane = lane_at(lane_index);
    if (dispatch_lane.dispatch_id == dispatch_id) dispatch_lane = {};
}

void WorkerThread::release_lane_unconditional(size_t lane_index) {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    lane_at(lane_index) = {};
}

void WorkerThread::complete_unpublished(WorkerDispatch dispatch, const std::string &error_message) {
    try {
        if (on_accept_) on_accept_(dispatch);
    } catch (...) {
        // The endpoint-publication error remains the primary task failure.
    }
    on_complete_(
        WorkerCompletion{dispatch.task_slot, dispatch.group_index, EndpointOutcome::ENDPOINT_FAILURE, error_message}
    );
}

void WorkerThread::stop() {
    std::lock_guard<std::mutex> admission_lk(admission_mu_);
    shutdown_.store(true, std::memory_order_release);
}

void WorkerThread::shutdown_child() {
    if (endpoint_) endpoint_->shutdown_child();
}

const WorkerEndpointCaps &WorkerThread::caps() const {
    if (!endpoint_) throw std::runtime_error("WorkerThread::caps: null endpoint");
    return endpoint_->caps();
}

int32_t WorkerThread::worker_id() const { return caps().worker_id; }

// =============================================================================
// WorkerThread — Scheduler-owned endpoint progress
// =============================================================================

void WorkerThread::progress() {
    if (shutdown_.load(std::memory_order_acquire)) {
        // A child finishing a control handler may overwrite an earlier stop
        // state, so the request remains level-triggered until terminalization.
        endpoint_->request_progress_stop();
    }

    // Every lane, oldest first. Only the active lane can hold an activation
    // while the launch depth is one; above it, a staged run authorized to
    // launch early keeps its own lane — the predecessor still owns the active
    // one — so its activation has to be published from there, and so does the
    // one behind it when the depth allows a third.
    // Reused rather than allocated: this runs on every scheduler progress tick,
    // once per worker, and the fixed array it replaced allocated nothing. Owned
    // by the progress thread alone — `WorkerManager::progress` is reached only
    // from the scheduler loop, under its loop mutex — and its last use here
    // precedes every callout that could re-enter, so nothing observes it
    // half-written. Sized in `start` alongside `lanes_`, whose count is fixed
    // from then on.
    std::vector<RunId> &activate = activation_scratch_;
    std::fill(activate.begin(), activate.end(), INVALID_RUN_ID);
    {
        std::lock_guard<std::mutex> admission_lk(admission_mu_);
        if (!shutdown_.load(std::memory_order_acquire)) {
            {
                std::lock_guard<std::mutex> lane_lk(lane_mu_);
                for (size_t index = 0; index < lanes_.size(); ++index) {
                    const LaneState &dispatch_lane = lanes_[index];
                    if (dispatch_lane.occupied && dispatch_lane.activation_requested) {
                        activate[index] = dispatch_lane.run_id;
                    }
                }
            }
            for (size_t index = 0; index < activate.size(); ++index) {
                if (activate[index] == INVALID_RUN_ID) continue;
                try {
                    if (endpoint_->activate_progress(activate[index])) {
                        std::lock_guard<std::mutex> lane_lk(lane_mu_);
                        LaneState &dispatch_lane = lanes_[index];
                        if (dispatch_lane.occupied && dispatch_lane.run_id == activate[index]) {
                            dispatch_lane.activation_requested = false;
                        }
                    }
                } catch (const std::exception &e) {
                    fail_progress_driver(std::string("activate_progress failed: ") + e.what());
                } catch (...) {
                    fail_progress_driver("activate_progress failed with unknown exception");
                }
            }
        }
    }

    WorkerEndpointProgress progress;
    try {
        if (endpoint_->poll_progress(progress)) finish_progress_dispatch(progress);
    } catch (const std::exception &e) {
        fail_progress_driver(std::string("poll_progress failed: ") + e.what());
    } catch (...) {
        fail_progress_driver("poll_progress failed with unknown exception");
    }
}

void WorkerThread::fail_submission(const WorkerDispatch &dispatch, const std::string &reason) {
    shutdown_.store(true, std::memory_order_release);
    bool endpoint_owned = false;
    try {
        endpoint_owned = endpoint_->report_submission_error(dispatch, reason);
    } catch (...) {
        endpoint_->request_progress_stop();
    }
    if (endpoint_owned) return;

    WorkerEndpointProgress progress;
    progress.kind = WorkerProgressKind::COMPLETED;
    progress.dispatch = dispatch;
    progress.completion = WorkerCompletion{
        dispatch.task_slot, dispatch.group_index, EndpointOutcome::ENDPOINT_FAILURE,
        "WorkerThread endpoint rejected publication: " + reason
    };
    finish_progress_dispatch(progress);
}

void WorkerThread::finish_progress_dispatch(const WorkerEndpointProgress &progress) {
    const WorkerDispatch &dispatch = progress.dispatch;
    if (progress.kind == WorkerProgressKind::FRAME_STAGED) {
        // The endpoint already owns the prepared frame. This cursor-only event
        // keeps the progress poll moving; acceptance, completion, and inflight
        // ownership intentionally remain unchanged until activation/terminal.
        //
        // What it does settle is whether that frame can be launched early at
        // all. Staging admits a validated-only frame too, and such a frame fell
        // back to depth one at the child: it holds no native preparation, so
        // authorizing it would spend an activation the child then declines.
        // Recording the disposition here is what lets the authorization refuse.
        {
            std::lock_guard<std::mutex> lane_lk(lane_mu_);
            const size_t staged_index = staged_lane_for_dispatch_locked(dispatch.dispatch_id);
            if (staged_index < lanes_.size()) {
                lane_at(staged_index).preparation_disposition = progress.preparation_disposition;
            }
        }
        // Announced because this is one of the two orders in which a staged run
        // can become launchable early: the run ahead of it may already have had
        // every dispatch accepted, and then nothing later says so. The callback
        // is non-throwing by contract, and a throw here would take the progress
        // driver down, so it is contained.
        if (on_staged_) {
            try {
                on_staged_(dispatch);
            } catch (...) {}
        }
        return;
    }

    if (progress.kind == WorkerProgressKind::ACCEPTED) {
        if (!accepted_dispatch_ids_.insert(dispatch.dispatch_id).second) return;
        try {
            if (on_accept_) on_accept_(dispatch);
        } catch (const std::exception &e) {
            accept_errors_[dispatch.dispatch_id] = std::string("WorkerThread accept fence failed: ") + e.what();
        } catch (...) {
            accept_errors_[dispatch.dispatch_id] = "WorkerThread accept fence failed with unknown exception";
        }
        return;
    }

#if SIMPLER_HOST_STRACE
    const bool trace_enabled = simpler::host_trace::enabled();
    RunId trace_run = INVALID_RUN_ID;
    uint64_t trace_hash = 0;
    std::string complete_attrs;
    if (trace_enabled) {
        trace_run = trace_run_id(ring_, dispatch.task_slot);
        trace_hash = trace_callable_hash(ring_, dispatch.task_slot);
        complete_attrs = trace_dispatch_attrs(trace_run, dispatch, endpoint_->caps(), "worker");
    }
#endif
    WorkerCompletion completion = progress.completion;
    if (accepted_dispatch_ids_.erase(dispatch.dispatch_id) == 0) {
        // This advances only the run-level waiter after a terminal endpoint
        // result. It does not manufacture the per-frame launch marker: that
        // sticky word is published exclusively by the native launch path.
        try {
            if (on_accept_) on_accept_(dispatch);
        } catch (const std::exception &e) {
            accept_errors_[dispatch.dispatch_id] = std::string("WorkerThread accept fence failed: ") + e.what();
        } catch (...) {
            accept_errors_[dispatch.dispatch_id] = "WorkerThread accept fence failed with unknown exception";
        }
    }
    auto accept_error = accept_errors_.find(dispatch.dispatch_id);
    if (accept_error != accept_errors_.end()) {
        if (completion.outcome == EndpointOutcome::SUCCESS) {
            completion.outcome = EndpointOutcome::ENDPOINT_FAILURE;
            completion.error_message = accept_error->second;
        }
        accept_errors_.erase(accept_error);
    }

#if SIMPLER_HOST_STRACE
    std::optional<simpler::host_trace::SpanScope> complete_trace;
    if (trace_enabled) {
        complete_attrs += " outcome=" + std::to_string(static_cast<int32_t>(completion.outcome));
        complete_trace.emplace(
            simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Complete), trace_run, trace_hash, 0,
            std::move(complete_attrs)
        );
    }
#endif
    on_complete_(std::move(completion));
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        for (LaneState &dispatch_lane : lanes_) {
            if (dispatch_lane.occupied && dispatch_lane.dispatch_id == dispatch.dispatch_id) {
                dispatch_lane = {};
                break;
            }
        }
    }
    inflight_.fetch_sub(1, std::memory_order_acq_rel);
}

void WorkerThread::fail_progress_driver(const std::string &reason) noexcept {
    shutdown_.store(true, std::memory_order_release);
    try {
        endpoint_->report_progress_error(reason);
    } catch (...) {
        endpoint_->request_progress_stop();
    }
}

size_t LocalMailboxEndpoint::available_request_transport() {
    std::lock_guard<std::mutex> mailbox_lk(mailbox_mu_);
    std::lock_guard<std::mutex> lk(progress_mu_);
    // A poisoned endpoint publishes nothing more, and reporting transport it
    // will refuse would make the caller claim a task to fail it.
    if (endpoint_poisoned_) return 0;
    reclaim_consumed_request_frames_locked();
    size_t free_frames = 0;
    for (const RequestFrameRecord &frame : request_frames_) {
        if (!frame.published) ++free_frames;
    }
    // An entry is as necessary as a frame: a hand-off needs both, and the entry
    // outlives the frame, so the narrower of the two is what can be admitted.
    size_t free_entries = 0;
    for (const EntryRecord &record : entries_) {
        if (!record.occupied) ++free_entries;
    }
    return std::min(free_frames, free_entries);
}

void LocalMailboxEndpoint::submit_progress(Ring *ring, const WorkerDispatch &dispatch) {
    if (ring == nullptr) throw std::invalid_argument("LocalMailboxEndpoint::submit_progress: null ring");
    TaskSlotState *slot_state = ring->slot_state(dispatch.task_slot);
    if (slot_state == nullptr) throw std::out_of_range("LocalMailboxEndpoint::submit_progress: invalid task slot");
    TaskSlotState &state = *slot_state;
    const TaskArgs &a = state.args(dispatch.group_index);
    if (!state.remote_sidecar_for(dispatch.group_index).empty()) {
        throw std::runtime_error("remote task sidecar is not supported by local mailbox");
    }
    if (state.run_id == INVALID_RUN_ID || state.pipeline_lease.reserved != 0 || state.pipeline_lease.generation == 0 ||
        (granted_sets_ > 1 && state.pipeline_lease.slot_id >= granted_sets_)) {
        throw std::runtime_error("task frame has an invalid pipeline lease identity");
    }
    const size_t blob_bytes = task_args_blob_size(a);
    if (blob_bytes > MAILBOX_ARGS_CAPACITY) {
        throw std::runtime_error(
            "args blob exceeds mailbox capacity: need " + std::to_string(blob_bytes) + ", capacity " +
            std::to_string(MAILBOX_ARGS_CAPACITY)
        );
    }

#if SIMPLER_HOST_STRACE
    std::optional<simpler::host_trace::SpanScope> frame_submit_trace;
    if (simpler::host_trace::enabled()) {
        frame_submit_trace.emplace(
            simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::FrameSubmit), state.run_id,
            trace_callable_hash(ring, dispatch.task_slot), 0,
            trace_dispatch_attrs(state.run_id, dispatch, caps_, "worker")
        );
    }
#endif

    // Linearize task-frame publication against base-frame controls. The
    // control mutex is released immediately after the task state release-store;
    // controls may still execute while the native run is active, subject to
    // the child-side registry lifetime gate.
    //
    // Acquiring it is not always cheap, and the cost is shared. This runs on
    // the Scheduler thread, which holds loop_mu_ and is the sole progress owner
    // for every endpoint; run_control_command holds this same mutex while it
    // blocks on the child, with no deadline by default. So while a control
    // command is outstanding on any one endpoint, no endpoint makes progress
    // and loop_mu_ stays held — and Orchestrator shares loop_mu_ with allocator
    // compaction. Blocking here is bounded only by the child's control latency,
    // not by anything this function does.
    std::lock_guard<std::mutex> mailbox_lk(mailbox_mu_);
    std::lock_guard<std::mutex> lk(progress_mu_);
    if (endpoint_poisoned_) {
        throw std::runtime_error("endpoint is poisoned: " + endpoint_poison_reason_);
    }
    // Anything the child has finished copying out comes back first, so a
    // submission that arrives right behind a hand-off finds a frame rather than
    // waiting for the next poll to notice the echo.
    reclaim_consumed_request_frames_locked();

    const size_t entry_index = free_entry_locked();
    if (entry_index >= entries_.size()) {
        // Admission is the caller's gate and it holds no run beyond the entry
        // count, so reaching here means the two disagree.
        poison_progress("no run-state entry is free for an admitted run");
        throw std::runtime_error("no run-state entry is free for an admitted run");
    }
    const size_t frame_index = free_request_frame_locked();
    if (frame_index >= request_frames_.size()) {
        // Unreachable while admission and this side agree, and that is the whole
        // reason it is left as a throw: reaching it means a publication caller
        // claimed work without consulting available_request_transport(), which
        // is a defect in the admitter and not backpressure. Saturation itself —
        // ordinary once the grant exceeds the frame count — is answered at the
        // gate, before anything is claimed, and never arrives here.
        //
        // The throw is handled by fail_submission, which does poison this
        // endpoint. That is correct for an admitter/endpoint disagreement and is
        // exactly why ordinary saturation must not take this path: a busy poll
        // would otherwise retire a healthy endpoint.
        throw std::runtime_error("no request frame is free for an admitted run");
    }

    EntryRecord &record = entries_[entry_index];
    MailboxRunEntry *entry = run_entry(entry_index);
    char *frame = task_frame(frame_index);

    // The entry, before the frame that names it. Its generation goes odd here
    // and even at release, so every cross-side reference to this run carries a
    // value no reuse of the entry repeats — which is what makes a stale
    // reference refusable instead of indistinguishable from a fresh one.
    const uint64_t entry_generation = next_entry_generation_++ * 2 + 1;
    const uint64_t slot_id = state.pipeline_lease.slot_id;
    const uint64_t task_slot = static_cast<uint64_t>(dispatch.task_slot);
    const uint64_t group_index = static_cast<uint64_t>(dispatch.group_index);
    const uint64_t group_size = static_cast<uint64_t>(state.group_size());
    entry->error = 0;
    entry->preparation_disposition = static_cast<int32_t>(MailboxPreparationDisposition::NONE);
    std::memset(entry->error_text, 0, MAILBOX_ERROR_MSG_SIZE);
    entry->run_epoch = 0;
    entry->protocol = MAILBOX_TASK_PROTOCOL_VERSION;
    entry->run_id = state.run_id;
    entry->slot_id = slot_id;
    entry->generation = state.pipeline_lease.generation;
    entry->dispatch_id = dispatch.dispatch_id;
    entry->task_slot = task_slot;
    entry->group_index = group_index;
    entry->group_size = group_size;
    // Both of these are read by the child to decide whether a frame is for this
    // entry at all, so they are released before the frame's own state word.
    store_entry_i32(&entry->accepted, 0);
    store_entry_u64(&entry->frame_consumed_generation, 0);
    store_entry_u64(&entry->entry_generation, entry_generation);
    store_entry_i32(&entry->state, static_cast<int32_t>(MailboxState::IDLE));

    // The frame carries the payload and the header that binds it to the entry.
    uint64_t reserved_callable = 0;
    std::memcpy(frame + MAILBOX_OFF_CALLABLE, &reserved_callable, sizeof(reserved_callable));
    std::memcpy(frame + MAILBOX_OFF_CONFIG, &state.config, sizeof(CallConfig));
    std::memcpy(frame + MAILBOX_OFF_PIPELINE_LEASE, &state.pipeline_lease, sizeof(PipelineSlotLease));
    std::memcpy(
        frame + MAILBOX_OFF_TASK_CALLABLE_HASH, state.callable.digest.data(),
        static_cast<size_t>(CALLABLE_HASH_DIGEST_SIZE)
    );

    // The versioned, length-prefixed Tensor blob (the L3->L2 wire; the child
    // materializes it to a ChipTensor blob before run).
    uint8_t *blob = reinterpret_cast<uint8_t *>(frame + MAILBOX_OFF_TASK_ARGS_BLOB);
    write_blob(blob, a);

    const uint64_t frame_generation = next_frame_generation_++;
    const uint64_t entry_index_wire = static_cast<uint64_t>(entry_index);
    std::memcpy(frame + MAILBOX_OFF_FRAME_ENTRY_INDEX, &entry_index_wire, sizeof(entry_index_wire));
    std::memcpy(frame + MAILBOX_OFF_FRAME_ENTRY_GENERATION, &entry_generation, sizeof(entry_generation));
    std::memcpy(frame + MAILBOX_OFF_FRAME_GENERATION, &frame_generation, sizeof(frame_generation));

    record = EntryRecord{};
    record.occupied = true;
    record.dispatch = dispatch;
    record.run_id = state.run_id;
    record.slot_id = slot_id;
    record.generation = state.pipeline_lease.generation;
    record.group_size = group_size;
    record.request_frame = frame_index;
    record.request_frame_generation = frame_generation;
    record.entry_generation = entry_generation;
    request_frames_[frame_index].published = true;
    request_frames_[frame_index].generation = frame_generation;
    request_frames_[frame_index].entry = entry_index;

    // Last: the frame's state word releases every byte written above it, on both
    // the frame and the entry it names.
    write_mailbox_state(dispatch.prepare_only ? MailboxState::PREPARE_READY : MailboxState::TASK_READY, frame);
}

bool LocalMailboxEndpoint::entry_identity_matches(const EntryRecord &record, const MailboxRunEntry *entry) const {
    // The generation first: it is what distinguishes this run from any later
    // occupant of the same entry, so a mismatch there makes every other field
    // irrelevant rather than merely inconsistent.
    if (load_entry_u64(&entry->entry_generation) != record.entry_generation) return false;
    return entry->protocol == MAILBOX_TASK_PROTOCOL_VERSION && entry->run_id == record.run_id &&
           entry->slot_id == record.slot_id && entry->generation == record.generation &&
           entry->dispatch_id == record.dispatch.dispatch_id &&
           entry->task_slot == static_cast<uint64_t>(record.dispatch.task_slot) &&
           entry->group_index == static_cast<uint64_t>(record.dispatch.group_index) &&
           entry->group_size == record.group_size;
}

bool LocalMailboxEndpoint::try_publish_activation(EntryRecord &record, MailboxRunEntry *entry) {
    if (!record.activation_requested || record.activation_published) return record.activation_published;
    if (!entry_identity_matches(record, entry)) {
        poison_progress("stale staged run-state entry identity before activation");
        return false;
    }
    // Parent-owned, identity-checked, and the only transition this side writes:
    // the child owns every other one, so an exchange that fails means the child
    // has already moved past FRAME_STAGED rather than that two writers raced.
    int32_t expected = static_cast<int32_t>(MailboxState::FRAME_STAGED);
    if (__atomic_compare_exchange_n(
            &entry->state, &expected, static_cast<int32_t>(MailboxState::ACTIVATE), false, __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE
        )) {
        record.activation_published = true;
    }
    return record.activation_published;
}

void LocalMailboxEndpoint::poison_progress(const std::string &reason) {
    if (endpoint_poisoned_) return;
    endpoint_poisoned_ = true;
    endpoint_poison_reason_ = reason;
}

bool LocalMailboxEndpoint::poisoned_progress_quiesced() {
    // SHUTDOWN is deliberately repeated. A child already inside a control
    // handler may publish CONTROL_DONE after the first store; the next store
    // restores the stop request and prevents a live native run from being
    // mistaken for a terminal parent completion.
    shutdown_child();
    if (child_pid_ > 0) return !check_child_death().empty();
    for (size_t index = 0; index < entries_.size(); ++index) {
        if (!entries_[index].occupied) continue;
        const auto state = static_cast<MailboxState>(load_entry_i32(&run_entry(index)->state));
        if (state != MailboxState::TASK_DONE && state != MailboxState::TASK_FAILED) return false;
    }
    return true;
}

WorkerCompletion LocalMailboxEndpoint::poisoned_completion(const EntryRecord &record) const {
    return WorkerCompletion{
        record.dispatch.task_slot, record.dispatch.group_index, EndpointOutcome::ENDPOINT_FAILURE,
        "LocalMailboxEndpoint progress failed: " + endpoint_poison_reason_
    };
}

WorkerCompletion LocalMailboxEndpoint::terminal_completion(
    const EntryRecord &record, const MailboxRunEntry *entry, MailboxState state
) const {
    // Both read after the acquire load of the terminal state that brought us
    // here, which is the child's final access to this entry — so from that
    // instant the rc and the text are this side's alone.
    const int32_t error_code = entry->error;
    WorkerCompletion completion{record.dispatch.task_slot, record.dispatch.group_index, EndpointOutcome::SUCCESS, {}};
    if (state == MailboxState::TASK_FAILED || error_code != 0) {
        const size_t text_len = ::strnlen(entry->error_text, MAILBOX_ERROR_MSG_SIZE);
        completion.outcome = EndpointOutcome::TASK_FAILURE;
        completion.error_message = "LocalMailboxEndpoint child failed (worker_id=" + std::to_string(caps_.worker_id) +
                                   ", code=" + std::to_string(error_code) +
                                   "): " + std::string(entry->error_text, text_len);
    }
    return completion;
}

bool LocalMailboxEndpoint::poll_progress(WorkerEndpointProgress &progress) {
    std::lock_guard<std::mutex> lk(progress_mu_);
    progress.preparation_disposition = MailboxPreparationDisposition::NONE;
    bool any_occupied = false;
    for (const EntryRecord &record : entries_)
        any_occupied = any_occupied || record.occupied;
    // Frames come back whether or not anything else is reportable: a hand-off
    // whose payload the child has copied out is finished with its frame even
    // while the run it carried is still executing, and that is the whole point
    // of the split.
    reclaim_consumed_request_frames_locked();
    if (!any_occupied) return false;

    if (!endpoint_poisoned_ && mailbox_control_timed_out_.load(std::memory_order_acquire)) {
        poison_progress("mailbox has an unresolved timed-out control command");
    }
    const auto now = std::chrono::steady_clock::now();
    if (!endpoint_poisoned_ && now >= next_liveness_check_) {
        next_liveness_check_ = now + kChildLivenessPollPeriod;
        std::string death = check_child_death();
        if (!death.empty()) {
            poison_progress(death);
            preserve_terminal_results_on_exit_ = true;
        }
    }

    const size_t entry_count = entries_.size();
    if (endpoint_poisoned_) {
        if (!poisoned_progress_quiesced()) return false;
        for (size_t offset = 0; offset < entry_count; ++offset) {
            size_t index = (poll_cursor_ + offset) % entry_count;
            EntryRecord &record = entries_[index];
            if (!record.occupied) continue;
            progress.kind = WorkerProgressKind::COMPLETED;
            progress.dispatch = record.dispatch;
            progress.completion = poisoned_completion(record);
            if (preserve_terminal_results_on_exit_) {
                // The child can publish between an earlier entry read and waitpid observing exit.
                // This acquire read follows confirmed exit; identity still belongs to this dispatch.
                MailboxRunEntry *entry = run_entry(index);
                const auto state = static_cast<MailboxState>(load_entry_i32(&entry->state));
                if ((state == MailboxState::TASK_DONE || state == MailboxState::TASK_FAILED) &&
                    entry_identity_matches(record, entry)) {
                    progress.completion = terminal_completion(record, entry, state);
                }
            }
            release_entry_locked(index);
            poll_cursor_ = (index + 1) % entry_count;
            return true;
        }
        return false;
    }

    for (size_t offset = 0; offset < entry_count; ++offset) {
        size_t index = (poll_cursor_ + offset) % entry_count;
        EntryRecord &record = entries_[index];
        if (!record.occupied) continue;
        MailboxRunEntry *entry = run_entry(index);
        const auto state = static_cast<MailboxState>(load_entry_i32(&entry->state));

        if (!record.accepted_reported && load_entry_i32(&entry->accepted) == MAILBOX_TASK_ACCEPTED) {
            if (!entry_identity_matches(record, entry)) {
                poison_progress("stale run-state entry identity at launch acceptance");
                break;
            }
            record.accepted_reported = true;
            progress.kind = WorkerProgressKind::ACCEPTED;
            progress.dispatch = record.dispatch;
            poll_cursor_ = (index + 1) % entry_count;
            return true;
        }

        if (state == MailboxState::FRAME_STAGED) {
            if (!entry_identity_matches(record, entry)) {
                poison_progress("stale run-state entry identity at endpoint staging");
                break;
            }
            if (record.activation_requested) {
                (void)try_publish_activation(record, entry);
                if (endpoint_poisoned_) break;
            }
            const auto disposition =
                static_cast<MailboxPreparationDisposition>(load_entry_i32(&entry->preparation_disposition));
            if (disposition != MailboxPreparationDisposition::VALIDATED_ONLY &&
                disposition != MailboxPreparationDisposition::NATIVE_PREPARED) {
                poison_progress("invalid preparation disposition at endpoint staging");
                break;
            }
            // Reported once when the run stages, and once more if that staged run is *later*
            // prepared natively: a run staged behind a predecessor that had not launched yet is
            // validated-only at that instant and holds native preparation only after the
            // predecessor reaches the device. The child publishes that promotion by writing this
            // word alone — never the state word, which only this endpoint's exchange above moves
            // out of FRAME_STAGED — so reporting the change is what keeps the run from waiting for
            // ordinary promotion instead of the authorization it has earned. Only ever a
            // promotion: the reverse direction would be a stale read, and is refused.
            const bool promoted = record.staged_reported &&
                                  record.reported_disposition == MailboxPreparationDisposition::VALIDATED_ONLY &&
                                  disposition == MailboxPreparationDisposition::NATIVE_PREPARED;
            if (!record.staged_reported || promoted) {
                record.staged_reported = true;
                record.reported_disposition = disposition;
                progress.kind = WorkerProgressKind::FRAME_STAGED;
                progress.dispatch = record.dispatch;
                progress.preparation_disposition = disposition;
                poll_cursor_ = (index + 1) % entry_count;
                return true;
            }
            continue;
        }

        if (state == MailboxState::TASK_DONE || state == MailboxState::TASK_FAILED) {
            if (!entry_identity_matches(record, entry)) {
                poison_progress("stale run-state entry identity at terminal completion");
                break;
            }
            progress.kind = WorkerProgressKind::COMPLETED;
            progress.dispatch = record.dispatch;
            progress.completion = terminal_completion(record, entry, state);
            release_entry_locked(index);
            poll_cursor_ = (index + 1) % entry_count;
            return true;
        }

        if (state != MailboxState::IDLE && state != MailboxState::TASK_READY && state != MailboxState::PREPARE_READY &&
            state != MailboxState::ACTIVATE && state != MailboxState::TASK_LAUNCHED) {
            poison_progress("run-state entry entered an invalid state " + std::to_string(static_cast<int32_t>(state)));
            break;
        }
    }

    if (!endpoint_poisoned_) return false;
    if (!poisoned_progress_quiesced()) return false;
    for (size_t index = 0; index < entry_count; ++index) {
        EntryRecord &record = entries_[index];
        if (!record.occupied) continue;
        progress.kind = WorkerProgressKind::COMPLETED;
        progress.dispatch = record.dispatch;
        progress.completion = poisoned_completion(record);
        release_entry_locked(index);
        poll_cursor_ = (index + 1) % entry_count;
        return true;
    }
    return false;
}

void LocalMailboxEndpoint::release_entry_locked(size_t index) {
    EntryRecord &record = entries_[index];
    // Even generation: the entry is free, and every reference still carrying the
    // odd one is now refusable rather than indistinguishable from a fresh run.
    // Published before the record is cleared, so nothing on the child side can
    // see a free entry still advertising this run's identity.
    MailboxRunEntry *entry = run_entry(index);
    store_entry_u64(&entry->entry_generation, record.entry_generation + 1);
    store_entry_i32(&entry->state, static_cast<int32_t>(MailboxState::IDLE));
    // A frame still published for this entry will never be echoed now, so it
    // comes back here rather than waiting for a level that cannot arrive.
    if (record.request_frame < request_frames_.size() && request_frames_[record.request_frame].entry == index) {
        request_frames_[record.request_frame].published = false;
        request_frames_[record.request_frame].entry = kNoRequestFrame;
    }
    record = EntryRecord{};
}

bool LocalMailboxEndpoint::activate_progress(RunId run_id) {
    std::unique_lock<std::mutex> lk(progress_mu_);
    if (endpoint_poisoned_) return false;
    for (size_t index = 0; index < entries_.size(); ++index) {
        EntryRecord &record = entries_[index];
        if (!record.occupied || !record.dispatch.prepare_only || record.run_id != run_id) continue;
#if SIMPLER_HOST_STRACE
        std::optional<simpler::host_trace::SpanScope> activate_trace;
        if (simpler::host_trace::enabled()) {
            std::ostringstream activate_attrs;
            activate_attrs << "run_id=" << run_id << " task_slot=" << record.dispatch.task_slot
                           << " group_index=" << record.dispatch.group_index << " worker_id=" << caps_.worker_id
                           << " dispatch_id=" << record.dispatch.dispatch_id
                           << " endpoint_kind=" << endpoint_kind_name(caps_.kind)
                           << " prepare_only=" << static_cast<int>(record.dispatch.prepare_only) << " role=worker";
            activate_trace.emplace(
                simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Activate), run_id, 0, 0,
                activate_attrs.str()
            );
        }
#endif
        record.activation_requested = true;
        (void)try_publish_activation(record, run_entry(index));
        lk.unlock();
        return true;
    }
    return false;
}

void LocalMailboxEndpoint::request_progress_stop() noexcept { shutdown_child(); }

void LocalMailboxEndpoint::report_progress_error(const std::string &reason) {
    std::lock_guard<std::mutex> lk(progress_mu_);
    poison_progress("endpoint progress driver failed: " + reason);
}

bool LocalMailboxEndpoint::report_submission_error(const WorkerDispatch &dispatch, const std::string &reason) {
    std::lock_guard<std::mutex> lk(progress_mu_);
    bool endpoint_owned = false;
    for (const EntryRecord &record : entries_) {
        endpoint_owned = endpoint_owned || (record.occupied && record.dispatch.dispatch_id == dispatch.dispatch_id);
    }
    poison_progress("endpoint submission failed: " + reason);
    return endpoint_owned;
}

// =============================================================================
// WorkerManager
// =============================================================================

void WorkerManager::add_next_level(void *mailbox, size_t mailbox_bytes, int child_pid, uint32_t granted_sets) {
    add_next_level_at(
        static_cast<int32_t>(next_level_entries_.size()), mailbox, mailbox_bytes, child_pid, granted_sets
    );
}

void WorkerManager::add_next_level_at(
    int32_t worker_id, void *mailbox, size_t mailbox_bytes, int child_pid, uint32_t granted_sets
) {
    if (worker_id < 0) throw std::invalid_argument("WorkerManager::add_next_level_at: negative worker_id");
    validate_mailbox_region("WorkerManager::add_next_level_at", mailbox, mailbox_bytes, granted_sets);
    next_level_entries_.push_back(LocalNextLevelEntry{worker_id, mailbox, mailbox_bytes, child_pid, granted_sets});
}

void WorkerManager::add_next_level_endpoint(std::unique_ptr<WorkerEndpoint> endpoint) {
    if (!endpoint) throw std::invalid_argument("WorkerManager::add_next_level_endpoint: null endpoint");
    next_level_endpoint_entries_.push_back(std::move(endpoint));
}

void WorkerManager::add_sub(void *mailbox, size_t mailbox_bytes, int child_pid) {
    // A SUB endpoint takes the default grant, and its region carries the same
    // entry table at the same offsets, so it is held to the same extent.
    validate_mailbox_region("WorkerManager::add_sub", mailbox, mailbox_bytes, /*granted_sets=*/1);
    sub_entries_.push_back(LocalSubEntry{mailbox, mailbox_bytes, child_pid});
}

void WorkerManager::start(
    Ring *ring, const OnCompleteFn &on_complete, const OnAcceptFn &on_accept, const OnStagedFn &on_staged
) {
    if (ring == nullptr) throw std::invalid_argument("WorkerManager::start: null ring");

    std::vector<int32_t> next_level_worker_ids;
    next_level_worker_ids.reserve(next_level_entries_.size() + next_level_endpoint_entries_.size());
    auto register_next_level_worker_id = [&](int32_t worker_id) {
        if (worker_id < 0) {
            throw std::runtime_error("WorkerManager::start: NEXT_LEVEL worker must have a stable worker_id");
        }
        if (std::find(next_level_worker_ids.begin(), next_level_worker_ids.end(), worker_id) !=
            next_level_worker_ids.end()) {
            throw std::runtime_error(
                "WorkerManager::start: duplicate NEXT_LEVEL worker_id " + std::to_string(worker_id)
            );
        }
        next_level_worker_ids.push_back(worker_id);
    };
    for (const auto &entry : next_level_entries_) {
        register_next_level_worker_id(entry.worker_id);
    }
    for (const auto &endpoint : next_level_endpoint_entries_) {
        register_next_level_worker_id(endpoint->caps().worker_id);
    }

    auto make_next_level_threads = [&]() {
        for (const auto &entry : next_level_entries_) {
            auto wt = std::make_unique<WorkerThread>();
            auto endpoint = std::make_unique<LocalMailboxEndpoint>(
                entry.worker_id, entry.mailbox, entry.mailbox_bytes, entry.child_pid, entry.granted_sets
            );
            wt->start(ring, on_complete, on_accept, on_staged, std::move(endpoint));
            next_level_threads_.push_back(std::move(wt));
        }
    };
    auto make_sub_threads = [&](const std::vector<LocalSubEntry> &entries,
                                std::vector<std::unique_ptr<WorkerThread>> &threads) {
        for (size_t i = 0; i < entries.size(); ++i) {
            auto wt = std::make_unique<WorkerThread>();
            auto endpoint = std::make_unique<LocalMailboxEndpoint>(
                static_cast<int32_t>(i), entries[i].mailbox, entries[i].mailbox_bytes, entries[i].child_pid
            );
            wt->start(ring, on_complete, on_accept, on_staged, std::move(endpoint));
            threads.push_back(std::move(wt));
        }
    };
    make_next_level_threads();
    for (auto &endpoint : next_level_endpoint_entries_) {
        auto wt = std::make_unique<WorkerThread>();
        wt->start(ring, on_complete, on_accept, on_staged, std::move(endpoint));
        next_level_threads_.push_back(std::move(wt));
    }
    next_level_endpoint_entries_.clear();
    make_sub_threads(sub_entries_, sub_threads_);
}

void WorkerManager::stop_workers() {
    for (auto &wt : next_level_threads_)
        wt->stop();
    for (auto &wt : sub_threads_)
        wt->stop();
}

void WorkerManager::stop() {
    stop_workers();
    next_level_threads_.clear();
    sub_threads_.clear();
}

void WorkerManager::progress() {
    for (auto &worker : next_level_threads_)
        worker->progress();
    for (auto &worker : sub_threads_)
        worker->progress();
}

WorkerThread *WorkerManager::get_worker_by_id(WorkerType type, int32_t worker_id) const {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    for (auto &wt : threads) {
        if (wt->worker_id() == worker_id) return wt.get();
    }
    return nullptr;
}

std::vector<int32_t> WorkerManager::next_level_worker_ids() const {
    std::vector<int32_t> worker_ids;
    worker_ids.reserve(next_level_threads_.size());
    for (const auto &worker : next_level_threads_) {
        worker_ids.push_back(worker->worker_id());
    }
    return worker_ids;
}

WorkerThread *WorkerManager::pick_idle_sub_excluding(const std::vector<WorkerThread *> &exclude) const {
    for (const auto &wt : sub_threads_) {
        if (!wt->idle()) continue;
        bool excluded = false;
        for (auto *ex : exclude) {
            if (ex == wt.get()) {
                excluded = true;
                break;
            }
        }
        if (!excluded) return wt.get();
    }
    return nullptr;
}

// =============================================================================
// LocalMailboxEndpoint — memory control (orch thread, concurrent with Scheduler progress)
// =============================================================================

static void write_control_args(char *mbox, uint64_t sub_cmd, uint64_t a0 = 0) {
    std::memcpy(mbox + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    std::memcpy(mbox + CTRL_OFF_ARG0, &a0, sizeof(uint64_t));
}

static uint64_t read_control_result(const char *mbox) {
    uint64_t r;
    std::memcpy(&r, mbox + CTRL_OFF_RESULT, sizeof(uint64_t));
    return r;
}

static void write_control_digest(char *mbox, const uint8_t *digest) {
    if (digest == nullptr) {
        std::memset(mbox + MAILBOX_OFF_CONTROL_CALLABLE_HASH, 0, CALLABLE_HASH_DIGEST_SIZE);
        return;
    }
    std::memcpy(mbox + MAILBOX_OFF_CONTROL_CALLABLE_HASH, digest, CALLABLE_HASH_DIGEST_SIZE);
}

// Issue a control sub-command and block until the child publishes
// CONTROL_DONE. Caller must hold `mailbox_mu_`. Registry controls may remain
// in CONTROL_REQUEST while the child owns an active native run or a live frame
// references the callable digest. The default infinite timeout waits for that
// deferral subject to child-liveness checks; an explicit timeout includes the
// deferred interval, and expiry poisons the endpoint. On a non-zero error code
// from the child, throws and leaves the mailbox in IDLE before unwinding (so
// the next claim starts from a clean state). The `op_name` is used only for the
// exception message.
void LocalMailboxEndpoint::run_control_command(const char *op_name, double timeout_s) {
    if (mailbox_control_timed_out_) {
        throw std::runtime_error(std::string(op_name) + " failed: mailbox has an unresolved timed-out control command");
    }
    int32_t zero_err = 0;
    std::memcpy(mbox() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
    std::memset(mbox() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
    write_mailbox_state(MailboxState::CONTROL_REQUEST);
    auto deadline = std::chrono::steady_clock::time_point::max();
    if (timeout_s >= 0.0) {
        deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_s));
    }
    auto next_liveness_check = std::chrono::steady_clock::now() + kChildLivenessPollPeriod;
    while (read_mailbox_state() != MailboxState::CONTROL_DONE) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            mailbox_control_timed_out_.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> progress_lk(progress_mu_);
                poison_progress(std::string(op_name) + " timed out waiting for CONTROL_DONE");
            }
            throw std::runtime_error(std::string(op_name) + " timed out waiting for CONTROL_DONE");
        }
        if (now >= next_liveness_check) {
            next_liveness_check = now + kChildLivenessPollPeriod;
            std::string death = check_child_death();
            if (!death.empty()) {
                // The mailbox is poisoned rather than reset to IDLE: with the
                // child gone no later command can complete, so admitting one
                // would restore the hang this check exists to break.
                mailbox_control_timed_out_.store(true, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> progress_lk(progress_mu_);
                    poison_progress(death);
                }
                throw std::runtime_error(std::string(op_name) + ": " + death);
            }
        }
    }
    int32_t err = 0;
    std::memcpy(&err, mbox() + MAILBOX_OFF_ERROR, sizeof(int32_t));
    if (err != 0) {
        std::string msg = read_error_msg(mbox());
        write_mailbox_state(MailboxState::IDLE);
        throw std::runtime_error(std::string(op_name) + " failed on child: " + msg);
    }
    write_mailbox_state(MailboxState::IDLE);
}

uint64_t LocalMailboxEndpoint::control_malloc(size_t size) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_MALLOC, static_cast<uint64_t>(size));
    run_control_command("control_malloc");
    return read_control_result(mbox());
}

uint64_t LocalMailboxEndpoint::control_committed_device_memory() {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_COMMITTED_DEVICE_MEMORY);
    run_control_command("control_committed_device_memory");
    return read_control_result(mbox());
}

DeviceMemoryInfo LocalMailboxEndpoint::control_device_memory_info() {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_DEVICE_MEMORY_INFO);
    run_control_command("control_device_memory_info");
    DeviceMemoryInfo info{};
    std::memcpy(&info, mbox() + CTRL_OFF_RESULT, sizeof(info));
    return info;
}

DfxFlushReport LocalMailboxEndpoint::control_dfx_flush(double timeout_s) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    // Checked under the mutex and before any frame byte is written. The other
    // control methods write their args first and let run_control_command throw,
    // which leaves a poisoned endpoint's frame mutated; a flush must not, since
    // the whole point of the poison is that this mailbox is never reused.
    if (mailbox_control_timed_out_.load(std::memory_order_acquire)) {
        throw std::runtime_error("control_dfx_flush failed: mailbox has an unresolved timed-out control command");
    }
    if (std::isnan(timeout_s)) {
        throw std::runtime_error("control_dfx_flush: timeout is not a number");
    }
    // The child's own budget travels in a0, so its native flush is bounded by
    // the caller's budget rather than by a ceiling of its own.
    //
    //   negative   unbounded, the convention `run_control_command` uses. The
    //              child's half is the largest millisecond count its `int`
    //              holds: finite, and far past any real flush.
    //   < 1 ms     refused here, before a frame byte is written. There is no
    //              time in which to flush, and issuing a command whose
    //              deadline has already passed would poison an otherwise
    //              healthy endpoint. Reachable: both callers hand out a
    //              clamped remaining budget, and zero is one of its values.
    //   finite     the request less `kChildFlushReplyMarginMs`, floored at
    //              1 ms. Shorter than this thread's wait, which makes a
    //              reported failure the likely answer and not a guaranteed
    //              one — the wait below still times out and poisons when the
    //              child does not answer in time, and that poison is what
    //              keeps a late reply away from a frame somebody else owns.
    //   huge       clamped to the same `int` ceiling, so no narrowing turns a
    //              long budget into a short one.
    uint64_t child_budget_ms = kUnboundedChildFlushBudgetMs;
    if (timeout_s >= 0.0) {
        const double requested_ms = timeout_s * 1000.0;
        if (requested_ms < 1.0) {
            throw std::runtime_error(
                "control_dfx_flush: no remaining budget to flush in; the close or flush deadline is already spent"
            );
        }
        const double budget_ms = requested_ms - static_cast<double>(kChildFlushReplyMarginMs);
        const double kMaxChildBudgetMs = static_cast<double>(kUnboundedChildFlushBudgetMs);
        child_budget_ms = budget_ms < 1.0               ? 1ULL :
                          budget_ms > kMaxChildBudgetMs ? kUnboundedChildFlushBudgetMs :
                                                          static_cast<uint64_t>(budget_ms);
    }
    write_control_args(mbox(), CTRL_DFX_FLUSH, child_budget_ms);
    run_control_command("control_dfx_flush", timeout_s);
    DfxFlushReport report{};
    std::memcpy(&report, mbox() + CTRL_OFF_RESULT, sizeof(report));
    return report;
}

void LocalMailboxEndpoint::control_prepare(const uint8_t *digest) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_PREPARE);
    write_control_digest(mbox(), digest);
    run_control_command("control_prepare");
}

void LocalMailboxEndpoint::control_register(const char *shm_name, size_t blob_size, const uint8_t *digest) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    // OFF_ERROR / OFF_ERROR_MSG are cleared by run_control_command — no
    // prelude memset needed (matches the other control_* methods).
    uint64_t sub_cmd = CTRL_REGISTER;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    uint64_t payload_size = static_cast<uint64_t>(blob_size);
    std::memcpy(mbox() + CTRL_OFF_ARG0, &payload_size, sizeof(uint64_t));
    write_control_digest(mbox(), digest);
    // Stage the NUL-terminated shm name in the args region. Pad with zeros so
    // stale bytes from a prior control op cannot leak into the child's decode.
    size_t name_len = std::strlen(shm_name);
    if (name_len + 1 > CTRL_SHM_NAME_BYTES) {
        throw std::runtime_error(std::string("control_register: shm name too long: ") + shm_name);
    }
    std::memcpy(mbox() + MAILBOX_OFF_ARGS, shm_name, name_len);
    std::memset(mbox() + MAILBOX_OFF_ARGS + name_len, 0, CTRL_SHM_NAME_BYTES - name_len);
    run_control_command("control_register");
}

void LocalMailboxEndpoint::control_unregister(const uint8_t *digest) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_UNREGISTER);
    write_control_digest(mbox(), digest);
    run_control_command("control_unregister");
}

void LocalMailboxEndpoint::control_remote_prepare_register(
    remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *, const void *, size_t
) {
    throw_unsupported_control("control_remote_prepare_register");
}

void LocalMailboxEndpoint::control_remote_commit_register(
    remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *
) {
    throw_unsupported_control("control_remote_commit_register");
}

void LocalMailboxEndpoint::control_remote_abort_register(
    remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *
) {
    throw_unsupported_control("control_remote_abort_register");
}

void LocalMailboxEndpoint::control_remote_unregister(remote_l3::RemoteRegistryTarget, CallableKind, const uint8_t *) {
    throw_unsupported_control("control_remote_unregister");
}

RemoteBufferHandle LocalMailboxEndpoint::control_remote_malloc(size_t) {
    throw_unsupported_control("control_remote_malloc");
}

void LocalMailboxEndpoint::control_remote_free(const RemoteBufferHandle &) {
    throw_unsupported_control("control_remote_free");
}

void LocalMailboxEndpoint::control_remote_copy_to(const RemoteBufferHandle &, uint64_t, const void *, size_t) {
    throw_unsupported_control("control_remote_copy_to");
}

void LocalMailboxEndpoint::control_remote_copy_from(void *, const RemoteBufferHandle &, uint64_t, size_t) {
    throw_unsupported_control("control_remote_copy_from");
}

RemoteBufferExport LocalMailboxEndpoint::control_remote_export(
    const RemoteBufferHandle &, uint64_t, uint64_t, uint32_t, const std::string &
) {
    throw_unsupported_control("control_remote_export");
}

RemoteBufferHandle LocalMailboxEndpoint::control_remote_import(int32_t, const RemoteBufferExport &, uint32_t) {
    throw_unsupported_control("control_remote_import");
}

void LocalMailboxEndpoint::control_remote_release_import(const RemoteBufferHandle &) {
    throw_unsupported_control("control_remote_release_import");
}

void LocalMailboxEndpoint::control_generic(
    uint64_t sub_cmd, const char *shm_name, size_t staged_payload_size, double timeout_s, const uint8_t *digest
) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    uint64_t payload_size = static_cast<uint64_t>(staged_payload_size);
    std::memcpy(mbox() + CTRL_OFF_ARG0, &payload_size, sizeof(uint64_t));
    write_control_digest(mbox(), digest);
    const char *name = shm_name ? shm_name : "";
    size_t name_len = std::strlen(name);
    if (name_len + 1 > CTRL_SHM_NAME_BYTES) {
        throw std::runtime_error(std::string("control_generic: shm name too long: ") + name);
    }
    if (name_len > 0) std::memcpy(mbox() + MAILBOX_OFF_ARGS, name, name_len);
    std::memset(mbox() + MAILBOX_OFF_ARGS + name_len, 0, CTRL_SHM_NAME_BYTES - name_len);
    run_control_command("control_generic", timeout_s);
}

void LocalMailboxEndpoint::control_free(uint64_t ptr) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_FREE, ptr);
    run_control_command("control_free");
}

void LocalMailboxEndpoint::control_copy_to(
    const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span
) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_copy_request(mbox(), CTRL_COPY_TO, dst, src, span);
    run_control_command("control_copy_to");
}

void LocalMailboxEndpoint::control_copy_from(
    const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span
) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_copy_request(mbox(), CTRL_COPY_FROM, dst, src, span);
    run_control_command("control_copy_from");
}

// Stage two NUL-terminated shm names at MAILBOX_OFF_ARGS: request first
// (CTRL_SHM_NAME_BYTES wide) then reply (CTRL_SHM_NAME_BYTES wide).  Pads each
// slot with zeros so stale bytes from a prior op cannot leak into the child's
// decode.  `reply_shm_name` may be empty (NUL byte) for release.
static void write_shm_name_pair(char *mbox, const char *request_shm_name, const char *reply_shm_name) {
    auto write_one = [&](char *dst, const char *name) {
        size_t n = name ? std::strlen(name) : 0;
        if (n + 1 > CTRL_SHM_NAME_BYTES) {
            throw std::runtime_error(std::string("control: shm name too long: ") + (name ? name : "(null)"));
        }
        if (n > 0) std::memcpy(dst, name, n);
        std::memset(dst + n, 0, CTRL_SHM_NAME_BYTES - n);
    };
    write_one(mbox + MAILBOX_OFF_ARGS, request_shm_name);
    write_one(mbox + MAILBOX_OFF_ARGS + CTRL_SHM_NAME_BYTES, reply_shm_name);
}

void LocalMailboxEndpoint::control_alloc_domain(const char *request_shm_name, const char *reply_shm_name) {
    if (!request_shm_name || !*request_shm_name || !reply_shm_name || !*reply_shm_name) {
        throw std::runtime_error("control_alloc_domain: request and reply shm names must be non-empty");
    }
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    uint64_t sub_cmd = CTRL_ALLOC_DOMAIN;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    write_shm_name_pair(mbox(), request_shm_name, reply_shm_name);
    run_control_command("control_alloc_domain");
}

void LocalMailboxEndpoint::control_release_domain(const char *request_shm_name) {
    if (!request_shm_name || !*request_shm_name) {
        throw std::runtime_error("control_release_domain: request shm name must be non-empty");
    }
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    uint64_t sub_cmd = CTRL_RELEASE_DOMAIN;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    write_shm_name_pair(mbox(), request_shm_name, "");
    run_control_command("control_release_domain");
}

void LocalMailboxEndpoint::control_comm_init(const char *request_shm_name) {
    if (!request_shm_name || !*request_shm_name) {
        throw std::runtime_error("control_comm_init: request shm name must be non-empty");
    }
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    uint64_t sub_cmd = CTRL_COMM_INIT;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    write_shm_name_pair(mbox(), request_shm_name, "");
    run_control_command("control_comm_init");
}

uint64_t WorkerThread::control_malloc(size_t size) {
    if (!endpoint_) throw std::runtime_error("control_malloc: null endpoint");
    return endpoint_->control_malloc(size);
}

uint64_t WorkerThread::control_committed_device_memory() {
    if (!endpoint_) throw std::runtime_error("control_committed_device_memory: null endpoint");
    return endpoint_->control_committed_device_memory();
}

DeviceMemoryInfo WorkerThread::control_device_memory_info() {
    if (!endpoint_) throw std::runtime_error("control_device_memory_info: null endpoint");
    return endpoint_->control_device_memory_info();
}

DfxFlushReport WorkerThread::control_dfx_flush(double timeout_s) {
    if (!endpoint_) throw std::runtime_error("control_dfx_flush: null endpoint");
    return endpoint_->control_dfx_flush(timeout_s);
}

void WorkerThread::control_prepare(const uint8_t *digest) {
    if (!endpoint_) throw std::runtime_error("control_prepare: null endpoint");
    endpoint_->control_prepare(digest);
}

void WorkerThread::control_register(const char *shm_name, size_t blob_size, const uint8_t *digest) {
    if (!endpoint_) throw std::runtime_error("control_register: null endpoint");
    endpoint_->control_register(shm_name, blob_size, digest);
}

void WorkerThread::control_unregister(const uint8_t *digest) {
    if (!endpoint_) throw std::runtime_error("control_unregister: null endpoint");
    endpoint_->control_unregister(digest);
}

void WorkerThread::control_remote_prepare_register(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest,
    const void *payload, size_t payload_size
) {
    if (!endpoint_) throw std::runtime_error("control_remote_prepare_register: null endpoint");
    endpoint_->control_remote_prepare_register(target_registry, callable_kind, digest, payload, payload_size);
}

void WorkerThread::control_remote_commit_register(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    if (!endpoint_) throw std::runtime_error("control_remote_commit_register: null endpoint");
    endpoint_->control_remote_commit_register(target_registry, callable_kind, digest);
}

void WorkerThread::control_remote_abort_register(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    if (!endpoint_) throw std::runtime_error("control_remote_abort_register: null endpoint");
    endpoint_->control_remote_abort_register(target_registry, callable_kind, digest);
}

void WorkerThread::control_remote_unregister(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    if (!endpoint_) throw std::runtime_error("control_remote_unregister: null endpoint");
    endpoint_->control_remote_unregister(target_registry, callable_kind, digest);
}

RemoteBufferHandle WorkerThread::control_remote_malloc(size_t size) {
    if (!endpoint_) throw std::runtime_error("control_remote_malloc: null endpoint");
    return endpoint_->control_remote_malloc(size);
}

void WorkerThread::control_remote_free(const RemoteBufferHandle &handle) {
    if (!endpoint_) throw std::runtime_error("control_remote_free: null endpoint");
    endpoint_->control_remote_free(handle);
}

void WorkerThread::control_remote_copy_to(
    const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size
) {
    if (!endpoint_) throw std::runtime_error("control_remote_copy_to: null endpoint");
    endpoint_->control_remote_copy_to(handle, offset, src, size);
}

void WorkerThread::control_remote_copy_from(void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size) {
    if (!endpoint_) throw std::runtime_error("control_remote_copy_from: null endpoint");
    endpoint_->control_remote_copy_from(dst, handle, offset, size);
}

RemoteBufferExport WorkerThread::control_remote_export(
    const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
    const std::string &transport_profile
) {
    if (!endpoint_) throw std::runtime_error("control_remote_export: null endpoint");
    return endpoint_->control_remote_export(handle, offset, size, access_flags, transport_profile);
}

RemoteBufferHandle WorkerThread::control_remote_import(
    int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
) {
    if (!endpoint_) throw std::runtime_error("control_remote_import: null endpoint");
    return endpoint_->control_remote_import(importer_worker_id, export_desc, requested_access_flags);
}

void WorkerThread::control_remote_release_import(const RemoteBufferHandle &handle) {
    if (!endpoint_) throw std::runtime_error("control_remote_release_import: null endpoint");
    endpoint_->control_remote_release_import(handle);
}

std::vector<uint8_t> WorkerThread::control_remote_domain(
    remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes, bool group_target
) {
    if (!endpoint_) throw std::runtime_error("control_remote_domain: null endpoint");
    return endpoint_->control_remote_domain(control_name, command_bytes, group_target);
}

void WorkerThread::control_generic(
    uint64_t sub_cmd, const char *shm_name, size_t payload_size, double timeout_s, const uint8_t *digest
) {
    if (!endpoint_) throw std::runtime_error("control_generic: null endpoint");
    endpoint_->control_generic(sub_cmd, shm_name, payload_size, timeout_s, digest);
}

void WorkerThread::control_free(uint64_t ptr) {
    if (!endpoint_) throw std::runtime_error("control_free: null endpoint");
    endpoint_->control_free(ptr);
}

void WorkerThread::control_copy_to(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span) {
    if (!endpoint_) throw std::runtime_error("control_copy_to: null endpoint");
    endpoint_->control_copy_to(dst, src, span);
}

void WorkerThread::control_copy_from(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span) {
    if (!endpoint_) throw std::runtime_error("control_copy_from: null endpoint");
    endpoint_->control_copy_from(dst, src, span);
}

void WorkerThread::control_alloc_domain(const char *request_shm_name, const char *reply_shm_name) {
    if (!endpoint_) throw std::runtime_error("control_alloc_domain: null endpoint");
    endpoint_->control_alloc_domain(request_shm_name, reply_shm_name);
}

void WorkerThread::control_release_domain(const char *request_shm_name) {
    if (!endpoint_) throw std::runtime_error("control_release_domain: null endpoint");
    endpoint_->control_release_domain(request_shm_name);
}

void WorkerThread::control_comm_init(const char *request_shm_name) {
    if (!endpoint_) throw std::runtime_error("control_comm_init: null endpoint");
    endpoint_->control_comm_init(request_shm_name);
}

bool WorkerManager::any_busy() const {
    for (auto &wt : next_level_threads_)
        if (wt->busy()) return true;
    for (auto &wt : sub_threads_)
        if (wt->busy()) return true;
    return false;
}

bool WorkerManager::has_staged_run(RunId run_id) const {
    for (const auto &worker : next_level_threads_)
        if (worker->has_staged_run(run_id)) return true;
    return false;
}

bool WorkerManager::activate_prepared_run(RunId run_id) {
    bool activated = false;
    for (const auto &worker : next_level_threads_)
        activated = worker->activate_prepared(run_id) || activated;
    return activated;
}

bool WorkerManager::authorize_staged_launch(RunId run_id) {
    bool authorized = false;
    for (const auto &worker : next_level_threads_)
        authorized = worker->authorize_staged_launch(run_id) || authorized;
    return authorized;
}

// =============================================================================
// Dynamic register/unregister broadcast (POSIX shm staging + parallel fan-out)
// =============================================================================

namespace {

// Process-wide monotonic counter so concurrent broadcasts do not collide on shm
// name. Atomic increment is enough — no need to lock.
std::atomic<uint64_t> g_shm_counter{0};

// Build the per-broadcast POSIX shm name. The name itself does NOT carry the
// leading '/' that shm_open requires (Python's multiprocessing.SharedMemory
// uses the same convention, so the child Python side reads the field as a
// plain name). Caller adds '/' when opening.
std::string make_shm_name() {
    char buf[CTRL_SHM_NAME_BYTES];
    int pid = static_cast<int>(getpid());
    uint64_t counter = g_shm_counter.fetch_add(1, std::memory_order_relaxed);
    int n = std::snprintf(buf, sizeof(buf), "simpler-cb-%d-%llu", pid, static_cast<unsigned long long>(counter));
    if (n < 0 || static_cast<size_t>(n) >= sizeof(buf)) {
        throw std::runtime_error("broadcast_register: shm name overflow");
    }
    return std::string(buf);
}

// Strip the outer "<op_name> failed on child: " prefix that
// run_control_command prepends to every control failure, so the broadcast
// caller can surface the child-side message (`register hash=sha256:...
// chip=<id>: <reason>`) directly under its own one-line Worker.register prefix.
std::string strip_control_prefix(const std::string &msg, const std::string &op_name) {
    const std::string needle = op_name + " failed on child: ";
    if (msg.compare(0, needle.size(), needle) == 0) {
        return msg.substr(needle.size());
    }
    return msg;
}

// RAII guard for a POSIX shm segment: create on construction, unlink on
// destruction. mmaps the region so the staged blob can be memcpy'd in
// place; the mmap is released in the destructor as well. The shm is only
// unlinked once — children open by name *before* this guard is destroyed.
class PosixShmHolder {
public:
    PosixShmHolder(const std::string &name, size_t size) :
        name_(name),
        size_(size) {
        std::string full_name = "/" + name_;
        fd_ = shm_open(full_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
        if (fd_ < 0) {
            throw std::runtime_error(
                std::string("broadcast_register: shm_open(") + full_name + ") failed: " + std::strerror(errno)
            );
        }
        if (ftruncate(fd_, static_cast<off_t>(size)) != 0) {
            int err = errno;
            ::close(fd_);
            shm_unlink(full_name.c_str());
            throw std::runtime_error(std::string("broadcast_register: ftruncate failed: ") + std::strerror(err));
        }
        addr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) {
            int err = errno;
            ::close(fd_);
            shm_unlink(full_name.c_str());
            addr_ = nullptr;
            throw std::runtime_error(std::string("broadcast_register: mmap failed: ") + std::strerror(err));
        }
    }
    ~PosixShmHolder() {
        if (addr_ != nullptr) munmap(addr_, size_);
        if (fd_ >= 0) ::close(fd_);
        std::string full_name = "/" + name_;
        shm_unlink(full_name.c_str());
    }
    PosixShmHolder(const PosixShmHolder &) = delete;
    PosixShmHolder &operator=(const PosixShmHolder &) = delete;

    void *addr() { return addr_; }
    const std::string &name() const { return name_; }

private:
    std::string name_;
    size_t size_{0};
    int fd_{-1};
    void *addr_{nullptr};
};

}  // namespace

void WorkerManager::control_prepare(int worker_id, const uint8_t *digest) {
    auto *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_prepare: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_prepare(digest);
}

void WorkerManager::control_alloc_domain(int worker_id, const char *request_shm_name, const char *reply_shm_name) {
    auto *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_alloc_domain: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_alloc_domain(request_shm_name, reply_shm_name);
}

void WorkerManager::control_release_domain(int worker_id, const char *request_shm_name) {
    auto *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_release_domain: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_release_domain(request_shm_name);
}

void WorkerManager::control_comm_init(int worker_id, const char *request_shm_name) {
    auto *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_comm_init: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_comm_init(request_shm_name);
}

ControlResult WorkerManager::control_digest_only(
    WorkerType type, int worker_id, uint64_t sub_cmd, const uint8_t *digest, double timeout_s
) {
    const char *type_name = (type == WorkerType::NEXT_LEVEL) ? "NEXT_LEVEL" : "SUB";
    ControlResult result{type_name, static_cast<int32_t>(worker_id), false, ""};
    WorkerThread *wt = get_worker_by_id(type, worker_id);
    if (wt == nullptr) {
        result.error_message = "invalid worker_id " + std::to_string(worker_id);
        return result;
    }
    try {
        wt->control_generic(sub_cmd, nullptr, 0, timeout_s, digest);
        result.ok = true;
    } catch (const std::exception &e) {
        result.error_message = strip_control_prefix(e.what(), "control_generic");
    }
    return result;
}

std::vector<uint8_t> WorkerManager::control_payload(
    WorkerType type, int worker_id, uint64_t sub_cmd, const void *payload, size_t payload_size, double timeout_s
) {
    if (payload == nullptr || payload_size == 0) {
        throw std::runtime_error("control_payload: payload must be non-empty");
    }
    WorkerThread *wt = get_worker_by_id(type, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_payload: invalid worker_id " + std::to_string(worker_id));
    }
    std::string shm_name = make_shm_name();
    PosixShmHolder shm(shm_name, payload_size);
    std::memcpy(shm.addr(), payload, payload_size);
    wt->control_generic(sub_cmd, shm_name.c_str(), payload_size, timeout_s, nullptr);
    auto *begin = static_cast<const uint8_t *>(shm.addr());
    return {begin, begin + payload_size};
}

ControlResult WorkerManager::control_remote_prepare_register(
    int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const void *payload,
    size_t payload_size, const uint8_t *digest
) {
    ControlResult result{"NEXT_LEVEL", static_cast<int32_t>(worker_id), false, ""};
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        result.error_message = "invalid worker_id " + std::to_string(worker_id);
        return result;
    }
    try {
        wt->control_remote_prepare_register(target_registry, callable_kind, digest, payload, payload_size);
        result.ok = true;
    } catch (const std::exception &e) {
        result.error_message = strip_control_prefix(e.what(), "control_remote_prepare_register");
    }
    return result;
}

ControlResult WorkerManager::control_remote_commit_register(
    int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    ControlResult result{"NEXT_LEVEL", static_cast<int32_t>(worker_id), false, ""};
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        result.error_message = "invalid worker_id " + std::to_string(worker_id);
        return result;
    }
    try {
        wt->control_remote_commit_register(target_registry, callable_kind, digest);
        result.ok = true;
    } catch (const std::exception &e) {
        result.error_message = strip_control_prefix(e.what(), "control_remote_commit_register");
    }
    return result;
}

ControlResult WorkerManager::control_remote_abort_register(
    int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    ControlResult result{"NEXT_LEVEL", static_cast<int32_t>(worker_id), false, ""};
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        result.error_message = "invalid worker_id " + std::to_string(worker_id);
        return result;
    }
    try {
        wt->control_remote_abort_register(target_registry, callable_kind, digest);
        result.ok = true;
    } catch (const std::exception &e) {
        result.error_message = strip_control_prefix(e.what(), "control_remote_abort_register");
    }
    return result;
}

ControlResult WorkerManager::control_remote_unregister(
    int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    ControlResult result{"NEXT_LEVEL", static_cast<int32_t>(worker_id), false, ""};
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        result.error_message = "invalid worker_id " + std::to_string(worker_id);
        return result;
    }
    try {
        wt->control_remote_unregister(target_registry, callable_kind, digest);
        result.ok = true;
    } catch (const std::exception &e) {
        result.error_message = strip_control_prefix(e.what(), "control_remote_unregister");
    }
    return result;
}

RemoteBufferHandle WorkerManager::control_remote_malloc(int worker_id, size_t size) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_remote_malloc: invalid worker_id " + std::to_string(worker_id));
    }
    return wt->control_remote_malloc(size);
}

void WorkerManager::control_remote_free(const RemoteBufferHandle &handle) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, handle.worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_remote_free: invalid worker_id " + std::to_string(handle.worker_id));
    }
    wt->control_remote_free(handle);
}

void WorkerManager::control_remote_copy_to(
    const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size
) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, handle.worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_remote_copy_to: invalid worker_id " + std::to_string(handle.worker_id));
    }
    wt->control_remote_copy_to(handle, offset, src, size);
}

void WorkerManager::control_remote_copy_from(
    void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size
) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, handle.worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_remote_copy_from: invalid worker_id " + std::to_string(handle.worker_id));
    }
    wt->control_remote_copy_from(dst, handle, offset, size);
}

RemoteBufferExport WorkerManager::control_remote_export(
    const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
    const std::string &transport_profile
) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, handle.owner_worker_id);
    if (wt == nullptr) {
        throw std::runtime_error(
            "control_remote_export: invalid owner worker_id " + std::to_string(handle.owner_worker_id)
        );
    }
    return wt->control_remote_export(handle, offset, size, access_flags, transport_profile);
}

RemoteBufferHandle WorkerManager::control_remote_import(
    int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, importer_worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_remote_import: invalid worker_id " + std::to_string(importer_worker_id));
    }
    return wt->control_remote_import(importer_worker_id, export_desc, requested_access_flags);
}

void WorkerManager::control_remote_release_import(const RemoteBufferHandle &handle) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, handle.worker_id);
    if (wt == nullptr) {
        throw std::runtime_error(
            "control_remote_release_import: invalid worker_id " + std::to_string(handle.worker_id)
        );
    }
    wt->control_remote_release_import(handle);
}

std::vector<uint8_t> WorkerManager::control_remote_domain(
    int worker_id, remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes, bool group_target
) {
    WorkerThread *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_remote_domain: invalid worker_id " + std::to_string(worker_id));
    }
    switch (control_name) {
    case remote_l3::ControlName::COMM_INIT:
    case remote_l3::ControlName::ALLOC_DOMAIN:
    case remote_l3::ControlName::RELEASE_DOMAIN:
    case remote_l3::ControlName::COPY_TO_DOMAIN:
    case remote_l3::ControlName::COPY_FROM_DOMAIN:
        return wt->control_remote_domain(control_name, command_bytes, group_target);
    default:
        throw std::runtime_error("control_remote_domain: control name is not a domain operation");
    }
}

std::vector<ControlResult>
WorkerManager::broadcast_register_all(const void *blob_ptr, size_t blob_size, const uint8_t *digest) {
    std::vector<ControlResult> results;
    results.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        results.push_back(ControlResult{"NEXT_LEVEL", next_level_threads_[i]->worker_id(), true, ""});
    }
    if (next_level_threads_.empty()) return results;

    std::string shm_name = make_shm_name();
    PosixShmHolder shm(shm_name, blob_size);
    std::memcpy(shm.addr(), blob_ptr, blob_size);

    // Fan out to every endpoint lane in parallel. Per-endpoint mailbox_mu_
    // is independent, so N control_register calls run concurrently — latency
    // is 1 × prepare_cost instead of N × prepare_cost.
    std::vector<std::thread> workers;
    workers.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        workers.emplace_back([this, i, digest, blob_size, name = shm.name(), &results]() {
            try {
                next_level_threads_[i]->control_register(name.c_str(), blob_size, digest);
            } catch (const std::exception &e) {
                results[i].ok = false;
                results[i].error_message = strip_control_prefix(e.what(), "control_register");
            }
        });
    }
    for (auto &t : workers)
        t.join();

    // shm is unlinked when `shm` goes out of scope. Children opened it by
    // name during control_register and have already closed their mappings
    // before publishing CONTROL_DONE — see python/simpler/worker.py.

    std::string hash = format_digest(digest);
    for (auto &result : results) {
        if (!result.ok && result.error_message.find("hash=") == std::string::npos) {
            result.error_message = "Worker.register(hash=" + hash + ") failed on next_level " +
                                   std::to_string(result.worker_id) + ": " + result.error_message;
        }
    }
    return results;
}

std::vector<std::string> WorkerManager::broadcast_unregister_all(const uint8_t *digest) {
    std::vector<std::string> errors;
    if (next_level_threads_.empty()) return errors;

    std::vector<std::string> per_worker(next_level_threads_.size());
    std::vector<std::thread> workers;
    workers.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        workers.emplace_back([this, i, digest, &per_worker]() {
            try {
                next_level_threads_[i]->control_unregister(digest);
            } catch (const std::exception &e) {
                std::string msg = strip_control_prefix(e.what(), "control_unregister");
                per_worker[i] = std::string("next_level worker_id ") +
                                std::to_string(next_level_threads_[i]->worker_id()) + ": " + msg;
            }
        });
    }
    for (auto &t : workers)
        t.join();

    for (auto &msg : per_worker) {
        if (!msg.empty()) errors.push_back(std::move(msg));
    }
    return errors;
}

std::vector<ControlResult> WorkerManager::broadcast_control_all(
    WorkerType type, uint64_t sub_cmd, const void *payload, size_t payload_size, const uint8_t *digest, double timeout_s
) {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    const char *type_name = (type == WorkerType::NEXT_LEVEL) ? "NEXT_LEVEL" : "SUB";

    std::vector<ControlResult> results;
    results.reserve(threads.size());
    for (size_t i = 0; i < threads.size(); ++i) {
        results.push_back(ControlResult{type_name, threads[i]->worker_id(), true, ""});
    }
    if (threads.empty()) return results;

    std::unique_ptr<PosixShmHolder> shm;
    std::string shm_name;
    if (payload != nullptr || payload_size != 0) {
        if (payload == nullptr || payload_size == 0) {
            throw std::runtime_error("broadcast_control_all: payload pointer and size must both be set");
        }
        shm_name = make_shm_name();
        shm = std::make_unique<PosixShmHolder>(shm_name, payload_size);
        std::memcpy(shm->addr(), payload, payload_size);
    }

    std::vector<std::thread> workers;
    workers.reserve(threads.size());
    for (size_t i = 0; i < threads.size(); ++i) {
        workers.emplace_back([&, i]() {
            try {
                threads[i]->control_generic(
                    sub_cmd, shm_name.empty() ? nullptr : shm_name.c_str(), payload_size, timeout_s, digest
                );
            } catch (const std::exception &e) {
                results[i].ok = false;
                results[i].error_message = strip_control_prefix(e.what(), "control_generic");
            }
        });
    }
    for (auto &t : workers)
        t.join();

    return results;
}
