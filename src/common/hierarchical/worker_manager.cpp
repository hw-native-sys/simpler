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
#include <cstdio>
#include <cstring>
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
void WorkerEndpoint::control_free(uint64_t) { throw_unsupported_control("control_free"); }
void WorkerEndpoint::control_copy_to(const BufferDescriptor &, const BufferDescriptor &, uint64_t) {
    throw_unsupported_control("control_copy_to");
}
void WorkerEndpoint::control_copy_from(const BufferDescriptor &, const BufferDescriptor &, uint64_t) {
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
void WorkerEndpoint::control_worker_chip_region_create(const char *, const char *) {
    throw_unsupported_control("control_worker_chip_region_create");
}
void WorkerEndpoint::control_worker_chip_region_release(uint64_t) {
    throw_unsupported_control("control_worker_chip_region_release");
}

void WorkerEndpoint::submit_progress(Ring *, const WorkerDispatch &) {
    throw std::runtime_error("progress submission is not supported by this WorkerEndpoint");
}
bool WorkerEndpoint::poll_progress(WorkerEndpointProgress &) { return false; }
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

LocalMailboxEndpoint::LocalMailboxEndpoint(int32_t worker_id, void *mailbox, int child_pid, uint32_t task_frame_count) :
    mailbox_(mailbox),
    task_frame_count_(task_frame_count),
    child_pid_(child_pid) {
    if (mailbox == nullptr) throw std::invalid_argument("LocalMailboxEndpoint: null mailbox");
    if (task_frame_count == 0 || task_frame_count > MAILBOX_TASK_FRAME_COUNT) {
        throw std::invalid_argument("LocalMailboxEndpoint: invalid task frame count");
    }
    caps_.worker_id = worker_id;
    caps_.max_inflight_tasks = task_frame_count;
    caps_.supports_frame_staging = task_frame_count > 1;
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

bool LocalMailboxEndpoint::read_task_accepted(const char *frame) const {
    const char *base = frame == nullptr ? mbox() : frame;
    const int32_t *ptr = reinterpret_cast<const int32_t *>(base + MAILBOX_OFF_ACCEPTED);
    int32_t v = 0;
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
    return v == MAILBOX_TASK_ACCEPTED;
}

void LocalMailboxEndpoint::clear_task_accepted(char *frame) {
    char *base = frame == nullptr ? mbox() : frame;
    int32_t *ptr = reinterpret_cast<int32_t *>(base + MAILBOX_OFF_ACCEPTED);
    int32_t v = 0;
    __atomic_store(ptr, &v, __ATOMIC_RELEASE);
}

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

// =============================================================================
// WorkerThread — lifecycle
// =============================================================================

void WorkerThread::start(
    Ring *ring, const std::function<void(WorkerCompletion)> &on_complete,
    const std::function<void(WorkerDispatch)> &on_accept, std::unique_ptr<WorkerEndpoint> endpoint
) {
    if (!endpoint) throw std::invalid_argument("WorkerThread::start: null endpoint");
    ring_ = ring;
    on_complete_ = on_complete;
    on_accept_ = on_accept;
    endpoint_ = std::move(endpoint);
    shutdown_ = false;
    if (endpoint_->caps().max_inflight_tasks == 0) {
        throw std::invalid_argument("WorkerThread::start: endpoint capacity is zero");
    }
    inflight_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        lanes_ = {};
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
        result = submit_dispatch(d, LaneKind::ACTIVE);
    } catch (const std::exception &e) {
        release_lane_unconditional(LaneKind::ACTIVE);
        complete_unpublished(d, std::string("WorkerThread::dispatch: submit failed: ") + e.what());
        return;
    } catch (...) {
        release_lane_unconditional(LaneKind::ACTIVE);
        complete_unpublished(d, "WorkerThread::dispatch: submit failed");
        return;
    }
    if (result == SubmitDispatchResult::SUBMITTED) return;
    release_lane_unconditional(LaneKind::ACTIVE);
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
    bool staged_lane_occupied = false;
    {
        std::lock_guard<std::mutex> lane_lk(lane_mu_);
        LaneState &staged = lane(LaneKind::STAGED);
        if (staged.occupied) {
            staged_lane_occupied = true;
        } else {
            staged.occupied = true;
            staged.run_id = slot->run_id;
        }
    }
    if (staged_lane_occupied) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: worker already owns a staged run");
        return;
    }
    SubmitDispatchResult result;
    try {
        result = submit_dispatch(d, LaneKind::STAGED, slot->run_id);
    } catch (const std::exception &e) {
        release_lane_unconditional(LaneKind::STAGED);
        complete_unpublished(d, std::string("WorkerThread::dispatch_prepared: submit failed: ") + e.what());
        return;
    } catch (...) {
        release_lane_unconditional(LaneKind::STAGED);
        complete_unpublished(d, "WorkerThread::dispatch_prepared: submit failed");
        return;
    }
    if (result == SubmitDispatchResult::SUBMITTED) return;
    release_lane_unconditional(LaneKind::STAGED);
    if (result == SubmitDispatchResult::STOPPING) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: worker is stopping");
    } else if (result == SubmitDispatchResult::CAPACITY_EXCEEDED) {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: endpoint capacity exceeded");
    } else {
        complete_unpublished(d, "WorkerThread::dispatch_prepared: staged lane identity changed before submission");
    }
}

WorkerThread::SubmitDispatchResult
WorkerThread::submit_dispatch(WorkerDispatch d, LaneKind lane_kind, RunId expected_run_id) {
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
        LaneState &dispatch_lane = lane(lane_kind);
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
    try {
        endpoint_->submit_progress(ring_, d);
    } catch (const std::exception &e) {
        fail_submission(d, std::string("submit_progress failed: ") + e.what());
    } catch (...) {
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
    return SubmitDispatchResult::SUBMITTED;
}

bool WorkerThread::activate_prepared(RunId run_id) {
    if (run_id == INVALID_RUN_ID) return false;
    std::lock_guard<std::mutex> admission_lk(admission_mu_);
    if (shutdown_.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    LaneState &active = lane(LaneKind::ACTIVE);
    LaneState &staged = lane(LaneKind::STAGED);
    if (!staged.occupied || staged.run_id != run_id || staged.dispatch_id == 0 || active.occupied) {
        return false;
    }
    active = staged;
    active.activation_requested = true;
    staged = {};
    return true;
}

bool WorkerThread::has_staged_run(RunId run_id) const {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    const LaneState &staged = lane(LaneKind::STAGED);
    return staged.occupied && staged.run_id == run_id;
}

bool WorkerThread::can_stage() const {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    return caps().supports_frame_staging && !lane(LaneKind::STAGED).occupied;
}

bool WorkerThread::idle() const {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    return !lane(LaneKind::ACTIVE).occupied;
}

void WorkerThread::release_lane(LaneKind kind, uint64_t dispatch_id) {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    LaneState &dispatch_lane = lane(kind);
    if (dispatch_lane.dispatch_id == dispatch_id) dispatch_lane = {};
}

void WorkerThread::release_lane_unconditional(LaneKind kind) {
    std::lock_guard<std::mutex> lane_lk(lane_mu_);
    lane(kind) = {};
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

    RunId activated = INVALID_RUN_ID;
    {
        std::lock_guard<std::mutex> admission_lk(admission_mu_);
        if (!shutdown_.load(std::memory_order_acquire)) {
            {
                std::lock_guard<std::mutex> lane_lk(lane_mu_);
                const LaneState &active = lane(LaneKind::ACTIVE);
                if (active.occupied && active.activation_requested) activated = active.run_id;
            }
            if (activated != INVALID_RUN_ID) {
                try {
                    if (endpoint_->activate_progress(activated)) {
                        std::lock_guard<std::mutex> lane_lk(lane_mu_);
                        LaneState &active = lane(LaneKind::ACTIVE);
                        if (active.occupied && active.run_id == activated) active.activation_requested = false;
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
        (task_frame_count_ > 1 && state.pipeline_lease.slot_id >= task_frame_count_)) {
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

    // The lease slot id is a native pipeline slot bounded by the runtime's
    // PipelineContract, not a mailbox frame number. A two-frame endpoint maps
    // the two 1:1; a single-frame endpoint always publishes to frame 0.
    const size_t frame_index = task_frame_count_ == 1 ? 0 : state.pipeline_lease.slot_id;
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
    FrameRecord &record = frames_[frame_index];
    char *frame = task_frame(frame_index);
    if (record.occupied || read_mailbox_state(frame) != MailboxState::IDLE) {
        poison_progress("pipeline-slot task frame is not reusable");
        throw std::runtime_error("pipeline-slot task frame is not reusable");
    }

    int32_t zero_err = 0;
    std::memcpy(frame + MAILBOX_OFF_ERROR, &zero_err, sizeof(zero_err));
    std::memset(frame + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
    clear_task_accepted(frame);
    const int32_t no_disposition = static_cast<int32_t>(MailboxPreparationDisposition::NONE);
    std::memcpy(frame + MAILBOX_OFF_PREPARATION_DISPOSITION, &no_disposition, sizeof(no_disposition));
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

    const uint64_t protocol = MAILBOX_TASK_PROTOCOL_VERSION;
    const uint64_t slot_id = state.pipeline_lease.slot_id;
    std::memcpy(frame + MAILBOX_OFF_FRAME_PROTOCOL, &protocol, sizeof(protocol));
    std::memcpy(frame + MAILBOX_OFF_FRAME_RUN_ID, &state.run_id, sizeof(state.run_id));
    std::memcpy(frame + MAILBOX_OFF_FRAME_SLOT_ID, &slot_id, sizeof(slot_id));
    std::memcpy(frame + MAILBOX_OFF_FRAME_GENERATION, &state.pipeline_lease.generation, sizeof(uint64_t));
    std::memcpy(frame + MAILBOX_OFF_FRAME_DISPATCH_ID, &dispatch.dispatch_id, sizeof(dispatch.dispatch_id));

    record.occupied = true;
    record.dispatch = dispatch;
    record.run_id = state.run_id;
    record.slot_id = slot_id;
    record.generation = state.pipeline_lease.generation;
    write_mailbox_state(dispatch.prepare_only ? MailboxState::PREPARE_READY : MailboxState::TASK_READY, frame);
}

bool LocalMailboxEndpoint::frame_identity_matches(const FrameRecord &record, const char *frame) const {
    uint64_t protocol = 0;
    RunId run_id = INVALID_RUN_ID;
    uint64_t slot_id = 0;
    uint64_t generation = 0;
    uint64_t dispatch_id = 0;
    PipelineSlotLease lease{};
    std::memcpy(&protocol, frame + MAILBOX_OFF_FRAME_PROTOCOL, sizeof(protocol));
    std::memcpy(&run_id, frame + MAILBOX_OFF_FRAME_RUN_ID, sizeof(run_id));
    std::memcpy(&slot_id, frame + MAILBOX_OFF_FRAME_SLOT_ID, sizeof(slot_id));
    std::memcpy(&generation, frame + MAILBOX_OFF_FRAME_GENERATION, sizeof(generation));
    std::memcpy(&dispatch_id, frame + MAILBOX_OFF_FRAME_DISPATCH_ID, sizeof(dispatch_id));
    std::memcpy(&lease, frame + MAILBOX_OFF_PIPELINE_LEASE, sizeof(lease));
    return protocol == MAILBOX_TASK_PROTOCOL_VERSION && run_id == record.run_id && slot_id == record.slot_id &&
           generation == record.generation && dispatch_id == record.dispatch.dispatch_id &&
           lease.slot_id == record.slot_id && lease.reserved == 0 && lease.generation == record.generation;
}

bool LocalMailboxEndpoint::try_publish_activation(FrameRecord &record, char *frame) {
    if (!record.activation_requested || record.activation_published) return record.activation_published;
    if (!frame_identity_matches(record, frame)) {
        poison_progress("stale staged frame identity before activation");
        return false;
    }
    if (mailbox_compare_exchange_state(frame, MailboxState::FRAME_STAGED, MailboxState::ACTIVATE)) {
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
    for (size_t index = 0; index < task_frame_count_; ++index) {
        if (!frames_[index].occupied) continue;
        MailboxState state = read_mailbox_state(task_frame(index));
        if (state != MailboxState::TASK_DONE && state != MailboxState::TASK_FAILED) return false;
    }
    return true;
}

WorkerCompletion LocalMailboxEndpoint::poisoned_completion(const FrameRecord &record) const {
    return WorkerCompletion{
        record.dispatch.task_slot, record.dispatch.group_index, EndpointOutcome::ENDPOINT_FAILURE,
        "LocalMailboxEndpoint progress failed: " + endpoint_poison_reason_
    };
}

bool LocalMailboxEndpoint::poll_progress(WorkerEndpointProgress &progress) {
    std::lock_guard<std::mutex> lk(progress_mu_);
    progress.preparation_disposition = MailboxPreparationDisposition::NONE;
    bool any_occupied = false;
    for (const FrameRecord &record : frames_)
        any_occupied = any_occupied || record.occupied;
    if (!any_occupied) return false;

    if (!endpoint_poisoned_ && mailbox_control_timed_out_.load(std::memory_order_acquire)) {
        poison_progress("mailbox has an unresolved timed-out control command");
    }
    const auto now = std::chrono::steady_clock::now();
    if (!endpoint_poisoned_ && now >= next_liveness_check_) {
        next_liveness_check_ = now + kChildLivenessPollPeriod;
        std::string death = check_child_death();
        if (!death.empty()) poison_progress(death);
    }

    if (endpoint_poisoned_) {
        if (!poisoned_progress_quiesced()) return false;
        for (size_t offset = 0; offset < task_frame_count_; ++offset) {
            size_t index = (poll_cursor_ + offset) % task_frame_count_;
            FrameRecord &record = frames_[index];
            if (!record.occupied) continue;
            progress.kind = WorkerProgressKind::COMPLETED;
            progress.dispatch = record.dispatch;
            progress.completion = poisoned_completion(record);
            record = FrameRecord{};
            poll_cursor_ = (index + 1) % task_frame_count_;
            return true;
        }
        return false;
    }

    for (size_t offset = 0; offset < task_frame_count_; ++offset) {
        size_t index = (poll_cursor_ + offset) % task_frame_count_;
        FrameRecord &record = frames_[index];
        if (!record.occupied) continue;
        char *frame = task_frame(index);
        MailboxState state = read_mailbox_state(frame);

        if (!record.accepted_reported && read_task_accepted(frame)) {
            if (!frame_identity_matches(record, frame)) {
                poison_progress("stale frame identity at launch acceptance");
                break;
            }
            record.accepted_reported = true;
            progress.kind = WorkerProgressKind::ACCEPTED;
            progress.dispatch = record.dispatch;
            poll_cursor_ = (index + 1) % task_frame_count_;
            return true;
        }

        if (state == MailboxState::FRAME_STAGED) {
            if (!frame_identity_matches(record, frame)) {
                poison_progress("stale frame identity at endpoint staging");
                break;
            }
            if (record.activation_requested) {
                (void)try_publish_activation(record, frame);
                if (endpoint_poisoned_) break;
            }
            if (!record.staged_reported) {
                int32_t disposition_value = 0;
                std::memcpy(&disposition_value, frame + MAILBOX_OFF_PREPARATION_DISPOSITION, sizeof(disposition_value));
                const auto disposition = static_cast<MailboxPreparationDisposition>(disposition_value);
                if (disposition != MailboxPreparationDisposition::VALIDATED_ONLY &&
                    disposition != MailboxPreparationDisposition::NATIVE_PREPARED) {
                    poison_progress("invalid preparation disposition at endpoint staging");
                    break;
                }
                record.staged_reported = true;
                progress.kind = WorkerProgressKind::FRAME_STAGED;
                progress.dispatch = record.dispatch;
                progress.preparation_disposition = disposition;
                poll_cursor_ = (index + 1) % task_frame_count_;
                return true;
            }
            continue;
        }

        if (state == MailboxState::TASK_DONE || state == MailboxState::TASK_FAILED) {
            if (!frame_identity_matches(record, frame)) {
                poison_progress("stale frame identity at terminal completion");
                break;
            }
            int32_t error_code = 0;
            std::memcpy(&error_code, frame + MAILBOX_OFF_ERROR, sizeof(error_code));
            progress.kind = WorkerProgressKind::COMPLETED;
            progress.dispatch = record.dispatch;
            progress.completion.task_slot = record.dispatch.task_slot;
            progress.completion.group_index = record.dispatch.group_index;
            if (state == MailboxState::TASK_FAILED || error_code != 0) {
                progress.completion.outcome = EndpointOutcome::TASK_FAILURE;
                progress.completion.error_message =
                    "LocalMailboxEndpoint child failed (worker_id=" + std::to_string(caps_.worker_id) +
                    ", code=" + std::to_string(error_code) + "): " + read_error_msg(frame);
            } else {
                progress.completion.outcome = EndpointOutcome::SUCCESS;
            }
            write_mailbox_state(MailboxState::IDLE, frame);
            record = FrameRecord{};
            poll_cursor_ = (index + 1) % task_frame_count_;
            return true;
        }

        if (state != MailboxState::TASK_READY && state != MailboxState::PREPARE_READY &&
            state != MailboxState::ACTIVATE && state != MailboxState::TASK_LAUNCHED) {
            poison_progress("task frame entered an invalid state " + std::to_string(static_cast<int32_t>(state)));
            break;
        }
    }

    if (!endpoint_poisoned_) return false;
    if (!poisoned_progress_quiesced()) return false;
    for (size_t index = 0; index < task_frame_count_; ++index) {
        FrameRecord &record = frames_[index];
        if (!record.occupied) continue;
        progress.kind = WorkerProgressKind::COMPLETED;
        progress.dispatch = record.dispatch;
        progress.completion = poisoned_completion(record);
        record = FrameRecord{};
        poll_cursor_ = (index + 1) % task_frame_count_;
        return true;
    }
    return false;
}

bool LocalMailboxEndpoint::activate_progress(RunId run_id) {
    std::unique_lock<std::mutex> lk(progress_mu_);
    if (endpoint_poisoned_) return false;
    for (size_t index = 0; index < task_frame_count_; ++index) {
        FrameRecord &record = frames_[index];
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
        char *frame = task_frame(index);
        (void)try_publish_activation(record, frame);
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
    for (const FrameRecord &record : frames_) {
        endpoint_owned = endpoint_owned || (record.occupied && record.dispatch.dispatch_id == dispatch.dispatch_id);
    }
    poison_progress("endpoint submission failed: " + reason);
    return endpoint_owned;
}

// =============================================================================
// WorkerManager
// =============================================================================

void WorkerManager::add_next_level(void *mailbox, int child_pid, uint32_t task_frame_count) {
    add_next_level_at(static_cast<int32_t>(next_level_entries_.size()), mailbox, child_pid, task_frame_count);
}

void WorkerManager::add_next_level_at(int32_t worker_id, void *mailbox, int child_pid, uint32_t task_frame_count) {
    if (worker_id < 0) throw std::invalid_argument("WorkerManager::add_next_level_at: negative worker_id");
    next_level_entries_.push_back(LocalNextLevelEntry{worker_id, mailbox, child_pid, task_frame_count});
}

void WorkerManager::add_next_level_endpoint(std::unique_ptr<WorkerEndpoint> endpoint) {
    if (!endpoint) throw std::invalid_argument("WorkerManager::add_next_level_endpoint: null endpoint");
    next_level_endpoint_entries_.push_back(std::move(endpoint));
}

void WorkerManager::add_sub(void *mailbox, int child_pid) { sub_entries_.push_back(LocalSubEntry{mailbox, child_pid}); }

void WorkerManager::start(Ring *ring, const OnCompleteFn &on_complete, const OnAcceptFn &on_accept) {
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
                entry.worker_id, entry.mailbox, entry.child_pid, entry.task_frame_count
            );
            wt->start(ring, on_complete, on_accept, std::move(endpoint));
            next_level_threads_.push_back(std::move(wt));
        }
    };
    auto make_sub_threads = [&](const std::vector<LocalSubEntry> &entries,
                                std::vector<std::unique_ptr<WorkerThread>> &threads) {
        for (size_t i = 0; i < entries.size(); ++i) {
            auto wt = std::make_unique<WorkerThread>();
            auto endpoint = std::make_unique<LocalMailboxEndpoint>(
                static_cast<int32_t>(i), entries[i].mailbox, entries[i].child_pid
            );
            wt->start(ring, on_complete, on_accept, std::move(endpoint));
            threads.push_back(std::move(wt));
        }
    };
    make_next_level_threads();
    for (auto &endpoint : next_level_endpoint_entries_) {
        auto wt = std::make_unique<WorkerThread>();
        wt->start(ring, on_complete, on_accept, std::move(endpoint));
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
    // Stop admission on every endpoint before asking any child to exit.  In
    // particular, direct-MPI L3 ranks have no out-of-band session owner: their
    // command loops leave only after this lifecycle SHUTDOWN reaches them.
    // Notify every lane before destroying any endpoint or its transport.
    for (auto &wt : next_level_threads_)
        wt->shutdown_child();
    for (auto &wt : sub_threads_)
        wt->shutdown_child();
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

void LocalMailboxEndpoint::control_copy_to(const BufferDescriptor &dst, const BufferDescriptor &src, uint64_t nbytes) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_copy_request(mbox(), CTRL_COPY_TO, dst, src, nbytes);
    run_control_command("control_copy_to");
}

void LocalMailboxEndpoint::control_copy_from(
    const BufferDescriptor &dst, const BufferDescriptor &src, uint64_t nbytes
) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_copy_request(mbox(), CTRL_COPY_FROM, dst, src, nbytes);
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

void LocalMailboxEndpoint::control_worker_chip_region_create(const char *request_shm_name, const char *reply_shm_name) {
    if (!request_shm_name || !*request_shm_name || !reply_shm_name || !*reply_shm_name) {
        throw std::runtime_error("control_worker_chip_region_create: request and reply shm names must be non-empty");
    }
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    uint64_t sub_cmd = CTRL_WORKER_CHIP_REGION_CREATE;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    write_shm_name_pair(mbox(), request_shm_name, reply_shm_name);
    run_control_command("control_worker_chip_region_create");
}

void LocalMailboxEndpoint::control_worker_chip_region_release(uint64_t region_id) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_WORKER_CHIP_REGION_RELEASE, region_id);
    run_control_command("control_worker_chip_region_release");
}

uint64_t WorkerThread::control_malloc(size_t size) {
    if (!endpoint_) throw std::runtime_error("control_malloc: null endpoint");
    return endpoint_->control_malloc(size);
}

uint64_t WorkerThread::control_committed_device_memory() {
    if (!endpoint_) throw std::runtime_error("control_committed_device_memory: null endpoint");
    return endpoint_->control_committed_device_memory();
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

void WorkerThread::control_copy_to(const BufferDescriptor &dst, const BufferDescriptor &src, uint64_t nbytes) {
    if (!endpoint_) throw std::runtime_error("control_copy_to: null endpoint");
    endpoint_->control_copy_to(dst, src, nbytes);
}

void WorkerThread::control_copy_from(const BufferDescriptor &dst, const BufferDescriptor &src, uint64_t nbytes) {
    if (!endpoint_) throw std::runtime_error("control_copy_from: null endpoint");
    endpoint_->control_copy_from(dst, src, nbytes);
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

void WorkerThread::control_worker_chip_region_create(const char *request_shm_name, const char *reply_shm_name) {
    if (!endpoint_) throw std::runtime_error("control_worker_chip_region_create: null endpoint");
    endpoint_->control_worker_chip_region_create(request_shm_name, reply_shm_name);
}

void WorkerThread::control_worker_chip_region_release(uint64_t region_id) {
    if (!endpoint_) throw std::runtime_error("control_worker_chip_region_release: null endpoint");
    endpoint_->control_worker_chip_region_release(region_id);
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

void WorkerManager::control_worker_chip_region_create(
    int worker_id, const char *request_shm_name, const char *reply_shm_name
) {
    auto *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_worker_chip_region_create: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_worker_chip_region_create(request_shm_name, reply_shm_name);
}

void WorkerManager::control_worker_chip_region_release(int worker_id, uint64_t region_id) {
    auto *wt = get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_worker_chip_region_release: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_worker_chip_region_release(region_id);
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
