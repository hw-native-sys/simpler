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

#include <acl/acl.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "common/support/native_run_execution_peer.h"
#include "device_runner.h"

static int device_syncs = 0;
extern "C" aclError aclrtSynchronizeDeviceWithTimeout(int32_t) {
    ++device_syncs;
    return ACL_SUCCESS;
}

static int stream_tokens[8];
static size_t streams_created = 0;
static int event_tokens[8];
static size_t events_created = 0;
static int event_error = 0;
static rtStream_t failing_sync_stream = nullptr;
static int device_frees = 0;
static rtStream_t publication_stream = nullptr;
static std::vector<rtStream_t> queried, synchronized, recorded;
extern "C" rtError_t rtMalloc(void **out, uint64_t bytes, uint32_t, uint16_t) {
    *out = std::malloc(bytes);
    return *out == nullptr ? -1 : 0;
}
extern "C" rtError_t rtFree(void *ptr) {
    ++device_frees;
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t, const void *src, uint64_t bytes, rtMemcpyKind_t) {
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t rtStreamCreate(rtStream_t *out, int32_t) {
    assert(streams_created < 8);
    *out = &stream_tokens[streams_created++];
    return 0;
}
extern "C" rtError_t rtStreamDestroy(rtStream_t) { return 0; }
extern "C" rtError_t rtStreamQuery(rtStream_t stream) {
    queried.push_back(stream);
    return 0;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t) {
    synchronized.push_back(stream);
    return stream == failing_sync_stream ? -778 : 0;
}
extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *model) {
    publication_stream = stream;
    *status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    *model = nullptr;
    return 0;
}
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *out, uint32_t) {
    if (event_error) return event_error;
    assert(events_created < 8);
    *out = &event_tokens[events_created++];
    return 0;
}
extern "C" aclError aclrtDestroyEvent(aclrtEvent) { return 0; }
extern "C" aclError aclrtRecordEvent(aclrtEvent, aclrtStream stream) {
    recorded.push_back(stream);
    return 0;
}
extern "C" aclError aclrtStreamGetId(aclrtStream, int32_t *out) {
    *out = 1;
    return 0;
}

class ProbeRunner : public DeviceRunner {
public:
    using DeviceRunnerBase::arm_run_fence;
    using DeviceRunnerBase::commit_boundary_wait;
    using DeviceRunnerBase::open_own_boundary_wait;
    using DeviceRunnerBase::queued_boundary_wait_count;
    using DeviceRunnerBase::run_fence;
    using DeviceRunnerBase::slot_persistent_args;
    MemoryAllocator &allocator() { return mem_alloc_; }
    using DeviceRunnerBase::poll_run_fence;
    using DeviceRunnerBase::record_run_boundary;
    using DeviceRunnerBase::wait_run_fence;
    void set_owner_streams(rtStream_t cpu, rtStream_t core) {
        stream_aicpu_ = cpu;
        stream_aicore_ = core;
    }
};

static void check_execution_streams() {
    ProbeRunner runner;
    // A5 launches on the bootstrap pair; A2/A3 creates its retained run pair.
    runner.set_owner_streams(&stream_tokens[6], &stream_tokens[7]);
    Runtime runtime;
    runtime.dev = {};
    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    assert(runner.poll_run_fence(*prepared) == SIMPLER_NATIVE_RUN_POLL_ERROR);
    assert(runner.wait_run_fence(*prepared) == PTO_RUNTIME_ERR_INVALID_STATE);
    assert(queried.empty() && synchronized.empty());
    assert(prepared->kernel_args.prepare_runtime_args(runtime, allocator, slot) == 0);
    // The production launch publishes before arming. Arming fails before any
    // kernel can reach RTS, and the returned owner still names that publication.
    event_error = -777;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(outcome.rc == -777);
    assert(outcome.progress == LaunchProgress::NotStarted && !outcome.poisoned());
    assert(outcome.active == nullptr && outcome.prepared != nullptr);
    assert(outcome.prepared->kernel_args.runtime_args_published());
    const auto bound = outcome.prepared->streams;
    assert(bound.ready() && bound.aicpu != bound.aicore);
    assert(publication_stream == bound.aicpu);
    assert(streams_created == 0 ? bound.aicore == &stream_tokens[7] : bound.aicore == &stream_tokens[1]);

    // A later change to the runner's owner fields cannot redirect this run's
    // event records or its unfenced completion/error fallback.
    runner.set_owner_streams(&stream_tokens[4], &stream_tokens[5]);
    event_error = 0;
    DeviceRunnerBase::PreparedExecution moved(std::move(*outcome.prepared));
    assert(!outcome.prepared->streams.ready());
    assert(runner.arm_run_fence(moved) == 0);
    assert(runner.record_run_boundary(moved, RunCompletionFence::StreamRole::Aicore) == 0);
    assert(recorded == std::vector<rtStream_t>{bound.aicore});
    // Only one side is recorded: this is the partial-launch fallback.
    assert(runner.poll_run_fence(moved) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    assert((queried == std::vector<rtStream_t>{bound.aicpu, bound.aicore}));
    assert(runner.wait_run_fence(moved) == 0);
    assert((synchronized == std::vector<rtStream_t>{bound.aicpu, bound.aicore}));
    assert(runner.record_run_boundary(moved, RunCompletionFence::StreamRole::Aicpu) == 0);
    assert((recorded == std::vector<rtStream_t>{bound.aicore, bound.aicpu}));
    moved.kernel_args.release_run_view();
    assert(release_slot_persistent_args(slot, allocator) == 0);
    runner.set_owner_streams(nullptr, nullptr);
}

// The fixture enters cleanup with the committed wait a partial launch leaves:
// no AICPU boundary proves consumption, so discharge must synchronize the pair.
static void check_cleanup_wait_failure(SimplerExecutionMode mode, int failing_side, bool already_poisoned) {
    std::fprintf(
        stderr, "cleanup: mode=%d failing_side=%d already_poisoned=%d\n", mode, failing_side, already_poisoned
    );
    ProbeRunner runner;
    assert(runner.execution_mode_latch().latch(mode) == 0);
    Runtime runtime;
    runtime.dev = {};
    const NativeRunIdentity id{2, 1, 1, 0};
    DeviceRunnerBase::PreparedExecution prepared(id, runtime, CallConfig{}, 0);
    prepared.streams = {&stream_tokens[4], &stream_tokens[5]};
    prepared.resources_owned = true;
    auto &slot = runner.slot_persistent_args(0);
    assert(prepared.kernel_args.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
    void *allocation = slot.runtime_args;
    assert(allocation != nullptr);
    events_created = 0;
    assert(runner.arm_run_fence(prepared) == 0);
    assert(runner.record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore) == 0);
    void *boundary = nullptr;
    assert(runner.open_own_boundary_wait(prepared, &boundary) == 0);
    assert(boundary != nullptr);
    assert(runner.commit_boundary_wait(QueuedStreamWaits::Shape::Own, prepared) == 0);
    assert(runner.queued_boundary_wait_count() == 1);
    if (already_poisoned) {
        static_cast<DeviceRunnerBase &>(runner).recover_device_or_mark_unusable(-779);
    }
    const int syncs_before = device_syncs;
    const int frees_before = device_frees;
    synchronized.clear();
    failing_sync_stream = failing_side == 1 ? prepared.streams.aicpu :
                          failing_side == 2 ? prepared.streams.aicore :
                                              nullptr;
    runner.abandon_prepared_execution(prepared);
    const bool poisoned = already_poisoned || failing_side != 0;
    assert(runner.can_accept_run() == !poisoned);
    assert(!prepared.resources_owned);
    assert(prepared.kernel_args.args.runtime_args == nullptr);
    assert(device_frees == frees_before);
    assert(runner.queued_boundary_wait_count() == (poisoned ? 1u : 0u));
    assert(runner.run_fence(0).has_device_references(id) == poisoned);
    if (poisoned) {
        // Quarantine clears reusable slot bookkeeping, not the live allocation
        // or the completion fence still named by the committed stream wait.
        assert(slot.runtime_args == nullptr && slot.runtime_bytes == 0);
        assert(!prepared.aicore_retirement_attempted);
    } else {
        assert(slot.runtime_args == allocation);
        assert(release_slot_persistent_args(slot, runner.allocator()) == 0);
    }
    if (already_poisoned) {
        assert(synchronized.empty());
    } else if (failing_side == 1) {
        assert((synchronized == std::vector<rtStream_t>{prepared.streams.aicpu}));
    } else {
        assert((synchronized == std::vector<rtStream_t>{prepared.streams.aicpu, prepared.streams.aicore}));
    }
    assert(device_syncs - syncs_before == (!already_poisoned && poisoned && mode == SIMPLER_MODE_PROGRAM ? 1 : 0));
    const auto sync_count = synchronized.size();
    const int syncs_after = device_syncs;
    runner.abandon_prepared_execution(prepared);
    assert(synchronized.size() == sync_count && device_syncs == syncs_after);
    assert(runner.queued_boundary_wait_count() == (poisoned ? 1u : 0u));
    failing_sync_stream = nullptr;
    // This fixture's allocations are host-backed RTS stubs; no device can
    // still hold them after the cleanup assertions have finished.
    if (poisoned) assert(runner.allocator().free(allocation) == 0);
}

int main() {
    check_execution_streams();
    {
        DeviceRunner runner;
        assert(runner.execution_mode_latch().latch(SIMPLER_MODE_PROGRAM) == 0);
        static_cast<DeviceRunnerBase &>(runner).recover_device_or_mark_unusable(-91);
        assert(device_syncs == 1);
    }
    {
        DeviceRunner runner;
        assert(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL) == 0);
        KernelContextOps ops;
        ops.get_current_device = [](void *, int *out) noexcept {
            *out = 0;
            return 0;
        };
        ops.get_current_context = [](void *, void **out) noexcept {
            *out = &stream_tokens[0];
            return 0;
        };
        ops.create_hidden_stream = [](void *, void **out) noexcept {
            *out = &stream_tokens[0];
            return 0;
        };
        ops.destroy_hidden_stream = [](void *, void *) noexcept {
            return 0;
        };
        ops.create_event = [](void *, void **out) noexcept {
            *out = &event_tokens[0];
            return 0;
        };
        ops.destroy_event = [](void *, void *) noexcept {
            return 0;
        };
        auto &state = runner.kernel_execution_state();
        assert(state.initialize(0, ops) == 0);
        static_cast<DeviceRunnerBase &>(runner).recover_device_or_mark_unusable(-91);
        static_cast<DeviceRunnerBase &>(runner).recover_device_or_mark_unusable(-92);
        assert(state.phase() == KernelContextPhase::Poisoned);
        assert(state.last_runtime_error() == -91);
        assert(device_syncs == 1);
        assert(state.close() == 0);
    }
    for (const auto mode : {SIMPLER_MODE_PROGRAM, SIMPLER_MODE_KERNEL}) {
        check_cleanup_wait_failure(mode, 0, false);
        check_cleanup_wait_failure(mode, 1, false);
        check_cleanup_wait_failure(mode, 2, false);
        check_cleanup_wait_failure(mode, 0, true);
    }
    std::puts("PASS: per-run streams and cleanup respect device ownership and wait failures");
}
extern "C" void unified_log_warn(const char *, const char *, ...) {}
extern "C" void unified_log_info(const char *, const char *, ...) {}
extern "C" void unified_log_error(const char *, const char *, ...) {}
