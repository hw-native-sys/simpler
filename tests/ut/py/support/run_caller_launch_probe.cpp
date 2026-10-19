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
#include <string>
#include <vector>
#include <unordered_map>
#include "common/support/native_run_execution_peer.h"
#include "device_runner.h"
#include "common/host_span.h"

static int tokens[64], next_stream, next_event;
static int fail_at, submits, syncs, copies;
static int allocation_attempts, fail_allocation;
static std::unordered_map<void *, size_t> live_allocations;
static bool capture;
static int metadata_extensions, sync_error;
extern "C" const char *get_platform();
extern "C" bool publish_runtime_chip_swimlane_extensions(Runtime *) noexcept {
    ++metadata_extensions;
    return true;
}
static void *failed_destroy_event;
static void *caller = &tokens[60];
static std::vector<std::string> trace;
static std::vector<void *> stream_trace, event_trace;
static int step(const char *op, void *stream, void *event = nullptr) {
    trace.emplace_back(op);
    stream_trace.push_back(stream);
    event_trace.push_back(event);
    return ++submits == fail_at ? -777 : 0;
}
extern "C" rtError_t rtMalloc(void **out, uint64_t bytes, uint32_t, uint16_t) {
    ++allocation_attempts;
    if (allocation_attempts == fail_allocation) {
        *out = nullptr;
        return -777;
    }
    *out = std::malloc(bytes);
    if (*out != nullptr) live_allocations.emplace(*out, bytes);
    return *out ? 0 : -1;
}
extern "C" rtError_t rtFree(void *ptr) {
    assert(live_allocations.erase(ptr) == 1);
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMallocHost(void **out, uint64_t bytes, uint16_t) {
    *out = std::malloc(bytes);
    return *out ? 0 : -1;
}
extern "C" rtError_t rtFreeHost(void *ptr) {
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t, const void *src, uint64_t bytes, rtMemcpyKind_t) {
    ++copies;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t
rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t stream) {
    const int rc = step("publish", stream);
    return rc == 0 ? rtMemcpy(dst, capacity, src, bytes, kind) : rc;
}
extern "C" rtError_t rtStreamCreate(rtStream_t *out, int32_t) {
    *out = &tokens[next_stream++];
    return 0;
}
extern "C" rtError_t rtStreamDestroy(rtStream_t) { return 0; }
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *out, uint32_t) {
    *out = &tokens[16 + next_event++];
    return 0;
}
extern "C" aclError aclrtDestroyEvent(aclrtEvent event) {
    if (event == failed_destroy_event) {
        failed_destroy_event = nullptr;
        return -778;
    }
    return 0;
}
extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) { return step("record", stream, event); }
extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) { return step("wait", stream, event); }
extern "C" aclError aclrtStreamGetId(aclrtStream, int32_t *out) {
    *out = 1;
    return 0;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream, int32_t) {
    ++syncs;
    return sync_error;
}
extern "C" aclError aclrtSynchronizeEventWithTimeout(aclrtEvent, int32_t) { return sync_error; }
extern "C" aclError aclrtQueryEventStatus(aclrtEvent, aclrtEventRecordedStatus *status) {
    *status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    return 0;
}
extern "C" aclError aclrtSynchronizeDeviceWithTimeout(int32_t) {
    assert(false);
    return -1;
}
extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *status, aclmdlRI *model) {
    *status = capture ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    *model = nullptr;
    return 0;
}
// Kernel submission, recovery and scheduler-export leaves are test transports.
// The production runner performs publication, arming, launch/drain decisions,
// both callbacks, and every event/reference step.
int DeviceRunnerBase::launch_aicore_kernel(rtStream_t stream, const KernelArgs &) { return step("core", stream); }
int host::LoadAicpuOp::LaunchBuiltInOp(rtStream_t stream, void *, size_t, int, const std::string &) {
    return step("cpu", stream);
}
class ProbeRunner : public DeviceRunner {
public:
    // The model has no device faults to reset; Partial ownership is still real.
    void recover_device_or_mark_unusable(int) override {}
    void initialize() {
        device_id_ = 0;
        assert(execution_mode_latch().latch(SIMPLER_MODE_KERNEL) == 0);
        KernelContextOps ops;
        ops.get_current_device = [](void *, int *out) noexcept {
            *out = 0;
            return 0;
        };
        ops.get_current_context = [](void *, void **out) noexcept {
            *out = &tokens[61];
            return 0;
        };
        ops.create_hidden_stream = [](void *, void **out) noexcept {
            *out = &tokens[next_stream++];
            return 0;
        };
        ops.destroy_hidden_stream = [](void *, void *) noexcept {
            return 0;
        };
        ops.create_event = [](void *, void **out) noexcept {
            *out = &tokens[16 + next_event++];
            return 0;
        };
        ops.destroy_event = [](void *, void *) noexcept {
            return 0;
        };
        assert(kernel_execution_state().initialize(0, ops) == 0);
    }
    void check_failed_event_close() {
        Runtime runtime;
        runtime.dev = {};
        const NativeRunIdentity id{1, 1, 1, 0};
        PreparedExecution prepared(id, runtime, CallConfig{}, 0);
        prepared.caller_stream = caller;
        prepared.streams = borrowed_run_streams();
        assert(arm_run_fence(prepared) == 0);
        retire_run_fence(prepared);
        // The last created event is the caller tail, not a context event.
        failed_destroy_event = &tokens[16 + next_event - 1];
        assert(finalize_common() == -778);
        assert(!kernel_execution_state().has_live_resources());
        assert(kernel_resources_live());
        assert(finalize_common() == 0);
    }
    void finish() {
        // This probe owns only host-backed stub objects. No device task can
        // retain the partial launches deliberately left uncompleted here.
        (void)abandon_common_after_device_failure();
        kernel_execution_state().close();
        device_id_ = -1;
    }
};
static void check(int fault, bool capturing) {
    fail_at = fault;
    capture = capturing;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    prepared->caller_stream = caller;
    assert(prepared->kernel_args.prepare_runtime_args(runtime, allocator, slot) == 0);
    const int prior_copies = copies;
    assert(simpler_validate_execution_context(nullptr) == PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    assert(simpler_validate_execution_context(&runner) == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (capturing) {
        assert(outcome.rc == PTO_RUNTIME_ERR_UNSUPPORTED);
        assert(outcome.progress == LaunchProgress::NotStarted);
        assert(trace.empty() && copies == prior_copies);
    } else if (fault != 0) {
        assert(outcome.rc == -777);
        assert(outcome.progress == LaunchProgress::Partial);
        assert(submits == fault);
        assert(outcome.active != nullptr && outcome.prepared == nullptr);
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        const auto streams = outcome.active->prepared->streams;
        assert(
            (trace == std::vector<std::string>{
                          "record", "wait", "publish", "record", "wait", "record", "core", "record", "cpu", "wait",
                          "record", "record", "wait", "record"
                      })
        );
        assert(
            (stream_trace == std::vector<void *>{
                                 caller, streams.aicpu, streams.aicpu, streams.aicpu, streams.aicore, streams.aicore,
                                 streams.aicore, streams.aicore, streams.aicpu, streams.aicpu, streams.aicpu,
                                 streams.aicpu, caller, caller
                             })
        );
    }
    if (!capturing && fault == 0) {
        // Each wait names its own record, not just the expected stream.
        for (const auto edge : {std::pair{0, 1}, std::pair{3, 4}, std::pair{7, 9}, std::pair{11, 12}}) {
            assert(event_trace[edge.first] != nullptr);
            assert(event_trace[edge.first] == event_trace[edge.second]);
        }
        assert(event_trace[0] != event_trace[3]);
        assert(event_trace[7] != event_trace[11]);
        assert(event_trace[13] != event_trace[11]);
    }
    runner.finish();
    assert(release_slot_persistent_args(slot, allocator) == 0);
}
static void check_metadata_publication(int fault, bool drain_fails = false) {
    metadata_extensions = 0;
    fail_at = fault;
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    uint64_t target = 0, source = 37;
    HostMetadataRegion region{&target, &source, sizeof(source)};
    assert(slot.staged_launch_metadata.prepare(&region, 1) == 0);
    source = 99;
    NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    prepared->caller_stream = caller;
    assert(prepared->kernel_args.prepare_runtime_args(runtime, allocator, slot) == 0);
    assert(target == 0 && copies == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (fault != 0) {
        assert(outcome.rc == -777 && outcome.progress == LaunchProgress::Partial);
        assert(submits == fault && outcome.active != nullptr && outcome.prepared == nullptr);
        assert(trace.back() == "publish");
        for (const auto &step : trace)
            assert(step != "core" && step != "cpu");
        assert(target == (fault == 3 ? 0u : 37u));
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        assert(target == 37);
        assert(
            (trace == std::vector<std::string>{
                          "record", "wait", "publish", "publish", "record", "wait", "record", "core", "record", "cpu",
                          "wait", "record", "record", "wait", "record"
                      })
        );
        const auto streams = outcome.active->prepared->streams;
        for (size_t i : {1u, 2u, 3u, 4u})
            assert(stream_trace[i] == streams.aicpu);
        assert(stream_trace[5] == streams.aicore);
        assert(event_trace[4] == event_trace[5] && event_trace[4] != nullptr);
    }
    // A5's production backend is linked into this probe so its private runtime
    // exporter can be observed without replacing the drain's readiness check.
    if (std::strcmp(get_platform(), "a5") == 0) {
        outcome.active->prepared->resources_owned = true;
        outcome.active->prepared->dfx.chip_swimlane_level = ChipSwimlaneLevel::TASK_TIMING;
        sync_error = drain_fails ? -779 : 0;
        (void)runner.drain_execution(*outcome.active);
        sync_error = 0;
        assert(metadata_extensions == (fault == 0 ? 1 : 0));
    }
    runner.finish();
    assert(release_slot_persistent_args(slot, allocator) == 0);
}
static void check_arena_growth(SimplerExecutionMode mode) {
    const size_t before = live_allocations.size();
    {
        DeviceRunner runner;
        assert(runner.execution_mode_latch().latch(mode) == 0);
        assert(runner.setup_static_arena(0, 4096, 2048, 1024) == 0);
        assert(runner.setup_static_arena(1, 4096, 2048, 1024) == 0);
        void *original[] = {
            runner.acquire_pooled_gm_heap(0), runner.acquire_pooled_gm_sm(0), runner.acquire_pooled_runtime_arena(0)
        };
        void *other_bank[] = {
            runner.acquire_pooled_gm_heap(1), runner.acquire_pooled_gm_sm(1), runner.acquire_pooled_runtime_arena(1)
        };
        for (void *base : original)
            std::memset(base, 0xa5, 1024);
        for (void *base : other_bank)
            std::memset(base, 0x5a, 1024);
        const size_t committed = live_allocations.size();
        for (int failure = 1; failure <= 3; ++failure) {
            fail_allocation = allocation_attempts + failure;
            assert(runner.setup_static_arena(0, 8192, 4096, 2048) != 0);
            assert(allocation_attempts == fail_allocation);
            fail_allocation = 0;
            assert(live_allocations.size() == committed);
            assert(runner.acquire_pooled_gm_heap(0) == original[0]);
            assert(runner.acquire_pooled_gm_sm(0) == original[1]);
            assert(runner.acquire_pooled_runtime_arena(0) == original[2]);
            for (void *base : original)
                for (size_t i = 0; i < 1024; ++i)
                    assert(static_cast<unsigned char *>(base)[i] == 0xa5);
            const int attempts = allocation_attempts;
            assert(runner.setup_static_arena(0, 4096, 2048, 1024) == 0);
            assert(allocation_attempts == attempts);
        }
        assert(runner.setup_static_arena(0, 8192, 4096, 2048) == 0);
        void *grown[] = {
            runner.acquire_pooled_gm_heap(0), runner.acquire_pooled_gm_sm(0), runner.acquire_pooled_runtime_arena(0)
        };
        for (size_t i = 0; i < 3; ++i)
            assert(grown[i] != original[i]);
        const int attempts = allocation_attempts;
        assert(runner.setup_static_arena(0, 4096, 2048, 1024) == 0);
        assert(allocation_attempts == attempts);
        assert(runner.acquire_pooled_gm_heap(0) == grown[0]);
        assert(runner.acquire_pooled_gm_sm(0) == grown[1]);
        assert(runner.acquire_pooled_runtime_arena(0) == grown[2]);
        assert(runner.acquire_pooled_gm_heap(1) == other_bank[0]);
        assert(runner.acquire_pooled_gm_sm(1) == other_bank[1]);
        assert(runner.acquire_pooled_runtime_arena(1) == other_bank[2]);
        for (void *base : other_bank)
            for (size_t i = 0; i < 1024; ++i)
                assert(static_cast<unsigned char *>(base)[i] == 0x5a);
    }
    assert(live_allocations.size() == before);
}
int main() {
    check_metadata_publication(0);
    check_metadata_publication(3);  // Metadata enqueue failure.
    check_metadata_publication(4);  // Descriptor enqueue after accepted metadata.

    if (std::strcmp(get_platform(), "a5") == 0) {
        check_metadata_publication(0, true);
        check_metadata_publication(3, true);
        check_metadata_publication(4, true);
    }
    check_arena_growth(SIMPLER_MODE_PROGRAM);
    check_arena_growth(SIMPLER_MODE_KERNEL);
    check(0, false);
    // Positions 6 and 11 are optional timing marks; their failure is deliberately non-fatal.
    for (int fault : {1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14})
        check(fault, false);
    check(0, true);
    next_stream = next_event = 0;
    ProbeRunner runner;
    runner.initialize();
    const int prior_copies = copies;
    assert(simpler_validate_caller_stream(nullptr, caller) == PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    assert(simpler_validate_caller_stream(&runner, nullptr) == PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    assert(simpler_validate_caller_stream(&runner, caller) == PTO_RUNTIME_ERR_UNSUPPORTED);
    capture = false;
    assert(simpler_validate_caller_stream(&runner, caller) == 0);
    assert(copies == prior_copies);
    runner.check_failed_event_close();
    runner.finish();
    std::puts("PASS: real launch/drain ordering, publication failures, and pre-publication capture refusal");
}
extern "C" void unified_log_warn(const char *, const char *, ...) {}
extern "C" void unified_log_info(const char *, const char *, ...) {}
extern "C" void unified_log_error(const char *, const char *, ...) {}
extern "C" void unified_log_debug(const char *, const char *, ...) {}
extern "C" void unified_log_host_span(const SimplerHostSpan *) {}
