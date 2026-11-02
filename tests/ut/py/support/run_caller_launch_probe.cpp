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
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
#include "native_run_context.h"
#include "task_args.h"
#include "common/support/native_run_execution_peer.h"
#include "device_runner.h"
#include "common/host_span.h"
#include "common/device_phase.h"

static int tokens[64], next_stream, next_event;
static int fail_at, submits, syncs, copies;
static int allocation_attempts, fail_allocation;
static std::unordered_map<void *, size_t> live_allocations;
static bool capture;
static int metadata_extensions, sync_error;
static bool owned_close_active, completion_events_ready = true;
static int force_resets, stale_resource_calls;
static bool fail_force_reset, fail_reset_probe;
static bool acl_initialized, fail_acl_finalize_once;
static int acl_finalize_calls;
static std::vector<void *> reset_allocations, reset_streams, reset_events;
static std::vector<void *> destroyed_streams, observed_streams;
static void *timing_target;
static const void *timing_source;
static size_t timing_bytes;
static void *result_target;
static const void *result_source;
bool device_phase_capture_enabled() { return true; }
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
static bool old_device_address(const void *address) {
    const auto target = reinterpret_cast<uintptr_t>(address);
    for (void *base : reset_allocations) {
        const auto found = live_allocations.find(base);
        if (found != live_allocations.end()) {
            const auto begin = reinterpret_cast<uintptr_t>(base);
            if (target >= begin && target - begin < found->second) return true;
        }
    }
    return false;
}
static bool stale_stream(void *stream) {
    return std::find(reset_streams.begin(), reset_streams.end(), stream) != reset_streams.end();
}
static bool stale_event(void *event) {
    return std::find(reset_events.begin(), reset_events.end(), event) != reset_events.end();
}
extern "C" rtError_t rtMalloc(void **out, uint64_t bytes, uint32_t, uint16_t) {
    ++allocation_attempts;
    if (allocation_attempts == fail_allocation) {
        *out = nullptr;
        return -777;
    }
    *out = std::malloc(bytes);
    if (*out != nullptr) {
        live_allocations.emplace(*out, bytes);
        std::memset(*out, 0xa5, bytes);
    }
    return *out ? 0 : -1;
}
extern "C" rtError_t rtFree(void *ptr) {
    if (std::find(reset_allocations.begin(), reset_allocations.end(), ptr) != reset_allocations.end()) {
        ++stale_resource_calls;
        return -983;
    }
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
    if (old_device_address(dst) || old_device_address(src)) {
        ++stale_resource_calls;
        return -983;
    }
    ++copies;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t
rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t stream) {
    const int rc = step("publish", stream);
    if (rc == 0 && dst == result_target) {
        assert(bytes == sizeof(uint64_t));
        result_source = src;
        return 0;
    }
    if (rc == 0 && dst == timing_target) {
        timing_source = src;
        timing_bytes = bytes;
        return 0;
    }
    return rc == 0 ? rtMemcpy(dst, capacity, src, bytes, kind) : rc;
}
extern "C" rtError_t rtStreamCreate(rtStream_t *out, int32_t) {
    *out = &tokens[next_stream++];
    return 0;
}
extern "C" rtError_t rtStreamDestroy(rtStream_t stream) {
    if (std::find(reset_streams.begin(), reset_streams.end(), stream) != reset_streams.end()) {
        ++stale_resource_calls;
        return -983;
    }
    assert(stream != caller);
    destroyed_streams.push_back(stream);
    return 0;
}
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *out, uint32_t) {
    *out = &tokens[16 + next_event++];
    return 0;
}
extern "C" aclError aclrtDestroyEvent(aclrtEvent event) {
    if (stale_event(event)) {
        ++stale_resource_calls;
        return -983;
    }
    if (event == failed_destroy_event) {
        failed_destroy_event = nullptr;
        return -778;
    }
    return 0;
}
extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    if (stale_event(event) || stale_stream(stream)) {
        ++stale_resource_calls;
        return -983;
    }
    return step("record", stream, event);
}
extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    if (stale_event(event) || stale_stream(stream)) {
        ++stale_resource_calls;
        return -983;
    }
    return step("wait", stream, event);
}
extern "C" aclError aclrtStreamGetId(aclrtStream, int32_t *out) {
    *out = 1;
    return 0;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t) {
    if (stale_stream(stream)) {
        ++stale_resource_calls;
        return -983;
    }
    observed_streams.push_back(stream);
    ++syncs;
    return sync_error;
}
extern "C" aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t) {
    if (stale_event(event)) {
        ++stale_resource_calls;
        return -983;
    }
    return sync_error;
}
extern "C" aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
    if (stale_event(event)) {
        ++stale_resource_calls;
        return -983;
    }
    *status = completion_events_ready ? ACL_EVENT_RECORDED_STATUS_COMPLETE : ACL_EVENT_RECORDED_STATUS_NOT_READY;
    return 0;
}
extern "C" aclError aclrtSynchronizeDeviceWithTimeout(int32_t) {
    assert(owned_close_active);
    return sync_error;
}
extern "C" rtError_t rtSetDevice(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" rtError_t rtDeviceReset(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" aclError aclInit(const char *) {
    assert(owned_close_active);
    if (acl_initialized) return 100002;
    acl_initialized = true;
    return 0;
}
extern "C" aclError aclFinalize() {
    assert(owned_close_active);
    ++acl_finalize_calls;
    if (fail_acl_finalize_once) {
        fail_acl_finalize_once = false;
        return -982;
    }
    acl_initialized = false;
    return 0;
}
extern "C" aclError aclrtSetDevice(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" aclError aclrtResetDevice(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" aclError aclrtResetDeviceForce(int32_t) {
    assert(owned_close_active);
    ++force_resets;
    if (fail_force_reset) return -980;
    for (const auto &allocation : live_allocations)
        reset_allocations.push_back(allocation.first);
    for (int index = 0; index != next_stream; ++index)
        reset_streams.push_back(&tokens[index]);
    for (int index = 0; index != next_event; ++index)
        reset_events.push_back(&tokens[16 + index]);
    return 0;
}
extern "C" aclError aclrtCreateStream(aclrtStream *stream) {
    if (owned_close_active && fail_reset_probe && force_resets != 0) return -981;
    return rtStreamCreate(stream, 0);
}
extern "C" aclError aclrtDestroyStream(aclrtStream stream) { return rtStreamDestroy(stream); }
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
static void initialize_borrowed_context(DeviceRunnerBase &runner) {
    assert(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL) == 0);
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
    assert(runner.kernel_execution_state().initialize(0, ops) == 0);
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
    using DeviceRunnerBase::ensure_device_run_result_region;
    using DeviceRunnerBase::ensure_device_wall_buffer;
    bool result_initialized() const { return device_run_result_initialized_[0]; }
    void release_result() {
        free_tensor(device_run_result_dev_ptrs_[0]);
        device_run_result_dev_ptrs_[0] = nullptr;
    }
    using DeviceRunnerBase::discharge_boundary_waits;
    using DeviceRunnerBase::slot_persistent_args;
    int finish_common() { return finalize_common(); }
    void *whole_event(const NativeRunIdentity &identity) {
        return run_fence(identity.pipeline_slot).boundary_event(identity, RunCompletionFence::StreamRole::Aicpu);
    }
    int retire_boundary(const NativeRunIdentity &identity) {
        return run_fence(identity.pipeline_slot).retire(identity);
    }
    void prepare_diagnostics(Runtime &runtime, int level, int pmu_event_type, int dump_level = 0) {
        max_block_dim_ = 1;
        CallConfig config;
        config.aicpu_thread_num = 2;
        config.enable_chip_swimlane = level;
        config.enable_pmu = pmu_event_type;
        config.enable_dump_args = dump_level;
        assert(prepare_launch_shape(runtime, config) == 0);
        PreparedExecution execution(NativeRunIdentity{}, runtime, config, 0);
        assert(
            runtime.dev.pmu_event_type ==
            (execution.dfx.pmu_enabled ? static_cast<uint32_t>(execution.dfx.pmu_event_type) : 0)
        );
    }
    bool timing_armed() const { return device_timing_armed_[0]; }
    void consume_timing() { device_timing_armed_[0] = false; }
    void release_timing() {
        free_tensor(device_wall_dev_ptrs_[0]);
        device_wall_dev_ptrs_[0] = nullptr;
    }
    MemoryAllocator &allocator() { return mem_alloc_; }
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
        void *control_stream = kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu);
        assert(finalize_common() == -778);
        assert(kernel_execution_state().has_live_resources());
        assert(kernel_execution_state().phase() == KernelContextPhase::Closing);
        assert(!kernel_execution_state().accepts_dispatch());
        assert(kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu) == control_stream);
        assert(kernel_resources_live());
        assert(finalize_common() == 0);
        assert(!kernel_execution_state().has_live_resources());
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
        assert(streams.aicpu == caller && streams.aicore != caller);
        assert(
            (trace ==
             std::vector<std::string>{
                 "publish", "record", "wait", "record", "core", "record", "cpu", "wait", "record", "record", "record"
             })
        );
        assert(
            (stream_trace == std::vector<void *>{
                                 caller, caller, streams.aicore, streams.aicore, streams.aicore, streams.aicore, caller,
                                 caller, caller, caller, caller
                             })
        );
    }
    if (!capturing && fault == 0) {
        // Each wait names its own record, not just the expected stream.
        for (const auto edge : {std::pair{1, 2}, std::pair{5, 7}}) {
            assert(event_trace[edge.first] != nullptr);
            assert(event_trace[edge.first] == event_trace[edge.second]);
        }
        assert(event_trace[1] != event_trace[5]);
        assert(event_trace[10] != event_trace[9]);
    }
    runner.finish();
    assert(release_slot_persistent_args(slot, allocator) == 0);
}
static void check_joined_callers(void *first_caller, void *second_caller, bool fail_proof = false) {
    fail_at = 0;
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    ProbeRunner runner;
    runner.initialize();
    Runtime first_runtime, second_runtime;
    first_runtime.dev = second_runtime.dev = {};
    MemoryAllocator allocator;
    SlotPersistentArgs first_slot, second_slot;
    const NativeRunIdentity first_id{1, 1, 1, 0}, second_id{2, 1, 2, 1};
    auto first = std::make_unique<DeviceRunnerBase::PreparedExecution>(first_id, first_runtime, CallConfig{}, 0);
    first->caller_stream = first_caller;
    first->joinable_boundary = true;
    assert(first->kernel_args.prepare_runtime_args(first_runtime, allocator, first_slot) == 0);
    auto preceding = runner.launch_execution(std::move(first), NativeRunExecutionTestPeer::mint(first_id));
    assert(preceding.rc == 0 && preceding.active != nullptr);
    void *whole_done = runner.whole_event(first_id);
    assert(whole_done != nullptr);
    auto second = std::make_unique<DeviceRunnerBase::PreparedExecution>(second_id, second_runtime, CallConfig{}, 1);
    second->caller_stream = second_caller;
    second->joinable_boundary = true;
    second->join = NativeRunJoin{preceding.active->prepared.get(), first_id};
    assert(second->kernel_args.prepare_runtime_args(second_runtime, allocator, second_slot) == 0);
    submits = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    fail_at = fail_proof ? 2 : 0;
    auto following = runner.launch_execution(std::move(second), NativeRunExecutionTestPeer::mint(second_id));
    assert(syncs == 0);
    assert(following.active != nullptr);
    const auto streams = following.active->prepared->streams;
    if (second_caller != nullptr) assert(streams.aicpu == second_caller);
    assert(trace[0] == "wait" && trace[1] == "record");
    assert(stream_trace[0] == streams.aicpu && stream_trace[1] == streams.aicpu);
    assert(event_trace[0] == whole_done);
    if (fail_proof) {
        assert(following.rc == -777 && following.progress == LaunchProgress::Partial);
        assert(trace.size() == 2);
        const size_t pending = runner.queued_boundary_wait_count();
        assert(pending == 2);
        for (int retry = 0; retry != 2; ++retry) {
            assert(runner.discharge_boundary_waits(*preceding.active->prepared, true) != 0);
            assert(runner.queued_boundary_wait_count() == 1);
            assert(runner.finish_common() != 0);
            assert(runner.queued_boundary_wait_count() == 1);
        }
    } else {
        assert(following.rc == 0 && following.progress == LaunchProgress::Complete);
        assert(trace[2] == "publish" && stream_trace[2] == streams.aicpu);
        assert(trace[3] == "record" && stream_trace[3] == streams.aicpu);
        assert(trace[4] == "wait" && stream_trace[4] == streams.aicore);
        assert(event_trace[3] == event_trace[4]);
        assert(runner.retire_boundary(first_id) == PTO_RUNTIME_ERR_INTERNAL);
        assert(runner.discharge_boundary_waits(*preceding.active->prepared, true) == 0);
        assert(runner.retire_boundary(first_id) == 0);
    }
    preceding.active->prepared->kernel_args.release_run_view();
    following.active->prepared->kernel_args.release_run_view();
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
        assert(target == (fault == 1 ? 0u : 37u));
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        assert(target == 37);
        assert(
            (trace == std::vector<std::string>{
                          "publish", "publish", "record", "wait", "record", "core", "record", "cpu", "wait", "record",
                          "record", "record"
                      })
        );
        const auto streams = outcome.active->prepared->streams;
        for (size_t i : {0u, 1u, 2u})
            assert(stream_trace[i] == streams.aicpu);
        assert(stream_trace[3] == streams.aicore);
        assert(event_trace[2] == event_trace[3] && event_trace[2] != nullptr);
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
static void check_level_publication(bool entry_route) {
    unsetenv("SIMPLER_PMU_EVENT_TYPE");
    next_stream = next_event = submits = copies = 0;
    fail_at = 0;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    auto &slot = runner.slot_persistent_args(0);
    for (const auto [level, pmu, dump_level] :
         {std::tuple{4, 0, 0},
          {1, 0, 0},
          {0, 0, 0},
          {3, 0, 0},
          {2, 0, 0},
          {2, 2, 0},
          {2, 4, 0},
          {2, 0, 0},
          {2, 8, 0},
          {0, 0, 3},
          {0, 0, 1},
          {0, 0, 0},
          {0, 0, 2},
          {0, 0, 3}}) {
        runner.prepare_diagnostics(runtime, level, pmu, dump_level);
        assert(runtime.dev.dump_args_level == static_cast<uint32_t>(dump_level));
        assert(runtime.dev.chip_swimlane_level == static_cast<uint32_t>(level));
        KernelArgsHelper helper;
        assert(helper.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
        runtime.dev.chip_swimlane_level = 99;
        runtime.dev.pmu_event_type = 99;
        runtime.dev.dump_args_level = 99;
        LaunchProgressSink sink;
        assert(helper.publish_runtime_args(entry_route, caller, sink) == 0);
        assert(sink.submitted());
        uint32_t published;
        const auto *device = reinterpret_cast<const std::byte *>(slot.runtime_args);
        std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, chip_swimlane_level), sizeof(published));
        assert(published == static_cast<uint32_t>(level));
        std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, pmu_event_type), sizeof(published));
        assert(published == static_cast<uint32_t>(pmu));
        std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, dump_args_level), sizeof(published));
        assert(published == static_cast<uint32_t>(dump_level));
    }
    assert(setenv("SIMPLER_PMU_EVENT_TYPE", "4", 1) == 0);
    runner.prepare_diagnostics(runtime, 0, 2);
    assert(runtime.dev.pmu_event_type == 4);
    KernelArgsHelper overridden;
    assert(overridden.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
    assert(setenv("SIMPLER_PMU_EVENT_TYPE", "1", 1) == 0);
    LaunchProgressSink sink;
    assert(overridden.publish_runtime_args(entry_route, caller, sink) == 0);
    uint32_t published;
    const auto *device = reinterpret_cast<const std::byte *>(slot.runtime_args);
    std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, pmu_event_type), sizeof(published));
    assert(published == 4);
    unsetenv("SIMPLER_PMU_EVENT_TYPE");
    assert(release_slot_persistent_args(slot, runner.allocator()) == 0);
    runner.finish();
}
static void check_result_prepare_refusal() {
    next_stream = next_event = 0;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    const NativeRunIdentity id{1, 1, 1, 0};
    DeviceRunnerBase::PreparedExecution prepared(id, runtime, CallConfig{}, 0);
    const int before = allocation_attempts;
    assert(runner.ensure_device_run_result_region(0, 0, prepared.kernel_args) != 0);
    assert(allocation_attempts == before);
    fail_allocation = allocation_attempts + 1;
    assert(runner.ensure_device_run_result_region(0, 1, prepared.kernel_args) != 0);
    assert(prepared.kernel_args.args.run_result_data_base == 0);
    fail_allocation = 0;
    for (int retry = 0; retry < 2; ++retry) {
        prepared.resources_owned = true;
        assert(runner.ensure_device_run_result_region(0, 1, prepared.kernel_args) == 0);
        assert(!runner.result_initialized());
        assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == sizeof(uint64_t));
        runner.abandon_prepared_execution(prepared);
        assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == 0);
    }
    runner.release_result();
    runner.finish();
}
static void check_result_publication(bool has_caller, int failure_step) {
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    fail_at = failure_step == 0 ? 0 : failure_step;
    result_source = nullptr;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    const NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    if (has_caller) prepared->caller_stream = caller;
    auto &slot = runner.slot_persistent_args(0);
    assert(runner.ensure_device_run_result_region(0, id.run_epoch, prepared->kernel_args) == 0);
    result_target = reinterpret_cast<void *>(prepared->kernel_args.args.run_result_data_base);
    assert(result_target != nullptr);
    assert(!runner.result_initialized() && copies == 0);
    assert(slot.staged_launch_metadata.size() == sizeof(uint64_t));
    // Allocator bytes can resemble a fully valid success for the requested epoch.
    DeviceRunResultRegion forged{};
    forged.published = id.run_epoch;
    forged.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    assert(device_run_result_terminal(forged, id.run_epoch).state == DeviceRunTerminalState::Succeeded);
    std::memcpy(result_target, &forged, sizeof(forged));
    assert(runner.read_device_run_result(0, id.run_epoch) == 0);
    assert(runner.device_run_result_read_status(0, id.run_epoch) == RunRecordRead::NotAttempted);
    assert(runner.device_run_terminal(0, id.run_epoch).state == DeviceRunTerminalState::Undecided);
    assert(copies == 0);
    assert(prepared->kernel_args.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (failure_step != 0) {
        assert(outcome.rc == -777 && outcome.progress == LaunchProgress::Partial);
        assert(runner.result_initialized() == (failure_step == 3));
        for (const auto &op : trace)
            assert(op != "core" && op != "cpu");
        if (!runner.result_initialized()) {
            const int before = copies;
            assert(runner.read_device_run_result(0, id.run_epoch) == 0);
            assert(copies == before);
            assert(runner.device_run_result_read_status(0, id.run_epoch) == RunRecordRead::NotAttempted);
        }
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        assert(runner.result_initialized());
        const size_t publication = 0;
        assert(trace[publication] == "publish" && trace[publication + 1] == "publish");
        assert(stream_trace[publication] == outcome.active->prepared->streams.aicpu);
        assert(stream_trace[publication + 1] == outcome.active->prepared->streams.aicpu);
    }
    if (outcome.active != nullptr) outcome.active->prepared->kernel_args.release_run_view();
    // Complete the accepted DMA after the per-run snapshot has been released.
    if (result_source != nullptr) {
        uint64_t before;
        std::memcpy(&before, result_target, sizeof(before));
        assert(before == id.run_epoch);
        std::memcpy(result_target, result_source, sizeof(uint64_t));
        uint64_t after;
        std::memcpy(&after, result_target, sizeof(after));
        assert(after == 0);
    }
    if (failure_step == 0) {
        assert(runner.read_device_run_result(0, id.run_epoch) == 0);
        assert(runner.device_run_result_read_status(0, id.run_epoch) == RunRecordRead::Ok);
        assert(runner.device_run_terminal(0, id.run_epoch).state == DeviceRunTerminalState::Undecided);
        const int before = allocation_attempts;
        KernelArgsHelper warm;
        assert(runner.ensure_device_run_result_region(0, 2, warm) == 0);
        assert(allocation_attempts == before);
        assert(warm.args.run_result_data_base == reinterpret_cast<uint64_t>(result_target));
        assert(slot.staged_launch_metadata.size() == 0);
    }
    result_target = nullptr;
    runner.release_result();
    runner.finish();
}
static void check_timing_prepare_refusal() {
    next_stream = next_event = 0;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    const NativeRunIdentity id{1, 1, 1, 0};
    DeviceRunnerBase::PreparedExecution prepared(id, runtime, CallConfig{}, 0);
    fail_allocation = allocation_attempts + 1;
    assert(runner.ensure_device_wall_buffer(0, prepared.kernel_args) == 0);
    assert(prepared.kernel_args.args.device_wall_data_base == 0);
    assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == 0);
    fail_allocation = 0;
    prepared.resources_owned = true;
    assert(runner.ensure_device_wall_buffer(0, prepared.kernel_args) == 0);
    const size_t reset_bytes = runner.slot_persistent_args(0).staged_launch_metadata.size();
    assert(reset_bytes != 0 && !runner.timing_armed());
    runner.abandon_prepared_execution(prepared);
    assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == 0);
    assert(!runner.timing_armed());
    prepared.resources_owned = true;
    assert(runner.ensure_device_wall_buffer(0, prepared.kernel_args) == 0);
    assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == reset_bytes);
    runner.abandon_prepared_execution(prepared);
    runner.release_timing();
    runner.finish();
}
static void check_timing_publication(int fault) {
    fail_at = fault;
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    timing_source = nullptr;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    const NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    prepared->caller_stream = caller;
    auto &slot = runner.slot_persistent_args(0);
    uint64_t scheduler_target = 0, scheduler_source = 37;
    const HostMetadataRegion region{&scheduler_target, &scheduler_source, sizeof(scheduler_source)};
    assert(slot.staged_launch_metadata.prepare(&region, 1) == 0);
    assert(runner.ensure_device_wall_buffer(0, prepared->kernel_args) == 0);
    timing_target = reinterpret_cast<void *>(prepared->kernel_args.args.device_wall_data_base);
    assert(timing_target != nullptr);
    constexpr int threads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    constexpr size_t bytes = device_phase_buffer_bytes(threads);
    assert(slot.staged_launch_metadata.size() == sizeof(scheduler_source) + bytes);
    std::memset(timing_target, 0xa5, bytes);
    assert(copies == 0 && !runner.timing_armed());
    assert(prepared->kernel_args.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (fault) {
        assert(outcome.rc == -777 && outcome.progress == LaunchProgress::Partial);
        assert(!runner.timing_armed());
        for (const auto &op : trace)
            assert(op != "core" && op != "cpu");
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        assert(runner.timing_armed());
        assert(scheduler_target == scheduler_source);
        assert(trace[0] == "publish" && trace[1] == "publish" && trace[2] == "publish");
        for (int i : {0, 1, 2})
            assert(stream_trace[i] == outcome.active->prepared->streams.aicpu);
        // The DMA reader outlives the per-run host snapshot, with no intervening wait.
        outcome.active->prepared->kernel_args.release_run_view();
        assert(timing_source != nullptr && timing_bytes == bytes);
        assert(static_cast<unsigned char *>(timing_target)[0] == 0xa5);
        std::memcpy(timing_target, timing_source, timing_bytes);
        DevicePhaseBufferStorage<threads> expected{};
        reset_device_phase_buffer(&expected, threads);
        assert(std::memcmp(timing_target, &expected, bytes) == 0);
        KernelArgsHelper unread;
        assert(runner.ensure_device_wall_buffer(0, unread) == 0);
        assert(unread.args.device_wall_data_base == 0);
        assert(slot.staged_launch_metadata.size() == 0);
    }
    runner.consume_timing();
    runner.release_timing();
    timing_target = nullptr;
    runner.finish();
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
struct ChipTaskArgs;
struct OrchContext;

class FatalCollectorProbeRunner : public DeviceRunner {
public:
    void initialize(bool borrowed) {
        if (borrowed) initialize_borrowed_context(*this);
        device_id_ = 0;
        timeout_configuration_rc_ = 0;
        max_block_dim_ = block_dim_ = 1;
        if (!borrowed) {
            assert(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM) == 0);
            // Program owns its persistent pair, including A5's fixed pair.
            assert(rtStreamCreate(&stream_aicpu_, 0) == 0);
            assert(rtStreamCreate(&stream_aicore_, 0) == 0);
        }
        assert(install_staged_workspace() == 0);
        assert(set_run_resource_grant(2));
        enable_automatic_admission();
        struct EntryPoints {
            void (*entry)(const ChipTaskArgs &);
            void (*bind)(OrchContext *);
            void (*finish)();
        };
        static EntryPoints empty_orchestration{[](const ChipTaskArgs &) {}, [](OrchContext *) {}, nullptr};
        CallableState callable;
        callable.hash = callable.chip_buffer_hash = 1;
        callable.signature = {ArgDirection::INOUT};
        if (std::strcmp(SIMPLER_RUNTIME_NAME, "host_build_graph") == 0) {
            callable.host_orch_func_ptr = &empty_orchestration;
        }
        callables_[1] = std::move(callable);
        chip_callable_buffers_[1] = ChipCallableBuffer{};
    }
    void initialize_scope_stats() {
        assert(
            scope_stats_collector_.init(
                1,
                [this](size_t bytes) {
                    return mem_alloc_.alloc(bytes);
                },
                nullptr,
                [this](void *ptr) {
                    return mem_alloc_.free(ptr);
                },
                device_id_
            ) == 0
        );
        assert(scope_stats_collector_.get_scope_stats_shm_device_ptr() != nullptr);
    }
    int fill_persistent_arch_fields(KernelArgs *, uint64_t) override { return 0; }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        *out = std::make_unique<PreparedExecution>(identity, runtime, config, slot);
        (*out)->resources_owned = true;
        int rc = ensure_device_run_result_region(slot, identity.run_epoch, (*out)->kernel_args);
        return rc == 0 ? (*out)->kernel_args.prepare_runtime_args(runtime, mem_alloc_, slot_persistent_args(slot)) : rc;
    }
};

static void check_standalone_collector_close(const std::string &scenario) {
    owned_close_active = true;
    completion_events_ready = false;
    fail_force_reset = scenario == "reset-failure";
    fail_reset_probe = scenario == "probe-failure";
    const bool borrowed = scenario == "borrowed";
    const bool confirmed = !borrowed && !fail_force_reset && !fail_reset_probe;
    const bool retry_acl_close = scenario == "acl-finalize-retry";
    acl_initialized = false;
    fail_acl_finalize_once = retry_acl_close;
    acl_finalize_calls = 0;
    fail_at = force_resets = stale_resource_calls = 0;
    sync_error = -739;
    next_stream = next_event = submits = syncs = copies = 0;
    reset_allocations.clear();
    reset_streams.clear();
    reset_events.clear();
    auto *runner = new FatalCollectorProbeRunner;
    runner->initialize(borrowed);
    if (retry_acl_close) assert(runner->ensure_acl_ready(0) == 0);
    runner->initialize_scope_stats();
    void *storage = std::aligned_alloc(get_runtime_alignment(), get_runtime_size());
    assert(storage != nullptr);
    std::memset(storage, 0, get_runtime_size());
    std::vector<uint8_t> output(16, 0x71);
    const uint32_t shape[]{16};
    ChipTensor tensor;
    tensor.init_external(
        output.data(), output.size(), shape, 1, DataType::UINT8, borrowed ? AddressSpace::DEVICE : AddressSpace::HOST
    );
    ChipStorageTaskArgs args;
    args.add_tensor(tensor);
    CallConfig config;
    config.aicpu_thread_num = 2;
    for (int ring = 0; ring != RUNTIME_ENV_RING_COUNT; ++ring) {
        config.runtime_env.ring_task_window[ring] = 4;
        config.runtime_env.ring_heap[ring] = 1024;
        config.runtime_env.ring_dep_pool[ring] = 4;
    }
    const NativeRunDescriptor descriptor{0, 0, 1, 1, 1, 1};
    assert(simpler_prepare_run(runner, storage, 1, &args, &config, &descriptor) == 0);
    assert(
        (borrowed ? simpler_launch_run_on_stream(runner, storage, caller, nullptr) :
                    simpler_launch_run(runner, storage)) == 0
    );
    const int original_error = simpler_wait_run(runner, storage);
    assert(original_error == -739);
    // Finalize the carrier before context teardown; outstanding native Runs
    // block context close.
    const int run_finalize_error = simpler_finalize_run(runner, storage);
    assert(run_finalize_error == original_error);
    assert(!runner->native_runs_outstanding());
    const uint64_t before = runner->device_health().generation();
    const auto allocations_before_close = live_allocations;
    int rc = finalize_device(runner);
    if (!confirmed) assert(live_allocations == allocations_before_close);
    if (retry_acl_close) {
        assert(rc == -982 && runner->device_id() == 0);
        assert(runner->device_health().generation() != before);
        assert(force_resets == 1 && acl_finalize_calls == 1 && acl_initialized);
        const uint64_t reset_generation = runner->device_health().generation();
        const auto operations = std::make_tuple(submits, syncs, copies, stale_resource_calls);
        rc = finalize_device(runner);
        assert(rc == 0 && force_resets == 1 && acl_finalize_calls == 2 && !acl_initialized);
        assert(runner->device_health().generation() == reset_generation);
        assert(operations == std::make_tuple(submits, syncs, copies, stale_resource_calls));
        std::fprintf(
            stderr, "standalone collector acl-finalize-retry: confirmed reset reused; original error=%d\n",
            original_error
        );
    }
    assert((rc == 0) == confirmed);
    assert((runner->device_health().generation() != before) == confirmed);
    assert(stale_resource_calls == 0);
    assert(output == std::vector<uint8_t>(output.size(), 0x71));
    if (borrowed) {
        assert(force_resets == 0 && runner->device_id() == 0);
        assert(!live_allocations.empty());
        std::free(storage);
        // The borrowed device has no reset proof; its context and backing stay pinned.
        std::fprintf(stderr, "standalone collector borrowed: context retained, no reset\n");
        return;
    }
    assert(force_resets != 0);
    if (!confirmed) {
        assert(runner->device_id() == 0);
        std::fprintf(stderr, "standalone collector %s: unconfirmed context retained\n", scenario.c_str());
        const int failed_reset_count = force_resets;
        const auto before_retry = std::make_tuple(submits, syncs, copies, stale_resource_calls);
        fail_force_reset = fail_reset_probe = false;
        assert(finalize_device(runner) == 0);
        assert(force_resets == failed_reset_count + 1);
        assert(runner->device_health().generation() != before);
        assert(stale_resource_calls == 0);
        assert(before_retry == std::make_tuple(submits, syncs, copies, stale_resource_calls));
        assert(output == std::vector<uint8_t>(output.size(), 0x71));
        std::fprintf(
            stderr, "standalone collector %s: same-runner retry confirmed; original error=%d\n", scenario.c_str(),
            original_error
        );
    }
    delete runner;
    assert(stale_resource_calls == 0);
    std::free(storage);
    for (const auto &allocation : live_allocations)
        std::free(allocation.first);
    live_allocations.clear();
    reset_allocations.clear();
    reset_streams.clear();
    reset_events.clear();
    completion_events_ready = true;
    sync_error = 0;
    owned_close_active = false;
    std::fprintf(stderr, "standalone collector confirmed: actual reset/probes/delete, no stale RTS\n");
}

int main(int argc, char **argv) {
    if (argc == 2) {
        check_standalone_collector_close(argv[1]);
        return 0;
    }
    check_standalone_collector_close("confirmed");
    check_joined_callers(nullptr, nullptr);
    check_joined_callers(nullptr, caller);
    check_joined_callers(caller, nullptr);
    check_joined_callers(caller, caller);
    check_joined_callers(caller, &tokens[62]);
    check_joined_callers(caller, &tokens[62], true);
    check_level_publication(false);
    check_level_publication(true);
    check_result_prepare_refusal();
    for (bool has_caller : {false, true})
        for (int failure_step : {0, 1, 2, 3})
            check_result_publication(has_caller, failure_step);
    check_timing_prepare_refusal();
    check_timing_publication(0);
    check_timing_publication(2);
    check_timing_publication(3);
    check_metadata_publication(0);
    check_metadata_publication(1);  // Metadata enqueue failure.
    check_metadata_publication(2);  // Descriptor enqueue after accepted metadata.

    if (std::strcmp(get_platform(), "a5") == 0) {
        check_metadata_publication(0, true);
        check_metadata_publication(1, true);
        check_metadata_publication(2, true);
    }
    check_arena_growth(SIMPLER_MODE_PROGRAM);
    check_arena_growth(SIMPLER_MODE_KERNEL);
    check(0, false);
    // Positions 4 and 9 are optional timing marks; their failure is deliberately non-fatal.
    for (int fault : {1, 2, 3, 5, 6, 7, 8, 10, 11})
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
extern "C" SimplerHostTraceState *unified_log_host_trace_state() { return nullptr; }
extern "C" uint64_t unified_log_host_trace_allocate_inv() { return 0; }
