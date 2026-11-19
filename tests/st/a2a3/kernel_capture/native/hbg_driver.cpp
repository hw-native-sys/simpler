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
#include <acl/error_codes/rt_error_codes.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "runtime_c_api.h"
#include "capture_submit_fault.h"
#include "call_config.h"
#include "task_args.h"
#include "capture_recovery.h"

namespace capture_test = simpler::capture_test;

using capture_test::check;
using capture_test::require;

static std::vector<uint8_t> read(const char *path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int main(int argc, char **argv) {
    require(argc == 7, "usage: device scenario callable cpu core dispatcher");
    const int device = std::atoi(argv[1]), scenario = std::atoi(argv[2]);
    const auto callable = read(argv[3]), cpu = read(argv[4]), core = read(argv[5]), dispatcher = read(argv[6]);
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(device), "set-device");
    aclrtStream caller = nullptr;
    check(aclrtCreateStream(&caller), "caller-stream");
    aclrtStream other_stream = nullptr;
    aclrtEvent dependency_event = nullptr;
    if (scenario >= 9 && scenario <= 11) {
        check(aclrtCreateStream(&other_stream), "failure-other-stream");
        check(aclrtCreateEventExWithFlag(&dependency_event, ACL_EVENT_SYNC), "failure-event");
    }
    void *input = nullptr, *output = nullptr, *host_input = nullptr, *host_output = nullptr;
    constexpr size_t bytes = 16 * sizeof(int32_t);
    check(aclrtMalloc(&input, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "input");
    check(aclrtMalloc(&output, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "output");
    check(aclrtMallocHost(&host_input, bytes), "host-input");
    check(aclrtMallocHost(&host_output, bytes), "host-output");
    for (int i = 0; i < 16; ++i)
        static_cast<int32_t *>(host_input)[i] = i;
    int32_t poison[16];
    for (auto &v : poison)
        v = -999;
    check(aclrtMemcpy(output, bytes, poison, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison-before-prepare");
    auto ctx = create_device_context();
    require(ctx != nullptr, "context");
    CallConfig config{};
    config.aicpu_thread_num = 2;
    check(
        simpler_kernel_mode_init(
            ctx, device, cpu.data(), cpu.size(), core.data(), core.size(), dispatcher.data(), dispatcher.size(),
            &config, 1
        ),
        "kernel-init"
    );
    check(simpler_kernel_mode_prepare_callable(ctx, 0, callable.data(), callable.size()), "register");
    std::vector<uint8_t> storage(get_runtime_size() + get_runtime_alignment());
    auto run = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(storage.data()) + get_runtime_alignment() - 1) & ~(get_runtime_alignment() - 1)
    );
    uint32_t shape[] = {16};
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor_external(input, shape, 1, DataType::INT32, AddressSpace::DEVICE, TensorTransfer::NONE));
    args.add_tensor(make_tensor_external(output, shape, 1, DataType::INT32, AddressSpace::DEVICE, TensorTransfer::NONE)
    );
    int32_t host_control = scenario == 6 ? 13 : 3;
    const uint32_t control_shape[]{1};
    args.add_tensor(
        make_tensor_external(&host_control, control_shape, 1, DataType::INT32, AddressSpace::HOST, TensorTransfer::NONE)
    );
    args.add_scalar(scenario);
    const bool caller_owns_run = scenario == 14 || scenario == 15;
    const bool cancellation_scenario = scenario >= 7 && scenario <= 11;
    const int capture_count = cancellation_scenario ? 2 : 1;
    int total_replays = 0;
    for (int cycle = 0; cycle < capture_count; ++cycle) {
        const bool failed_end = scenario >= 9 && scenario <= 11 && cycle == 0;
        const bool partial_submission = scenario == 11 && cycle == 0;
        NativeRunDescriptor descriptor{};
        descriptor.generation = cycle + 1;
        descriptor.run_epoch = cycle + 1;
        if (!caller_owns_run) descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
        const size_t publishes_before = capture_test::prepare_snapshot(ctx, scenario);
        check(simpler_prepare_run(ctx, run, 0, &args, &config, &descriptor), "prepare-once");
        capture_test::check_prepare(ctx, scenario, publishes_before);
        host_control = 93;  // Changing the source must not change the prepared task arguments.
        RuntimeHandle reference = nullptr;
        if (!caller_owns_run) check(simpler_retain_run_resources(ctx, run, caller, &reference), "retain-once");
        aclmdlRI model = nullptr;
        check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture-begin");
        check(aclrtMemcpyAsync(input, bytes, host_input, bytes, ACL_MEMCPY_HOST_TO_DEVICE, caller), "capture-input");
        if (scenario == 5) {
            const int rc = simpler_launch_run_on_stream(ctx, run, caller, run);
            std::printf("capture-predecessor-rejected rc=%d expected=%d\n", rc, PTO_RUNTIME_ERR_UNSUPPORTED);
            require(rc == PTO_RUNTIME_ERR_UNSUPPORTED, "reject-capture-predecessor");
        }
        if (partial_submission) {
            check(aclmdlRICaptureBegin(other_stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "conflict-other-begin");
            check(aclrtRecordEvent(dependency_event, other_stream), "conflict-other-record");
            arm_capture_submit_fault(caller, dependency_event);
        }
        const bool probe_publications = capture_test::needs_publications(scenario) || scenario == 14;
        if (probe_publications) capture_test::begin_capture_publication_probe();
        const int launch_rc = simpler_launch_run_on_stream(ctx, run, caller, nullptr);
        const auto submission =
            probe_publications ? capture_test::finish_capture_publication_probe() : capture_test::CapturedSubmission{};
        if (partial_submission) {
            const auto fault = finish_capture_submit_fault();
            std::printf(
                "partial-capture core=%d cpu=%d after=%d boundary=%d conflict=%d launch=%d\n", fault.accepted_core,
                fault.cpu_calls, fault.submissions_after_failure, fault.boundary_rc, fault.conflict_rc, launch_rc
            );
            require(
                fault.fired && fault.accepted_core == 1 && fault.cpu_calls == 0 &&
                    fault.submissions_after_failure == 0 && fault.boundary_rc == 0 &&
                    fault.conflict_rc == ACL_ERROR_STREAM_CAPTURE_CONFLICT &&
                    launch_rc == ACL_ERROR_STREAM_CAPTURE_CONFLICT,
                "partial-capture-stops-submission"
            );
        } else {
            check(launch_rc, "capture-launch-once");
        }
        capture_test::check_publications(scenario, submission);
        if (scenario == 4) {
            const int rc = simpler_launch_run_on_stream(ctx, run, caller, nullptr);
            std::printf("capture-duplicate-rejected rc=%d expected=%d\n", rc, PTO_RUNTIME_ERR_INVALID_STATE);
            require(rc == PTO_RUNTIME_ERR_INVALID_STATE, "reject-duplicate-capture-launch");
        }
        if (!partial_submission)
            check(
                aclrtMemcpyAsync(host_output, bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST, caller), "capture-output"
            );
        if (failed_end && scenario == 9) {
            check(aclrtRecordEvent(dependency_event, caller), "fork-record");
            check(aclrtStreamWaitEvent(other_stream, dependency_event), "fork-wait");
            check(
                aclrtMemcpyAsync(input, bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, other_stream), "fork-copy"
            );
        }
        if (failed_end && scenario == 10) {
            check(aclmdlRICaptureBegin(other_stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "conflict-other-begin");
            check(aclrtRecordEvent(dependency_event, caller), "conflict-record");
            const int conflict_rc = aclrtStreamWaitEvent(other_stream, dependency_event);
            std::printf("cross-model-wait rc=%d\n", conflict_rc);
            require(conflict_rc == ACL_ERROR_STREAM_CAPTURE_CONFLICT, "cross-model-conflict");
        }
        const bool cancel_recording = cancellation_scenario && cycle == 0;
        if (cancel_recording) {
            if (scenario == 8) {
                const int rc = aclmdlRICaptureEnd(caller, nullptr);
                std::printf(
                    "capture-end-rejected rc=%d expected=%d|%d\n", rc, ACL_ERROR_INVALID_PARAM,
                    ACL_ERROR_RT_PARAM_INVALID
                );
                require(
                    rc == ACL_ERROR_INVALID_PARAM || rc == ACL_ERROR_RT_PARAM_INVALID, "capture-end-invalid-argument"
                );
            }
            aclmdlRICaptureStatus status{};
            aclmdlRI recording_model = nullptr;
            check(aclmdlRICaptureGetInfo(caller, &status, &recording_model), "capture-info-before-cancel");
            require(status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE && recording_model != nullptr, "capture-still-active");
            const int finalize_rc = simpler_finalize_run(ctx, run);
            std::printf("finalize-capture rc=%d expected=%d\n", finalize_rc, launch_rc);
            require(finalize_rc == launch_rc, "finalize-during-capture");
            require(
                simpler_unregister_callable(ctx, 0) == PTO_RUNTIME_ERR_INVALID_STATE,
                "external-reference-retains-callable"
            );
        }
        const int end_rc = aclmdlRICaptureEnd(caller, &model);
        if (failed_end) {
            const int expected_end =
                scenario == 9 ? ACL_ERROR_RT_STREAM_UNJOINED : ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED;
            std::printf("failed-capture-end rc=%d expected=%d model=%p\n", end_rc, expected_end, model);
            require(end_rc == expected_end && model == nullptr, "failed-end-no-model");
            if (scenario != 9) {
                aclmdlRI other_model = nullptr;
                const int other_rc = aclmdlRICaptureEnd(other_stream, &other_model);
                require(
                    other_rc == ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED && other_model == nullptr,
                    "failed-other-end-no-model"
                );
            }
            aclmdlRICaptureStatus other_status{};
            aclmdlRI other_model = nullptr;
            check(aclmdlRICaptureGetInfo(other_stream, &other_status, &other_model), "other-info-after-end");
            require(other_status == ACL_MODEL_RI_CAPTURE_STATUS_NONE && other_model == nullptr, "other-capture-ended");
        } else {
            check(end_rc, "capture-end");
            require(model != nullptr, "capture-returned-model");
        }
        aclmdlRICaptureStatus ended_status{};
        aclmdlRI ended_model = nullptr;
        check(aclmdlRICaptureGetInfo(caller, &ended_status, &ended_model), "capture-info-after-end");
        require(ended_status == ACL_MODEL_RI_CAPTURE_STATUS_NONE && ended_model == nullptr, "capture-ended");
        if (!cancel_recording && !caller_owns_run) check(simpler_finalize_run(ctx, run), "finalize-host-run");
        int32_t observed[16];
        check(aclrtMemcpy(observed, bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read-before-replay");
        require(std::memcmp(poison, observed, bytes) == 0, "capture-did-not-execute");
        if (scenario == 13) {
            capture_test::inject_boot_failure();
            for (int fault_round = 0; fault_round < 2; ++fault_round) {
                check(aclmdlRIExecuteAsync(model, caller), "boot-failure-replay");
                ++total_replays;
                const int rc = aclrtSynchronizeStreamWithTimeout(caller, 10000);
                std::printf("boot-failure=%d sync=%d\n", fault_round + 1, rc);
                require(rc == ACL_ERROR_RT_MODEL_EXECUTE, "boot-failure-reported");
                check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "boot-failure-execution-ended");
                capture_test::check_boot_failure();
                check(aclrtMemcpy(observed, bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read-boot-output");
                require(std::memcmp(poison, observed, bytes) == 0, "boot-failure-no-user-task");
                require(
                    simpler_unregister_callable(ctx, 0) == PTO_RUNTIME_ERR_INVALID_STATE,
                    "boot-failure-retains-callable"
                );
            }
            // Undo only the injected corruption, after the failed invocations ended.
            capture_test::restore_boot_failure();
        }
        if (scenario == 12) {
            for (int fault_round = 0; fault_round < 2; ++fault_round) {
                for (int i = 0; i < 16; ++i) {
                    static_cast<int32_t *>(host_input)[i] = 100 * fault_round + i;
                    static_cast<int32_t *>(host_output)[i] = -888;
                }
                static_cast<int32_t *>(host_input)[0] = -12345 - fault_round;
                check(aclrtMemcpy(output, bytes, poison, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison-before-fault");
                check(aclmdlRIExecuteAsync(model, caller), "fault-replay");
                ++total_replays;
                const int rc = aclrtSynchronizeStreamWithTimeout(caller, 10000);
                std::printf("fault-replay=%d sync=%d\n", fault_round + 1, rc);
                require(rc == ACL_ERROR_RT_MODEL_EXECUTE, "scheduler-failure-reported");
                check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "fault-execution-ended");
                capture_test::check_progress_timeout();
                // Captured D2H can be skipped on failure. Read after stream completion.
                check(aclrtMemcpy(observed, bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read-fault-output");
                for (int i = 0; i < 16; ++i)
                    require(observed[i] == static_cast<int32_t *>(host_input)[i] + 3, "fault-round-executed");
                std::printf("fault-replay=%d executed=16\n", fault_round + 1);
            }
        }
        const int replays = (scenario == 3 || scenario == 15 || cancel_recording) ? 0 : 3;
        total_replays += replays;
        for (int round = 0; round < replays; ++round) {
            host_control = 1000 + round;
            for (int i = 0; i < 16; ++i) {
                static_cast<int32_t *>(host_input)[i] = 100 * round + i;
                static_cast<int32_t *>(host_output)[i] = -888;
            }
            check(aclrtMemcpy(output, bytes, poison, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison-before-replay");
            check(aclmdlRIExecuteAsync(model, caller), "rts-replay");
            check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "replay-sync");
            for (int i = 0; i < 16; ++i)
                require(
                    static_cast<int32_t *>(host_output)[i] ==
                        100 * round + i + (scenario == 6 ? 20 : (scenario == 12 ? 3 : 10)),
                    "fresh-output"
                );
            std::printf("replay=%d verified=16\n", round + 1);
            if (scenario == 14 && round == 0) {
                check(simpler_wait_run_completion(ctx, run), "retire-first-replay");
                // Reuse the same slot while the caller still owns the original run and model.
                std::vector<uint8_t> larger_storage(get_runtime_size() + get_runtime_alignment());
                const auto aligned =
                    (reinterpret_cast<uintptr_t>(larger_storage.data()) + get_runtime_alignment() - 1) &
                    ~(get_runtime_alignment() - 1);
                auto larger_run = reinterpret_cast<void *>(aligned);
                ChipStorageTaskArgs larger_args;
                for (int i = 0; i < args.tensor_count(); ++i)
                    larger_args.add_tensor(args.tensor(i));
                larger_args.add_scalar(16);
                NativeRunDescriptor larger_descriptor{};
                larger_descriptor.generation = larger_descriptor.run_epoch = 2;
                check(
                    simpler_prepare_run(ctx, larger_run, 0, &larger_args, &config, &larger_descriptor),
                    "prepare-larger-run"
                );
                capture_test::begin_capture_publication_probe();
                check(simpler_launch_run_on_stream(ctx, larger_run, caller, nullptr), "launch-larger-run");
                const auto larger_submission = capture_test::finish_capture_publication_probe();
                require(
                    submission.cpu_launches.size() == 1 && larger_submission.cpu_launches.size() == 1 &&
                        submission.cpu_launches.front().image_bytes > 0 &&
                        larger_submission.cpu_launches.front().image_bytes >
                            submission.cpu_launches.front().image_bytes,
                    "larger-image"
                );
                // Only A5 has resident scheduler worker contexts in a separate metadata upload.
                const auto scheduler_upload = [](const capture_test::CapturedSubmission &captured) {
                    const auto contexts = captured.launches.at(0).worker_context_base;
                    for (const auto &copy : captured.publications) {
                        const auto target = reinterpret_cast<uintptr_t>(copy.destination);
                        if (contexts >= target && contexts - target < copy.bytes) return copy;
                    }
                    return capture_test::CapturedPublication{};
                };
                if (submission.launches.at(0).worker_context_base != 0) {
                    const auto first = scheduler_upload(submission);
                    const auto second = scheduler_upload(larger_submission);
                    require(
                        first.source && second.source && first.source != second.source &&
                            first.destination != second.destination,
                        "independent-scheduler-storage"
                    );
                }
                check(simpler_finalize_run(ctx, larger_run), "finalize-larger-run");
                check(aclrtMemcpy(observed, bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read-larger-output");
                for (int i = 0; i < 16; ++i)
                    require(observed[i] == i + 14, "larger-output");
            }
        }
        if (model != nullptr) check(aclmdlRIDestroy(model), "destroy-model");
        if (caller_owns_run) check(simpler_finalize_run(ctx, run), "finalize-after-destroy");
        else check(simpler_release_run_resources(ctx, reference), "release-after-destroy");
        if (cancel_recording) std::printf("cancelled-capture-cleaned cycle=%d\n", cycle);
    }
    if (dependency_event != nullptr) check(aclrtDestroyEvent(dependency_event), "destroy-failure-event");
    if (other_stream != nullptr) check(aclrtDestroyStream(other_stream), "destroy-failure-other-stream");
    check(finalize_device(ctx), "finalize-context");
    destroy_device_context(ctx);
    check(aclrtFree(input), "free-input");
    check(aclrtFree(output), "free-output");
    check(aclrtFreeHost(host_input), "free-host-input");
    check(aclrtFreeHost(host_output), "free-host-output");
    check(aclrtDestroyStream(caller), "destroy-caller");
    check(aclrtResetDevice(device), "reset-device");
    check(aclFinalize(), "aclFinalize");
    std::printf(
        "hbg_capture PASS scenario=%d replays=%d prepare=%d launch=%d\n", scenario, total_replays, capture_count,
        capture_count
    );
}
