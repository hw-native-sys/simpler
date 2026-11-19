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

#include "capture_submit_fault.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include "runtime/mem.h"
#include "common/kernel_args.h"
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"

namespace capture_test = simpler::capture_test;

namespace {
thread_local bool armed = false;
thread_local bool observe_publications = false;
thread_local capture_test::CapturedSubmission captured_submission;
thread_local void *caller_stream = nullptr;
thread_local void *conflict_event = nullptr;
thread_local CaptureSubmitFaultResult result;

template <typename Function>
Function next_symbol(const char *name) {
    auto symbol = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
    if (symbol == nullptr) {
        std::fprintf(stderr, "missing RTS symbol: %s\n", name);
        std::abort();
    }
    return symbol;
}

void note_submission() {
    if (armed && result.fired) ++result.submissions_after_failure;
}
}  // namespace

namespace simpler::capture_test {

void begin_capture_publication_probe() {
    captured_submission = {};
    observe_publications = true;
}

CapturedSubmission finish_capture_publication_probe() {
    observe_publications = false;
    return std::move(captured_submission);
}

}  // namespace simpler::capture_test

void arm_capture_submit_fault(void *caller, void *event) {
    result = {};
    caller_stream = caller;
    conflict_event = event;
    armed = true;
}

CaptureSubmitFaultResult finish_capture_submit_fault() {
    armed = false;
    return result;
}

extern "C" rtError_t rtKernelLaunchWithHandleV2(
    void *handle, uint64_t key, uint32_t blocks, rtArgsEx_t *args, rtSmDesc_t *sm, rtStream_t stream,
    const rtTaskCfgInfo_t *config
) {
    static auto real = next_symbol<decltype(&rtKernelLaunchWithHandleV2)>("rtKernelLaunchWithHandleV2");
    note_submission();
    const int rc = real(handle, key, blocks, args, sm, stream, config);
    if (armed && rc == 0) ++result.accepted_core;
    if (observe_publications && rc == 0 && args != nullptr && args->args != nullptr &&
        args->argsSize == sizeof(AicoreLaunchArgs)) {
        AicoreLaunchArgs launch{};
        std::memcpy(&launch, args->args, sizeof(launch));
        captured_submission.launches.push_back({launch.runtime_args, captured_submission.publications.size()});
    }
    return rc;
}

extern "C" rtError_t rtsLaunchCpuKernel(
    const rtFuncHandle function, uint32_t blocks, rtStream_t stream, const rtKernelLaunchCfg_t *config,
    rtCpuKernelArgs_t *args
) {
    static auto real = next_symbol<decltype(&rtsLaunchCpuKernel)>("rtsLaunchCpuKernel");
    note_submission();
    if (armed) ++result.cpu_calls;
    return real(function, blocks, stream, config, args);
}

extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    static auto real = next_symbol<decltype(&aclrtRecordEvent)>("aclrtRecordEvent");
    note_submission();
    const int rc = real(event, stream);
    if (armed && result.accepted_core > 0 && !result.fired) {
        result.boundary_rc = rc;
        if (rc != 0) return rc;
        static auto wait = next_symbol<decltype(&aclrtStreamWaitEvent)>("aclrtStreamWaitEvent");
        // The record and core launch above reached RTS. Force a real model
        // conflict here, before Simpler can enter its AICPU submission stage.
        result.conflict_rc = wait(caller_stream, conflict_event);
        result.fired = true;
        return result.conflict_rc;
    }
    return rc;
}

extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    static auto real = next_symbol<decltype(&aclrtStreamWaitEvent)>("aclrtStreamWaitEvent");
    note_submission();
    return real(stream, event);
}

extern "C" aclError aclrtMemcpyAsync(
    void *destination, size_t capacity, const void *source, size_t bytes, aclrtMemcpyKind kind, aclrtStream stream
) {
    static auto real = next_symbol<decltype(&aclrtMemcpyAsync)>("aclrtMemcpyAsync");
    note_submission();
    return real(destination, capacity, source, bytes, kind, stream);
}

extern "C" rtError_t rtMemcpyAsync(
    void *destination, uint64_t capacity, const void *source, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t stream
) {
    static auto real = next_symbol<decltype(&rtMemcpyAsync)>("rtMemcpyAsync");
    const int rc = real(destination, capacity, source, bytes, kind, stream);
    if (observe_publications && rc == 0 && kind == RT_MEMCPY_HOST_TO_DEVICE)
        captured_submission.publications.push_back({destination, source, bytes});
    return rc;
}
