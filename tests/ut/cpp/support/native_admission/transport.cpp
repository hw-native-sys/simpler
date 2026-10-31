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
// CPU-only transport. Production runner, allocator, ledger, maker and C API own the policy.
#include "transport.h"
#include "arg_direction.h"
#include "runtime/rt.h"
#include "acl/acl.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "kernel_platform_ops.h"
#include "platform_comm/comm.h"
#include "run_retention_probe.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace native_admission_transport {
Control &control() {
    static Control c;
    return c;
}
}  // namespace native_admission_transport
namespace {
std::unordered_map<void *, size_t> allocations;
std::unordered_set<void *> streams;
}  // namespace
namespace native_admission_transport {
void release_provider_state() {
    for (auto &entry : allocations)
        std::free(entry.first);
    allocations.clear();
    for (void *stream : streams)
        std::free(stream);
    streams.clear();
    control().bytes = 0;
}
}  // namespace native_admission_transport
extern "C" {
// The CPU transport has no device SDMA channels to warm up.
uint32_t dma_workspace_channel_count(void) { return 0; }
int rtMalloc(void **out, uint64_t bytes, uint32_t, uint16_t) {
    auto &c = native_admission_transport::control();
    ++c.allocations;
    *out = nullptr;
    if (c.allocations == c.fail_allocation || c.bytes > c.ceiling || bytes > c.ceiling - c.bytes) return -12;
    if (posix_memalign(out, CALLABLE_ALIGN, bytes) != 0 || !*out) return -12;
    std::memset(*out, 0, bytes);
    ++c.successful_allocations;
    allocations[*out] = bytes;
    c.bytes += bytes;
    return 0;
}
int rtFree(void *ptr) {
    if (native_admission_transport::control().retired) {
        ++native_admission_transport::control().calls_after_retire;
        return -4244;
    }
    auto &c = native_admission_transport::control();
    if (c.initialize && std::exchange(c.fail_free_once, false)) return -4327;
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return ptr == nullptr ? 0 : -1;
    native_admission_transport::control().bytes -= it->second;
    allocations.erase(it);
    std::free(ptr);
    return 0;
}
int rtMallocHost(void **ptr, uint64_t bytes, uint16_t) {
    *ptr = std::malloc(bytes);
    return *ptr == nullptr ? PTO_RUNTIME_ERR_INTERNAL : 0;
}
int rtFreeHost(void *ptr) {
    std::free(ptr);
    return 0;
}
int rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t) {
    return rtMemcpy(dst, capacity, src, bytes, kind);
}
int rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t) {
    if (const int rc = native_admission_transport::control().copy_rc) return rc;
    if (bytes > capacity || (bytes && (!dst || !src))) return -2;
    std::memcpy(dst, src, bytes);
    return 0;
}
int rtSetDevice(int32_t) { return native_admission_transport::control().attach_rc; }
int rtDeviceReset(int32_t) {
    auto &c = native_admission_transport::control();
    c.retired = true;
    if (c.reset_error == 0) native_admission_transport::release_provider_state();
    return c.reset_error;
}
int rtStreamCreate(rtStream_t *out, int32_t) {
    auto &c = native_admission_transport::control();
    if (++c.stream_create_calls == c.fail_stream_create) return -4242;
    *out = std::malloc(1);
    if (*out != nullptr) streams.insert(*out);
    return *out ? 0 : -12;
}
int rtStreamDestroy(rtStream_t p) {
    if (native_admission_transport::control().retired) {
        ++native_admission_transport::control().calls_after_retire;
        return -4244;
    }
    streams.erase(p);
    std::free(p);
    return 0;
}
int rtStreamQuery(rtStream_t) { return 0; }
int rtStreamSynchronize(rtStream_t) { return 0; }
int rtRegisterAllKernel(const rtDevBinary_t *, void **out) {
    auto &c = native_admission_transport::control();
    if (c.initialize && std::exchange(c.throw_kernel_registration, false)) throw std::bad_alloc();
    *out = std::malloc(1);
    return *out ? 0 : -12;
}
int rtUnregisterAllKernel(void *p) {
    std::free(p);
    return 0;
}
int rtKernelLaunchWithHandleV2(void *, uint64_t, uint32_t, rtArgsEx_t *, void *, rtStream_t, rtTaskCfgInfo_t *) {
    auto &c = native_admission_transport::control();
    if (!c.initialize) return -1;
    ++c.aicore_launch_calls;
    return 0;
}
const char *aclGetRecentErrMsg() { return "native admission CPU transport"; }
int aclFinalize() { return 0; }
int aclrtGetMemInfo(int, size_t *free, size_t *total) {
    *free = *total = 1ULL << 30;
    return 0;
}
int aclrtCreateEventExWithFlag(aclrtEvent *out, uint32_t) {
    ++native_admission_transport::control().event_creates;
    *out = std::malloc(1);
    return *out ? 0 : -12;
}
int aclrtDestroyEvent(aclrtEvent p) {
    if (std::exchange(native_admission_transport::control().fail_event_destroy_once, false)) return -4332;
    ++native_admission_transport::control().event_destroys;
    std::free(p);
    return 0;
}
int aclrtRecordEvent(aclrtEvent, aclrtStream) { return 0; }
int aclrtResetEvent(aclrtEvent, aclrtStream) { return 0; }
int aclrtQueryEventStatus(aclrtEvent, aclrtEventRecordedStatus *out) {
    *out = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    return 0;
}
int aclrtSynchronizeEventWithTimeout(aclrtEvent, int32_t) { return 0; }
int aclrtEventGetTimestamp(aclrtEvent, uint64_t *out) {
    *out = 1;
    return 0;
}
int aclrtStreamWaitEvent(aclrtStream, aclrtEvent) { return 0; }
int aclrtSynchronizeStream(aclrtStream) { return 0; }
int aclrtSynchronizeStreamWithTimeout(aclrtStream, int32_t) {
    auto &c = native_admission_transport::control();
    if (c.sync_hook) c.sync_hook();
    if (c.initialize && ++c.sync_calls == c.fail_sync_call) return -4328;
    return 0;
}
int aclrtGetStreamResLimit(aclrtStream, int, uint32_t *out) {
    *out = 1;
    return 0;
}
int aclrtMemset(void *p, size_t capacity, int32_t value, size_t bytes) {
    if (bytes > capacity) return -2;
    std::memset(p, value, bytes);
    return 0;
}
int aclrtSetOpExecuteTimeOutV2(uint64_t wanted, uint64_t *actual) {
    auto &c = native_admission_transport::control();
    ++c.op_timeout_calls;
    c.requested_op_timeout_us = wanted;
    if (c.op_timeout_rc != 0) return c.op_timeout_rc;
    *actual = c.replace_op_timeout ? c.actual_op_timeout_us : wanted;
    return 0;
}
int aclrtStreamGetId(aclrtStream, int32_t *out) {
    *out = 1;
    return 0;
}
int aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *status, aclmdlRI *model) {
    const auto &c = native_admission_transport::control();
    *status = static_cast<aclmdlRICaptureStatus>(c.capture_status);
    *model = c.capture_model;
    return c.capture_query_rc;
}
int aclmdlRIDestroyRegisterCallback(aclmdlRI, aclrtCallback callback, void *data) {
    auto &c = native_admission_transport::control();
    ++c.capture_callback_calls;
    if (c.capture_callback_rc != 0) return c.capture_callback_rc;
    c.capture_destroy_callback = callback;
    c.capture_destroy_data = data;
    return 0;
}
int dlog_setlevel(int, int, int) { return 0; }
}
namespace host {
LoadAicpuOp::~LoadAicpuOp() = default;
int LoadAicpuOp::Finalize(rtStream_t) {
    auto &c = native_admission_transport::control();
    ++c.loader_finalize_calls;
    return c.loader_finalize_error;
}
void LoadAicpuOp::ForgetWithoutUnload() {}
int LoadAicpuOp::BootstrapDispatcher(const void *, size_t, const void *, size_t, rtStream_t, int) {
    auto &c = native_admission_transport::control();
    if (!c.initialize) return -1;
    ++c.bootstrap_calls;
    return 0;
}
int LoadAicpuOp::Init(const std::vector<std::string> &) {
    auto &c = native_admission_transport::control();
    if (!c.initialize) return -1;
    ++c.loader_init_calls;
    return 0;
}
int LoadAicpuOp::LaunchBuiltInOp(rtStream_t, void *, size_t, int, const std::string &name) {
    auto &c = native_admission_transport::control();
    if (!c.initialize) return -1;
    c.launched_symbols.push_back(name);
    return 0;
}
}  // namespace host

KernelContextOps make_onboard_kernel_context_ops() {
    if (!native_admission_transport::control().initialize) return {};
    return {
        nullptr,
        [](void *, int *device) noexcept {
            *device = 0;
            return 0;
        },
        [](void *, void **context) noexcept {
            static int borrowed_context;
            *context = &borrowed_context;
            return 0;
        },
        [](void *, void **stream) noexcept {
            return rtStreamCreate(stream, 0);
        },
        [](void *, void *stream) noexcept {
            return rtStreamDestroy(stream);
        },
        [](void *, void **event) noexcept {
            return aclrtCreateEventExWithFlag(event, 0);
        },
        [](void *, void *event) noexcept {
            return aclrtDestroyEvent(event);
        }
    };
}

// The late-read hardware probe needs real stream-pair ownership. This transport
// cannot retire that ownership; native admission/close uses the real base runner.
int RunRetentionProbePeer::retire_predecessor_ownership(DeviceRunnerBase &, PreparedExecution &) {
    return PTO_RUNTIME_ERR_UNSUPPORTED;
}

void RunRetentionProbePeer::adopt_drain_ownership(DeviceRunnerBase &, const PreparedExecution &) {
    std::fputs("native admission CPU transport cannot adopt hardware drain ownership\n", stderr);
    std::abort();
}
