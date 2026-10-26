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
#include "runtime/rt.h"
#include "acl/acl.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "kernel_platform_ops.h"
#include "platform_comm/comm.h"
#include "run_retention_probe.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <stdexcept>

namespace hbg_host_transport {
int Control::enqueue(std::string operation, void *stream, void *event) {
    operations.push_back({std::move(operation), stream, event});
    return ++step == fail_step ? -4242 : 0;
}
void reset_trace() {
    control().operations.clear();
    control().synchronized.clear();
    control().launches.clear();
    control().step = control().fail_step = 0;
}
Control &control() {
    static Control c;
    return c;
}
}  // namespace hbg_host_transport
namespace {
std::unordered_map<void *, size_t> allocations;
}
namespace hbg_host_transport {
bool live_allocation(const void *address, size_t bytes) {
    const auto begin = reinterpret_cast<uintptr_t>(address);
    for (const auto &[allocation, size] : allocations) {
        const auto base = reinterpret_cast<uintptr_t>(allocation);
        if (begin >= base && begin - base <= size && bytes <= size - (begin - base)) return true;
    }
    return false;
}
}  // namespace hbg_host_transport
extern "C" {
int dlclose(void *handle) noexcept {
    hbg_host_transport::control().released_orchestration_modules.push_back(handle);
    return 0;
}
int rtEventWorkModeGet(uint8_t *out) {
    *out = 1;
    return 0;
}
int rtEventWorkModeSet(uint8_t) { return 0; }
int aclrtMemsetAsync(void *p, size_t capacity, int32_t value, size_t bytes, aclrtStream stream) {
    const int rc = hbg_host_transport::control().enqueue("clear", stream);
    return rc == 0 ? aclrtMemset(p, capacity, value, bytes) : rc;
}
// The CPU transport has no device SDMA channels to warm up.
uint32_t dma_workspace_channel_count(void) { return 0; }
int rtMalloc(void **out, uint64_t bytes, uint32_t, uint16_t) {
    auto &c = hbg_host_transport::control();
    if (c.launching) ++c.forbidden_launch_calls;
    *out = nullptr;
    *out = std::calloc(1, bytes);
    if (!*out) return -12;
    allocations[*out] = bytes;
    c.allocated_bytes += bytes;
    return 0;
}
int rtFree(void *ptr) {
    auto &c = hbg_host_transport::control();
    if (c.launching) ++c.forbidden_launch_calls;
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return ptr == nullptr ? 0 : -1;
    hbg_host_transport::control().allocated_bytes -= it->second;
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
    if (hbg_host_transport::control().launching) ++hbg_host_transport::control().forbidden_launch_calls;
    if (bytes > capacity || (bytes && (!dst || !src))) return -2;
    std::memcpy(dst, src, bytes);
    return 0;
}
int rtSetDevice(int32_t) { return 0; }
int rtDeviceReset(int32_t) { return 0; }
int rtStreamCreate(rtStream_t *out, int32_t) {
    *out = std::malloc(1);
    return *out ? 0 : -12;
}
int rtStreamDestroy(rtStream_t p) {
    std::free(p);
    return 0;
}
int rtStreamQuery(rtStream_t) { return 0; }
int rtStreamSynchronize(rtStream_t) { return 0; }
int rtRegisterAllKernel(const rtDevBinary_t *, void **out) {
    auto &c = hbg_host_transport::control();
    ++c.module_registrations;
    if (c.launching) ++c.forbidden_launch_calls;
    *out = reinterpret_cast<void *>(uintptr_t{31});
    return 0;
}
int rtUnregisterAllKernel(void *) { return 0; }
int rtKernelLaunchWithHandleV2(
    void *, uint64_t, uint32_t, rtArgsEx_t *args, void *, rtStream_t stream, rtTaskCfgInfo_t *
) {
    auto &c = hbg_host_transport::control();
    const int rc = c.enqueue("core", stream);
    if (rc == 0)
        c.launches.push_back(
            {"core", stream, {static_cast<uint8_t *>(args->args), static_cast<uint8_t *>(args->args) + args->argsSize}}
        );
    return rc;
}
const char *aclGetRecentErrMsg() { return "HBG host submission CPU transport"; }
int aclFinalize() { return 0; }
int aclrtGetMemInfo(int, size_t *free, size_t *total) {
    *free = *total = 1ULL << 30;
    return 0;
}
int aclrtCreateEventExWithFlag(aclrtEvent *out, uint32_t) {
    *out = std::malloc(1);
    return *out ? 0 : -12;
}
int aclrtDestroyEvent(aclrtEvent p) {
    std::free(p);
    return 0;
}
int aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    return hbg_host_transport::control().enqueue("record", stream, event);
}
int aclrtResetEvent(aclrtEvent, aclrtStream) { return 0; }
int aclrtQueryEventStatus(aclrtEvent, aclrtEventRecordedStatus *out) {
    auto &c = hbg_host_transport::control();
    ++c.query_calls;
    *out = c.complete ? ACL_EVENT_RECORDED_STATUS_COMPLETE : ACL_EVENT_RECORDED_STATUS_NOT_READY;
    return c.query_rc;
}
int aclrtSynchronizeEventWithTimeout(aclrtEvent, int32_t) { return 0; }
int aclrtEventGetTimestamp(aclrtEvent, uint64_t *out) {
    *out = 1;
    return 0;
}
int aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    return hbg_host_transport::control().enqueue("wait", stream, event);
}
int aclrtSynchronizeStream(aclrtStream) { return 0; }
int aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t) {
    auto &c = hbg_host_transport::control();
    c.synchronized.push_back(stream);
    if (c.launching) ++c.forbidden_launch_calls;
    if (++c.sync_calls == c.fail_sync) return -8282;
    return c.sync_rc;
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
    *actual = wanted;
    return 0;
}
int aclrtStreamGetId(aclrtStream, int32_t *out) {
    *out = 1;
    return 0;
}
int aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *status, aclmdlRI *model) {
    *status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    *model = nullptr;
    return 0;
}
int dlog_setlevel(int, int, int) { return 0; }
}
namespace host {
LoadAicpuOp::~LoadAicpuOp() = default;
int LoadAicpuOp::Finalize() { return 0; }
void LoadAicpuOp::ForgetWithoutUnload() {}
int LoadAicpuOp::BootstrapDispatcher(const void *, size_t, const void *, size_t, rtStream_t, int) { return -1; }
int LoadAicpuOp::Init(const std::vector<std::string> &) { return -1; }
int LoadAicpuOp::LaunchBuiltInOp(rtStream_t stream, void *payload, size_t bytes, int, const std::string &name) {
    auto &c = hbg_host_transport::control();
    if (c.throw_cpu) throw std::runtime_error("AICPU transport failed");
    const int rc = c.enqueue(name, stream);
    if (rc == 0)
        c.launches.push_back(
            {name, stream, {static_cast<uint8_t *>(payload), static_cast<uint8_t *>(payload) + bytes}}
        );
    return rc;
}
}  // namespace host

KernelContextOps make_onboard_kernel_context_ops() {
    KernelContextOps ops;
    ops.get_current_device = [](void *, int *out) noexcept {
        *out = 0;
        return 0;
    };
    ops.get_current_context = [](void *, void **out) noexcept {
        *out = reinterpret_cast<void *>(uintptr_t{17});
        return 0;
    };
    ops.create_hidden_stream = [](void *, void **out) noexcept {
        return rtStreamCreate(out, 0);
    };
    ops.destroy_hidden_stream = [](void *, void *stream) noexcept {
        return rtStreamDestroy(stream);
    };
    ops.create_event = [](void *, void **out) noexcept {
        return aclrtCreateEventExWithFlag(out, 0);
    };
    ops.destroy_event = [](void *, void *event) noexcept {
        return aclrtDestroyEvent(event);
    };
    return ops;
}

// The late-read hardware probe needs real stream-pair ownership. This transport
// cannot retire that ownership; native admission/close uses the real base runner.
int RunRetentionProbePeer::retire_predecessor_ownership(DeviceRunnerBase &, PreparedExecution &) {
    return PTO_RUNTIME_ERR_UNSUPPORTED;
}

void RunRetentionProbePeer::adopt_drain_ownership(DeviceRunnerBase &, const PreparedExecution &) {
    std::fputs("HBG host submission CPU transport cannot adopt hardware drain ownership\n", stderr);
    std::abort();
}
