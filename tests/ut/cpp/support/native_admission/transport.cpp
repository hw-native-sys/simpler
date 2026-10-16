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

namespace native_admission_transport {
Control &control() {
    static Control c;
    return c;
}
}  // namespace native_admission_transport
namespace {
std::unordered_map<void *, size_t> allocations;
}
extern "C" {
// The CPU transport has no device SDMA channels to warm up.
uint32_t dma_workspace_channel_count(void) { return 0; }
int rtMalloc(void **out, uint64_t bytes, uint32_t, uint16_t) {
    auto &c = native_admission_transport::control();
    ++c.allocations;
    *out = nullptr;
    if (c.allocations == c.fail_allocation || c.bytes > c.ceiling || bytes > c.ceiling - c.bytes) return -12;
    *out = std::calloc(1, bytes);
    if (!*out) return -12;
    ++c.successful_allocations;
    allocations[*out] = bytes;
    c.bytes += bytes;
    return 0;
}
int rtFree(void *ptr) {
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return ptr == nullptr ? 0 : -1;
    native_admission_transport::control().bytes -= it->second;
    allocations.erase(it);
    std::free(ptr);
    return 0;
}
int rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t) {
    if (const int rc = native_admission_transport::control().copy_rc) return rc;
    if (bytes > capacity || (bytes && (!dst || !src))) return -2;
    std::memcpy(dst, src, bytes);
    return 0;
}
int rtSetDevice(int32_t) { return native_admission_transport::control().attach_rc; }
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
    *out = std::malloc(1);
    return *out ? 0 : -12;
}
int rtUnregisterAllKernel(void *p) {
    std::free(p);
    return 0;
}
int rtKernelLaunchWithHandleV2(void *, uint64_t, uint32_t, rtArgsEx_t *, void *, rtStream_t, rtTaskCfgInfo_t *) {
    return -1;
}
const char *aclGetRecentErrMsg() { return "native admission CPU transport"; }
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
int aclrtSynchronizeStreamWithTimeout(aclrtStream, int32_t) { return 0; }
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
int LoadAicpuOp::LaunchBuiltInOp(rtStream_t, void *, size_t, int, const std::string &) { return -1; }
}  // namespace host

KernelContextOps make_onboard_kernel_context_ops() { return {}; }

// The late-read hardware probe needs real stream-pair ownership. This transport
// cannot retire that ownership; native admission/close uses the real base runner.
int RunRetentionProbePeer::retire_predecessor_ownership(DeviceRunnerBase &, PreparedExecution &) {
    return PTO_RUNTIME_ERR_UNSUPPORTED;
}

void RunRetentionProbePeer::adopt_drain_ownership(DeviceRunnerBase &, const PreparedExecution &) {
    std::fputs("native admission CPU transport cannot adopt hardware drain ownership\n", stderr);
    std::abort();
}

#ifdef SIMPLER_UT_TRB_RUNTIME
#include "host/host_graph_exporter.h"
extern "C" bool dep_gen_host_graph_active() { return false; }
extern "C" int dep_gen_host_graph_emit(const char *) { return -1; }
extern "C" int dep_gen_host_graph_take(simpler::dfx::host_graph::HostGraphExport *) { return -1; }
#endif
