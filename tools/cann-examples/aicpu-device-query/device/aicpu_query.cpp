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
//
// aicpu_query.cpp — device-side AICPU SO that runs HAL queries.
//
// Two exports:
//   simpler_aicpu_init  — no-op, present because LoadAicpuOp::Init resolves it
//                         via rtsFuncGetByName. Returns 0.
//   simpler_aicpu_query — the actual workhorse: reads (module, infoType) pairs
//                         from a GM input buffer, calls halGetDeviceInfo for
//                         each, writes results to a GM output buffer.
//
// I/O contract (matches host_main.cpp's struct layout):
//   KernelArgs.device_args -> &DeviceArgs (in GM)
//     DeviceArgs.q_input_addr  (offset 96)  -> &QueryRequest[]   (in GM)
//     DeviceArgs.q_input_count (offset 104) -> count of requests
//     DeviceArgs.q_output_addr (offset 112) -> &QueryResult[]    (in GM)
//
//   QueryRequest  { int32 module_type; int32 info_type; }            // 8 B
//   QueryResult   { int32 rc; int32 _pad; int64 value; }              // 16 B
//
// The dispatcher bootstrap path already lands this SO at the preinstall
// directory; the host registers it via rtsBinaryLoadFromFile + invokes
// `simpler_aicpu_query` via rtsLaunchCpuKernel.

#include <cstdint>
#include <cstring>

#include <driver/ascend_hal_base.h>

namespace {

constexpr uint32_t kHalSuccess = 0;

// Layout of the device_args struct we share with host. Only the three
// query-related qwords beyond the dispatcher's existing layout matter here.
struct DeviceArgs {
    uint64_t reserved_pre[12];  // 0..95   — unused on this path
    uint64_t q_input_addr;      // 96
    uint64_t q_input_count;     // 104
    uint64_t q_output_addr;     // 112
};

// KernelArgs is the standard envelope CANN passes to AICPU kernels.
// device_args is the only field we care about.
struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

#pragma pack(push, 4)
struct QueryRequest {
    int32_t module_type;
    int32_t info_type;
};
struct QueryResult {
    int32_t rc;
    int32_t _pad;
    int64_t value;
};
#pragma pack(pop)
static_assert(sizeof(QueryRequest) == 8, "QueryRequest size drift");
static_assert(sizeof(QueryResult) == 16, "QueryResult size drift");

extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);

constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelError = 3;

void DiagLog(const char *msg) { DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[aicpu-query] %s", msg); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    // No-op. LoadAicpuOp::Init resolves this symbol via rtsFuncGetByName
    // and treats failure to resolve as fatal.
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_query(void *args) {
    if (args == nullptr) {
        DiagLog("simpler_aicpu_query: args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<DeviceArgs *>(k->device_args);
    if (d == nullptr) {
        DiagLog("simpler_aicpu_query: device_args==nullptr");
        return 1;
    }
    if (d->q_input_addr == 0 || d->q_output_addr == 0 || d->q_input_count == 0) {
        DiagLog("simpler_aicpu_query: empty I/O buffers");
        return 1;
    }

    auto *requests = reinterpret_cast<const QueryRequest *>(d->q_input_addr);
    auto *results = reinterpret_cast<QueryResult *>(d->q_output_addr);
    const uint64_t n = d->q_input_count;

    // Device-side HAL uses local device id 0 to mean "myself" (validated via
    // the earlier kernel.cpp probe — using host's logical did fails rc=1).
    const uint32_t self_did = 0;

    for (uint64_t i = 0; i < n; ++i) {
        int64_t value = 0;
        drvError_t rc = halGetDeviceInfo(self_did, requests[i].module_type, requests[i].info_type, &value);
        results[i].rc = static_cast<int32_t>(rc);
        results[i]._pad = 0;
        results[i].value = (rc == kHalSuccess) ? value : 0;
    }

    return 0;
}

}  // extern "C"
