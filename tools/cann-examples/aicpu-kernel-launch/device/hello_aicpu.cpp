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
// hello_aicpu.cpp — the absolute minimum AICPU SO that demonstrates "you
// launched it and it ran." Two exports, no business logic:
//
//   simpler_aicpu_init  — no-op. Required because LoadAicpuOp::Init resolves
//                         this symbol via rtsFuncGetByName and treats a miss
//                         as fatal. Returning 0 is enough.
//   simpler_aicpu_run   — reads DeviceArgs.input_token (so the host can
//                         prove the kernel saw its input), calls one
//                         halGetDeviceInfo to show device-side CANN works,
//                         and writes a HelloResult to GM at
//                         DeviceArgs.result_addr.
//
// I/O contract (matches host_main.cpp's struct layout):
//   KernelArgs.device_args -> &DeviceArgs (in GM)
//     DeviceArgs.result_addr  (offset 96)  -> &HelloResult (in GM, 32 B)
//     DeviceArgs.input_token  (offset 104) -> host-supplied nonce
//
// Output schema:
//   HelloResult {
//     uint64 magic         // = 0xDEADBEEFC0FFEE01 — proves kernel ran
//     uint64 echoed_token  // = DeviceArgs.input_token
//     int32  hal_rc        // halGetDeviceInfo(AICPU, CORE_NUM) rc
//     int32  _pad
//     int64  hal_value     // hal value or 0 on error
//   }
//
// The dispatcher bootstrap (host_main.cpp::Bootstrap) lands this SO at the
// preinstall directory; the host then loads it via rtsBinaryLoadFromFile and
// invokes `simpler_aicpu_run` via rtsLaunchCpuKernel. Same pipeline as the
// production runtime — see src/common/aicpu_dispatcher/README.md.

#include <cstdint>
#include <cstring>

#include <driver/ascend_hal_base.h>

namespace {

constexpr uint64_t kMagic = 0xDEADBEEFC0FFEE01ULL;
constexpr uint32_t kHalSuccess = 0;

// Mirrors host_main.cpp::DeviceArgs. We only touch the fields at the offsets
// our run() needs; the dispatcher uses offsets 96..128 during bootstrap, but
// by the time run() is invoked the host has rewritten the layout below.
struct DeviceArgs {
    uint64_t reserved_pre[12];  // 0..95
    uint64_t result_addr;       // 96
    uint64_t input_token;       // 104
};

// KernelArgs is CANN's standard envelope for AICPU kernels. The
// device_args slot at offset 40 (5 qwords in) is the only field the
// dispatcher and our run() care about.
struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

#pragma pack(push, 4)
struct HelloResult {
    uint64_t magic;
    uint64_t echoed_token;
    int32_t hal_rc;
    int32_t _pad;
    int64_t hal_value;
};
#pragma pack(pop)
static_assert(sizeof(HelloResult) == 32, "HelloResult size drift");

// CANN ships a device-side logger as a weak symbol — DlogRecord lands in
// the CANN device log (visible via msnpureport / plog), nothing else.
// Match the device-query precedent: log a single line at entry so a reader
// running this for the first time has proof their kernel was actually
// invoked even before the D2H readback runs.
extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);

constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelInfo = 1;

void DiagLog(const char *msg) { DlogRecord(kDlogModuleCcecpu, kDlogLevelInfo, "[hello-aicpu] %s", msg); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    // No-op. LoadAicpuOp::Init resolves this symbol via rtsFuncGetByName
    // and treats failure to resolve as fatal — but it doesn't care about
    // the return value beyond zero/non-zero.
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_run(void *args) {
    DiagLog("simpler_aicpu_run entered");
    if (args == nullptr) {
        DiagLog("args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<DeviceArgs *>(k->device_args);
    if (d == nullptr || d->result_addr == 0) {
        DiagLog("device_args missing or result_addr=0");
        return 1;
    }

    auto *out = reinterpret_cast<HelloResult *>(d->result_addr);
    out->magic = kMagic;
    out->echoed_token = d->input_token;
    out->_pad = 0;

    // Single device-side HAL call: ask "how many AICPU cores does this die
    // expose?" — the answer (6 on a3, 6 on a5) is independently verifiable
    // from the host-side `tools/cann-examples/query` tool. Local device id
    // 0 means "myself" device-side; using the host's logical did returns
    // rc=1 (see aicpu-device-query's writeup for why).
    constexpr uint32_t self_did = 0;
    constexpr int kModuleAicpu = 1;
    constexpr int kInfoCoreNum = 1;
    int64_t hal_value = 0;
    drvError_t hal_rc = halGetDeviceInfo(self_did, kModuleAicpu, kInfoCoreNum, &hal_value);
    out->hal_rc = static_cast<int32_t>(hal_rc);
    out->hal_value = (hal_rc == kHalSuccess) ? hal_value : 0;

    return 0;
}

}  // extern "C"
