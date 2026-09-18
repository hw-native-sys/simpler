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

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

#include "../protocol.h"
#include "load_aicpu_op.h"

namespace {
std::unique_ptr<host::LoadAicpuOp> g_loader;

std::vector<char> read_file(const char *path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
}  // namespace

extern "C" __attribute__((visibility("default"))) int
rts_error_probe_init(int device, uintptr_t stream, const char *dispatcher_path, const char *probe_path) {
    if (g_loader || stream == 0 || dispatcher_path == nullptr || probe_path == nullptr) return -1;
    auto dispatcher = read_file(dispatcher_path);
    auto probe = read_file(probe_path);
    if (dispatcher.empty() || probe.empty()) return -1;
    auto loader = std::make_unique<host::LoadAicpuOp>();
    int rc = loader->BootstrapDispatcher(
        dispatcher.data(), dispatcher.size(), probe.data(), probe.size(), reinterpret_cast<rtStream_t>(stream), device
    );
    if (rc != 0) return rc;
    rc = loader->Init({});
    if (rc != 0) return rc;
    g_loader = std::move(loader);
    return 0;
}

extern "C" __attribute__((visibility("default"))) int
rts_error_probe_launch(uintptr_t stream, int32_t status, int32_t aicpu_num) {
    if (!g_loader || stream == 0 || aicpu_num <= 0) return -1;
    CallerProbeArgs args{status, 0};
    return g_loader->LaunchBuiltInOp(
        reinterpret_cast<rtStream_t>(stream), &args, sizeof(args), aicpu_num, host::KernelNames::RunName
    );
}
