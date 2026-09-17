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
#include <runtime/rt.h>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <vector>
#include <type_traits>

namespace {
struct Binary {
    std::vector<char> bytes;
    void *handle{nullptr};
};
struct Args {
    void *input;
    void *output;
    uint64_t count;
    float scalar;
    uint32_t reserved{0};
};
static_assert(std::is_trivially_copyable_v<Args> && std::is_standard_layout_v<Args>);
static_assert(
    offsetof(Args, input) == 0 && offsetof(Args, output) == 8 && offsetof(Args, count) == 16 &&
    offsetof(Args, scalar) == 24
);
}  // namespace
extern "C" int ordinary_open(const char *path, void **out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good() || !out) return -1;
    auto *binary = new Binary{{std::istreambuf_iterator<char>(file), {}}};
    rtDevBinary_t descriptor{};
    descriptor.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    descriptor.data = binary->bytes.data();
    descriptor.length = binary->bytes.size();
    const auto rc = rtRegisterAllKernel(&descriptor, &binary->handle);
    if (rc != 0) {
        delete binary;
        return rc;
    }
    *out = binary;
    return 0;
}
extern "C" int ordinary_launch(void *opaque, void *stream, void *input, void *output, uint64_t count, float scalar) {
    Args packet{input, output, count, scalar};
    rtArgsEx_t args{};
    args.args = &packet;
    args.argsSize = sizeof(packet);
    rtTaskCfgInfo_t config{};
    config.schemMode = RT_SCHEM_MODE_BATCH;
    return rtKernelLaunchWithHandleV2(static_cast<Binary *>(opaque)->handle, 0, 1, &args, nullptr, stream, &config);
}
extern "C" int ordinary_close(void *opaque) {
    auto *binary = static_cast<Binary *>(opaque);
    const auto rc = rtDevBinaryUnRegister(binary->handle);
    if (rc == 0) delete binary;
    return rc;
}
