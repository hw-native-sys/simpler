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

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include "host_device_memory.h"

namespace {

HostDeviceMemoryConfig cfg(uint64_t data_bytes = 128, uint32_t signal_count = 2, uint32_t flags = 0) {
    return HostDeviceMemoryConfig{data_bytes, signal_count, flags};
}

HostDeviceMemory *make_memory(const HostDeviceMemoryConfig &c) {
    size_t bytes = host_device_memory_required_bytes(&c);
    void *base = nullptr;
    EXPECT_EQ(posix_memalign(&base, 64, bytes), 0);
    auto *mem = host_device_memory_wrap(base, base, bytes, &c, 1, free);
    EXPECT_NE(mem, nullptr);
    return mem;
}

}  // namespace

TEST(HostDeviceMemoryTest, RejectsInvalidConfig) {
    auto no_data = cfg(0, 1);
    EXPECT_EQ(host_device_memory_required_bytes(&no_data), 0u);

    auto no_signals = cfg(64, 0);
    EXPECT_EQ(host_device_memory_required_bytes(&no_signals), 0u);
}

TEST(HostDeviceMemoryTest, InfoReturnsPointersAndShape) {
    auto c = cfg(256, 3, 7);
    HostDeviceMemory *mem = make_memory(c);

    HostDeviceMemoryInfo info{};
    EXPECT_EQ(host_device_memory_info(mem, &info), HDMEM_OK);
    EXPECT_NE(info.host_ptr, 0u);
    EXPECT_NE(info.device_ptr, 0u);
    EXPECT_EQ(info.host_ptr, info.device_ptr);
    EXPECT_EQ(info.data_bytes, 256u);
    EXPECT_EQ(info.signal_count, 3u);
    EXPECT_EQ(info.flags, 7u);

    host_device_memory_destroy(mem);
}

TEST(HostDeviceMemoryTest, CpuWriteReadRoundTrip) {
    HostDeviceMemory *mem = make_memory(cfg());
    const char msg[] = "shared-memory";

    EXPECT_EQ(host_device_memory_write(mem, 8, msg, sizeof(msg) - 1), HDMEM_OK);

    uint8_t out[32]{};
    EXPECT_EQ(host_device_memory_read(mem, 8, out, sizeof(msg) - 1), HDMEM_OK);
    EXPECT_EQ(std::memcmp(out, msg, sizeof(msg) - 1), 0);

    host_device_memory_destroy(mem);
}

TEST(HostDeviceMemoryTest, L2WriteCpuReadAndCpuWriteL2Read) {
    HostDeviceMemory *mem = make_memory(cfg());
    const char l2_msg[] = "from-l2";
    const char cpu_msg[] = "from-cpu";

    EXPECT_EQ(host_device_memory_write_l2_for_test(mem, 0, l2_msg, sizeof(l2_msg) - 1), HDMEM_OK);
    uint8_t cpu_out[16]{};
    EXPECT_EQ(host_device_memory_read(mem, 0, cpu_out, sizeof(l2_msg) - 1), HDMEM_OK);
    EXPECT_EQ(std::memcmp(cpu_out, l2_msg, sizeof(l2_msg) - 1), 0);

    EXPECT_EQ(host_device_memory_write(mem, 32, cpu_msg, sizeof(cpu_msg) - 1), HDMEM_OK);
    uint8_t l2_out[16]{};
    EXPECT_EQ(host_device_memory_read_l2_for_test(mem, 32, l2_out, sizeof(cpu_msg) - 1), HDMEM_OK);
    EXPECT_EQ(std::memcmp(l2_out, cpu_msg, sizeof(cpu_msg) - 1), 0);

    host_device_memory_destroy(mem);
}

TEST(HostDeviceMemoryTest, BoundsReject) {
    HostDeviceMemory *mem = make_memory(cfg(16, 1));
    const char payload[] = "abcd";
    uint8_t out[4]{};

    EXPECT_EQ(host_device_memory_write(mem, 13, payload, 4), HDMEM_ERR_INVALID);
    EXPECT_EQ(host_device_memory_read(mem, 13, out, 4), HDMEM_ERR_INVALID);
    EXPECT_EQ(host_device_memory_write(mem, 16, payload, 0), HDMEM_OK);
    EXPECT_EQ(host_device_memory_read(mem, 16, out, 0), HDMEM_OK);

    host_device_memory_destroy(mem);
}

TEST(HostDeviceMemoryTest, NotifyWait) {
    HostDeviceMemory *mem = make_memory(cfg(64, 2));

    EXPECT_EQ(host_device_memory_wait(mem, 0, 1, 0), HDMEM_ERR_WOULD_BLOCK);
    EXPECT_EQ(host_device_memory_notify_l2_for_test(mem, 0, 5), HDMEM_OK);
    EXPECT_EQ(host_device_memory_wait(mem, 0, 5, 0), HDMEM_OK);

    EXPECT_EQ(host_device_memory_wait_l2_for_test(mem, 1, 7, 0), HDMEM_ERR_WOULD_BLOCK);
    EXPECT_EQ(host_device_memory_notify(mem, 1, 7), HDMEM_OK);
    EXPECT_EQ(host_device_memory_wait_l2_for_test(mem, 1, 7, 0), HDMEM_OK);

    EXPECT_EQ(host_device_memory_notify(mem, 2, 1), HDMEM_ERR_INVALID);
    EXPECT_EQ(host_device_memory_wait(mem, 2, 1, 0), HDMEM_ERR_INVALID);

    host_device_memory_destroy(mem);
}
