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

#include "host_device_channel.h"

namespace {

HostDeviceChannelConfig cfg(uint32_t c2l = 2, uint32_t l2c = 2, uint32_t depth = 4, uint32_t bytes = 64) {
    return HostDeviceChannelConfig{c2l, l2c, depth, bytes, 0};
}

HostDeviceChannel *make_channel(const HostDeviceChannelConfig &c) {
    size_t bytes = host_device_channel_required_bytes(&c);
    void *base = nullptr;
    EXPECT_EQ(posix_memalign(&base, 64, bytes), 0);
    auto *ch = host_device_channel_wrap(base, base, bytes, &c, 1, free);
    EXPECT_NE(ch, nullptr);
    return ch;
}

}  // namespace

TEST(HostDeviceChannelTest, RejectsInvalidConfig) {
    auto bad_depth = cfg(1, 1, 3, 64);
    EXPECT_EQ(host_device_channel_required_bytes(&bad_depth), 0u);

    auto too_large = cfg(1, 1, 4, HDCH_MAX_INLINE_BYTES + 1);
    EXPECT_EQ(host_device_channel_required_bytes(&too_large), 0u);
}

TEST(HostDeviceChannelTest, CpuToL2SendRecvRoundTrip) {
    auto c = cfg();
    HostDeviceChannel *ch = make_channel(c);

    const char msg[] = "hello-l2";
    EXPECT_EQ(host_device_channel_send_cpu(ch, 7, msg, sizeof(msg) - 1, 42, 0), HDCH_OK);

    uint8_t out[64]{};
    size_t nbytes = 0;
    uint64_t cid = 0;
    uint32_t route = 0;
    EXPECT_EQ(host_device_channel_recv_l2_for_test(ch, out, sizeof(out), &nbytes, &cid, &route, 0), HDCH_OK);
    EXPECT_EQ(nbytes, sizeof(msg) - 1);
    EXPECT_EQ(memcmp(out, msg, nbytes), 0);
    EXPECT_EQ(route, 7u);
    EXPECT_EQ(cid, 42u);

    host_device_channel_destroy(ch);
}

TEST(HostDeviceChannelTest, L2ToCpuSendRecvRoundTrip) {
    auto c = cfg();
    HostDeviceChannel *ch = make_channel(c);

    const char msg[] = "hello-cpu";
    EXPECT_EQ(host_device_channel_send_l2_for_test(ch, 3, msg, sizeof(msg) - 1, 99, 0), HDCH_OK);

    uint8_t out[64]{};
    size_t nbytes = 0;
    uint64_t cid = 0;
    uint32_t route = 0;
    EXPECT_EQ(host_device_channel_recv_cpu(ch, out, sizeof(out), &nbytes, &cid, &route, 0), HDCH_OK);
    EXPECT_EQ(nbytes, sizeof(msg) - 1);
    EXPECT_EQ(memcmp(out, msg, nbytes), 0);
    EXPECT_EQ(route, 3u);
    EXPECT_EQ(cid, 99u);

    host_device_channel_destroy(ch);
}

TEST(HostDeviceChannelTest, FullAndEmptyReturnWouldBlock) {
    auto c = cfg(1, 1, 2, 16);
    HostDeviceChannel *ch = make_channel(c);
    const char payload[] = "x";

    EXPECT_EQ(host_device_channel_recv_cpu(ch, nullptr, 0, nullptr, nullptr, nullptr, 0), HDCH_ERR_INVALID);
    EXPECT_EQ(host_device_channel_send_cpu(ch, 0, payload, 1, 0, 0), HDCH_OK);
    EXPECT_EQ(host_device_channel_send_cpu(ch, 0, payload, 1, 1, 0), HDCH_OK);
    EXPECT_EQ(host_device_channel_send_cpu(ch, 0, payload, 1, 2, 0), HDCH_ERR_WOULD_BLOCK);

    uint8_t out[16]{};
    size_t nbytes = 0;
    uint64_t cid = 0;
    uint32_t route = 0;
    EXPECT_EQ(host_device_channel_recv_l2_for_test(ch, out, sizeof(out), &nbytes, &cid, &route, 0), HDCH_OK);
    EXPECT_EQ(host_device_channel_send_cpu(ch, 0, payload, 1, 3, 0), HDCH_OK);

    host_device_channel_destroy(ch);
}

TEST(HostDeviceChannelTest, MessageTooLarge) {
    auto c = cfg(1, 1, 4, 4);
    HostDeviceChannel *ch = make_channel(c);
    const char payload[] = "12345";

    EXPECT_EQ(host_device_channel_send_cpu(ch, 0, payload, 5, 0, 0), HDCH_ERR_MSG_TOO_LARGE);

    host_device_channel_destroy(ch);
}
