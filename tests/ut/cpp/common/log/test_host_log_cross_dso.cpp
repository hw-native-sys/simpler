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

#include <fstream>
#include <string>

#include <dlfcn.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "common/host_log_state.h"
#include "host_log.h"

using simpler::log::LogLevel;

TEST(HostLogCrossDsoTest, BoundConsumerUsesProcessOwnedFileSink) {
    char directory_template[] = "/tmp/simpler-host-log-cross-dso-XXXXXX";
    char *directory = mkdtemp(directory_template);
    ASSERT_NE(directory, nullptr);

    HostLogger &owner = HostLogger::get_instance();
    owner.set_log_directory(directory);
    owner.set_level(LogLevel::ERROR);

    void *handle = dlopen(TEST_HOST_LOG_CONSUMER_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();

    dlerror();
    auto bind = reinterpret_cast<SimplerHostLogBindStateFn>(dlsym(handle, "simpler_host_log_bind_state"));
    ASSERT_NE(bind, nullptr) << dlerror();
    ASSERT_EQ(bind(owner.state()), 0);

    dlerror();
    auto emit = reinterpret_cast<void (*)()>(dlsym(handle, "test_host_log_consumer_emit"));
    ASSERT_NE(emit, nullptr) << dlerror();
    dlerror();
    auto start_writer = reinterpret_cast<int (*)()>(dlsym(handle, "test_host_log_consumer_start_writer"));
    ASSERT_NE(start_writer, nullptr) << dlerror();

    owner.log(LogLevel::ERROR, "owner", "owner-record");
    emit();
    auto flush = reinterpret_cast<int (*)()>(dlsym(handle, "test_host_log_consumer_flush"));
    ASSERT_NE(flush, nullptr);
    ASSERT_EQ(flush(), 1) << "a bound runtime consumer must drain the process owner's accepted records";
    EXPECT_EQ(start_writer(), 1) << "a bound consumer should recognize the already-published owner sink";

    const std::string path = std::string(directory) + "/host." + std::to_string(static_cast<int>(getpid())) + ".log";
    std::ifstream input(path);
    ASSERT_TRUE(input.good());
    const std::string captured((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();

    EXPECT_NE(captured.find("][ERROR] owner: owner-record\n"), std::string::npos);
    EXPECT_NE(captured.find("][ERROR] consumer: cross-dso-record\n"), std::string::npos);
    ASSERT_TRUE(owner.prepare_to_fork());
    EXPECT_EQ(start_writer(), 0) << "a transient bound DSO must not become the process sink owner";
    EXPECT_EQ(dlclose(handle), 0);
    EXPECT_EQ(unlink(path.c_str()), 0);
    EXPECT_EQ(rmdir(directory), 0);
}
