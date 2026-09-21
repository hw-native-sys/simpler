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

// A bound consumer DSO submits through the process owner's queue, and enqueue
// copies the record into the owner's slot rather than retaining the caller's
// storage. This pins the consequence: unloading such a module cannot lose the
// records it already submitted, even though dlclose discards the mapping they
// were formatted in. DeviceRunner::unload_executor_binaries() does exactly this
// to the sim AICPU SO on every teardown.

#include <fstream>
#include <string>

#include <dlfcn.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "common/host_log_state.h"
#include "host_log.h"

using simpler::log::LogLevel;

namespace {

size_t count_occurrences(const std::string &haystack, const std::string &needle) {
    size_t total = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + needle.size())) {
        ++total;
    }
    return total;
}

}  // namespace

TEST(HostLogUnloadTest, UnloadedModuleLeavesNoRecordsInItsPrivateBuffer) {
    constexpr int kRecords = 200;

    char directory_template[] = "/tmp/simpler-host-log-unload-XXXXXX";
    char *directory = mkdtemp(directory_template);
    ASSERT_NE(directory, nullptr);

    HostLogger &owner = HostLogger::get_instance();
    owner.set_level(LogLevel::DEBUG);
    owner.set_log_directory(directory);

    void *handle = dlopen(TEST_HOST_LOG_UNLOAD_CONSUMER_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();

    dlerror();
    auto bind = reinterpret_cast<int (*)(SimplerHostLogState *)>(dlsym(handle, "test_host_log_unload_bind"));
    ASSERT_NE(bind, nullptr) << dlerror();
    dlerror();
    auto emit = reinterpret_cast<void (*)(int)>(dlsym(handle, "test_host_log_unload_emit"));
    ASSERT_NE(emit, nullptr) << dlerror();

    ASSERT_EQ(bind(owner.state()), 0);
    emit(kRecords);
    ASSERT_EQ(dlclose(handle), 0);
    // After the unload, so this asserts the owner can still write records whose
    // producer is gone. The writer owns the destination, so nothing is on disk
    // until this returns — reading the file without it races the writer thread.
    ASSERT_TRUE(owner.flush());

    const std::string path = std::string(directory) + "/host." + std::to_string(static_cast<int>(getpid())) + ".log";
    std::ifstream input(path);
    ASSERT_TRUE(input.good());
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();

    EXPECT_EQ(count_occurrences(contents, "] unloaded_module: record="), static_cast<size_t>(kRecords));

    EXPECT_EQ(unlink(path.c_str()), 0);
    EXPECT_EQ(rmdir(directory), 0);
}
