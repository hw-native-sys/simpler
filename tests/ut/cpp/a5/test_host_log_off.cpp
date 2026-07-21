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

// HostLogger filtering: severity floor + INFO verbosity threshold.
// Drives the singleton via direct setters (no env vars; Python pushes via
// nanobind in production), captures stderr, and asserts on the buffered output.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

#include "host_log.h"

using simpler::log::LogLevel;

namespace {

struct CapturedStdio {
    std::string out;
    std::string err;
};

CapturedStdio run_with_config(LogLevel level, int info_v, void (*fn)()) {
    fflush(stdout);
    fflush(stderr);
    FILE *out_tmp = tmpfile();
    FILE *err_tmp = tmpfile();
    int saved_out = dup(fileno(stdout));
    int saved_err = dup(fileno(stderr));
    dup2(fileno(out_tmp), fileno(stdout));
    dup2(fileno(err_tmp), fileno(stderr));

    HostLogger::get_instance().set_level(level);
    HostLogger::get_instance().set_info_v(info_v);

    fn();

    fflush(stdout);
    fflush(stderr);
    dup2(saved_out, fileno(stdout));
    dup2(saved_err, fileno(stderr));
    close(saved_out);
    close(saved_err);

    auto slurp = [](FILE *f) {
        std::string s;
        rewind(f);
        char buf[512];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            s.append(buf, n);
        }
        fclose(f);
        return s;
    };
    return {slurp(out_tmp), slurp(err_tmp)};
}

}  // namespace

TEST(HostLogTest, NulLevelMutesAllSeverities) {
    auto captured = run_with_config(LogLevel::NUL, 0, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "err-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "dbg-msg");
        HostLogger::get_instance().log_info_v(9, "fn", "v9-msg");
        HostLogger::get_instance().log_info_v(0, "fn", "v0-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err, "");
}

TEST(HostLogTest, ErrorLevelEmitsErrorOnly) {
    auto captured = run_with_config(LogLevel::ERROR, 5, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "err-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log_info_v(9, "fn", "v9-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_NE(captured.err.find("err-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("warn-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("v9-msg"), std::string::npos);
}

TEST(HostLogTest, InfoVerbosityCutsBelowThreshold) {
    // Threshold V5: V0..V4 silenced, V5..V9 printed.
    auto captured = run_with_config(LogLevel::INFO, 5, [] {
        HostLogger::get_instance().log_info_v(0, "fn", "v0-msg");
        HostLogger::get_instance().log_info_v(4, "fn", "v4-msg");
        HostLogger::get_instance().log_info_v(5, "fn", "v5-msg");
        HostLogger::get_instance().log_info_v(9, "fn", "v9-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err.find("v0-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("v4-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("v5-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("v9-msg"), std::string::npos);
}

TEST(HostLogTest, EmitPrefixHasTimestampAndTid) {
    auto captured = run_with_config(LogLevel::INFO, 5, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "marker");
    });
    // Expected shape: "[YYYY-MM-DD HH:MM:SS.uuuuuu][T0x...][ERROR] fn: marker\n"
    ASSERT_FALSE(captured.err.empty());
    EXPECT_EQ(captured.err[0], '[');
    // Year must be 4 ASCII digits.
    for (int i = 1; i <= 4; ++i) {
        EXPECT_GE(captured.err[i], '0');
        EXPECT_LE(captured.err[i], '9');
    }
    EXPECT_EQ(captured.err[5], '-');
    // Thread-id segment "[T0x" must appear before the level tag.
    auto tid_pos = captured.err.find("][T0x");
    auto level_pos = captured.err.find("][ERROR]");
    ASSERT_NE(tid_pos, std::string::npos);
    ASSERT_NE(level_pos, std::string::npos);
    EXPECT_LT(tid_pos, level_pos);
    // Body still present.
    EXPECT_NE(captured.err.find("marker"), std::string::npos);
}

TEST(HostLogTest, AllOutputGoesToStderr) {
    auto captured = run_with_config(LogLevel::DEBUG, 0, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "e");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "w");
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "d");
        HostLogger::get_instance().log_info_v(0, "fn", "i0");
        HostLogger::get_instance().log_info_v(9, "fn", "i9");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_NE(captured.err.find("e"), std::string::npos);
    EXPECT_NE(captured.err.find("w"), std::string::npos);
    EXPECT_NE(captured.err.find("d"), std::string::npos);
    EXPECT_NE(captured.err.find("i0"), std::string::npos);
    EXPECT_NE(captured.err.find("i9"), std::string::npos);
}
