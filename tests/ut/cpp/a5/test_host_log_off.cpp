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

// Covers PTO_LOG_LEVEL=off: when set, every level including ALWAYS
// must be fully muted. Regression guard for the bug where invalid level
// strings were silently mapped to INFO in the host path and ALWAYS
// bypassed the filter.

#include <cstdio>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "host_log.h"

namespace {

struct CapturedStdio {
    std::string out;
    std::string err;
};

CapturedStdio run_with_level(const char *level_value, void (*fn)()) {
    // Redirect stdout/stderr to temp files so we can read the bytes back.
    fflush(stdout);
    fflush(stderr);
    FILE *out_tmp = tmpfile();
    FILE *err_tmp = tmpfile();
    int saved_out = dup(fileno(stdout));
    int saved_err = dup(fileno(stderr));
    dup2(fileno(out_tmp), fileno(stdout));
    dup2(fileno(err_tmp), fileno(stderr));

    if (level_value != nullptr) {
        setenv("PTO_LOG_LEVEL", level_value, 1);
    } else {
        unsetenv("PTO_LOG_LEVEL");
    }
    HostLogger::get_instance().reinitialize();

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

TEST(HostLogOffTest, OffMutesEverythingIncludingAlways) {
    auto captured = run_with_level("off", [] {
        HostLogger::get_instance().log(HostLogLevel::ERROR, "err-msg");
        HostLogger::get_instance().log(HostLogLevel::WARN, "warn-msg");
        HostLogger::get_instance().log(HostLogLevel::INFO, "info-msg");
        HostLogger::get_instance().log(HostLogLevel::DEBUG, "dbg-msg");
        HostLogger::get_instance().log(HostLogLevel::ALWAYS, "always-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err, "");
}

TEST(HostLogOffTest, ErrorLevelStillEmitsErrorAndAlways) {
    auto captured = run_with_level("error", [] {
        HostLogger::get_instance().log(HostLogLevel::ERROR, "err-msg");
        HostLogger::get_instance().log(HostLogLevel::INFO, "info-msg");
        HostLogger::get_instance().log(HostLogLevel::ALWAYS, "always-msg");
    });
    EXPECT_NE(captured.err.find("err-msg"), std::string::npos);
    EXPECT_EQ(captured.out.find("info-msg"), std::string::npos);
    EXPECT_NE((captured.out + captured.err).find("always-msg"), std::string::npos);
}
