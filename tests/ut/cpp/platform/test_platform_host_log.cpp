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

/**
 * UT for the host-side logging singleton (HostLogger).
 *
 * Covers the parts that are testable in pure userspace:
 *   - level filtering (is_enabled)
 *   - PTO_LOG_LEVEL env-var parsing via reinitialize()
 *
 * Hardware-specific sinks (e.g. device-side ring buffers) are out of scope.
 */

#include <gtest/gtest.h>

#include <cstdlib>

#include "host_log.h"

namespace {

// Test fixture: save/restore PTO_LOG_LEVEL so tests are independent regardless
// of execution order or shell environment.
class HostLoggerTest : public ::testing::Test {
protected:
    std::string saved_level_;
    bool had_level_ = false;

    void SetUp() override {
        const char *env = std::getenv("PTO_LOG_LEVEL");
        had_level_ = env != nullptr;
        if (had_level_) saved_level_ = env;
        unsetenv("PTO_LOG_LEVEL");
    }

    void TearDown() override {
        if (had_level_) {
            setenv("PTO_LOG_LEVEL", saved_level_.c_str(), 1);
        } else {
            unsetenv("PTO_LOG_LEVEL");
        }
        HostLogger::get_instance().reinitialize();  // restore logger to real env
    }
};

}  // namespace

// ---------- is_enabled at default level ----------

TEST_F(HostLoggerTest, DefaultLevel_InfoEnabled_DebugDisabled) {
    // Arrange: no env var -> default INFO.
    HostLogger::get_instance().reinitialize();

    // Assert
    HostLogger &logger = HostLogger::get_instance();
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::ERROR));
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::WARN));
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::INFO));
    EXPECT_FALSE(logger.is_enabled(HostLogLevel::DEBUG));
}

TEST_F(HostLoggerTest, AlwaysLevel_IsAlwaysEnabled) {
    HostLogger::get_instance().reinitialize();
    // ALWAYS == -1, so it should pass the <= check for any current_level_ >= 0.
    EXPECT_TRUE(HostLogger::get_instance().is_enabled(HostLogLevel::ALWAYS));
}

// ---------- env-var parsing ----------

TEST_F(HostLoggerTest, EnvLevelError_SilencesInfoAndWarn) {
    setenv("PTO_LOG_LEVEL", "error", 1);
    HostLogger::get_instance().reinitialize();

    HostLogger &logger = HostLogger::get_instance();
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::ERROR));
    EXPECT_FALSE(logger.is_enabled(HostLogLevel::WARN));
    EXPECT_FALSE(logger.is_enabled(HostLogLevel::INFO));
    EXPECT_FALSE(logger.is_enabled(HostLogLevel::DEBUG));
}

TEST_F(HostLoggerTest, EnvLevelDebug_EnablesAllNumericLevels) {
    setenv("PTO_LOG_LEVEL", "debug", 1);
    HostLogger::get_instance().reinitialize();

    HostLogger &logger = HostLogger::get_instance();
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::ERROR));
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::WARN));
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::INFO));
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::DEBUG));
}

TEST_F(HostLoggerTest, EnvLevelMixedCase_IsCaseInsensitive) {
    setenv("PTO_LOG_LEVEL", "WaRn", 1);
    HostLogger::get_instance().reinitialize();

    HostLogger &logger = HostLogger::get_instance();
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::WARN));
    EXPECT_FALSE(logger.is_enabled(HostLogLevel::INFO));
}

TEST_F(HostLoggerTest, EnvLevelUnknown_FallsBackToInfo) {
    setenv("PTO_LOG_LEVEL", "not_a_real_level", 1);
    HostLogger::get_instance().reinitialize();

    HostLogger &logger = HostLogger::get_instance();
    EXPECT_TRUE(logger.is_enabled(HostLogLevel::INFO));
    EXPECT_FALSE(logger.is_enabled(HostLogLevel::DEBUG));
}

// ---------- Smoke: log(...) does not crash ----------

TEST_F(HostLoggerTest, LogAboveLevel_DoesNotCrash) {
    setenv("PTO_LOG_LEVEL", "info", 1);
    HostLogger::get_instance().reinitialize();

    // Act: exercise the log formatter path.  Output goes to stdout/stderr; the
    // observable property we assert is that no exception is thrown and the
    // process does not abort.
    HostLogger::get_instance().log(HostLogLevel::INFO, "unit-test info %d", 42);
    HostLogger::get_instance().log(HostLogLevel::DEBUG, "this should be filtered");
    SUCCEED();
}
