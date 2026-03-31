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

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>
#include <string>

/**
 * Get the current stack trace, including file paths and line numbers.
 * Implemented in common.cpp.
 */
std::string get_stacktrace(int skip_frames = 1);

/**
 * Assertion failure exception with condition, file, line, and stack trace.
 */
class AssertionError : public std::runtime_error {
 public:
    AssertionError(const char* condition, const char* file, int line);

    const char* condition() const { return condition_; }
    const char* file() const { return file_; }
    int line() const { return line_; }

 private:
    const char* condition_;
    const char* file_;
    int line_;
};

/**
 * Assertion failure handler.
 * Implemented in common.cpp.
 */
[[noreturn]] void assert_impl(const char* condition, const char* file, int line);

/**
 * debug_assert macro:
 * checks the condition in debug builds and throws with a stack trace on failure.
 * It is a no-op in release builds (NDEBUG).
 */
#ifdef NDEBUG
#define debug_assert(cond) ((void)0)
#else
#define debug_assert(cond)                          \
    do {                                            \
        if (!(cond)) {                              \
            assert_impl(#cond, __FILE__, __LINE__); \
        }                                           \
    } while (0)
#endif

/**
 * always_assert macro:
 * checks the condition in both debug and release builds.
 */
#define always_assert(cond)                         \
    do {                                            \
        if (!(cond)) {                              \
            assert_impl(#cond, __FILE__, __LINE__); \
        }                                           \
    } while (0)

#define PTO_PRAGMA(x) _Pragma(#x)

#if defined(__clang__)
#define MAYBE_UNINITIALIZED_BEGIN                          \
    PTO_PRAGMA(clang diagnostic push)                      \
    PTO_PRAGMA(clang diagnostic ignored "-Wuninitialized") \
    PTO_PRAGMA(clang diagnostic ignored "-Wsometimes-uninitialized")
#define MAYBE_UNINITIALIZED_END PTO_PRAGMA(clang diagnostic pop)
#elif defined(__GNUC__)
#define MAYBE_UNINITIALIZED_BEGIN                        \
    PTO_PRAGMA(GCC diagnostic push)                      \
    PTO_PRAGMA(GCC diagnostic ignored "-Wuninitialized") \
    PTO_PRAGMA(GCC diagnostic ignored "-Wmaybe-uninitialized")
#define MAYBE_UNINITIALIZED_END PTO_PRAGMA(GCC diagnostic pop)
#else
#define MAYBE_UNINITIALIZED_BEGIN
#define MAYBE_UNINITIALIZED_END
#endif
