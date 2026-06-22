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
 * Host-side definition of the assert_compat.h diagnostics.
 *
 * Linked only into host-side targets that include the unified Tensor but do
 * NOT link the per-arch runtime orchestration/common.cpp (which provides the
 * richer, LOG_ERROR-backed implementation). The nanobind binding is the
 * primary consumer: Tensor::make ODR-uses Tensor::init_external, which calls
 * always_assert. This keeps the binding self-contained without dragging in a
 * runtime-arch-specific translation unit.
 */
#include "assert_compat.h"

#include <cstdio>
#include <cstdlib>

#ifdef __linux__
#include <execinfo.h>
#endif

std::string get_stacktrace(int skip_frames) {
    (void)skip_frames;
    std::string result;
#ifdef __linux__
    const int max_frames = 64;
    void *buffer[max_frames];
    int nframes = backtrace(buffer, max_frames);
    char **symbols = backtrace_symbols(buffer, nframes);
    if (symbols) {
        result = "Stack trace:\n";
        for (int i = skip_frames; i < nframes; i++) {
            result += "  ";
            result += symbols[i];
            result += "\n";
        }
        free(symbols);
    }
#else
    result = "(Stack trace is only available on Linux)\n";
#endif
    return result;
}

static std::string build_assert_message(const char *condition, const char *file, int line) {
    std::string msg = "Assertion failed: " + std::string(condition) + "\n";
    msg += "  Location: " + std::string(file) + ":" + std::to_string(line) + "\n";
    msg += get_stacktrace(3);
    return msg;
}

AssertionError::AssertionError(const char *condition, const char *file, int line) :
    std::runtime_error(build_assert_message(condition, file, line)),
    condition_(condition),
    file_(file),
    line_(line) {}

[[noreturn]] void assert_impl(const char *condition, const char *file, int line) {
    throw AssertionError(condition, file, line);
}
