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
 * common.h stubs: a failed always_assert throws instead of aborting.
 *
 * Weak, so a case that supplies its own definition gets its own. What a failed
 * assertion does is behaviour a test may want to observe differently —
 * test_<arch>_fatal throws a sentinel it catches by value — and
 * src/common/task_interface/assert_compat.cpp is the real host-side
 * definition the nanobind-adjacent targets compile. Strong here, either of
 * those would be a duplicate-symbol error rather than an override.
 */

#include <stdexcept>
#include <string>

__attribute__((weak)) std::string get_stacktrace(int /* skip_frames */) {
    return "<stacktrace not available in test stubs>";
}

class AssertionError : public std::runtime_error {
public:
    AssertionError(const char *condition, const char *file, int line) :
        std::runtime_error(std::string("Assertion failed: ") + condition + " at " + file + ":" + std::to_string(line)),
        condition_(condition),
        file_(file),
        line_(line) {}

    const char *condition() const { return condition_; }
    const char *file() const { return file_; }
    int line() const { return line_; }

private:
    const char *condition_;
    const char *file_;
    int line_;
};

[[noreturn]] __attribute__((weak)) void assert_impl(const char *condition, const char *file, int line) {
    throw AssertionError(condition, file, line);
}
