#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

/**
 * 获取当前调用栈信息（包含文件路径和行号）
 * 实现在 common.cpp 中
 */
std::string get_stacktrace(int skip_frames = 1);

/**
 * 断言失败异常，包含文件、行号、条件和调用栈信息
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
 * 断言失败时的处理函数
 * 实现在 common.cpp 中
 */
[[noreturn]] void assert_impl(const char* condition, const char* file, int line);

/**
 * debug_assert 宏 - 在 debug 模式下检查条件，失败时抛出异常并打印调用栈
 * 在 release 模式 (NDEBUG) 下为空操作
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
 * always_assert 宏 - 无论 debug 还是 release 模式都检查条件
 */
#define always_assert(cond)                         \
    do {                                            \
        if (!(cond)) {                              \
            assert_impl(#cond, __FILE__, __LINE__); \
        }                                           \
    } while (0)
