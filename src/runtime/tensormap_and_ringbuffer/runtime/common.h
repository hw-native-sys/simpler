#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

#ifdef __linux__
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <memory>
#endif

/**
 * 使用 addr2line 将地址转换为 文件:行号 信息
 */
#ifdef __linux__
inline std::string addr_to_line(const char* executable, void* addr) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "addr2line -e %s -f -C -p %p 2>/dev/null", executable, addr);

    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = popen(cmd, "r");
    if (pipe) {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        pclose(pipe);
        // 移除末尾换行符
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
            result.pop_back();
        }
    }

    // 如果 addr2line 失败或返回 "??"，返回空
    if (result.empty() || result.find("??") != std::string::npos) {
        return "";
    }
    return result;
}
#endif

/**
 * 获取当前调用栈信息（包含文件路径和行号）
 */
inline std::string get_stacktrace(int skip_frames = 1) {
    std::string result;
#ifdef __linux__
    const int max_frames = 64;
    void* buffer[max_frames];
    int nframes = backtrace(buffer, max_frames);
    char** symbols = backtrace_symbols(buffer, nframes);

    // 获取当前可执行文件路径
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
    } else {
        exe_path[0] = '\0';
    }

    if (symbols) {
        result = "调用栈:\n";
        for (int i = skip_frames; i < nframes; i++) {
            std::string frame_info;

            // 尝试使用 addr2line 获取精确的文件:行号
            if (exe_path[0] != '\0') {
                std::string addr2line_result = addr_to_line(exe_path, buffer[i]);
                if (!addr2line_result.empty()) {
                    frame_info = addr2line_result;
                }
            }

            // 如果 addr2line 失败，使用 backtrace_symbols 的输出
            if (frame_info.empty()) {
                std::string frame(symbols[i]);

                // 尝试 demangle C++ 符号
                size_t start = frame.find('(');
                size_t end = frame.find('+', start);
                if (start != std::string::npos && end != std::string::npos) {
                    std::string mangled = frame.substr(start + 1, end - start - 1);
                    int status;
                    char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                    if (status == 0 && demangled) {
                        frame = frame.substr(0, start + 1) + demangled + frame.substr(end);
                        free(demangled);
                    }
                }
                frame_info = frame;
            }

            char buf[16];
            snprintf(buf, sizeof(buf), "  #%d ", i - skip_frames);
            result += buf + frame_info + "\n";
        }
        free(symbols);
    }
#else
    result = "(调用栈仅在 Linux 上可用)\n";
#endif
    return result;
}

/**
 * 断言失败异常，包含文件、行号、条件和调用栈信息
 */
class AssertionError : public std::runtime_error {
public:
    AssertionError(const char* condition, const char* file, int line)
        : std::runtime_error(build_message(condition, file, line)), condition_(condition), file_(file), line_(line) {}

    const char* condition() const { return condition_; }
    const char* file() const { return file_; }
    int line() const { return line_; }

private:
    static std::string build_message(const char* condition, const char* file, int line) {
        std::string msg = "断言失败: " + std::string(condition) + "\n";
        msg += "  位置: " + std::string(file) + ":" + std::to_string(line) + "\n";
        msg += get_stacktrace(3);  // 跳过 build_message, 构造函数, debug_assert_impl
        return msg;
    }

    const char* condition_;
    const char* file_;
    int line_;
};

/**
 * 断言失败时的处理函数
 */
[[noreturn]] inline void assert_impl(const char* condition, const char* file, int line) {
    // 打印错误信息到 stderr
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "断言失败: %s\n", condition);
    fprintf(stderr, "位置: %s:%d\n", file, line);
    fprintf(stderr, "%s", get_stacktrace(2).c_str());
    fprintf(stderr, "========================================\n\n");
    fflush(stderr);

    // 抛出异常，允许测试框架捕获
    throw AssertionError(condition, file, line);
}

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
