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
#include <vector>
#endif

/**
 * 使用 addr2line 将地址转换为 文件:行号 信息
 * 使用 -i 标志展开内联，返回第一行（最内层实际代码位置）
 * 如果存在内联，同时通过 inline_chain 返回外层调用链
 */
#ifdef __linux__
inline std::string addr_to_line(const char* executable, void* addr,
                                std::string* inline_chain = nullptr) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "addr2line -e %s -f -C -p -i %p 2>/dev/null", executable, addr);

    std::array<char, 256> buffer;
    std::string raw_output;

    FILE* pipe = popen(cmd, "r");
    if (pipe) {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            raw_output += buffer.data();
        }
        pclose(pipe);
    }

    if (raw_output.empty() || raw_output.find("??") != std::string::npos) {
        return "";
    }

    // 按行分割
    std::vector<std::string> lines;
    size_t pos = 0;
    while (pos < raw_output.size()) {
        size_t nl = raw_output.find('\n', pos);
        if (nl == std::string::npos) nl = raw_output.size();
        std::string line = raw_output.substr(pos, nl - pos);
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
        pos = nl + 1;
    }

    if (lines.empty()) return "";

    // 第一行是最内层的实际代码位置，后续行是外层内联调用者
    if (inline_chain && lines.size() > 1) {
        *inline_chain = "";
        for (size_t j = 1; j < lines.size(); j++) {
            *inline_chain += "    [inlined by] " + lines[j] + "\n";
        }
    }

    return lines.front();
}
#endif

/**
 * 获取当前调用栈信息（包含文件路径和行号）
 * 通过 dladdr 定位每个栈帧所在的共享库，并用相对地址调用 addr2line
 */
inline std::string get_stacktrace(int skip_frames = 1) {
    std::string result;
#ifdef __linux__
    const int max_frames = 64;
    void* buffer[max_frames];
    int nframes = backtrace(buffer, max_frames);
    char** symbols = backtrace_symbols(buffer, nframes);

    if (symbols) {
        result = "调用栈:\n";
        for (int i = skip_frames; i < nframes; i++) {
            std::string frame_info;

            // backtrace() 返回的是返回地址（call 指令的下一条指令）
            // 减 1 使地址落在 call 指令内部，避免解析到下一个函数
            void* addr = (void*)((char*)buffer[i] - 1);

            // 使用 dladdr 获取栈帧所在的共享库信息
            Dl_info dl_info;
            std::string inline_chain;
            if (dladdr(addr, &dl_info) && dl_info.dli_fname) {
                // 计算相对于共享库基地址的偏移
                void* rel_addr = (void*)((char*)addr - (char*)dl_info.dli_fbase);
                std::string addr2line_result = addr_to_line(dl_info.dli_fname, rel_addr, &inline_chain);

                // 如果相对地址失败，尝试用绝对地址（适用于非 PIE 可执行文件）
                if (addr2line_result.empty()) {
                    addr2line_result = addr_to_line(dl_info.dli_fname, addr, &inline_chain);
                }

                if (!addr2line_result.empty()) {
                    frame_info = std::string(dl_info.dli_fname) + ": " + addr2line_result;
                }
            }

            // 如果 addr2line 失败，使用 backtrace_symbols 的输出并 demangle
            if (frame_info.empty()) {
                std::string frame(symbols[i]);

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
            if (!inline_chain.empty()) {
                result += inline_chain;
            }
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
