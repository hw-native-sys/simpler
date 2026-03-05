#include "common.h"

#ifdef __linux__
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <vector>
#endif

/**
 * 使用 addr2line 将地址转换为 文件:行号 信息
 * 使用 -i 标志展开内联，返回第一行（最内层实际代码位置）
 * 如果存在内联，同时通过 inline_chain 返回外层调用链
 */
#ifdef __linux__
static std::string addr_to_line(const char* executable, void* addr,
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
std::string get_stacktrace(int skip_frames) {
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

            void* addr = (void*)((char*)buffer[i] - 1);

            Dl_info dl_info;
            std::string inline_chain;
            if (dladdr(addr, &dl_info) && dl_info.dli_fname) {
                void* rel_addr = (void*)((char*)addr - (char*)dl_info.dli_fbase);
                std::string addr2line_result = addr_to_line(dl_info.dli_fname, rel_addr, &inline_chain);

                if (addr2line_result.empty()) {
                    addr2line_result = addr_to_line(dl_info.dli_fname, addr, &inline_chain);
                }

                if (!addr2line_result.empty()) {
                    frame_info = std::string(dl_info.dli_fname) + ": " + addr2line_result;
                }
            }

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

// AssertionError 构造函数
static std::string build_assert_message(const char* condition, const char* file, int line) {
    std::string msg = "断言失败: " + std::string(condition) + "\n";
    msg += "  位置: " + std::string(file) + ":" + std::to_string(line) + "\n";
    msg += get_stacktrace(3);
    return msg;
}

AssertionError::AssertionError(const char* condition, const char* file, int line)
    : std::runtime_error(build_assert_message(condition, file, line)),
      condition_(condition), file_(file), line_(line) {}

[[noreturn]] void assert_impl(const char* condition, const char* file, int line) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "断言失败: %s\n", condition);
    fprintf(stderr, "位置: %s:%d\n", file, line);
    fprintf(stderr, "%s", get_stacktrace(2).c_str());
    fprintf(stderr, "========================================\n\n");
    fflush(stderr);

    throw AssertionError(condition, file, line);
}
