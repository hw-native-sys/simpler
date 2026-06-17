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

#ifdef __linux__
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <vector>

inline std::string addr_to_line(const char *executable, void *addr, std::string *inline_chain = nullptr)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "addr2line -e %s -f -C -p -i %p 2>/dev/null", executable, addr);

    std::array<char, 256> buffer;
    std::string raw_output;

    FILE *pipe = popen(cmd, "r");
    if (pipe)
    {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) raw_output += buffer.data();
        pclose(pipe);
    }

    if (raw_output.empty() || raw_output.find("??") != std::string::npos) return "";

    std::vector<std::string> lines;
    size_t pos = 0;
    while (pos < raw_output.size())
    {
        size_t nl = raw_output.find('\n', pos);
        if (nl == std::string::npos) nl = raw_output.size();
        std::string line = raw_output.substr(pos, nl - pos);
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
        pos = nl + 1;
    }

    if (lines.empty()) return "";

    if (inline_chain && lines.size() > 1)
    {
        *inline_chain = "";
        for (size_t j = 1; j < lines.size(); j++) *inline_chain += "    [inlined by] " + lines[j] + "\n";
    }

    return lines.front();
}
#endif

inline std::string get_stacktrace(int skip_frames)
{
    (void)skip_frames;  // May be unused on non-Linux platforms
    std::string result;
#ifdef __linux__
    const int max_frames = 64;
    void *buffer[max_frames];
    int nframes = backtrace(buffer, max_frames);
    char **symbols = backtrace_symbols(buffer, nframes);

    if (symbols)
    {
        result = "Stack trace:\n";
        for (int i = skip_frames; i < nframes; i++)
        {
            std::string frame_info;

            void *addr = (void *)((char *)buffer[i] - 1);

            Dl_info dl_info;
            std::string inline_chain;
            if (dladdr(addr, &dl_info) && dl_info.dli_fname)
            {
                void *rel_addr = (void *)((char *)addr - (char *)dl_info.dli_fbase);
                std::string addr2line_result = addr_to_line(dl_info.dli_fname, rel_addr, &inline_chain);

                if (addr2line_result.empty()) addr2line_result = addr_to_line(dl_info.dli_fname, addr, &inline_chain);

                if (!addr2line_result.empty()) frame_info = std::string(dl_info.dli_fname) + ": " + addr2line_result;
            }

            if (frame_info.empty())
            {
                std::string frame(symbols[i]);

                size_t start = frame.find('(');
                size_t end = frame.find('+', start);
                if (start != std::string::npos && end != std::string::npos)
                {
                    std::string mangled = frame.substr(start + 1, end - start - 1);
                    int status;
                    char *demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                    if (status == 0 && demangled)
                    {
                        frame = frame.substr(0, start + 1) + demangled + frame.substr(end);
                        free(demangled);
                    }
                }
                frame_info = frame;
            }

            char buf[16];
            snprintf(buf, sizeof(buf), "  #%d ", i - skip_frames);
            result += buf + frame_info + "\n";
            if (!inline_chain.empty()) result += inline_chain;
        }
        free(symbols);
    }
#else
    result = "(Stack trace is only available on Linux)\n";
#endif
    return result;
}

inline std::string build_assert_message(const char *condition, const char *file, int line)
{
    std::string msg = "Assertion failed: " + std::string(condition) + "\n";
    msg += "  Location: " + std::string(file) + ":" + std::to_string(line) + "\n";
    msg += get_stacktrace(3);
    return msg;
}

class AssertionError : public std::runtime_error
{
public:
    AssertionError(const char *condition, const char *file, int line) :
        std::runtime_error(build_assert_message(condition, file, line)),
        condition_(condition),
        file_(file),
        line_(line)
    {}

    const char *condition() const
    {
        return condition_;
    }
    const char *file() const
    {
        return file_;
    }
    int line() const
    {
        return line_;
    }

private:
    const char *condition_;
    const char *file_;
    int line_;
};

[[noreturn]] inline void assert_impl(const char *condition, const char *file, int line)
{
    throw AssertionError(condition, file, line);
}

#ifdef NDEBUG
#define debug_assert(cond) ((void)0)
#else
#define debug_assert(cond)                          \
    do {                                            \
        if (!(cond))                                \
        {                                           \
            assert_impl(#cond, __FILE__, __LINE__); \
        }                                           \
    } while (0)
#endif

#define always_assert(cond)                         \
    do {                                            \
        if (!(cond))                                \
        {                                           \
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
