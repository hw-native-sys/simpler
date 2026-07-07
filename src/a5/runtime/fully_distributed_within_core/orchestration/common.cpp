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

struct PTO2Runtime;

// Unified-log error sink. Forward-declared here rather than pulled via
// common/unified_log.h: that header lives under common/log/include, which is
// not on the orchestration .so build's include path. The symbol resolves at
// link time for the runtime targets, and at dlopen time for the orchestration
// .so (against the executor's unified_log_device), so onboard diagnostics still
// reach the CANN device log.
extern "C" void unified_log_error(const char *func, const char *fmt, ...);

// Plain global (not thread_local) to avoid glibc TLSDESC stale-resolution
// crash (BZ #32412) when the orchestration SO is dlclose'd/re-dlopen'd
// between execution rounds.  All orchestrator threads bind the same rt
// value, so per-thread storage is unnecessary.
//
// On the CCEC AICore build this MUST be __gm__ (a GM-resident global) — CCEC
// forbids writable BSS/file-scope storage in __aicore__ functions. __gm__
// globals live in global memory (a valid AICore address space) and are
// permitted (cf. s_dump_gd in dist_engine.cpp). All 9 AICore workers share
// the same GM and bind the same rt, so the single shared slot is correct.
// On host/AICPU builds __gm__ is empty → plain BSS global, unchanged.
//
// EXTERNAL + weak (NOT an anonymous-namespace static): the a5 AICore image
// compiles this TU TWICE (once for AIC, once for AIV) and links both object
// sets into one binary. With internal linkage each object gets its OWN private
// g_current_runtime slot; framework_bind_runtime / framework_current_runtime are
// weak so the linker keeps ONE of each, and if the surviving bind and current
// come from opposite TUs they touch DIFFERENT slots — bind writes one, current
// reads the other (nullptr) → orchestration's current_runtime()->ops derefs
// null and faults right at the orchestration entry (observed: every core stuck
// at the replay crumb, nondeterministic 507000/507046). A single weak external
// symbol collapses to ONE shared slot across both TUs, matching the single weak
// bind/current definitions. The unique name avoids clashing with the AICPU .so's
// own g_current_runtime (that binary is linked separately).
__attribute__((weak)) __gm__ PTO2Runtime *pto_fdwc_g_current_runtime = nullptr;

// __attribute__((weak)): the a5 AICore runtime binary compiles this TU twice
// (once for AIC, once for AIV) and links both object sets into one image with
// `ld -m aicorelinux`. Without weak, the extern "C" definitions collide as
// duplicate symbols at that final link (same as dist_core_main, which is weak
// for the same reason). Weak keeps a single definition; g_current_runtime is a
// shared __gm__ slot so bind/current stay consistent across all cores.
extern "C" __attribute__((visibility("default"))) __attribute__((weak)) __aicore__ void framework_bind_runtime(__gm__ PTO2Runtime *rt) {
    pto_fdwc_g_current_runtime = rt;
}

// Keep current_runtime local to this .so so orchestration helpers do not
// accidentally bind to the AICPU binary's same-named symbol.
extern "C" __attribute__((visibility("hidden"))) __attribute__((weak)) __aicore__ __gm__ PTO2Runtime *framework_current_runtime() { return pto_fdwc_g_current_runtime; }

/**
 * Use addr2line to convert an address to file:line information.
 * Uses the -i flag to expand inlines; returns the first line (innermost actual code location).
 * If inlining is present, also returns the outer call chain via inline_chain.
 */
#ifdef __linux__
static std::string addr_to_line(const char *executable, void *addr, std::string *inline_chain = nullptr) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "addr2line -e %s -f -C -p -i %p 2>/dev/null", executable, addr);

    std::array<char, 256> buffer;
    std::string raw_output;

    FILE *pipe = popen(cmd, "r");
    if (pipe) {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            raw_output += buffer.data();
        }
        pclose(pipe);
    }

    if (raw_output.empty() || raw_output.find("??") != std::string::npos) {
        return "";
    }

    // Split by lines
    std::vector<std::string> lines;
    size_t pos = 0;
    while (pos < raw_output.size()) {
        size_t nl = raw_output.find('\n', pos);
        if (nl == std::string::npos) nl = raw_output.size();
        std::string line = raw_output.substr(pos, nl - pos);
        while (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!line.empty()) lines.push_back(line);
        pos = nl + 1;
    }

    if (lines.empty()) return "";

    // First line is the innermost actual code location; subsequent lines are outer inline callers
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
 * Get current stack trace information (including file paths and line numbers).
 * Uses dladdr to locate the shared library for each stack frame, then calls addr2line with relative addresses.
 */
std::string get_stacktrace(int skip_frames) {
    (void)skip_frames;  // May be unused on non-Linux platforms
    std::string result;
#ifdef __linux__
    const int max_frames = 64;
    void *buffer[max_frames];
    int nframes = backtrace(buffer, max_frames);
    char **symbols = backtrace_symbols(buffer, nframes);

    if (symbols) {
        result = "Stack trace:\n";
        for (int i = skip_frames; i < nframes; i++) {
            std::string frame_info;

            void *addr = (void *)((char *)buffer[i] - 1);

            Dl_info dl_info;
            std::string inline_chain;
            if (dladdr(addr, &dl_info) && dl_info.dli_fname) {
                void *rel_addr = (void *)((char *)addr - (char *)dl_info.dli_fbase);
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
                    char *demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
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
    result = "(Stack trace is only available on Linux)\n";
#endif
    return result;
}

// AssertionError constructor
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
    // Use unified_log_error directly rather than the LOG_ERROR macro: that macro
    // lives in pto_orchestration_api.h and expands to
    // current_runtime()->ops->log_error, but the ops table's definition pulls in
    // pto_types.h (Arg → __aicore__-only to_u64), which the AICore build of this
    // TU cannot compile. unified_log_error reaches the same sink without that
    // dependency.
    unified_log_error(__FUNCTION__, "\n========================================");
    unified_log_error(__FUNCTION__, "Assertion failed: %s", condition);
    unified_log_error(__FUNCTION__, "Location: %s:%d", file, line);
    unified_log_error(__FUNCTION__, "%s", get_stacktrace(2).c_str());
    unified_log_error(__FUNCTION__, "========================================\n");

    throw AssertionError(condition, file, line);
}
