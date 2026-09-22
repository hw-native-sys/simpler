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
 * unified_log.h stubs: the log calls runtime headers make, printed to stderr.
 *
 * Its own file because src/common/log/unified_log_host.cpp defines the same
 * seven symbols against the real HostLogger. A case wanting that behaviour
 * compiles the logger and leaves this out; the two are alternatives, and the
 * linker can only say so when they are separable.
 */

#include <cstdarg>
#include <cstdio>

extern "C" {

int unified_log_host_span_enabled() { return 0; }

void unified_log_host_span(const struct SimplerHostSpan *) {}

void unified_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[WARN]  %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_timing(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[TIMING] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_info(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[INFO] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_debug(const char * /* func */, const char * /* fmt */, ...) {
    // Suppress debug in tests
}

}  // extern "C"
