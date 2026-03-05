/**
 * Device logging implementation for AICPU kernel
 */

#include "aicpu/device_log.h"
#include "dlog_pub.h"  // CANN dlog API
#include <cstdarg>
#include <cstdio>

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_warn = false;
bool g_is_log_enable_error = false;

const char* TILE_FWK_DEVICE_MACHINE = "AI_CPU";

void init_log_switch() {
    g_is_log_enable_debug = CheckLogLevel(AICPU, DLOG_DEBUG);
    g_is_log_enable_info = CheckLogLevel(AICPU, DLOG_INFO);
    g_is_log_enable_warn = CheckLogLevel(AICPU, DLOG_WARN);
    g_is_log_enable_error = CheckLogLevel(AICPU, DLOG_ERROR);
}

// =============================================================================
// Platform-Specific Logging Functions (Real Hardware: use CANN dlog API)
// =============================================================================

void dev_log_debug(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    // Add quotes around message to match original behavior (#fmt stringification)
    dlog_debug(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_info(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_info(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_warn(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_warn(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_error(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_error(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_always(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_error(AICPU, "[ALWAYS] %lu %s\n\"%s\"", GET_TID(), func, buffer);
}
