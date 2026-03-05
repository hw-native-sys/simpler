/**
 * @file unified_log_device.cpp
 * @brief Unified logging - Device implementation
 */

#include "common/unified_log.h"
#include "aicpu/device_log.h"

#include <cstdarg>

void unified_log_error(const char* func, const char* fmt, ...) {
    if (!is_log_enable_error()) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dev_log_error(func, "%s", buffer);
}

void unified_log_warn(const char* func, const char* fmt, ...) {
    if (!is_log_enable_warn()) {
        return;
    }
    
    va_list args;
    va_start(args, fmt);
    
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    dev_log_warn(func, "%s", buffer);
}

void unified_log_info(const char* func, const char* fmt, ...) {
    if (!is_log_enable_info()) {
        return;
    }
    
    va_list args;
    va_start(args, fmt);
    
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    dev_log_info(func, "%s", buffer);
}

void unified_log_debug(const char* func, const char* fmt, ...) {
    if (!is_log_enable_debug()) {
        return;
    }
    
    va_list args;
    va_start(args, fmt);
    
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    dev_log_debug(func, "%s", buffer);
}

void unified_log_always(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dev_log_always(func, "%s", buffer);
}