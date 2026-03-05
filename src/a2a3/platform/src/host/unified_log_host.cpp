/**
 * @file unified_log_host.cpp
 * @brief Unified logging - Host implementation
 */

#include "common/unified_log.h"
#include "host_log.h"

#include <cstdarg>
#include <cstdio>

void unified_log_error(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::ERROR, "%s: %s", func, buffer);
}

void unified_log_warn(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    HostLogger::get_instance().log(HostLogLevel::WARN, "%s: %s", func, buffer);
}

void unified_log_info(const char* func, const char* fmt, ...) {
    if (!HostLogger::get_instance().is_enabled(HostLogLevel::INFO)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::INFO, "%s: %s", func, buffer);
}

void unified_log_debug(const char* func, const char* fmt, ...) {
    if (!HostLogger::get_instance().is_enabled(HostLogLevel::DEBUG)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::DEBUG, "%s: %s", func, buffer);
}

void unified_log_always(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::ALWAYS, "%s: %s", func, buffer);
}