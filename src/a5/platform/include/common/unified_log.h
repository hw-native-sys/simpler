/**
 * @file unified_log.h
 * @brief Unified logging interface using link-time polymorphism
 *
 * Provides unified logging API across Host and Device platforms.
 * Implementation is automatically selected at link time:
 * - Host builds link unified_log_host.cpp
 * - Device builds link unified_log_device.cpp
 */

#ifndef PLATFORM_UNIFIED_LOG_H_
#define PLATFORM_UNIFIED_LOG_H_

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Unified logging functions
void unified_log_error(const char* func, const char* fmt, ...);
void unified_log_warn(const char* func, const char* fmt, ...);
void unified_log_info(const char* func, const char* fmt, ...);
void unified_log_debug(const char* func, const char* fmt, ...);
void unified_log_always(const char* func, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// Convenience macros (automatically capture function name)
#define LOG_ERROR(fmt, ...) unified_log_error(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  unified_log_warn(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  unified_log_info(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) unified_log_debug(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_ALWAYS(fmt, ...) unified_log_always(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)

#endif  // PLATFORM_UNIFIED_LOG_H_

