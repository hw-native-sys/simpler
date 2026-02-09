/**
 * @file device_log.h
 * @brief Unified Device Logging Interface for AICPU
 *
 * Provides logging macros that work on both real hardware (using CANN dlog)
 * and simulation (using printf). Uses conditional compilation to select
 * the appropriate backend with a layered design to minimize code duplication.
 *
 * Platform Support:
 * - a2a3: Real hardware with CANN dlog API
 * - a2a3sim: Host-based simulation using printf
 */

#ifndef PLATFORM_DEVICE_LOG_H_
#define PLATFORM_DEVICE_LOG_H_

#include <cstdio>
#include <cstdint>

// =============================================================================
// Platform Detection and Thread ID
// =============================================================================

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#define GET_TID() syscall(SYS_gettid)
#else
#define GET_TID() 0
#endif

// =============================================================================
// Log Enable Flags
// =============================================================================

// Unified: flags defined in platform-specific device_log.cpp
extern bool g_is_log_enable_debug;
extern bool g_is_log_enable_info;
extern bool g_is_log_enable_warn;
extern bool g_is_log_enable_error;

// Platform constant (defined in platform-specific device_log.cpp)
extern const char* TILE_FWK_DEVICE_MACHINE;

// =============================================================================
// Platform-Specific Logging Functions (Low-Level Layer)
// =============================================================================

// Platform-specific logging functions (implemented in device_log.cpp)
void dev_log_debug(const char* func, const char* fmt, ...);
void dev_log_info(const char* func, const char* fmt, ...);
void dev_log_warn(const char* func, const char* fmt, ...);
void dev_log_error(const char* func, const char* fmt, ...);

// =============================================================================
// High-Level Logging Macros (Platform-Independent Layer)
// =============================================================================

#define D_DEV_LOGD(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_debug()) { \
            dev_log_debug(__FUNCTION__, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define D_DEV_LOGI(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_info()) { \
            dev_log_info(__FUNCTION__, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define D_DEV_LOGW(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_warn()) { \
            dev_log_warn(__FUNCTION__, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define D_DEV_LOGE(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_error()) { \
            dev_log_error(__FUNCTION__, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

// =============================================================================
// Convenience Macros
// =============================================================================

#define DEV_INFO(fmt, args...)  D_DEV_LOGI(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...)  D_DEV_LOGW(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_ERROR(fmt, args...) D_DEV_LOGE(TILE_FWK_DEVICE_MACHINE, fmt, ##args)

// =============================================================================
// Platform-Specific Assertion
// =============================================================================

// =============================================================================
// Assertion (Unified: both platforms use assert)
// =============================================================================

#include <cassert>
#define DEV_ASSERT(condition) assert(condition)

// =============================================================================
// Conditional Check Macros
// =============================================================================

#define DEV_CHECK_COND_RETURN_VOID(cond, fmt, ...)          \
    do {                                                    \
        if (!(cond)) {                                      \
            DEV_ERROR(fmt, ##__VA_ARGS__);                  \
            DEV_ASSERT(0);                                  \
            return;                                         \
        }                                                   \
    } while(0)

#define DEV_CHECK_COND_RETURN(cond, retval, fmt, ...)       \
    do {                                                    \
        if (!(cond)) {                                      \
            DEV_ERROR(fmt, ##__VA_ARGS__);                  \
            DEV_ASSERT(0);                                  \
            return (retval);                                \
        }                                                   \
    } while(0)

#define DEV_CHECK_POINTER_NULL_RETURN_VOID(ptr, fmt, ...)   \
    do {                                                    \
        if ((ptr) == nullptr) {                             \
            DEV_ERROR(fmt, ##__VA_ARGS__);                  \
            DEV_ASSERT(0);                                  \
            return;                                         \
        }                                                   \
    } while(0)

// =============================================================================
// Helper Functions
// =============================================================================

// Check if log level is enabled (inline for efficiency)
inline bool is_log_enable_debug() { return g_is_log_enable_debug; }
inline bool is_log_enable_info()  { return g_is_log_enable_info; }
inline bool is_log_enable_warn()  { return g_is_log_enable_warn; }
inline bool is_log_enable_error() { return g_is_log_enable_error; }

// Initialize log switch (platform-specific implementation)
void init_log_switch();

#endif  // PLATFORM_DEVICE_LOG_H_
