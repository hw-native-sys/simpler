/**
 * Device Logging Header for AICPU Simulation
 *
 * Provides printf-based logging for simulating device-side logging.
 * Replaces CANN dlog calls with standard printf.
 */

#pragma once

#include <cstdio>
#include <cstdint>

// Log enable flags (always enabled in simulation)
static bool g_is_log_enable_debug = true;
static bool g_is_log_enable_info = true;
static bool g_is_log_enable_warn = true;
static bool g_is_log_enable_error = true;

static inline bool is_log_enable_debug() { return g_is_log_enable_debug; }
static inline bool is_log_enable_info() { return g_is_log_enable_info; }
static inline bool is_log_enable_warn() { return g_is_log_enable_warn; }
static inline bool is_log_enable_error() { return g_is_log_enable_error; }

// Thread ID helper (simplified for simulation)
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#define GET_TID() syscall(__NR_gettid)
#else
#define GET_TID() 0
#endif

constexpr const char* TILE_FWK_DEVICE_MACHINE = "SIM_CPU";

inline bool is_debug_mode() { return g_is_log_enable_debug; }

// Simple printf-based logging macros
#define D_DEV_LOGD(MODE_NAME, fmt, ...)                                   \
    do {                                                                  \
        if (is_log_enable_debug()) {                                       \
            printf("[DEBUG][%s] %s: " fmt "\n", MODE_NAME, __FUNCTION__, ##__VA_ARGS__); \
        }                                                                 \
    } while (false)

#define D_DEV_LOGI(MODE_NAME, fmt, ...)                                   \
    do {                                                                  \
        if (is_log_enable_info()) {                                        \
            printf("[INFO][%s] %s: " fmt "\n", MODE_NAME, __FUNCTION__, ##__VA_ARGS__);  \
        }                                                                 \
    } while (false)

#define D_DEV_LOGW(MODE_NAME, fmt, ...)                                   \
    do {                                                                  \
        if (is_log_enable_warn()) {                                        \
            printf("[WARN][%s] %s: " fmt "\n", MODE_NAME, __FUNCTION__, ##__VA_ARGS__);  \
        }                                                                 \
    } while (false)

#define D_DEV_LOGE(MODE_NAME, fmt, ...)                                   \
    do {                                                                  \
        if (is_log_enable_error()) {                                       \
            printf("[ERROR][%s] %s: " fmt "\n", MODE_NAME, __FUNCTION__, ##__VA_ARGS__); \
        }                                                                 \
    } while (false)

#define DEV_DEBUG(fmt, args...) D_DEV_LOGD(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_INFO(fmt, args...)  D_DEV_LOGI(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...)  D_DEV_LOGW(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_ERROR(fmt, args...) D_DEV_LOGE(TILE_FWK_DEVICE_MACHINE, fmt, ##args)

#define DEV_ASSERT_MSG(expr, fmt, args...)                           \
    do {                                                             \
        if (!(expr)) {                                               \
            DEV_ERROR("Assertion failed (%s): " fmt, #expr, ##args); \
        }                                                            \
    } while (0)

#define DEV_ASSERT(expr)                               \
    do {                                               \
        if (!(expr)) {                                 \
            DEV_ERROR("Assertion failed (%s)", #expr); \
        }                                              \
    } while (0)

#define DEV_DEBUG_ASSERT(expr)                                                      \
    do {                                                                            \
        if (!(expr)) {                                                              \
            DEV_ERROR("Assertion failed at %s:%d (%s)", __FILE__, __LINE__, #expr); \
        }                                                                           \
    } while (0)

#define DEV_DEBUG_ASSERT_MSG(expr, fmt, args...) DEV_ASSERT_MSG(expr, fmt, ##args)

// No-op initialization for simulation
inline void init_log_switch() {}
