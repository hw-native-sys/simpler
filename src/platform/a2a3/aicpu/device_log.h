/**
 * Device logging header for AICPU kernel
 */

#pragma once

#include <sys/syscall.h>
#include <unistd.h>

#include <cassert>

#include "dlog_pub.h"

extern bool g_is_log_enable_debug;
extern bool g_is_log_enable_info;
extern bool g_is_log_enable_warn;
extern bool g_is_log_enable_error;

static inline bool is_log_enable_debug() { return g_is_log_enable_debug; }
static inline bool is_log_enable_info() { return g_is_log_enable_info; }
static inline bool is_log_enable_warn() { return g_is_log_enable_warn; }
static inline bool is_log_enable_error() { return g_is_log_enable_error; }

#define GET_TID() syscall(__NR_gettid)
constexpr const char *TILE_FWK_DEVICE_MACHINE = "AI_CPU";

inline bool is_debug_mode() { return g_is_log_enable_debug; }

#define D_DEV_LOGD(MODE_NAME, fmt, ...)                                                 \
    do {                                                                                \
        if (is_log_enable_debug()) {                                                       \
            dlog_debug(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                               \
    } while (false)

#define D_DEV_LOGI(MODE_NAME, fmt, ...)                                                \
    do {                                                                               \
        if (is_log_enable_info()) {                                                       \
            dlog_info(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                              \
    } while (false)

#define D_DEV_LOGW(MODE_NAME, fmt, ...)                                                \
    do {                                                                               \
        if (is_log_enable_warn()) {                                                       \
            dlog_warn(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                              \
    } while (false)

#define D_DEV_LOGE(MODE_NAME, fmt, ...)                                                 \
    do {                                                                                \
        if (is_log_enable_error()) {                                                       \
            dlog_error(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                               \
    } while (false)

#define DEV_DEBUG(fmt, args...) D_DEV_LOGD(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_INFO(fmt, args...) D_DEV_LOGI(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...) D_DEV_LOGW(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_ERROR(fmt, args...) D_DEV_LOGE(TILE_FWK_DEVICE_MACHINE, fmt, ##args)

#define DEV_ASSERT_MSG(expr, fmt, args...)                           \
    do {                                                             \
        if (!(expr)) {                                               \
            DEV_ERROR("Assertion failed (%s): " fmt, #expr, ##args); \
            assert(0);                                               \
        }                                                            \
    } while (0)

#define DEV_ASSERT(expr)                               \
    do {                                               \
        if (!(expr)) {                                 \
            DEV_ERROR("Assertion failed (%s)", #expr); \
            assert(0);                                 \
        }                                              \
    } while (0)

#define DEV_DEBUG_ASSERT(expr)                                                      \
    do {                                                                            \
        if (!(expr)) {                                                              \
            DEV_ERROR("Assertion failed at %s:%d (%s)", __FILE__, __LINE__, #expr); \
            assert(0);                                                              \
        }                                                                           \
    } while (0)

#define DEV_DEBUG_ASSERT_MSG(expr, fmt, args...) DEV_ASSERT_MSG(expr, fmt, ##args)

void init_log_switch();
