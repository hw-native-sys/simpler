/**
 * Device logging header for AICPU kernel
 */

#pragma once

#include <sys/syscall.h>
#include <unistd.h>

#include <cassert>

#include "dlog_pub.h"

extern bool g_isLogEnableDebug;
extern bool g_isLogEnableInfo;
extern bool g_isLogEnableWarn;
extern bool g_isLogEnableError;

static inline bool IsLogEnableDebug() { return g_isLogEnableDebug; }
static inline bool IsLogEnableInfo() { return g_isLogEnableInfo; }
static inline bool IsLogEnableWarn() { return g_isLogEnableWarn; }
static inline bool IsLogEnableError() { return g_isLogEnableError; }

#define GET_TID() syscall(__NR_gettid)
constexpr const char *TILE_FWK_DEVICE_MACHINE = "AI_CPU";

inline bool IsDebugMode() { return g_isLogEnableDebug; }

#define D_DEV_LOGD(MODE_NAME, fmt, ...)                                                 \
    do {                                                                                \
        if (IsLogEnableDebug()) {                                                       \
            dlog_debug(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                               \
    } while (false)

#define D_DEV_LOGI(MODE_NAME, fmt, ...)                                                \
    do {                                                                               \
        if (IsLogEnableInfo()) {                                                       \
            dlog_info(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                              \
    } while (false)

#define D_DEV_LOGW(MODE_NAME, fmt, ...)                                                \
    do {                                                                               \
        if (IsLogEnableWarn()) {                                                       \
            dlog_warn(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                              \
    } while (false)

#define D_DEV_LOGE(MODE_NAME, fmt, ...)                                                 \
    do {                                                                                \
        if (IsLogEnableError()) {                                                       \
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

void InitLogSwitch();
