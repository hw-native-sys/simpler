/**
 * @file device_log.cpp
 * @brief Simulation Platform Log Implementation
 *
 * Provides log enable flags and initialization for simulation environment.
 * Log levels can be controlled via PTO_LOG_LEVEL environment variable.
 */

#include "aicpu/device_log.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

// =============================================================================
// Log Enable Flags (Simulation: controlled by PTO_LOG_LEVEL)
// =============================================================================

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = true;
bool g_is_log_enable_warn = true;
bool g_is_log_enable_error = true;

// =============================================================================
// Platform Constant
// =============================================================================

const char* TILE_FWK_DEVICE_MACHINE = "SIM_CPU";

// =============================================================================
// Log Initialization (Read from PTO_LOG_LEVEL environment variable)
// =============================================================================

void init_log_switch() {
    // Read PTO_LOG_LEVEL environment variable
    const char* level_str = std::getenv("PTO_LOG_LEVEL");
    
    if (level_str != nullptr) {
        // Convert to lowercase for comparison
        char level[64];
        strncpy(level, level_str, sizeof(level) - 1);
        level[sizeof(level) - 1] = '\0';
        
        for (char* p = level; *p; ++p) {
            *p = tolower(*p);
        }
        
        // Parse log level
        if (strcmp(level, "silent") == 0 || strcmp(level, "error") == 0) {
            // Silent/Error: only errors
            g_is_log_enable_debug = false;
            g_is_log_enable_info = false;
            g_is_log_enable_warn = false;
            g_is_log_enable_error = true;
        } else if (strcmp(level, "normal") == 0 || strcmp(level, "info") == 0) {
            // Normal/Info: info, warn, error (default)
            g_is_log_enable_debug = false;
            g_is_log_enable_info = true;
            g_is_log_enable_warn = true;
            g_is_log_enable_error = true;
        } else if (strcmp(level, "verbose") == 0 || strcmp(level, "debug") == 0) {
            // Verbose/Debug: all levels
            g_is_log_enable_debug = true;
            g_is_log_enable_info = true;
            g_is_log_enable_warn = true;
            g_is_log_enable_error = true;
        } else {
            // Unknown value: default to INFO
            g_is_log_enable_debug = false;
            g_is_log_enable_info = true;
            g_is_log_enable_warn = true;
            g_is_log_enable_error = true;
        }
    } else {
        // No environment variable: default to INFO level
        g_is_log_enable_debug = false;
        g_is_log_enable_info = true;
        g_is_log_enable_warn = true;
        g_is_log_enable_error = true;
    }
}

// =============================================================================
// Platform-Specific Logging Functions (Simulation: use printf)
// =============================================================================

void dev_log_debug(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[DEBUG] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void dev_log_info(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[INFO] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void dev_log_warn(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[WARN] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void dev_log_error(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[ERROR] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void dev_log_always(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[ALWAYS] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}
