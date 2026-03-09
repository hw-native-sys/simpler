/**
 * @file host_log.h
 * @brief Unified Host Logging System
 *
 * Provides thread-safe logging interface for Host-side C++ code.
 * Integrates with Python logging system via environment variables.
 *
 * Environment Variables:
 * - PTO_LOG_LEVEL: error/warn/info/debug (default: info)
 * - PTO_LOG_FILE: Optional log file path (default: stdout/stderr)
 *
 * Log Levels (1-to-1 mapping with Python logging):
 * - ERROR: Only errors and failures
 * - WARN: Warnings and above
 * - INFO: Key progress steps and above (default)
 * - DEBUG: Detailed debug info and above
 */

#ifndef PLATFORM_HOST_LOG_H_
#define PLATFORM_HOST_LOG_H_

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>

// =============================================================================
// Log Level Enum
// =============================================================================

enum class HostLogLevel {
    ALWAYS = -1, // always logging
    ERROR = 0,   // error level only
    WARN = 1,    // warn level and above
    INFO = 2,    // info level and above (default)
    DEBUG = 3    // debug level (all messages)
};

// =============================================================================
// HostLogger Class (Singleton)
// =============================================================================

class HostLogger {
public:
    // Get singleton instance
    static HostLogger& get_instance();

    // Log a message with specified level
    void log(HostLogLevel level, const char* format, ...);
    
    // Check if a log level is enabled
    bool is_enabled(HostLogLevel level) const;

    // Reinitialize from environment (useful if env vars change)
    void reinitialize();

private:
    HostLogger();
    ~HostLogger();
    
    // Delete copy/move constructors
    HostLogger(const HostLogger&) = delete;
    HostLogger& operator=(const HostLogger&) = delete;
    HostLogger(HostLogger&&) = delete;
    HostLogger& operator=(HostLogger&&) = delete;

    // Initialize from environment variables
    void init_from_env();
    
    // Get level name string
    const char* get_level_name(HostLogLevel level) const;
    
    // Get output file handle (FILE* for stdout/stderr or file)
    FILE* get_output_file(HostLogLevel level);

    // Member variables
    HostLogLevel current_level_;
    std::string log_file_path_;
    FILE* log_file_handle_;
    std::mutex mutex_;
    bool initialized_;
};

// =============================================================================
// Logging Macros (High-Level Interface)
// =============================================================================

#define HOST_LOG_ERROR(fmt, ...) \
    do { \
        HostLogger::get_instance().log(HostLogLevel::ERROR, fmt, ##__VA_ARGS__); \
    } while(0)

#define HOST_LOG_WARN(fmt, ...) \
    do { \
        HostLogger::get_instance().log(HostLogLevel::WARN, fmt, ##__VA_ARGS__); \
    } while(0)

#define HOST_LOG_INFO(fmt, ...) \
    do { \
        if (HostLogger::get_instance().is_enabled(HostLogLevel::INFO)) { \
            HostLogger::get_instance().log(HostLogLevel::INFO, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define HOST_LOG_DEBUG(fmt, ...) \
    do { \
        if (HostLogger::get_instance().is_enabled(HostLogLevel::DEBUG)) { \
            HostLogger::get_instance().log(HostLogLevel::DEBUG, fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#endif // PLATFORM_HOST_LOG_H_

