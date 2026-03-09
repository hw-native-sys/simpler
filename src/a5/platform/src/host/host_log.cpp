/**
 * @file host_log.cpp
 * @brief Implementation of Unified Host Logging System
 */

#include "host_log.h"

#include <cstdlib>
#include <cstring>
#include <ctime>

// =============================================================================
// HostLogger Implementation
// =============================================================================

HostLogger& HostLogger::get_instance() {
    static HostLogger instance;
    return instance;
}

HostLogger::HostLogger() 
    : current_level_(HostLogLevel::INFO),
      log_file_path_(""),
      log_file_handle_(nullptr),
      initialized_(false) {
    init_from_env();
}

HostLogger::~HostLogger() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_handle_ != nullptr && 
        log_file_handle_ != stdout && 
        log_file_handle_ != stderr) {
        fclose(log_file_handle_);
        log_file_handle_ = nullptr;
    }
}

void HostLogger::init_from_env() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Parse PTO_LOG_LEVEL environment variable
    const char* level_str = std::getenv("PTO_LOG_LEVEL");
    if (level_str != nullptr) {
        std::string level(level_str);
        
        // Convert to lowercase for comparison
        for (char& c : level) {
            c = std::tolower(c);
        }
        
        // Map log level strings to enum values (1-to-1 mapping)
        if (level == "error") {
            current_level_ = HostLogLevel::ERROR;
        } else if (level == "warn") {
            current_level_ = HostLogLevel::WARN;
        } else if (level == "info") {
            current_level_ = HostLogLevel::INFO;
        } else if (level == "debug") {
            current_level_ = HostLogLevel::DEBUG;
        } else {
            // Default to INFO for unknown values
            current_level_ = HostLogLevel::INFO;
        }
    } else {
        // Default to INFO
        current_level_ = HostLogLevel::INFO;
    }
    
    // Parse PTO_LOG_FILE environment variable
    const char* file_path = std::getenv("PTO_LOG_FILE");
    if (file_path != nullptr && strlen(file_path) > 0) {
        log_file_path_ = file_path;
        
        // Close previous file handle if it exists
        if (log_file_handle_ != nullptr && 
            log_file_handle_ != stdout && 
            log_file_handle_ != stderr) {
            fclose(log_file_handle_);
        }
        
        // Open log file in append mode
        log_file_handle_ = fopen(log_file_path_.c_str(), "a");
        if (log_file_handle_ == nullptr) {
            // Fall back to stderr if file cannot be opened
            fprintf(stderr, "[ERROR] Failed to open log file: %s\n", log_file_path_.c_str());
            log_file_handle_ = nullptr;
            log_file_path_.clear();
        }
    }
    
    initialized_ = true;
}

void HostLogger::reinitialize() {
    initialized_ = false;
    init_from_env();
}

bool HostLogger::is_enabled(HostLogLevel level) const {
    return static_cast<int>(level) <= static_cast<int>(current_level_);
}

const char* HostLogger::get_level_name(HostLogLevel level) const {
    switch (level) {
        case HostLogLevel::ERROR:
            return "ERROR";
        case HostLogLevel::WARN:
            return "WARN";
        case HostLogLevel::INFO:
            return "INFO";
        case HostLogLevel::DEBUG:
            return "DEBUG";
        case HostLogLevel::ALWAYS:
            return "ALWAYS";
        default:
            return "UNKNOWN";
    }
}

FILE* HostLogger::get_output_file(HostLogLevel level) {
    // If log file is configured, use it for all levels
    if (log_file_handle_ != nullptr) {
        return log_file_handle_;
    }
    
    // Otherwise, use stderr for ERROR/WARN, stdout for INFO/DEBUG
    if (level == HostLogLevel::ERROR || level == HostLogLevel::WARN) {
        return stderr;
    } else {
        return stdout;
    }
}

void HostLogger::log(HostLogLevel level, const char* format, ...) {
    // Check if this level is enabled
    if (!is_enabled(level)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get output file
    FILE* output = get_output_file(level);
    if (output == nullptr) {
        return;
    }
    
    // Print log level prefix (matching Python format)
    fprintf(output, "[%s] ", get_level_name(level));
    
    // Print formatted message
    va_list args;
    va_start(args, format);
    vfprintf(output, format, args);
    va_end(args);
    
    // Add newline if not already present
    if (format[strlen(format) - 1] != '\n') {
        fprintf(output, "\n");
    }
    
    // Flush to ensure immediate output
    fflush(output);
}

