/**
 * @file core_type.h
 * @brief Core type enumeration and utilities for AICore platform
 *
 * This header defines the CoreType enumeration used across the platform
 * and runtime layers to specify which AICore type a task should run on.
 */

#ifndef PLATFORM_COMMON_CORE_TYPE_H_
#define PLATFORM_COMMON_CORE_TYPE_H_

#include <cstdint>
#include <cstring>

/**
 * Core type enumeration
 *
 * Specifies which AICore type a task should run on.
 * AIC (AICore Compute) handles compute-intensive operations.
 * AIV (AICore Vector) handles vector/SIMD operations.
 *
 * Note: Using int32_t for binary compatibility with Handshake protocol.
 */
enum class CoreType : int32_t {
    AIC = 0,  // AICore Compute
    AIV = 1   // AICore Vector
};

/**
 * Convert core type string to enum value
 *
 * @param type_str  String representation ("aic" or "aiv", case-insensitive)
 * @return Core type enum value, or CoreType::AIC if invalid
 */
inline CoreType core_type_from_string(const char* type_str) {
    if (type_str == nullptr) {
        return CoreType::AIC;
    }
    if (strcmp(type_str, "aic") == 0 || strcmp(type_str, "AIC") == 0) {
        return CoreType::AIC;
    }
    if (strcmp(type_str, "aiv") == 0 || strcmp(type_str, "AIV") == 0) {
        return CoreType::AIV;
    }
    return CoreType::AIC;
}

/**
 * Convert core type enum to string representation
 *
 * @param core_type  Core type enum value
 * @return String representation ("AIC", "AIV", or "UNKNOWN")
 */
inline const char* core_type_to_string(CoreType core_type) {
    switch (core_type) {
        case CoreType::AIC: return "AIC";
        case CoreType::AIV: return "AIV";
        default: return "UNKNOWN";
    }
}

#endif  // PLATFORM_COMMON_CORE_TYPE_H_
