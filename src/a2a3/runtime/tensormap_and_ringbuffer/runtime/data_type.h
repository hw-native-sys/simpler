/**
 * Data Type Definitions for Orchestration Build Graph Runtime
 *
 * Defines supported data types and helper functions for element size calculation.
 */

#ifndef ORCH_BUILD_GRAPH_DATA_TYPE_H
#define ORCH_BUILD_GRAPH_DATA_TYPE_H

#include <cstdint>

/**
 * Supported data types for tensor elements
 */
enum class DataType : uint32_t {
    FLOAT32,   // 4 bytes
    FLOAT16,   // 2 bytes
    INT32,     // 4 bytes
    INT16,     // 2 bytes
    INT8,      // 1 byte
    UINT8,     // 1 byte
    BFLOAT16,  // 2 bytes
    INT64,     // 8 bytes
    UINT64,    // 8 bytes
    DATA_TYPE_NUM,
};

/**
 * Get the size in bytes of a single element of the given data type
 *
 * @param dtype Data type
 * @return Size in bytes (0 for unknown types)
 */
inline uint64_t get_element_size(DataType dtype) {
    // 这里的顺序要和enum的顺序严格一致
    static uint64_t data_type_size[static_cast<int>(DataType::DATA_TYPE_NUM)] = {
        4, // case DataType::FLOAT32
        2, // DataType::FLOAT16
        4, // DataType::INT32
        2, // DataType::INT16
        1, // DataType::INT8
        1, // DataType::UINT8
        2, // DataType::BFLOAT16
        8, // DataType::INT64
        8, // DataType::UINT64
    };
    return data_type_size[static_cast<int>(dtype)];
}

/**
 * Get the name of a data type as a string
 *
 * @param dtype Data type
 * @return String name of the data type
 */
inline const char* get_dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return "FLOAT32";
        case DataType::FLOAT16:
            return "FLOAT16";
        case DataType::INT32:
            return "INT32";
        case DataType::INT16:
            return "INT16";
        case DataType::INT8:
            return "INT8";
        case DataType::UINT8:
            return "UINT8";
        case DataType::BFLOAT16:
            return "BFLOAT16";
        case DataType::INT64:
            return "INT64";
        case DataType::UINT64:
            return "UINT64";
        default:
            return "UNKNOWN";
    }
}

#endif  // ORCH_BUILD_GRAPH_DATA_TYPE_H
