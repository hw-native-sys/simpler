/**
 * TaskArg - Tagged union for orchestration function arguments
 *
 * Each TaskArg carries either a Tensor (ptr/shape/ndims/dtype) or a Scalar
 * (uint64_t value). Host side builds a TaskArg[] array which is copied to
 * device; AICPU reads fields directly.
 *
 * This struct is trivially copyable (required for DMA) and fixed at 48 bytes.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "data_type.h"

constexpr int TASK_ARG_MAX_DIMS = 5;

enum class TaskArgKind : uint32_t {
    TENSOR = 0,
    SCALAR = 1,
};

struct TaskArg {
    TaskArgKind kind;         // 4B: TENSOR or SCALAR

    union {
        struct {                                    // --- Tensor metadata ---
            uint64_t data;                          // Host/device memory address
            uint32_t shapes[TASK_ARG_MAX_DIMS];     // Shape per dim (element count)
            uint32_t ndims;                         // Number of dimensions (1..5)
            DataType dtype;                         // DataType : uint32_t
        } tensor;                                   // 36B

        uint64_t scalar;                            // --- Scalar value ---  8B
    };

    // Compute total bytes for this tensor from shape x element_size
    uint64_t nbytes() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < tensor.ndims; i++)
            total *= tensor.shapes[i];
        return total * get_element_size(tensor.dtype);
    }

    // Get raw pointer to tensor data
    template<typename T>
    T* data() const {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(tensor.data));
    }

    // Reinterpret scalar bits as target type (compliant type-punning via memcpy)
    template<typename T>
    T value_as() const {
        static_assert(sizeof(T) <= sizeof(uint64_t), "");
        T result;
        memcpy(&result, &scalar, sizeof(T));
        return result;
    }
};

static_assert(std::is_trivially_copyable<TaskArg>::value, "TaskArg must be trivially copyable for DMA");
static_assert(sizeof(TaskArg) == 48, "TaskArg size must be exactly 48B for stable ABI");
