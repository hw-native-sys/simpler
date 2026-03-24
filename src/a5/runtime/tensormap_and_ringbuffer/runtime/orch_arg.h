/**
 * OrchArg - Tagged union for orchestration function arguments
 *
 * Each OrchArg carries either a Tensor (ptr/shape/ndims/dtype) or a Scalar
 * (uint64_t value). Host side builds an OrchArg[] array which is copied to
 * device; AICPU reads fields directly.
 *
 * This struct is trivially copyable (required for DMA) and fixed at 48 bytes.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "data_type.h"

constexpr int ORCH_ARG_MAX_DIMS = 5;

enum class OrchArgKind : uint32_t {
    TENSOR = 0,
    SCALAR = 1,
};

struct Tensor;  // Forward declaration — full definition in tensor.h

struct OrchArg {
    OrchArgKind kind;         // 4B: TENSOR or SCALAR

    union {
        struct {                                    // --- Tensor metadata ---
            uint64_t data;                          // Device memory address
            uint32_t shapes[ORCH_ARG_MAX_DIMS];     // Shape per dim (element count)
            uint32_t ndims;                         // Number of dimensions (1..5)
            DataType dtype;                         // DataType : uint32_t
        } tensor;                                   // 36B

        uint64_t scalar;                            // --- Scalar value ---  8B
    };

    // ==================== AICPU-side convenience methods ====================

    // Convert to runtime Tensor using golden.py's original shape.
    // For simple cases where golden shape == kernel shape.
    // When reshape is needed, read fields manually and call make_tensor_external.
    // Defined in pto_orchestration_api.h (needs make_tensor_external).
    Tensor to_tensor(bool manual_dep = false, int32_t version = 0) const;

    // Get raw pointer to tensor data (eliminates verbose double-cast)
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

static_assert(std::is_trivially_copyable<OrchArg>::value, "OrchArg must be trivially copyable for DMA");
static_assert(sizeof(OrchArg) == 48, "OrchArg size must be exactly 48B for stable ABI");
