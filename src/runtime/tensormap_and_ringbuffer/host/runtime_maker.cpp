/**
 * Runtime Builder - rt2 Implementation (Device Orchestration)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions for rt2 runtime.
 * Supports device orchestration where AICPU thread 3 runs the orchestrator.
 *
 * init_runtime_impl:
 *   - Converts host pointers to device pointers based on arg_types
 *   - Copies orchestration SO to device memory
 *   - Sets up runtime state for device orchestration
 *
 * validate_runtime_impl:
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#include "../runtime/runtime.h"
#include "../runtime/pto_shared_memory.h"
#include "common/unified_log.h"
#include "host/pto_runtime_c_api.h"  // For ArgType enum
#include "common/platform_config.h"
#include <stdint.h>
#include <stddef.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sys/time.h>

// Helper: return current time in milliseconds
static long long _now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

// Max args for device orchestration
#define RT2_MAX_DEVICE_ARGS 32

/**
 * Parse an environment variable as uint64_t with optional power-of-2 constraint.
 * Returns the parsed value on success, or 0 if unset or validation fails.
 */
static uint64_t parse_env_uint64(const char* name, uint64_t min_val, bool require_power_of_2) {
    const char* env = std::getenv(name);
    if (!env) return 0;
    char* endptr;
    errno = 0;
    unsigned long long val = strtoull(env, &endptr, 10);
    if (errno == ERANGE || endptr == env || *endptr != '\0' || val < min_val) {
        LOG_WARN("%s=%s invalid (must be a valid integer >= %lu), ignored",
                 name, env, (unsigned long)min_val);
        return 0;
    }
    if (require_power_of_2 && (val & (val - 1)) != 0) {
        LOG_WARN("%s=%s invalid (must be a power of 2, >= %lu), ignored",
                 name, env, (unsigned long)min_val);
        return 0;
    }
    return static_cast<uint64_t>(val);
}

/**
 * Initialize a pre-allocated runtime for device orchestration.
 *
 * For rt2 runtime, orchestration runs on AICPU thread 3 (device-side).
 * This function:
 * - Converts host pointers to device pointers based on arg_types
 * - Copies input data to device
 * - Records output tensors for copy-back
 * - Copies orchestration SO to device memory
 * - Sets up runtime state for device orchestration
 *
 * @param runtime           Pointer to pre-constructed Runtime
 * @param orch_so_binary    Orchestration shared library binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration function (unused)
 * @param func_args         Arguments for orchestration
 * @param func_args_count   Number of arguments
 * @param arg_types         Array describing each argument's type (ArgType enum)
 * @param arg_sizes         Array of sizes for pointer arguments (0 for scalars)
 * @return 0 on success, -1 on failure
 */
extern "C" int init_runtime_impl(Runtime *runtime,
                    const uint8_t* orch_so_binary,
                    size_t orch_so_size,
                    const char* orch_func_name,
                    uint64_t* func_args,
                    int func_args_count,
                    int* arg_types,
                    uint64_t* arg_sizes,
                    const int* kernel_func_ids,
                    const uint8_t* const* kernel_binaries,
                    const size_t* kernel_sizes,
                    int kernel_count) {
    // Suppress unused parameter warning
    (void)orch_func_name;

    // Validate inputs
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }

    // Register kernel binaries via platform-provided upload function
    if (kernel_count > 0 && kernel_func_ids != nullptr &&
        kernel_binaries != nullptr && kernel_sizes != nullptr) {
        LOG_INFO("Registering %d kernel(s) in init_runtime_impl", kernel_count);
        for (int i = 0; i < kernel_count; i++) {
            uint64_t addr = runtime->host_api.upload_kernel_binary(
                kernel_func_ids[i], kernel_binaries[i], kernel_sizes[i]);
            if (addr == 0) {
                LOG_ERROR("Failed to upload kernel binary for func_id=%d", kernel_func_ids[i]);
                return -1;
            }
            runtime->set_function_bin_addr(kernel_func_ids[i], addr);
        }
    }

    if (orch_so_binary == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration SO binary is required for device orchestration");
        return -1;
    }

    if (arg_types == nullptr || arg_sizes == nullptr) {
        LOG_ERROR("arg_types and arg_sizes are required for device orchestration");
        return -1;
    }

    if (func_args_count > RT2_MAX_DEVICE_ARGS) {
        LOG_ERROR("Too many arguments: %d (max %d)", func_args_count, RT2_MAX_DEVICE_ARGS);
        return -1;
    }

    LOG_INFO("RT2 init: %d arguments, device orchestration mode", func_args_count);

    long long t_total_start = _now_ms();

    // Convert host pointers to device pointers based on arg_types
    uint64_t device_args[RT2_MAX_DEVICE_ARGS];

    long long t_args_start = _now_ms();
    for (int i = 0; i < func_args_count; i++) {
        switch (arg_types[i]) {
            case ARG_SCALAR:
                // Scalar value, pass directly
                device_args[i] = func_args[i];
                break;

            case ARG_INPUT_PTR: {
                // Input pointer: allocate device memory, copy data
                void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                size_t size = arg_sizes[i];

                void* dev_ptr = runtime->host_api.device_malloc(size);
                if (dev_ptr == nullptr) {
                    LOG_ERROR("Failed to allocate device memory for arg %d", i);
                    return -1;
                }

                int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
                if (rc != 0) {
                    LOG_ERROR("Failed to copy arg %d to device", i);
                    runtime->host_api.device_free(dev_ptr);
                    return -1;
                }

                device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                // Record for cleanup (no copy-back needed)
                runtime->record_tensor_pair(nullptr, dev_ptr, size);
                LOG_INFO("  Arg %d (input): %zu bytes at %p", i, size, dev_ptr);
                break;
            }

            case ARG_OUTPUT_PTR: {
                // Output pointer: allocate device memory, record for copy-back
                void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                size_t size = arg_sizes[i];

                void* dev_ptr = runtime->host_api.device_malloc(size);
                if (dev_ptr == nullptr) {
                    LOG_ERROR("Failed to allocate device memory for arg %d", i);
                    return -1;
                }

                device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                // Record for copy-back during finalize
                runtime->record_tensor_pair(host_ptr, dev_ptr, size);
                LOG_INFO("  Arg %d (output): %zu bytes at %p", i, size, dev_ptr);
                break;
            }

            case ARG_INOUT_PTR: {
                // Input/output pointer: allocate, copy, record for copy-back
                void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                size_t size = arg_sizes[i];

                void* dev_ptr = runtime->host_api.device_malloc(size);
                if (dev_ptr == nullptr) {
                    LOG_ERROR("Failed to allocate device memory for arg %d", i);
                    return -1;
                }

                int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
                if (rc != 0) {
                    LOG_ERROR("Failed to copy arg %d to device", i);
                    runtime->host_api.device_free(dev_ptr);
                    return -1;
                }

                device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                // Record for copy-back during finalize
                runtime->record_tensor_pair(host_ptr, dev_ptr, size);
                LOG_INFO("  Arg %d (inout): %zu bytes at %p", i, size, dev_ptr);
                break;
            }

            default:
                LOG_ERROR("Unknown arg_type %d for arg %d", arg_types[i], i);
                return -1;
        }
    }
    long long t_args_end = _now_ms();

    // Copy orchestration SO to device memory (AICPU cannot access host memory)
    long long t_so_start = _now_ms();
    void* dev_so = runtime->host_api.device_malloc(orch_so_size);
    if (dev_so == nullptr) {
        LOG_ERROR("Failed to allocate device memory for orchestration SO");
        return -1;
    }
    int rc = runtime->host_api.copy_to_device(dev_so, orch_so_binary, orch_so_size);
    if (rc != 0) {
        LOG_ERROR("Failed to copy orchestration SO to device");
        runtime->host_api.device_free(dev_so);
        return -1;
    }
    // Copy SO binary into Runtime's internal storage (device_orch_so_storage_)
    // Pass the HOST pointer (orch_so_binary), not the device pointer (dev_so)
    // AICPU Thread 3 will read from get_device_orch_so_data() which returns this storage
    runtime->set_device_orch_so(orch_so_binary, orch_so_size);
    runtime->record_tensor_pair(nullptr, dev_so, orch_so_size);
    LOG_INFO("Orchestration SO: %zu bytes copied to device", orch_so_size);
    long long t_so_end = _now_ms();

    // Read ready queue shard count from environment for AICPU scheduler
    {
        const char* env_shards = std::getenv("PTO2_READY_QUEUE_SHARDS");
        if (env_shards) {
            char* endptr;
            long val = strtol(env_shards, &endptr, 10);
            if (endptr != env_shards && *endptr == '\0' && val >= 1 && val <= PLATFORM_MAX_AICPU_THREADS) {
                runtime->ready_queue_shards = static_cast<int>(val);
            } else {
                LOG_WARN("PTO2_READY_QUEUE_SHARDS=%s is invalid or out of range [1,%d], using default %d",
                         env_shards, PLATFORM_MAX_AICPU_THREADS, RUNTIME_DEFAULT_READY_QUEUE_SHARDS);
                runtime->ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;
            }
        }
        LOG_INFO("Ready queue shards: %d", runtime->ready_queue_shards);
    }

    // Read ring buffer size overrides from environment
    {
        runtime->pto2_task_window_size  = parse_env_uint64("PTO2_RING_TASK_WINDOW", 4, true);
        runtime->pto2_heap_size         = parse_env_uint64("PTO2_RING_HEAP", 1024, true);
        runtime->pto2_dep_list_pool_size = parse_env_uint64("PTO2_RING_DEP_POOL", 16, false);
        if (runtime->pto2_task_window_size || runtime->pto2_heap_size || runtime->pto2_dep_list_pool_size) {
            LOG_INFO("Ring buffer overrides: task_window=%lu heap=%lu dep_pool=%lu",
                     (unsigned long)(runtime->pto2_task_window_size ? runtime->pto2_task_window_size : PTO2_TASK_WINDOW_SIZE),
                     (unsigned long)(runtime->pto2_heap_size ? runtime->pto2_heap_size : PTO2_HEAP_SIZE),
                     (unsigned long)(runtime->pto2_dep_list_pool_size ? runtime->pto2_dep_list_pool_size : PTO2_DEP_LIST_POOL_SIZE));
        }
    }

    // Resolve effective sizes (env override or compile-time default)
    uint64_t eff_heap_size = runtime->pto2_heap_size ? runtime->pto2_heap_size : PTO2_HEAP_SIZE;
    uint64_t eff_task_window_size = runtime->pto2_task_window_size ? runtime->pto2_task_window_size : PTO2_TASK_WINDOW_SIZE;
    uint64_t eff_dep_list_pool_size = runtime->pto2_dep_list_pool_size ? runtime->pto2_dep_list_pool_size : PTO2_DEP_LIST_POOL_SIZE;

    // Allocate GM heap for orchestrator output buffers
    long long t_heap_start = _now_ms();
    void* gm_heap = runtime->host_api.device_malloc(eff_heap_size);
    long long t_heap_end = _now_ms();
    if (gm_heap == nullptr) {
        LOG_ERROR("Failed to allocate GM heap");
        return -1;
    }
    runtime->record_tensor_pair(nullptr, gm_heap, eff_heap_size);
    runtime->set_pto2_gm_heap(gm_heap);

    // Allocate PTO2 shared memory
    long long t_sm_start = _now_ms();
    uint64_t sm_size = pto2_sm_calculate_size(eff_task_window_size, eff_dep_list_pool_size);
    void* sm_ptr = runtime->host_api.device_malloc(sm_size);
    long long t_sm_end = _now_ms();
    if (sm_ptr == nullptr) {
        LOG_ERROR("Failed to allocate PTO2 shared memory");
        return -1;
    }
    runtime->set_pto2_gm_sm_ptr(sm_ptr);
    runtime->record_tensor_pair(nullptr, sm_ptr, static_cast<size_t>(sm_size));

    // Set up device orchestration state
    runtime->set_orch_built_on_host(false);
    runtime->set_orch_args(device_args, func_args_count);

    LOG_INFO("Device orchestration ready: %d args", func_args_count);

    long long t_total_end = _now_ms();
    LOG_INFO("TIMING: args_malloc_copy = %lldms", t_args_end - t_args_start);
    LOG_INFO("TIMING: orch_so_copy = %lldms", t_so_end - t_so_start);
    LOG_INFO("TIMING: gm_heap_alloc(1GB) = %lldms", t_heap_end - t_heap_start);
    LOG_INFO("TIMING: shared_mem_alloc = %lldms", t_sm_end - t_sm_start);
    LOG_INFO("TIMING: total_init_runtime_impl = %lldms", t_total_end - t_total_start);

    return 0;
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Frees device memory for recorded tensors
 * 3. Clears tensor pair state
 *
 * @param runtime  Pointer to Runtime
 * @return 0 on success, -1 on failure
 */
extern "C" int validate_runtime_impl(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }

    int rc = 0;

    LOG_INFO("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorPair* tensor_pairs = runtime->get_tensor_pairs();
    int tensor_pair_count = runtime->get_tensor_pair_count();

    LOG_INFO("Tensor pairs to process: %d", tensor_pair_count);

    // PTO2 (device orchestration): graph output may be in packed buffer
    void* pto2_sm = runtime->get_pto2_gm_sm_ptr();
    uint64_t graph_out_ptr = 0;
    uint64_t graph_out_size = 0;

    if (pto2_sm != nullptr) {
        // Copy header from device to host to read graph_output_ptr/size
        PTO2SharedMemoryHeader host_header;
        int hdr_rc = runtime->host_api.copy_from_device(&host_header, pto2_sm, sizeof(PTO2SharedMemoryHeader));
        if (hdr_rc == 0) {
            graph_out_ptr = host_header.graph_output_ptr;
            graph_out_size = host_header.graph_output_size;
            if (graph_out_ptr != 0) {
                LOG_INFO("Graph output buffer: ptr=0x%lx, size=%lu", (unsigned long)graph_out_ptr, (unsigned long)graph_out_size);
            }
        } else {
            LOG_WARN("Failed to copy PTO2 header from device");
        }
    }

    bool first_output_tensor = true;
    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair& pair = tensor_pairs[i];

        // Skip if device pointer is null
        if (pair.dev_ptr == nullptr) {
            LOG_WARN("Tensor %d has null device pointer, skipping", i);
            continue;
        }

        // If host pointer is null, this is a device-only allocation (no copy-back)
        if (pair.host_ptr == nullptr) {
            LOG_INFO("Tensor %d: device-only allocation (no copy-back)", i);
            continue;
        }

        void* src_ptr = pair.dev_ptr;
        size_t copy_size = pair.size;

        // Use graph_output_ptr for the first output tensor if available
        if (first_output_tensor && graph_out_ptr != 0 && graph_out_size > 0) {
            src_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(graph_out_ptr));
            copy_size = static_cast<size_t>(graph_out_size);
            LOG_INFO("Using packed output buffer for tensor %d", i);
            first_output_tensor = false;
        }

        int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, src_ptr, copy_size);
        if (copy_rc != 0) {
            LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
            rc = copy_rc;
        } else {
            LOG_INFO("Tensor %d: %zu bytes copied to host", i, pair.size);
        }
    }

    // Cleanup device tensors
    LOG_INFO("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr != nullptr) {
            runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
        }
    }
    LOG_INFO("Freed %d device allocations", tensor_pair_count);

    // Clear tensor pairs
    runtime->clear_tensor_pairs();

    LOG_INFO("=== Finalize Complete ===");

    return rc;
}
