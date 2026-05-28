/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pto_runtime_c_api.h"

#include "host/pto_cuda_host_schedule_abi.h"
#include "host/pto_cuda_persistent_device_abi.h"
#include "platform_comm/comm.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {

struct PtoCudaRuntime {
    uint32_t reserved = 0;
};

struct PtoCudaCallableHeader {
    uint32_t version;
    uint32_t op;
    const void *image;
    size_t image_size;
    const char *entry_name;
    uint32_t grid_dim;
    uint32_t block_dim;
    size_t shared_mem_bytes;
};

struct PreparedCallable {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    uint32_t op = 0;
    uint32_t grid_dim = 0;
    uint32_t block_dim = 0;
    uint32_t stream_id = 0;
    size_t shared_mem_bytes = 0;
};

constexpr uint32_t kStreamPoolSize = 4;

class CudaDeviceRunner {
public:
    ~CudaDeviceRunner() { finalize(); }

    int init(int device_id) {
        device_id_ = device_id;
        CUresult cu_rc = cuInit(0);
        if (cu_rc != CUDA_SUCCESS) {
            return -1;
        }
        cudaError_t rc = cudaSetDevice(device_id_);
        if (rc != cudaSuccess) {
            return -1;
        }
        streams_.resize(kStreamPoolSize, nullptr);
        for (auto &stream : streams_) {
            rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (rc != cudaSuccess) {
                finalize();
                return -1;
            }
        }
        return 0;
    }

    int finalize() {
        for (auto &entry : prepared_) {
            if (entry.second.module != nullptr) {
                cuModuleUnload(entry.second.module);
                entry.second.module = nullptr;
            }
        }
        prepared_.clear();
        for (auto &stream : streams_) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
        streams_.clear();
        return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
    }

    void *malloc(size_t size) {
        if (size == 0) {
            return nullptr;
        }
        void *ptr = nullptr;
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return nullptr;
        }
        if (cudaMalloc(&ptr, size) != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void free(void *ptr) {
        if (ptr == nullptr) {
            return;
        }
        cudaSetDevice(device_id_);
        cudaFree(ptr);
    }

    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t size) {
        if (dev_ptr == nullptr || host_ptr == nullptr) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }
        cudaStream_t stream = default_stream();
        if (stream == nullptr) {
            return -1;
        }
        cudaError_t rc = cudaMemcpyAsync(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream);
        if (rc != cudaSuccess) {
            return -1;
        }
        return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -1;
    }

    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t size) {
        if (host_ptr == nullptr || dev_ptr == nullptr) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }
        cudaStream_t stream = default_stream();
        if (stream == nullptr) {
            return -1;
        }
        cudaError_t rc = cudaMemcpyAsync(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost, stream);
        if (rc != cudaSuccess) {
            return -1;
        }
        return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -1;
    }

    int prepare(int32_t callable_id, const PtoCudaHostCallable *callable) {
        auto *header = static_cast<const PtoCudaCallableHeader *>(static_cast<const void *>(callable));
        if (header == nullptr || header->image == nullptr || header->image_size == 0 || header->entry_name == nullptr) {
            return -1;
        }
        if (header->version != 1 && header->version != 2) {
            return -1;
        }
        if (header->op != PTO_CUDA_HOST_OP_VECTOR_ADD_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_SCALE_F32 &&
            header->op != PTO_CUDA_HOST_OP_VECTOR_AXPY_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_UNARY_F32 &&
            header->op != PTO_CUDA_HOST_OP_VECTOR_AFFINE_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_TRIAD_F32 &&
            header->op != PTO_CUDA_HOST_OP_VECTOR_QUAD_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS_F32 &&
            header->op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_TASKS &&
            header->op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_GRID &&
            header->op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_QUEUE &&
            header->op != PTO_CUDA_PERSISTENT_OP_DAG_F32_RING) {
            return -1;
        }
        uint32_t stream_id = 0;
        if (header->version >= 2) {
            stream_id = callable->stream_id;
        }
        if (stream_id >= streams_.size()) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }

        unregister(callable_id);

        std::vector<char> image(
            static_cast<const char *>(header->image), static_cast<const char *>(header->image) + header->image_size
        );
        if (image.empty() || image.back() != '\0') {
            image.push_back('\0');
        }

        PreparedCallable prepared;
        CUresult cu_rc = cuModuleLoadData(&prepared.module, image.data());
        if (cu_rc != CUDA_SUCCESS) {
            return -1;
        }
        cu_rc = cuModuleGetFunction(&prepared.function, prepared.module, header->entry_name);
        if (cu_rc != CUDA_SUCCESS) {
            cuModuleUnload(prepared.module);
            return -1;
        }

        prepared.op = header->op;
        prepared.grid_dim = header->grid_dim;
        prepared.block_dim = header->block_dim;
        prepared.stream_id = stream_id;
        prepared.shared_mem_bytes = header->shared_mem_bytes;
        prepared_[callable_id] = prepared;
        return 0;
    }

    int unregister(int32_t callable_id) {
        auto it = prepared_.find(callable_id);
        if (it == prepared_.end()) {
            return 0;
        }
        if (it->second.module != nullptr) {
            cuModuleUnload(it->second.module);
        }
        prepared_.erase(it);
        return 0;
    }

    int run(int32_t callable_id, const void *args, PtoRunTiming *out_timing) {
        if (out_timing != nullptr) {
            std::memset(out_timing, 0, sizeof(*out_timing));
        }
        if (args == nullptr) {
            return -1;
        }
        auto it = prepared_.find(callable_id);
        if (it == prepared_.end()) {
            return -1;
        }
        PreparedCallable &prepared = it->second;
        if (prepared.op != PTO_CUDA_HOST_OP_VECTOR_ADD_F32 && prepared.op != PTO_CUDA_HOST_OP_VECTOR_SCALE_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_AXPY_F32 && prepared.op != PTO_CUDA_HOST_OP_VECTOR_UNARY_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_AFFINE_F32 && prepared.op != PTO_CUDA_HOST_OP_VECTOR_TRIAD_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_QUAD_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS_F32 &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_TASKS &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_GRID &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_QUEUE &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_DAG_F32_RING) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }
        cudaStream_t stream = stream_for(prepared.stream_id);
        if (stream == nullptr) {
            return -1;
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
            if (start != nullptr) cudaEventDestroy(start);
            if (stop != nullptr) cudaEventDestroy(stop);
            return -1;
        }

        auto host_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, stream);
        const float *a = nullptr;
        const float *b = nullptr;
        const float *c = nullptr;
        const float *d = nullptr;
        float *out = nullptr;
        float alpha = 0.0f;
        float beta = 0.0f;
        uint64_t n = 0;
        const PtoCudaPersistentVectorAddTask *tasks = nullptr;
        uint64_t task_count = 0;
        uint32_t worker_blocks_per_task = 1;
        const PtoCudaPersistentVectorAddQueueState *queue_state = nullptr;
        const PtoCudaPersistentDagState *dag_state = nullptr;
        void *kernel_args[8] = {};
        if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_ADD_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorAddArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_SCALE_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorScaleArgs *>(args);
            if (typed_args->a == nullptr || typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            out = typed_args->out;
            alpha = typed_args->alpha;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &out;
            kernel_args[2] = &alpha;
            kernel_args[3] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_UNARY_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorUnaryArgs *>(args);
            if (typed_args->a == nullptr || typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &out;
            kernel_args[2] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_AXPY_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorAxpyArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            alpha = typed_args->alpha;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &alpha;
            kernel_args[4] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_AFFINE_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorAffineArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            alpha = typed_args->alpha;
            beta = typed_args->beta;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &alpha;
            kernel_args[4] = &beta;
            kernel_args[5] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_TRIAD_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorTernaryArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->c == nullptr ||
                typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            c = typed_args->c;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &c;
            kernel_args[3] = &out;
            kernel_args[4] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_QUAD_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorQuaternaryArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->c == nullptr ||
                typed_args->d == nullptr || typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            c = typed_args->c;
            d = typed_args->d;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &c;
            kernel_args[3] = &d;
            kernel_args[4] = &out;
            kernel_args[5] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorGenericArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->tensor_arg_count < 2 || typed_args->scalar_arg_count < 2 ||
                typed_args->tensor_args[0] == nullptr || typed_args->tensor_args[1] == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            c = typed_args->tensor_args[0];
            d = typed_args->tensor_args[1];
            alpha = typed_args->scalar_args[0];
            beta = typed_args->scalar_args[1];
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &c;
            kernel_args[4] = &d;
            kernel_args[5] = &alpha;
            kernel_args[6] = &beta;
            kernel_args[7] = &n;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_TASKS) {
            auto *typed_args = static_cast<const PtoCudaPersistentVectorAddArgs *>(args);
            if (typed_args->tasks == nullptr || typed_args->task_count == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            tasks = typed_args->tasks;
            task_count = typed_args->task_count;
            kernel_args[0] = &tasks;
            kernel_args[1] = &task_count;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_GRID) {
            auto *typed_args = static_cast<const PtoCudaPersistentVectorAddGridArgs *>(args);
            if (typed_args->tasks == nullptr || typed_args->task_count == 0 ||
                typed_args->worker_blocks_per_task == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            tasks = typed_args->tasks;
            task_count = typed_args->task_count;
            worker_blocks_per_task = typed_args->worker_blocks_per_task;
            kernel_args[0] = &tasks;
            kernel_args[1] = &task_count;
            kernel_args[2] = &worker_blocks_per_task;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_QUEUE) {
            auto *typed_args = static_cast<const PtoCudaPersistentVectorAddQueueArgs *>(args);
            if (typed_args->state == nullptr) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            queue_state = typed_args->state;
            kernel_args[0] = &queue_state;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_DAG_F32_RING) {
            auto *typed_args = static_cast<const PtoCudaPersistentDagArgs *>(args);
            if (typed_args->state == nullptr) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            dag_state = typed_args->state;
            kernel_args[0] = &dag_state;
        }
        CUresult cu_rc = cuLaunchKernel(
            prepared.function, prepared.grid_dim, 1, 1, prepared.block_dim, 1, 1, prepared.shared_mem_bytes,
            reinterpret_cast<CUstream>(stream), kernel_args, nullptr
        );
        cudaEventRecord(stop, stream);
        cudaError_t sync_rc = cudaStreamSynchronize(stream);
        auto host_stop = std::chrono::steady_clock::now();

        if (out_timing != nullptr) {
            out_timing->host_wall_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(host_stop - host_start).count()
            );
            float elapsed_ms = 0.0F;
            if (cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess) {
                out_timing->device_wall_ns = static_cast<uint64_t>(elapsed_ms * 1000000.0F);
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (cu_rc != CUDA_SUCCESS || sync_rc != cudaSuccess) {
            return -1;
        }
        return 0;
    }

private:
    cudaStream_t stream_for(uint32_t stream_id) {
        if (stream_id >= streams_.size()) {
            return nullptr;
        }
        return streams_[stream_id];
    }

    cudaStream_t default_stream() { return stream_for(0); }

    int device_id_ = 0;
    std::vector<cudaStream_t> streams_;
    std::unordered_map<int32_t, PreparedCallable> prepared_;
};

CudaDeviceRunner *runner(DeviceContextHandle ctx) { return static_cast<CudaDeviceRunner *>(ctx); }

}  // namespace

extern "C" {

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new CudaDeviceRunner());
    } catch (...) {
        return nullptr;
    }
}

void destroy_device_context(DeviceContextHandle ctx) { delete runner(ctx); }

size_t get_runtime_size(void) { return sizeof(PtoCudaRuntime); }

void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) {
    if (ctx == nullptr) return nullptr;
    try {
        return runner(ctx)->malloc(size);
    } catch (...) {
        return nullptr;
    }
}

void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) {
    if (ctx == nullptr) return;
    try {
        runner(ctx)->free(dev_ptr);
    } catch (...) {}
}

int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size
) {
    (void)aicpu_binary;
    (void)aicpu_size;
    (void)aicore_binary;
    (void)aicore_size;
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->init(device_id);
    } catch (...) {
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->finalize();
    } catch (...) {
        return -1;
    }
}

int prepare_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
    if (ctx == nullptr || callable == nullptr) return -1;
    try {
        return runner(ctx)->prepare(callable_id, static_cast<const PtoCudaHostCallable *>(callable));
    } catch (...) {
        return -1;
    }
}

int run_prepared(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, int block_dim,
    int aicpu_thread_num, int enable_l2_swimlane, int enable_dump_tensor, int enable_pmu, int enable_dep_gen,
    const char *output_prefix, PtoRunTiming *out_timing
) {
    (void)runtime;
    (void)block_dim;
    (void)aicpu_thread_num;
    (void)enable_l2_swimlane;
    (void)enable_dump_tensor;
    (void)enable_pmu;
    (void)enable_dep_gen;
    (void)output_prefix;
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->run(callable_id, args, out_timing);
    } catch (...) {
        return -1;
    }
}

int unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->unregister(callable_id);
    } catch (...) {
        return -1;
    }
}

size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) {
    (void)ctx;
    return 0;
}

size_t get_host_dlopen_count(DeviceContextHandle ctx) {
    (void)ctx;
    return 0;
}

int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
    (void)ctx;
    (void)device_id;
    return 0;
}

void *create_comm_stream_ctx(DeviceContextHandle ctx) {
    (void)ctx;
    return nullptr;
}

int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
    (void)ctx;
    (void)stream;
    return 0;
}

CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) {
    (void)rank;
    (void)nranks;
    (void)stream;
    (void)rootinfo_path;
    return nullptr;
}

int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) {
    (void)h;
    (void)win_size;
    (void)device_ctx_out;
    return -1;
}

int comm_get_local_window_base(CommHandle h, uint64_t *base_out) {
    (void)h;
    (void)base_out;
    return -1;
}

int comm_get_window_size(CommHandle h, size_t *size_out) {
    (void)h;
    (void)size_out;
    return -1;
}

int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) {
    (void)h;
    (void)rank_ids;
    (void)rank_count;
    (void)domain_rank;
    (void)window_offset;
    (void)window_size;
    (void)device_ctx_out;
    return -1;
}

int comm_alloc_domain_windows(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    size_t window_size, uint64_t *device_ctx_out, uint64_t *local_window_base_out
) {
    (void)h;
    (void)allocation_id;
    (void)rank_ids;
    (void)rank_count;
    (void)domain_rank;
    (void)window_size;
    (void)device_ctx_out;
    (void)local_window_base_out;
    return -1;
}

int comm_release_domain_windows(CommHandle h, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank) {
    (void)h;
    (void)allocation_id;
    (void)rank_count;
    (void)domain_rank;
    return -1;
}

int comm_barrier(CommHandle h) {
    (void)h;
    return -1;
}

int comm_destroy(CommHandle h) {
    (void)h;
    return 0;
}

}  // extern "C"
