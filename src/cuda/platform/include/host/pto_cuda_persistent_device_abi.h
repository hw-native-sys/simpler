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

#ifndef SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_PERSISTENT_DEVICE_ABI_H_
#define SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_PERSISTENT_DEVICE_ABI_H_

#include <stddef.h>
#include <stdint.h>

enum PtoCudaPersistentDeviceOp : uint32_t {
    PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_TASKS = 1001,
    PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_QUEUE = 1002,
    PTO_CUDA_PERSISTENT_OP_DAG_F32_RING = 1003,
    PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_GRID = 1004,
};

struct PtoCudaPersistentCallable {
    uint32_t version;
    uint32_t op;
    const void *image;
    size_t image_size;
    const char *entry_name;
    uint32_t grid_dim;
    uint32_t block_dim;
    size_t shared_mem_bytes;
};

struct PtoCudaPersistentVectorAddTask {
    const float *a;
    const float *b;
    float *out;
    uint64_t n;
};

struct PtoCudaPersistentVectorAddArgs {
    const PtoCudaPersistentVectorAddTask *tasks;
    uint64_t task_count;
};

struct PtoCudaPersistentVectorAddGridArgs {
    const PtoCudaPersistentVectorAddTask *tasks;
    uint64_t task_count;
    uint32_t worker_blocks_per_task;
};

struct PtoCudaPersistentVectorAddQueueState {
    const PtoCudaPersistentVectorAddTask *tasks;
    uint64_t task_count;
    uint32_t *ready_queue;
    uint32_t *ready_flags;
    uint32_t queue_capacity;
    uint32_t *queue_head;
    uint32_t *queue_tail;
    uint32_t *completed_count;
};

struct PtoCudaPersistentVectorAddQueueArgs {
    const PtoCudaPersistentVectorAddQueueState *state;
};

struct PtoCudaPersistentDagTask {
    uint32_t func_id;
    const float *a;
    const float *b;
    float *out;
    uint64_t n;
    uint32_t dependent_begin;
    uint32_t dependent_count;
    uint32_t initial_fanin;
};

struct PtoCudaPersistentDagState {
    const PtoCudaPersistentDagTask *tasks;
    uint64_t task_count;
    const uint32_t *dependents;
    uint32_t *fanin;
    uint32_t *ready_queue;
    uint32_t *ready_flags;
    uint32_t queue_capacity;
    uint32_t *queue_head;
    uint32_t *queue_tail;
    uint32_t *completed_count;
};

struct PtoCudaPersistentDagArgs {
    const PtoCudaPersistentDagState *state;
};

#endif  // SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_PERSISTENT_DEVICE_ABI_H_
