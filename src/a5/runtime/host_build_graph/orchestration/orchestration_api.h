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
/**
 * Orchestration API for host_build_graph.
 *
 * Orchestration sources include only this header and interact with the runtime
 * through the function-pointer table embedded in OrchestrationRuntime.
 */

#ifndef SRC_A5_RUNTIME_HOST_BUILD_GRAPH_ORCHESTRATION_ORCHESTRATION_API_H_
#define SRC_A5_RUNTIME_HOST_BUILD_GRAPH_ORCHESTRATION_ORCHESTRATION_API_H_

#include <stddef.h>
#include <stdint.h>

#include "common/core_type.h"
#include "task_args.h"
#include "tensor_info.h"

typedef struct OrchestrationRuntime OrchestrationRuntime;

typedef struct OrchestrationRuntimeOps {
    int (*add_task)(OrchestrationRuntime *runtime, uint64_t *args, int num_args, int func_id, CoreType core_type);
    int (*set_tensor_info_to_task)(
        OrchestrationRuntime *runtime, int task_id, const TensorInfo *tensor_info, int tensor_count
    );
    void (*add_successor)(OrchestrationRuntime *runtime, int from_task, int to_task);
    // Host-build-graph orch owns its entry-tensor H2D (the orch runs on host
    // and may need host-side pointers for control tensors). This call
    // registers a host<->device mapping so runtime_maker's validate path can
    // D2H copy-back and free at finalize. TMARB has no equivalent — its
    // runtime_maker handles tensor uploads directly because its orch is
    // device-side.
    void (*record_tensor_pair)(OrchestrationRuntime *runtime, void *host_ptr, void *dev_ptr, size_t size);
    int (*get_task_count)(OrchestrationRuntime *runtime);
    void (*print_runtime)(OrchestrationRuntime *runtime);

    void *(*device_malloc)(OrchestrationRuntime *runtime, size_t size);
    void (*device_free)(OrchestrationRuntime *runtime, void *ptr);
    int (*copy_to_device)(OrchestrationRuntime *runtime, void *dev_ptr, const void *host_ptr, size_t size);
} OrchestrationRuntimeOps;

struct OrchestrationRuntime {
    const OrchestrationRuntimeOps *ops;
};

static inline int
add_task(OrchestrationRuntime *runtime, uint64_t *args, int num_args, int func_id, CoreType core_type) {
    return runtime->ops->add_task(runtime, args, num_args, func_id, core_type);
}

static inline int
set_tensor_info_to_task(OrchestrationRuntime *runtime, int task_id, const TensorInfo *tensor_info, int tensor_count) {
    return runtime->ops->set_tensor_info_to_task(runtime, task_id, tensor_info, tensor_count);
}

static inline int add_task_with_tensor_info(
    OrchestrationRuntime *runtime, uint64_t *args, int num_args, int func_id, CoreType core_type,
    const TensorInfo *tensor_info, int tensor_count
) {
    int task_id = add_task(runtime, args, num_args, func_id, core_type);
    if (task_id < 0) {
        return task_id;
    }
    if (set_tensor_info_to_task(runtime, task_id, tensor_info, tensor_count) != 0) {
        return -1;
    }
    return task_id;
}

static inline void add_successor(OrchestrationRuntime *runtime, int from_task, int to_task) {
    runtime->ops->add_successor(runtime, from_task, to_task);
}

static inline void record_tensor_pair(OrchestrationRuntime *runtime, void *host_ptr, void *dev_ptr, size_t size) {
    runtime->ops->record_tensor_pair(runtime, host_ptr, dev_ptr, size);
}

static inline int get_task_count(OrchestrationRuntime *runtime) { return runtime->ops->get_task_count(runtime); }

static inline void print_runtime(OrchestrationRuntime *runtime) { runtime->ops->print_runtime(runtime); }

static inline void *device_malloc(OrchestrationRuntime *runtime, size_t size) {
    return runtime->ops->device_malloc(runtime, size);
}

static inline void device_free(OrchestrationRuntime *runtime, void *ptr) { runtime->ops->device_free(runtime, ptr); }

static inline int copy_to_device(OrchestrationRuntime *runtime, void *dev_ptr, const void *host_ptr, size_t size) {
    return runtime->ops->copy_to_device(runtime, dev_ptr, host_ptr, size);
}

typedef int (*OrchestrationFunc)(OrchestrationRuntime *runtime, const ChipStorageTaskArgs &orch_args);

#endif  // SRC_A5_RUNTIME_HOST_BUILD_GRAPH_ORCHESTRATION_ORCHESTRATION_API_H_
