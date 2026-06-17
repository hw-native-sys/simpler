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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <vector>

#include "common/core_type.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"
#include "pto2_dispatch_payload.h"
#include "task_args.h"

#define RUNTIME_MAX_ARGS 128
#define RUNTIME_MAX_WORKER 72  // 24 AIC + 48 AIV cores
#define RUNTIME_MAX_FUNC_ID 1024
#define RUNTIME_MAX_ORCH_SO_SIZE (4 * 1024 * 1024)  // 4MB max for orchestration SO
#define RUNTIME_MAX_ORCH_SYMBOL_NAME 64

// Default ready queue shards: one shard per worker thread (total minus orchestrator)
constexpr int RUNTIME_DEFAULT_READY_QUEUE_SHARDS = PLATFORM_MAX_AICPU_THREADS - 1;

struct Handshake
{
    volatile uint32_t aicpu_ready;        // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;        // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;               // Init: PTO2DispatchPayload* (set before aicpu_ready); runtime: unused
    volatile CoreType core_type;          // Core type: CoreType::AIC or CoreType::AIV
    volatile uint32_t physical_core_id;   // Physical core ID
    volatile uint32_t aicpu_regs_ready;   // AICPU register init done: 0=pending, 1=done
    volatile uint32_t aicore_regs_ready;  // AICore ID reported: 0=pending, 1=done
} __attribute__((aligned(64)));

struct TensorPair
{
    void *host_ptr;
    void *dev_ptr;
    size_t size;
};

struct HostApi
{
    void *(*device_malloc)(size_t size);
    void (*device_free)(void *dev_ptr);
    int (*copy_to_device)(void *dev_ptr, const void *host_ptr, size_t size);
    int (*copy_from_device)(void *host_ptr, const void *dev_ptr, size_t size);
    int (*device_memset)(void *dev_ptr, int value, size_t size);
    int (*setup_static_arena)(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);
    void *(*acquire_pooled_gm_heap)();
    void *(*acquire_pooled_gm_sm)();
    void *(*acquire_pooled_runtime_arena)();
    uint64_t (*upload_chip_callable_buffer)(const void *callable);
};

struct Task
{
    int func_id;
    uint64_t function_bin_addr;
};

class Runtime
{
public:
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];  // Worker (AICore) handshake buffers
    int worker_count;                       // Number of active workers

    int aicpu_thread_num;
    int ready_queue_shards;  // Number of ready queue shards (1..MAX_AICPU_THREADS, default MAX-1)

    // Ring buffer size overrides (0 = use compile-time defaults)
    uint64_t task_window_size;
    uint64_t heap_size;
    uint64_t dep_pool_size;

    // PTO2 integration: kernel_id -> GM function_bin_addr mapping
    // NOTE: Made public for direct access from aicore code
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    bool orch_to_sched;

private:
    // Kernel binary tracking for cleanup
    int registered_kernel_func_ids_[RUNTIME_MAX_FUNC_ID];
    int registered_kernel_count_;

    void *gm_sm_ptr_;                        // GM pointer to PTO2 shared memory (device)
    void *gm_heap_ptr_;                      // GM heap for orchestrator output buffers (device)
    void *slot_states_ptr_;                  // Pointer to PTO2TaskSlotState array (scheduler-private, for profiling)
    ChipStorageTaskArgs orch_args_storage_;  // Copy of args for device

    void *prebuilt_arena_base_;
    size_t prebuilt_runtime_offset_;

    uint64_t dev_orch_so_addr_;
    uint64_t dev_orch_so_size_;
    int32_t active_callable_id_;
    bool register_new_callable_id_;
    char device_orch_func_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];
    char device_orch_config_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];

public:
    Runtime()
    {
        // NOTE: host_api is initialized in InitRuntime() (host-only code)
        // because the CApi functions don't exist when compiled for device.

        // Initialize handshake buffers
        memset(workers, 0, sizeof(workers));
        worker_count = 0;
        aicpu_thread_num = 1;
        ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;
        task_window_size = 0;
        heap_size = 0;
        dep_pool_size = 0;
        orch_to_sched = false;

        // Initialize device orchestration state
        gm_sm_ptr_ = nullptr;
        gm_heap_ptr_ = nullptr;
        slot_states_ptr_ = nullptr;
        orch_args_storage_.clear();
        prebuilt_arena_base_ = nullptr;
        prebuilt_runtime_offset_ = 0;

        // Initialize device orchestration SO binary
        dev_orch_so_addr_ = 0;
        dev_orch_so_size_ = 0;
        active_callable_id_ = -1;
        register_new_callable_id_ = false;
        device_orch_func_name_[0] = '\0';
        device_orch_config_name_[0] = '\0';

        // Initialize kernel binary tracking
        registered_kernel_count_ = 0;

        // Initialize function address mapping
        for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) func_id_to_addr_[i] = 0;
    }

    void *get_gm_sm_ptr() const
    {
        return gm_sm_ptr_;
    }
    void *get_gm_heap_ptr() const
    {
        return gm_heap_ptr_;
    }
    const ChipStorageTaskArgs &get_orch_args() const
    {
        return orch_args_storage_;
    }
    void set_gm_sm_ptr(void *p)
    {
        gm_sm_ptr_ = p;
    }
    void set_gm_heap(void *p)
    {
        gm_heap_ptr_ = p;
    }
    void set_slot_states_ptr(void *p)
    {
        slot_states_ptr_ = p;
    }
    void set_orch_args(const ChipStorageTaskArgs &args)
    {
        orch_args_storage_ = args;
    }

    void set_prebuilt_arena(void *arena_base, size_t runtime_off)
    {
        prebuilt_arena_base_ = arena_base;
        prebuilt_runtime_offset_ = runtime_off;
    }
    void *get_prebuilt_arena_base() const
    {
        return prebuilt_arena_base_;
    }
    size_t get_prebuilt_runtime_offset() const
    {
        return prebuilt_runtime_offset_;
    }

    // Device orchestration SO binary (for dlopen on AICPU thread 3)
    void set_dev_orch_so(uint64_t dev_addr, uint64_t size)
    {
        dev_orch_so_addr_ = dev_addr;
        dev_orch_so_size_ = size;
    }
    uint64_t get_dev_orch_so_addr() const
    {
        return dev_orch_so_addr_;
    }
    uint64_t get_dev_orch_so_size() const
    {
        return dev_orch_so_size_;
    }
    void set_active_callable_id(int32_t callable_id, bool is_new)
    {
        active_callable_id_ = callable_id;
        register_new_callable_id_ = is_new;
    }
    int32_t get_active_callable_id() const
    {
        return active_callable_id_;
    }
    bool register_new_callable_id() const
    {
        return register_new_callable_id_;
    }
    void set_device_orch_func_name(const char *name)
    {
        if (name == nullptr)
        {
            device_orch_func_name_[0] = '\0';
            return;
        }
        std::strncpy(device_orch_func_name_, name, RUNTIME_MAX_ORCH_SYMBOL_NAME - 1);
        device_orch_func_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME - 1] = '\0';
    }
    const char *get_device_orch_func_name() const
    {
        return device_orch_func_name_;
    }
    void set_device_orch_config_name(const char *name)
    {
        if (name == nullptr)
        {
            device_orch_config_name_[0] = '\0';
            return;
        }
        std::strncpy(device_orch_config_name_, name, RUNTIME_MAX_ORCH_SYMBOL_NAME - 1);
        device_orch_config_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME - 1] = '\0';
    }
    const char *get_device_orch_config_name() const
    {
        return device_orch_config_name_;
    }

    uint64_t get_function_bin_addr(int func_id) const
    {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }
    void set_function_bin_addr(int func_id, uint64_t addr)
    {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return;
        if (addr != 0 && func_id_to_addr_[func_id] == 0)
        {
            if (registered_kernel_count_ < RUNTIME_MAX_FUNC_ID)
            {
                registered_kernel_func_ids_[registered_kernel_count_++] = func_id;
            }
            else
            {}
        }
        func_id_to_addr_[func_id] = addr;
    }
    void replay_function_bin_addr(int func_id, uint64_t addr)
    {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return;
        func_id_to_addr_[func_id] = addr;
    }

    int get_registered_kernel_count() const
    {
        return registered_kernel_count_;
    }
    int get_registered_kernel_func_id(int index) const
    {
        if (index < 0 || index >= registered_kernel_count_) return -1;
        return registered_kernel_func_ids_[index];
    }
    void clear_registered_kernels()
    {
        registered_kernel_count_ = 0;
    }

    int get_task_count() const
    {
        return 0;
    }

    Task *get_task([[maybe_unused]] int taskId)
    {
        return nullptr;
    }

    // Host API function pointers for device memory operations
    // NOTE: Placed at end of class to avoid affecting device memory layout
    HostApi host_api;

    std::vector<TensorPair> tensor_pairs_;
};

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
