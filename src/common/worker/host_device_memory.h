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

#ifndef SRC_COMMON_WORKER_HOST_DEVICE_MEMORY_H_
#define SRC_COMMON_WORKER_HOST_DEVICE_MEMORY_H_

#include <stddef.h>
#include <stdint.h>

#include "pto_runtime_c_api.h"

static constexpr uint32_t HDMEM_MAGIC = 0x48444D45U;  // "HDME"
static constexpr uint32_t HDMEM_VERSION = 1;

static constexpr int HDMEM_OK = 0;
static constexpr int HDMEM_ERR_WOULD_BLOCK = -11;
static constexpr int HDMEM_ERR_INVALID = -22;
static constexpr int HDMEM_ERR_BACKEND = -5;

struct alignas(64) HostDeviceMemoryHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t signal_count;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t total_bytes;
    uint64_t fatal_status;
    uint64_t reserved[3];
};

struct alignas(64) HostDeviceSignalSlot {
    volatile uint64_t value;
    uint64_t reserved[7];
};

struct HostDeviceMemory {
    void *device_base;
    void *host_base;
    size_t bytes;
    uint32_t owns_host_allocation;
    void (*free_host_allocation)(void *);
};

size_t host_device_memory_required_bytes(const HostDeviceMemoryConfig *cfg);
int host_device_memory_init_region(void *host_base, size_t bytes, const HostDeviceMemoryConfig *cfg);
HostDeviceMemory *host_device_memory_wrap(
    void *device_base, void *host_base, size_t bytes, const HostDeviceMemoryConfig *cfg, uint32_t owns_host_allocation,
    void (*free_host_allocation)(void *)
);
void host_device_memory_destroy(HostDeviceMemory *mem);
int host_device_memory_info(HostDeviceMemory *mem, HostDeviceMemoryInfo *info);
int host_device_memory_read(HostDeviceMemory *mem, uint64_t offset, void *dst, size_t nbytes);
int host_device_memory_write(HostDeviceMemory *mem, uint64_t offset, const void *src, size_t nbytes);
int host_device_memory_notify(HostDeviceMemory *mem, uint32_t signal_id, uint64_t value);
int host_device_memory_wait(HostDeviceMemory *mem, uint32_t signal_id, uint64_t target, uint32_t timeout_us);

int host_device_memory_read_l2_for_test(HostDeviceMemory *mem, uint64_t offset, void *dst, size_t nbytes);
int host_device_memory_write_l2_for_test(HostDeviceMemory *mem, uint64_t offset, const void *src, size_t nbytes);
int host_device_memory_notify_l2_for_test(HostDeviceMemory *mem, uint32_t signal_id, uint64_t value);
int host_device_memory_wait_l2_for_test(HostDeviceMemory *mem, uint32_t signal_id, uint64_t target, uint32_t timeout_us);

#endif  // SRC_COMMON_WORKER_HOST_DEVICE_MEMORY_H_
