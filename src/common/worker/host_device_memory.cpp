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

#include "host_device_memory.h"

#include <string.h>

#include <chrono>
#include <limits>
#include <new>
#include <thread>

namespace {

bool checked_add_size(size_t a, size_t b, size_t *out) {
    if (out == nullptr || a > std::numeric_limits<size_t>::max() - b) return false;
    *out = a + b;
    return true;
}

bool checked_mul_size(size_t a, size_t b, size_t *out) {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<size_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool checked_align_up(size_t v, size_t alignment, size_t *out) {
    if (alignment == 0 || (alignment & (alignment - 1U)) != 0) return false;
    size_t biased = 0;
    if (!checked_add_size(v, alignment - 1U, &biased)) return false;
    *out = biased & ~(alignment - 1U);
    return true;
}

bool valid_cfg(const HostDeviceMemoryConfig *cfg) {
    return cfg != nullptr && cfg->data_bytes > 0 && cfg->signal_count > 0;
}

bool compute_layout(const HostDeviceMemoryConfig *cfg, size_t *data_offset, size_t *total_bytes) {
    if (!valid_cfg(cfg) || data_offset == nullptr || total_bytes == nullptr) return false;
    if (cfg->data_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;

    size_t signal_bytes = 0;
    size_t signals_end = 0;
    size_t offset = 0;
    size_t raw_total = 0;
    size_t required = 0;
    if (!checked_mul_size(static_cast<size_t>(cfg->signal_count), sizeof(HostDeviceSignalSlot), &signal_bytes)) {
        return false;
    }
    if (!checked_add_size(sizeof(HostDeviceMemoryHeader), signal_bytes, &signals_end)) return false;
    if (!checked_align_up(signals_end, 64, &offset)) return false;
    if (!checked_add_size(offset, static_cast<size_t>(cfg->data_bytes), &raw_total)) return false;
    if (!checked_align_up(raw_total, 64, &required)) return false;

    *data_offset = offset;
    *total_bytes = required;
    return true;
}

HostDeviceMemoryHeader *header(HostDeviceMemory *mem) {
    return reinterpret_cast<HostDeviceMemoryHeader *>(mem->host_base);
}

const HostDeviceMemoryHeader *header_const(const HostDeviceMemory *mem) {
    return reinterpret_cast<const HostDeviceMemoryHeader *>(mem->host_base);
}

HostDeviceSignalSlot *signals(HostDeviceMemory *mem) {
    return reinterpret_cast<HostDeviceSignalSlot *>(reinterpret_cast<uint8_t *>(mem->host_base) + sizeof(HostDeviceMemoryHeader));
}

void *host_data_ptr(HostDeviceMemory *mem) {
    auto *hdr = header(mem);
    return reinterpret_cast<uint8_t *>(mem->host_base) + hdr->data_offset;
}

uint64_t device_data_addr(HostDeviceMemory *mem) {
    auto *hdr = header(mem);
    return reinterpret_cast<uint64_t>(mem->device_base) + hdr->data_offset;
}

bool valid_region(HostDeviceMemory *mem) {
    if (mem == nullptr || mem->host_base == nullptr || mem->device_base == nullptr) return false;
    auto *hdr = header(mem);
    return hdr->magic == HDMEM_MAGIC && hdr->version == HDMEM_VERSION && hdr->total_bytes <= mem->bytes;
}

bool valid_range(HostDeviceMemory *mem, uint64_t offset, size_t nbytes) {
    if (!valid_region(mem)) return false;
    auto *hdr = header(mem);
    return offset <= hdr->data_bytes && nbytes <= hdr->data_bytes - offset;
}

int read_impl(HostDeviceMemory *mem, uint64_t offset, void *dst, size_t nbytes) {
    if ((dst == nullptr && nbytes != 0) || !valid_range(mem, offset, nbytes)) return HDMEM_ERR_INVALID;
    if (nbytes != 0) memcpy(dst, reinterpret_cast<uint8_t *>(host_data_ptr(mem)) + offset, nbytes);
    return HDMEM_OK;
}

int write_impl(HostDeviceMemory *mem, uint64_t offset, const void *src, size_t nbytes) {
    if ((src == nullptr && nbytes != 0) || !valid_range(mem, offset, nbytes)) return HDMEM_ERR_INVALID;
    if (nbytes != 0) memcpy(reinterpret_cast<uint8_t *>(host_data_ptr(mem)) + offset, src, nbytes);
    return HDMEM_OK;
}

int notify_impl(HostDeviceMemory *mem, uint32_t signal_id, uint64_t value) {
    if (!valid_region(mem)) return HDMEM_ERR_INVALID;
    auto *hdr = header(mem);
    if (signal_id >= hdr->signal_count) return HDMEM_ERR_INVALID;
    __atomic_store_n(&signals(mem)[signal_id].value, value, __ATOMIC_RELEASE);
    return HDMEM_OK;
}

int wait_impl(HostDeviceMemory *mem, uint32_t signal_id, uint64_t target, uint32_t timeout_us) {
    if (!valid_region(mem)) return HDMEM_ERR_INVALID;
    auto *hdr = header(mem);
    if (signal_id >= hdr->signal_count) return HDMEM_ERR_INVALID;
    auto start = std::chrono::steady_clock::now();
    while (true) {
        uint64_t value = __atomic_load_n(&signals(mem)[signal_id].value, __ATOMIC_ACQUIRE);
        if (value >= target) return HDMEM_OK;
        if (timeout_us == 0) return HDMEM_ERR_WOULD_BLOCK;
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        if (elapsed.count() >= timeout_us) return HDMEM_ERR_WOULD_BLOCK;
        std::this_thread::yield();
    }
}

}  // namespace

size_t host_device_memory_required_bytes(const HostDeviceMemoryConfig *cfg) {
    size_t data_offset = 0;
    size_t total_bytes = 0;
    if (!compute_layout(cfg, &data_offset, &total_bytes)) return 0;
    return total_bytes;
}

int host_device_memory_init_region(void *host_base, size_t bytes, const HostDeviceMemoryConfig *cfg) {
    if (host_base == nullptr) return HDMEM_ERR_INVALID;
    size_t data_offset = 0;
    size_t required = 0;
    if (!compute_layout(cfg, &data_offset, &required)) return HDMEM_ERR_INVALID;
    if (required == 0 || bytes < required) return HDMEM_ERR_INVALID;
    memset(host_base, 0, required);
    auto *hdr = reinterpret_cast<HostDeviceMemoryHeader *>(host_base);
    hdr->magic = HDMEM_MAGIC;
    hdr->version = HDMEM_VERSION;
    hdr->flags = cfg->flags;
    hdr->signal_count = cfg->signal_count;
    hdr->data_offset = data_offset;
    hdr->data_bytes = cfg->data_bytes;
    hdr->total_bytes = required;
    return HDMEM_OK;
}

HostDeviceMemory *host_device_memory_wrap(
    void *device_base, void *host_base, size_t bytes, const HostDeviceMemoryConfig *cfg, uint32_t owns_host_allocation,
    void (*free_host_allocation)(void *)
) {
    if (device_base == nullptr || host_base == nullptr || !valid_cfg(cfg)) return nullptr;
    int rc = host_device_memory_init_region(host_base, bytes, cfg);
    if (rc != HDMEM_OK) return nullptr;
    HostDeviceMemory *mem = new (std::nothrow) HostDeviceMemory();
    if (mem == nullptr) return nullptr;
    mem->device_base = device_base;
    mem->host_base = host_base;
    mem->bytes = host_device_memory_required_bytes(cfg);
    mem->owns_host_allocation = owns_host_allocation;
    mem->free_host_allocation = free_host_allocation;
    return mem;
}

void host_device_memory_destroy(HostDeviceMemory *mem) {
    if (mem == nullptr) return;
    if (mem->owns_host_allocation && mem->free_host_allocation != nullptr && mem->host_base != nullptr) {
        mem->free_host_allocation(mem->host_base);
    }
    delete mem;
}

int host_device_memory_info(HostDeviceMemory *mem, HostDeviceMemoryInfo *info) {
    if (!valid_region(mem) || info == nullptr) return HDMEM_ERR_INVALID;
    const auto *hdr = header_const(mem);
    info->host_ptr = reinterpret_cast<uint64_t>(mem->host_base) + hdr->data_offset;
    info->device_ptr = device_data_addr(mem);
    info->data_bytes = hdr->data_bytes;
    info->signal_count = hdr->signal_count;
    info->flags = hdr->flags;
    return HDMEM_OK;
}

int host_device_memory_read(HostDeviceMemory *mem, uint64_t offset, void *dst, size_t nbytes) {
    return read_impl(mem, offset, dst, nbytes);
}

int host_device_memory_write(HostDeviceMemory *mem, uint64_t offset, const void *src, size_t nbytes) {
    return write_impl(mem, offset, src, nbytes);
}

int host_device_memory_notify(HostDeviceMemory *mem, uint32_t signal_id, uint64_t value) {
    return notify_impl(mem, signal_id, value);
}

int host_device_memory_wait(HostDeviceMemory *mem, uint32_t signal_id, uint64_t target, uint32_t timeout_us) {
    return wait_impl(mem, signal_id, target, timeout_us);
}

int host_device_memory_read_l2_for_test(HostDeviceMemory *mem, uint64_t offset, void *dst, size_t nbytes) {
    return read_impl(mem, offset, dst, nbytes);
}

int host_device_memory_write_l2_for_test(HostDeviceMemory *mem, uint64_t offset, const void *src, size_t nbytes) {
    return write_impl(mem, offset, src, nbytes);
}

int host_device_memory_notify_l2_for_test(HostDeviceMemory *mem, uint32_t signal_id, uint64_t value) {
    return notify_impl(mem, signal_id, value);
}

int host_device_memory_wait_l2_for_test(HostDeviceMemory *mem, uint32_t signal_id, uint64_t target, uint32_t timeout_us) {
    return wait_impl(mem, signal_id, target, timeout_us);
}
