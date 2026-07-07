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
 * @file device_malloc.cpp
 * @brief Device Memory Allocation for Real Hardware
 *
 * Implements HBM allocation using HAL memory API (halMemAlloc/halMemFree).
 * These symbols are resolved at runtime via dlsym from libascend_hal.so,
 * which is already loaded in the AICPU scheduler process.
 */

#include "aicpu/device_malloc.h"
#include "common/unified_log.h"

#include <dlfcn.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

using HalMemAllocFn = int (*)(void **pp, unsigned long long size, unsigned long long flag);
using HalMemFreeFn = int (*)(void *pp);
using HalMemCtlFn = int (*)(int type, void *param_value, size_t param_value_size, void *out_value, size_t *out_size_ret);

static HalMemAllocFn g_halMemAlloc = nullptr;
static HalMemFreeFn g_halMemFree = nullptr;
static HalMemCtlFn g_halMemCtl = nullptr;
static bool g_hal_resolved = false;

// Double-page-table "nocache" alias offset. On A5 HBM is mapped twice — a
// cacheable view and an uncacheable view separated by a fixed VA offset. The
// distributed engine's cross-core shared segment (DistGlobal) MUST be accessed
// through the uncacheable alias: its Coherent<T> path uses plain loads/stores +
// device atomics and relies on every AICore seeing the same HBM word with no
// per-core cached copy. Without the alias, each AICore caches its own copy and
// the startup barrier / drain counters never converge (device watchdog timeout).
// 0 = not yet queried; SENTINEL = queried and unavailable (fall back to cached).
static constexpr uint64_t kNocacheOffsetUnavailable = ~0ULL;
static uint64_t g_nocache_offset = 0;

// halMemCtl ctrlType (ascend_hal_define.h): Inpara is devid, Outpara is the
// nocache VA offset. Kept as a local constant to avoid pulling the driver
// header into this TU.
static constexpr int kCtrlTypeGetDoublePgtableOffset = 3;

static void resolve_hal_mem_functions() {
    if (g_hal_resolved) {
        return;
    }
    g_halMemAlloc = reinterpret_cast<HalMemAllocFn>(dlsym(RTLD_DEFAULT, "halMemAlloc"));
    g_halMemFree = reinterpret_cast<HalMemFreeFn>(dlsym(RTLD_DEFAULT, "halMemFree"));
    g_halMemCtl = reinterpret_cast<HalMemCtlFn>(dlsym(RTLD_DEFAULT, "halMemCtl"));
    if (g_halMemAlloc == nullptr || g_halMemFree == nullptr) {
        LOG_ERROR("Failed to resolve halMemAlloc/halMemFree: %s", dlerror());
        g_halMemAlloc = nullptr;
        g_halMemFree = nullptr;
    }
    if (g_halMemCtl == nullptr) {
        LOG_WARN("halMemCtl not available; distributed GM segment stays cacheable (%s)", dlerror());
    }
    g_hal_resolved = true;
}

// Query (once) the cached→uncached VA alias offset. Returns kNocacheOffsetUnavailable
// if the driver/device does not support the double page table, in which case the
// caller returns the plain cached pointer.
static uint64_t nocache_alias_offset() {
    if (g_nocache_offset != 0) {
        return g_nocache_offset;
    }
    if (g_halMemCtl == nullptr) {
        g_nocache_offset = kNocacheOffsetUnavailable;
        return g_nocache_offset;
    }
    uint32_t devid = 0;  // local device
    uint64_t offset = 0;
    size_t out_size = sizeof(offset);
    int rc = g_halMemCtl(kCtrlTypeGetDoublePgtableOffset, &devid, sizeof(devid), &offset, &out_size);
    if (rc != 0 || offset == 0) {
        LOG_INFO_V9("[dmalloc] halMemCtl(GET_DOUBLE_PGTABLE_OFFSET) unavailable: rc=%d offset=0x%lx", rc, offset);
        g_nocache_offset = kNocacheOffsetUnavailable;
    } else {
        LOG_INFO_V9("[dmalloc] nocache alias offset = 0x%lx", offset);
        g_nocache_offset = offset;
    }
    return g_nocache_offset;
}

void *aicpu_device_malloc(size_t size) {
    resolve_hal_mem_functions();

    if (g_halMemAlloc == nullptr) {
        LOG_ERROR("halMemAlloc not available, cannot allocate device memory");
        return nullptr;
    }

    void *ptr = nullptr;
    // halMemAlloc flag layout (ascend_hal_define.h):
    //   bit0~9:   devid (0 for local device)
    //   bit10~13: virt mem type (MEM_SVM=0x0 << 10)
    //   bit14~16: phy mem type  (MEM_TYPE_HBM=0x1 << 14)
    constexpr unsigned long long MEM_TYPE_HBM = 0x1ULL << 14;
    unsigned long long flag = MEM_TYPE_HBM;
    int rc = g_halMemAlloc(&ptr, static_cast<unsigned long long>(size), flag);
    if (rc != 0 || ptr == nullptr) {
        LOG_ERROR("halMemAlloc failed: rc=%d size=%zu flag=0x%llx", rc, size, flag);
        return nullptr;
    }
    // Return the uncacheable alias so cross-core sharing of this segment is
    // coherent (see nocache_alias_offset). If the double page table is
    // unavailable, fall back to the cached pointer.
    const uint64_t off = nocache_alias_offset();
    if (off != kNocacheOffsetUnavailable) {
        return reinterpret_cast<void *>(reinterpret_cast<uint64_t>(ptr) + off);
    }
    return ptr;
}

unsigned long long aicpu_device_nocache_offset() {
    resolve_hal_mem_functions();
    const uint64_t off = nocache_alias_offset();
    return (off == kNocacheOffsetUnavailable) ? 0ULL : static_cast<unsigned long long>(off);
}

void aicpu_device_probe_uncacheable(void *va, struct AicpuUncacheProbe *out) {
    if (out == nullptr) {
        return;
    }
    *out = AicpuUncacheProbe{};
    resolve_hal_mem_functions();
    out->hal_ctl_resolved = (g_halMemCtl != nullptr) ? 1 : 0;
    if (g_halMemCtl == nullptr) {
        return;
    }
    // CTRL_TYPE_GET_DOUBLE_PGTABLE_OFFSET = 3: Inpara devid, Outpara nocache off.
    {
        uint32_t devid = 0;
        uint64_t offset = 0;
        size_t out_size = sizeof(offset);
        out->dpt_rc = g_halMemCtl(3, &devid, sizeof(devid), &offset, &out_size);
        out->dpt_offset = offset;
    }
    // CTRL_TYPE_SUPPORT_FEATURE = 2: query supported-feature bitmask (devid in).
    {
        uint32_t devid = 0;
        uint64_t bits = 0;
        size_t out_size = sizeof(bits);
        out->feature_rc = g_halMemCtl(2, &devid, sizeof(devid), &bits, &out_size);
        out->feature_bits = bits;
    }
    // CTRL_TYPE_GET_DCACHE_ADDR = 8: given a VA, ask for its dcache/alias addr.
    {
        uint64_t in_va = reinterpret_cast<uint64_t>(va);
        uint64_t addr = 0;
        size_t out_size = sizeof(addr);
        out->dcache_rc = g_halMemCtl(8, &in_va, sizeof(in_va), &addr, &out_size);
        out->dcache_addr = addr;
    }
}

void aicpu_device_free(void *ptr) {
    if (ptr == nullptr) {
        return;
    }

    resolve_hal_mem_functions();

    if (g_halMemFree == nullptr) {
        LOG_ERROR("halMemFree not available, cannot free device memory");
        return;
    }
    // Strip the uncacheable alias offset (if applied at alloc) to recover the
    // original cached VA halMemAlloc handed back — halMemFree expects that one.
    const uint64_t off = nocache_alias_offset();
    void *base = ptr;
    if (off != kNocacheOffsetUnavailable && reinterpret_cast<uint64_t>(ptr) >= off) {
        base = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(ptr) - off);
    }
    int rc = g_halMemFree(base);
    if (rc != 0) {
        LOG_ERROR("halMemFree failed: rc=%d ptr=%p", rc, base);
    }
}
