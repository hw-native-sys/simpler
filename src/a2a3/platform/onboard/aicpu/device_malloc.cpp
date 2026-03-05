/**
 * @file device_malloc.cpp
 * @brief Device Memory Allocation for Real Hardware (a2a3)
 *
 * Implements HBM allocation using HAL memory API (halMemAlloc/halMemFree).
 * These symbols are resolved at runtime via dlsym from libascend_hal.so,
 * which is already loaded in the AICPU scheduler process.
 */

#include "aicpu/device_malloc.h"
#include "aicpu/device_log.h"

#include <dlfcn.h>
#include <cstdlib>

using HalMemAllocFn = int (*)(void** pp, unsigned long long size, unsigned long long flag);
using HalMemFreeFn = int (*)(void* pp);

static HalMemAllocFn g_halMemAlloc = nullptr;
static HalMemFreeFn g_halMemFree = nullptr;
static bool g_hal_resolved = false;

static void resolve_hal_mem_functions() {
    if (g_hal_resolved) {
        return;
    }
    g_halMemAlloc = reinterpret_cast<HalMemAllocFn>(dlsym(RTLD_DEFAULT, "halMemAlloc"));
    g_halMemFree = reinterpret_cast<HalMemFreeFn>(dlsym(RTLD_DEFAULT, "halMemFree"));
    if (g_halMemAlloc == nullptr || g_halMemFree == nullptr) {
        DEV_ERROR("Failed to resolve halMemAlloc/halMemFree: %s", dlerror());
        g_halMemAlloc = nullptr;
        g_halMemFree = nullptr;
    }
    g_hal_resolved = true;
}

void* aicpu_device_malloc(size_t size) {
    resolve_hal_mem_functions();

    if (g_halMemAlloc == nullptr) {
        DEV_ERROR("halMemAlloc not available, cannot allocate device memory");
        return nullptr;
    }

    void* ptr = nullptr;
    // halMemAlloc flag layout (ascend_hal_define.h):
    //   bit0~9:   devid (0 for local device)
    //   bit10~13: virt mem type (MEM_SVM=0x0 << 10)
    //   bit14~16: phy mem type  (MEM_TYPE_HBM=0x1 << 14)
    constexpr unsigned long long MEM_TYPE_HBM = 0x1ULL << 14;
    unsigned long long flag = MEM_TYPE_HBM;
    int rc = g_halMemAlloc(&ptr, static_cast<unsigned long long>(size), flag);
    if (rc != 0 || ptr == nullptr) {
        DEV_ERROR("halMemAlloc failed: rc=%d size=%zu flag=0x%llx", rc, size, flag);
        return nullptr;
    }
    return ptr;
}

void aicpu_device_free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    resolve_hal_mem_functions();

    if (g_halMemFree == nullptr) {
        DEV_ERROR("halMemFree not available, cannot free device memory");
        return;
    }
    int rc = g_halMemFree(ptr);
    if (rc != 0) {
        DEV_ERROR("halMemFree failed: rc=%d ptr=%p", rc, ptr);
    }
}
