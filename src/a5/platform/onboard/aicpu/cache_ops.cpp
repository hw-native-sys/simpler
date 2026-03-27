#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"

void cache_invalidate_range(const void* addr, size_t size) {
    if (size == 0) {
        return;
    }
    const size_t kCacheLineSize = 64;
    uintptr_t start = (uintptr_t)addr & ~(kCacheLineSize - 1);
    uintptr_t end = ((uintptr_t)addr + size + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    for (uintptr_t p = start; p < end; p += kCacheLineSize) {
        __asm__ __volatile__("dc civac, %0" ::"r"(p) : "memory");
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
}
