#include <cstddef>

#include "aicpu/platform_regs.h"

void cache_invalidate_range(const void* /* addr */, size_t /* size */) {
    // No-op on simulation: no hardware cache to invalidate
}
