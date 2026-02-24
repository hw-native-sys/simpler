#include <cstdint>
#include "aicpu/platform_regs.h"

static uint64_t g_platform_regs = 0;

void set_platform_regs(uint64_t regs) {
    g_platform_regs = regs;
}

uint64_t get_platform_regs() {
    return g_platform_regs;
}
