#include "host/platform_compile_info.h"
#include "host/runtime_compile_info.h"
#include <string.h>

extern "C" {

ToolchainType get_incore_compiler(void) {
    if (strcmp(get_platform(), "a2a3") == 0) return TOOLCHAIN_CCEC;
    return TOOLCHAIN_HOST_GXX_15;
}

ToolchainType get_orchestration_compiler(void) {
    // tensormap_and_ringbuffer: a2a3 needs aarch64 cross-compile (AICPU is aarch64)
    if (strcmp(get_platform(), "a2a3") == 0) return TOOLCHAIN_AARCH64_GXX;
    return TOOLCHAIN_HOST_GXX;
}

}
