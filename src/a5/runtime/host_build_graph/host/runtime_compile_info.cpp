#include "host/platform_compile_info.h"
#include "host/runtime_compile_info.h"
#include <string.h>

extern "C" {

ToolchainType get_incore_compiler(void) {
    if (strcmp(get_platform(), "a2a3") == 0) return TOOLCHAIN_CCEC;
    return TOOLCHAIN_HOST_GXX_15;
}

ToolchainType get_orchestration_compiler(void) {
    // host_build_graph: always host g++ (orchestration runs on host)
    return TOOLCHAIN_HOST_GXX;
}

}
