#include "aicpu/device_log.h"
#include "runtime.h"

// The aicpu_build_graph runtime expects the graph building program to be
// provided by the example and compiled into the AICPU shared library.
//
// Example location (by convention):
//   examples/<example>/kernels/aicpu/build_graph_aicpu.cpp
//
// CodeRunner will include that directory when compiling the AICPU binary for
// this runtime. This weak default keeps the runtime linkable even when no
// example builder is provided (it fails loudly at runtime).
extern "C" __attribute__((weak)) int build_graph_aicpu(Runtime* runtime) {
    (void)runtime;
    DEV_ERROR("%s", "build_graph_aicpu() not provided by example");
    return -1;
}

