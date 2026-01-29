/**
 * AICore Kernel Wrapper for Simulation
 *
 * Provides a wrapper around aicore_execute for dlsym lookup.
 * This allows adding pre/post processing around kernel execution.
 */

#include "aicore.h"
#include "runtime.h"

// Declare the original function (defined in aicore_executor.cpp with weak linkage)
void aicore_execute(__gm__ Runtime* runtime, int block_idx, int core_type);

// Wrapper with extern "C" for dlsym lookup
extern "C" void aicore_execute_wrapper(__gm__ Runtime* runtime, int block_idx, int core_type) {
    aicore_execute(runtime, block_idx, core_type);
}
