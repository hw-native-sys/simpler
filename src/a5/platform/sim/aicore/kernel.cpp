/**
 * AICore Kernel Wrapper for Simulation
 *
 * Provides a wrapper around aicore_execute for dlsym lookup.
 * Sets up per-thread simulated register base before calling the executor.
 */

#include <cstdint>
#include "aicore/aicore.h"
#include "common/core_type.h"
#include "common/platform_config.h"
#include "runtime.h"

// Thread-local simulated register base (declared in inner_kernel.h)
thread_local volatile uint8_t* g_sim_reg_base = nullptr;

// Thread-local simulated physical core ID (declared in inner_kernel.h)
thread_local uint32_t g_sim_physical_core_id = 0;

// Declare the original function (defined in aicore_executor.cpp with weak linkage)
void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type);

// Wrapper with extern "C" for dlsym lookup
// NOTE: physical_core_id stays in wrapper signature (DeviceRunner passes it for register indexing)
extern "C" void aicore_execute_wrapper(__gm__ Runtime* runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs) {
    // Set up simulated register base for this thread.
    // regs points to an array of uint64_t base addresses (one per core).
    // physical_core_id indexes into it to get this core's register block.
    if (regs != 0) {
        uint64_t* regs_array = reinterpret_cast<uint64_t*>(regs);
        g_sim_reg_base = reinterpret_cast<volatile uint8_t*>(regs_array[physical_core_id]);
    }

    g_sim_physical_core_id = physical_core_id;

    aicore_execute(runtime, block_idx, core_type);
}
