/**
 * Platform Configuration Implementation - a2a3 Platform
 *
 * This file implements platform-specific runtime configuration functions
 * for the a2a3 platform, specifically the core topology initialization.
 */

#include "platform_config.h"
#include "runtime.h"
#include <iostream>

/**
 * Initialize core topology for a2a3 platform
 *
 * a2a3 topology characteristics:
 * - Each block has 3 cores: 1 AIC + 2 AIV
 * - Total cores = block_dim * 3
 * - AIC cores are indexed [0, block_dim)
 * - AIV cores are indexed [block_dim, block_dim + 2*block_dim)
 * - block_idx mapping:
 *   - AIC core i: block_idx = i (direct mapping)
 *   - AIV core at position (block_dim + 2*b + offset): block_idx = b
 *     where offset is 0 or 1 for the two AIV cores in block b
 *
 * This mapping must match the platform kernel's block_idx calculation:
 * - AICore kernel (kernel.cpp): block_idx = get_block_idx()
 * - AIVector kernel (kernel.cpp): block_idx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num()
 */
extern "C" int init_runtime_core_topology(Runtime* runtime, int block_dim) {
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }

    if (block_dim <= 0) {
        std::cerr << "Error: Invalid block_dim: " << block_dim << "\n";
        return -1;
    }

    // Calculate core counts
    int num_aic = block_dim;           // 1 AIC per block
    int num_aiv = block_dim * 2;       // 2 AIV per block
    int total_cores = num_aic + num_aiv;

    if (total_cores > RUNTIME_MAX_WORKER) {
        std::cerr << "Error: Total cores (" << total_cores
                  << ") exceeds RUNTIME_MAX_WORKER (" << RUNTIME_MAX_WORKER << ")\n";
        return -1;
    }

    // Initialize topology metadata
    runtime->core_topology.total_cores = total_cores;

    // Configure AIC cores: [0, block_dim)
    // Each AIC core i belongs to block i
    for (int i = 0; i < num_aic; i++) {
        runtime->core_topology.cores[i].core_id = i;
        runtime->core_topology.cores[i].block_idx = i;
        runtime->core_topology.cores[i].core_type = 0;  // AIC
    }

    // Configure AIV cores: [block_dim, block_dim + 2*block_dim)
    // For each block b:
    //   - First AIV:  core_id = block_dim + 2*b,     block_idx = b
    //   - Second AIV: core_id = block_dim + 2*b + 1, block_idx = b
    for (int b = 0; b < block_dim; b++) {
        int aiv_core_0 = num_aic + b * 2;      // First AIV of block b
        int aiv_core_1 = num_aic + b * 2 + 1;  // Second AIV of block b

        runtime->core_topology.cores[aiv_core_0].core_id = aiv_core_0;
        runtime->core_topology.cores[aiv_core_0].block_idx = b;
        runtime->core_topology.cores[aiv_core_0].core_type = 1;  // AIV

        runtime->core_topology.cores[aiv_core_1].core_id = aiv_core_1;
        runtime->core_topology.cores[aiv_core_1].block_idx = b;
        runtime->core_topology.cores[aiv_core_1].core_type = 1;  // AIV
    }

    // Mark topology as initialized
    runtime->core_topology.initialized = true;

    std::cout << "Platform config: Initialized a2a3 core topology - "
              << num_aic << " AIC + " << num_aiv << " AIV = "
              << total_cores << " total cores\n";

    return 0;
}
