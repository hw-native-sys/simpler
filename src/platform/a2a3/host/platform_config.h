/**
 * Platform Configuration - a2a3 Platform
 *
 * This header declares platform-specific initialization functions
 * for configuring the runtime with a2a3 platform characteristics.
 */

#ifndef PLATFORM_CONFIG_H
#define PLATFORM_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
class Runtime;

// =============================================================================
// Platform-specific execution configuration
// =============================================================================

/**
 * Maximum number of AICPU threads for task scheduling
 * a2a3 platform supports up to 4 AICPU threads
 */
#define PLATFORM_MAX_AICPU_THREADS 4

/**
 * Maximum number of AIC cores per AICPU thread
 * a2a3 platform: each block has 1 AIC, max 24 blocks per thread
 */
#define PLATFORM_MAX_AIC_PER_THREAD 24

/**
 * Maximum number of AIV cores per AICPU thread
 * a2a3 platform: each block has 2 AIV, max 24 blocks per thread = 48 AIV
 */
#define PLATFORM_MAX_AIV_PER_THREAD 48

/**
 * Maximum total cores (AIC + AIV) per AICPU thread
 */
#define PLATFORM_MAX_CORES_PER_THREAD (PLATFORM_MAX_AIC_PER_THREAD + PLATFORM_MAX_AIV_PER_THREAD)

/**
 * Initialize runtime core topology for a2a3 platform
 *
 * This function configures the core topology based on a2a3 platform
 * characteristics:
 * - Each block has 3 cores: 1 AIC + 2 AIV
 * - AIC cores: [0, block_dim)
 * - AIV cores: [block_dim, block_dim + 2*block_dim)
 * - block_idx mapping:
 *   - AIC core i: block_idx = i
 *   - AIV core (block_dim + 2*b + offset): block_idx = b
 *
 * @param runtime   Pointer to Runtime to configure
 * @param block_dim Number of blocks in the configuration
 * @return 0 on success, negative on error
 */
int init_runtime_core_topology(Runtime* runtime, int block_dim);

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_CONFIG_H
