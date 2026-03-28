/**
 * @file aicore.h
 * @brief AICore Platform Abstraction Layer
 *
 * Provides unified AICore qualifiers and macros for both real hardware
 * and simulation environments. Uses conditional compilation to select
 * the appropriate implementation.
 *
 * Platform Support:
 * - a5: Real Ascend hardware with CANN compiler
 * - a5sim: Host-based simulation using standard C++
 */

#ifndef PLATFORM_AICORE_H_
#define PLATFORM_AICORE_H_

// =============================================================================
// Common Memory Qualifiers (All Platforms)
// =============================================================================

#include "common/qualifier.h"

// =============================================================================
// Platform-Specific Definitions
// =============================================================================
// Platform-specific macros (__aicore__, dcci) are defined in inner_kernel.h
// The build system selects the correct implementation based on platform:
// - src/a5/platform/onboard/aicore/inner_kernel.h (real hardware)
// - src/a5/platform/sim/aicore/inner_kernel.h (simulation)

#include "inner_kernel.h"

#endif  // PLATFORM_AICORE_H_
