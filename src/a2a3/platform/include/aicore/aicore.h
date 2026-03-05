/**
 * @file aicore.h
 * @brief AICore Platform Abstraction Layer
 *
 * Provides unified AICore qualifiers and macros for both real hardware
 * and simulation environments. Uses conditional compilation to select
 * the appropriate implementation.
 *
 * Platform Support:
 * - a2a3: Real Ascend hardware with CANN compiler
 * - a2a3sim: Host-based simulation using standard C++
 */

#ifndef PLATFORM_AICORE_H_
#define PLATFORM_AICORE_H_

// =============================================================================
// Common Memory Qualifiers (All Platforms)
// =============================================================================

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __in__
#define __in__
#endif

#ifndef __out__
#define __out__
#endif

// =============================================================================
// Platform-Specific Definitions
// =============================================================================
// Platform-specific macros (__aicore__, dcci) are defined in inner_kernel.h
// The build system selects the correct implementation based on platform:
// - src/platform/a2a3/aicore/inner_kernel.h (real hardware)
// - src/platform/a2a3sim/aicore/inner_kernel.h (simulation)

#include "inner_kernel.h"

#endif  // PLATFORM_AICORE_H_
