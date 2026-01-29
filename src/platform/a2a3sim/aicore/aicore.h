/**
 * AICore Simulation Header
 *
 * Provides empty macros/stubs for simulating AICore execution on host.
 * Device-specific qualifiers become no-ops since we use unified host memory.
 */

#ifndef AICORE_SIM_H
#define AICORE_SIM_H

// Empty qualifiers - no special memory spaces on host
#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __in__
#define __in__
#endif

#ifndef __out__
#define __out__
#endif

// Cache coherency - no-op on host (unified memory)
#define ENTIRE_DATA_CACHE 0
#define CACHELINE_OUT 0
#define dcci(addr, mode, opt) ((void)0)

#endif  // AICORE_SIM_H
