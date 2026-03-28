/**
 * @file qualifier.h
 * @brief Memory qualifier fallbacks for non-AICore builds
 *
 * On real AICore the compiler provides __gm__, __global__, etc. natively.
 * For AICPU, Host, and simulation builds these are no-ops.
 */

#ifndef PLATFORM_COMMON_QUALIFIER_H_
#define PLATFORM_COMMON_QUALIFIER_H_

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

#endif  // PLATFORM_COMMON_QUALIFIER_H_
