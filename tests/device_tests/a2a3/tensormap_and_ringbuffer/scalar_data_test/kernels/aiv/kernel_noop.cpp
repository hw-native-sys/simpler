/**
 * No-op Kernel
 *
 * Empty kernel used to trigger runtime allocation for tensors passed
 * as OUTPUT/INOUT via add_inout(). The runtime allocates HeapRing memory
 * and writes initial values before dispatching this task; the kernel
 * itself does not read or modify any data.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) { (void)args; }
