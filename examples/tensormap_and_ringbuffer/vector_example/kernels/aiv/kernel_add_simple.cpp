/**
 * Simple Add Kernel for debugging
 *
 * Does not use PTO-ISA, just plain C++ to verify function call works.
 */

#include <cstdint>
#include <cstdio>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    fprintf(stderr, "[kernel_entry] entered, args=%p\n", (void*)args);
    fflush(stderr);

    __gm__ float* src0 = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* src1 = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* out = reinterpret_cast<__gm__ float*>(args[2]);
    int size = static_cast<int>(args[3]);

    fprintf(stderr, "[kernel_entry] src0=%p src1=%p out=%p size=%d\n",
            (void*)src0, (void*)src1, (void*)out, size);
    fflush(stderr);

    // Simple element-wise addition
    for (int i = 0; i < size && i < 16; i++) {
        out[i] = src0[i] + src1[i];
    }

    fprintf(stderr, "[kernel_entry] done\n");
    fflush(stderr);
}
