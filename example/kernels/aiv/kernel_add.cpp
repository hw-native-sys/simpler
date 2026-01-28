/**
 * Element-wise Tensor Addition Kernel
 *
 * Implements: out[i] = src0[i] + src1[i]
 *
 * This kernel performs element-wise addition of two tensors. It's compiled
 * separately as a standalone kernel and linked with the dispatcher using
 * function pointers, demonstrating the separation pattern used in production
 * systems where kernel binaries are loaded dynamically.
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

/**
 * Element-wise addition kernel implementation
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src0 pointer (first input tensor)
 *              args[1] = src1 pointer (second input tensor)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_add(__gm__ int64_t* args) {
    // Unpack arguments
    __gm__ float* src0 = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* src1 = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* out = reinterpret_cast<__gm__ float*>(args[2]);
    int size = static_cast<int>(args[3]);

    // Perform computation
    for (int i = 0; i < size; i++) {
        out[i] = src0[i] + src1[i];
    }
}
