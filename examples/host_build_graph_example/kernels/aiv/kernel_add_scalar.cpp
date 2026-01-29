/**
 * Scalar Addition Kernel
 *
 * Implements: out[i] = src[i] + scalar
 *
 * This kernel adds a scalar value to each element of a tensor. It's compiled
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
 * Scalar addition kernel implementation
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src pointer (input tensor)
 *              args[1] = scalar value (as uint64_t, needs conversion to float)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_add_scalar(__gm__ int64_t* args) {
    // Unpack arguments
    __gm__ float* src = reinterpret_cast<__gm__ float*>(args[0]);

    // Convert scalar from uint64_t to float
    union {
        uint64_t u64;
        float f32;
    } converter;
    converter.u64 = args[1];
    float scalar = converter.f32;

    __gm__ float* out = reinterpret_cast<__gm__ float*>(args[2]);
    int size = static_cast<int>(args[3]);

    // Perform computation
    for (int i = 0; i < size; i++) {
        out[i] = src[i] + scalar;
    }
}
