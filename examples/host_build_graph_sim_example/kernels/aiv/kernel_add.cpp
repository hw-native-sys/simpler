/**
 * Element-wise Tensor Addition Kernel (Simulation)
 *
 * Implements: out[i] = src0[i] + src1[i]
 *
 * This is a simple loop-based implementation for simulation.
 * The real a2a3 version uses PTO tile-based operations.
 */

#include <cstdint>

/**
 * Element-wise addition kernel implementation
 *
 * @param args  Argument array:
 *              args[0] = src0 pointer (first input tensor)
 *              args[1] = src1 pointer (second input tensor)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" void kernel_add(int64_t* args) {
    float* src0 = reinterpret_cast<float*>(args[0]);
    float* src1 = reinterpret_cast<float*>(args[1]);
    float* out = reinterpret_cast<float*>(args[2]);
    int size = static_cast<int>(args[3]);

    for (int i = 0; i < size; i++) {
        out[i] = src0[i] + src1[i];
    }
}
