/**
 * Tensor + Scalar Addition Kernel (Simulation)
 *
 * Implements: out[i] = src[i] + scalar
 *
 * This is a simple loop-based implementation for simulation.
 */

#include <cstdint>

/**
 * Tensor + scalar addition kernel implementation
 *
 * @param args  Argument array:
 *              args[0] = src pointer (input tensor)
 *              args[1] = scalar (encoded as uint64_t, needs conversion)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" void kernel_add_scalar(int64_t* args) {
    float* src = reinterpret_cast<float*>(args[0]);

    // Convert scalar from uint64_t encoding to float
    union {
        uint64_t u64;
        float f32;
    } conv;
    conv.u64 = static_cast<uint64_t>(args[1]);
    float scalar = conv.f32;

    float* out = reinterpret_cast<float*>(args[2]);
    int size = static_cast<int>(args[3]);

    for (int i = 0; i < size; i++) {
        out[i] = src[i] + scalar;
    }
}
