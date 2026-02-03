/**
 * Softmax Preparation Kernel (AIV) - Multi-Head
 *
 * Performs row-wise softmax operations:
 *   sij_scale[h] = sij[h] * scale (in-place)
 *   mij[h] = max(sij_scale[h])
 *   pij[h] = exp(sij_scale[h] - mij[h])
 *   lij[h] = sum(pij[h])
 *
 * Memory layout:
 *   sij: (num_heads, block_size)  in-place scaled
 *   pij: (num_heads, block_size)  output
 *   mij: (num_heads,)             output
 *   lij: (num_heads,)             output
 */
#include <cstdint>
#include <pto/pto-inst.hpp>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

__aicore__ static inline float my_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;

    float result = 1.0f + x * (1.0f + x * (0.5f + x * (0.166666667f +
                   x * (0.041666667f + x * (0.008333333f + x * 0.001388889f)))));

    int n = static_cast<int>(x * 1.4426950408889634f);
    union { uint32_t i; float f; } bias;
    bias.i = static_cast<uint32_t>((n + 127)) << 23;

    return result * bias.f;
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* sij = reinterpret_cast<__gm__ float*>(args[0]);   // (num_heads, block_size)
    union { uint64_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    __gm__ float* pij = reinterpret_cast<__gm__ float*>(args[2]);    // (num_heads, block_size)
    __gm__ float* mij = reinterpret_cast<__gm__ float*>(args[3]);    // (num_heads,)
    __gm__ float* lij = reinterpret_cast<__gm__ float*>(args[4]);    // (num_heads,)
    int num_heads  = static_cast<int>(args[5]);
    int block_size = static_cast<int>(args[6]);
    int valid_len  = static_cast<int>(args[7]);

    const float NEG_INF = -1e30f;

    for (int h = 0; h < num_heads; h++) {
        __gm__ float* sij_h = sij + h * block_size;
        __gm__ float* pij_h = pij + h * block_size;

        // Scale and find row max
        float max_val = NEG_INF;
        for (int j = 0; j < block_size; j++) {
            float val = (j < valid_len) ? sij_h[j] * scale_value : NEG_INF;
            sij_h[j] = val;
            if (val > max_val) max_val = val;
        }
        mij[h] = max_val;

        // Exp and row sum
        float sum_val = 0.0f;
        for (int j = 0; j < block_size; j++) {
            float exp_val = (j < valid_len) ? my_exp(sij_h[j] - max_val) : 0.0f;
            pij_h[j] = exp_val;
            sum_val += exp_val;
        }
        lij[h] = sum_val;
    }
}
