/**
 * Online Softmax Update + Normalize Kernel (AIV) - Multi-Head
 *
 * Processes ALL heads in one task.
 *
 * Memory layout:
 *   mij:    (num_heads,)             current block row max
 *   lij:    (num_heads,)             current block row sum
 *   oi_new: (num_heads, head_dim)    current block PV output
 *   mi:     (num_heads,)             accumulated max  (in/out)
 *   li:     (num_heads,)             accumulated sum  (in/out)
 *   oi:     (num_heads, head_dim)    accumulated out  (in/out)
 *   dst:    (num_heads, head_dim)    final output (written when is_last)
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
    __gm__ float* mij    = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* lij    = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* oi_new = reinterpret_cast<__gm__ float*>(args[2]);
    __gm__ float* mi     = reinterpret_cast<__gm__ float*>(args[3]);
    __gm__ float* li     = reinterpret_cast<__gm__ float*>(args[4]);
    __gm__ float* oi     = reinterpret_cast<__gm__ float*>(args[5]);
    int is_first  = static_cast<int>(args[6]);
    int is_last   = static_cast<int>(args[7]);
    __gm__ float* dst    = reinterpret_cast<__gm__ float*>(args[8]);
    int num_heads = static_cast<int>(args[9]);
    int head_dim  = static_cast<int>(args[10]);

    for (int h = 0; h < num_heads; h++) {
        __gm__ float* oi_new_h = oi_new + h * head_dim;
        __gm__ float* oi_h     = oi     + h * head_dim;

        if (is_first) {
            mi[h] = mij[h];
            li[h] = lij[h];
            for (int d = 0; d < head_dim; d++) {
                oi_h[d] = oi_new_h[d];
            }
        } else {
            float mi_new = (mi[h] > mij[h]) ? mi[h] : mij[h];
            float alpha = my_exp(mi[h] - mi_new);
            float beta  = my_exp(mij[h] - mi_new);

            li[h] = alpha * li[h] + beta * lij[h];
            for (int d = 0; d < head_dim; d++) {
                oi_h[d] = alpha * oi_h[d] + beta * oi_new_h[d];
            }
            mi[h] = mi_new;
        }

        // Fused normalize on last block
        if (is_last) {
            float inv_li = 1.0f / li[h];
            __gm__ float* dst_h = dst + h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                dst_h[d] = oi_h[d] * inv_li;
            }
        }
    }
}
