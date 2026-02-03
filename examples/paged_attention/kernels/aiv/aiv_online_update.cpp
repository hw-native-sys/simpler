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

static inline float my_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;

    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;

    float n_float = x * inv_ln2;
    int n = (int)(n_float + (n_float >= 0.0f ? 0.5f : -0.5f));
    float r = x - n * ln2;

    float result = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * 0.041666668f)));

    union { float f; int i; } bias;
    bias.i = (n + 127) << 23;
    return result * bias.f;
}

extern "C" void aiv_online_update(int64_t* args) {
    float* mij    = reinterpret_cast<float*>(args[0]);
    float* lij    = reinterpret_cast<float*>(args[1]);
    float* oi_new = reinterpret_cast<float*>(args[2]);
    float* mi     = reinterpret_cast<float*>(args[3]);
    float* li     = reinterpret_cast<float*>(args[4]);
    float* oi     = reinterpret_cast<float*>(args[5]);
    int is_first  = static_cast<int>(args[6]);
    int is_last   = static_cast<int>(args[7]);
    float* dst    = reinterpret_cast<float*>(args[8]);
    int num_heads = static_cast<int>(args[9]);
    int head_dim  = static_cast<int>(args[10]);

    for (int h = 0; h < num_heads; h++) {
        float* oi_new_h = oi_new + h * head_dim;
        float* oi_h     = oi     + h * head_dim;

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
            float* dst_h = dst + h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                dst_h[d] = oi_h[d] * inv_li;
            }
        }
    }
}
