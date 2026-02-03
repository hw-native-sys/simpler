/**
 * Softmax Prepare Kernel (AIV) - Multi-Head
 *
 * Computes for all heads h:
 *   sij_scale[h] = sij[h] * scale
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

extern "C" void aiv_softmax_prepare(int64_t* args) {
    float* sij = reinterpret_cast<float*>(args[0]);   // (num_heads, block_size)
    union { uint64_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    float* pij = reinterpret_cast<float*>(args[2]);    // (num_heads, block_size)
    float* mij = reinterpret_cast<float*>(args[3]);    // (num_heads,)
    float* lij = reinterpret_cast<float*>(args[4]);    // (num_heads,)
    int num_heads  = static_cast<int>(args[5]);
    int block_size = static_cast<int>(args[6]);
    int valid_len  = static_cast<int>(args[7]);

    const float NEG_INF = -1e30f;

    for (int h = 0; h < num_heads; h++) {
        float* sij_h = sij + h * block_size;
        float* pij_h = pij + h * block_size;

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
