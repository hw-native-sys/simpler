/**
 * Softmax Prepare Kernel (AIV) - No external library calls
 * Computes: pij, mij, lij from sij with scale
 */
#include <cstdint>

// Inline exp approximation (no library call)
static inline float my_exp(float x) {
    // Clamp to avoid overflow/underflow
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    
    // Range reduction: x = n*ln2 + r
    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;
    
    float n_float = x * inv_ln2;
    int n = (int)(n_float + (n_float >= 0.0f ? 0.5f : -0.5f));
    float r = x - n * ln2;
    
    // Polynomial approx for exp(r) where |r| < ln2/2
    float r2 = r * r;
    float result = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * 0.041666668f)));
    
    // Multiply by 2^n
    union { float f; int i; } bias;
    bias.i = (n + 127) << 23;
    return result * bias.f;
}

extern "C" void aiv_softmax_prepare(int64_t* args) {
    float* sij = reinterpret_cast<float*>(args[0]);
    union { uint64_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    float* pij = reinterpret_cast<float*>(args[2]);
    float* mij = reinterpret_cast<float*>(args[3]);
    float* lij = reinterpret_cast<float*>(args[4]);
    int q_tile = static_cast<int>(args[5]);
    int block_size = static_cast<int>(args[6]);
    int valid_len = static_cast<int>(args[7]);

    const float NEG_INF = -1e30f;

    for (int i = 0; i < q_tile; i++) {
        // Scale and find row max
        float max_val = NEG_INF;
        for (int j = 0; j < block_size; j++) {
            float val;
            if (j < valid_len) {
                val = sij[i * block_size + j] * scale_value;
            } else {
                val = NEG_INF;
            }
            sij[i * block_size + j] = val;
            if (val > max_val) max_val = val;
        }
        mij[i] = max_val;

        // Exp and row sum
        float sum_val = 0.0f;
        for (int j = 0; j < block_size; j++) {
            float exp_val;
            if (j < valid_len) {
                exp_val = my_exp(sij[i * block_size + j] - max_val);
            } else {
                exp_val = 0.0f;
            }
            pij[i * block_size + j] = exp_val;
            sum_val += exp_val;
        }
        lij[i] = sum_val;
    }
}
