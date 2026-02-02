/**
 * Online Softmax Update + Normalize Kernel (AIV) - No external library calls
 *
 * Fused kernel: online softmax accumulation + optional final normalization.
 *
 * When is_last == 0: performs standard online softmax update.
 * When is_last == 1: after the update, normalizes and writes result to dst.
 *
 * This eliminates the separate aiv_normalize kernel, saving one task launch
 * per (batch, head) and avoiding an extra read of oi/li from memory.
 */
#include <cstdint>

// Inline exp approximation (no external library calls)
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

/**
 * Fused online update + normalize kernel
 *
 * @param args  Argument array:
 *   args[0]  = mij      (q_tile,)       current block row max
 *   args[1]  = lij      (q_tile,)       current block row sum
 *   args[2]  = oi_new   (q_tile, head_dim) current block PV output
 *   args[3]  = mi       (q_tile,)       accumulated max  (in/out)
 *   args[4]  = li       (q_tile,)       accumulated sum  (in/out)
 *   args[5]  = oi       (q_tile, head_dim) accumulated out (in/out)
 *   args[6]  = is_first 1 if first block, 0 otherwise
 *   args[7]  = is_last  1 if last block, 0 otherwise
 *   args[8]  = dst      output pointer (only written when is_last == 1)
 *   args[9]  = q_tile
 *   args[10] = head_dim
 */
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
    int q_tile    = static_cast<int>(args[9]);
    int head_dim  = static_cast<int>(args[10]);

    // ---- Online Softmax Update ----
    if (is_first) {
        for (int i = 0; i < q_tile; i++) {
            mi[i] = mij[i];
            li[i] = lij[i];
        }
        for (int i = 0; i < q_tile * head_dim; i++) {
            oi[i] = oi_new[i];
        }
    } else {
        for (int i = 0; i < q_tile; i++) {
            float mi_new = (mi[i] > mij[i]) ? mi[i] : mij[i];
            float alpha = my_exp(mi[i] - mi_new);
            float beta  = my_exp(mij[i] - mi_new);

            li[i] = alpha * li[i] + beta * lij[i];

            for (int j = 0; j < head_dim; j++) {
                oi[i * head_dim + j] = alpha * oi[i * head_dim + j]
                                     + beta  * oi_new[i * head_dim + j];
            }
            mi[i] = mi_new;
        }
    }

    // ---- Fused Normalize (only on last block) ----
    if (is_last) {
        for (int i = 0; i < q_tile; i++) {
            float inv_li = 1.0f / li[i];
            for (int j = 0; j < head_dim; j++) {
                dst[i * head_dim + j] = oi[i * head_dim + j] * inv_li;
            }
        }
    }
}
