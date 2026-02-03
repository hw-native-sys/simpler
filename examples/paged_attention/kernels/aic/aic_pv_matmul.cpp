/**
 * PV MatMul Kernel (AIC) - Multi-Head
 *
 * Computes: oi_new[h] = pij[h] @ vj[kv_h]  for all heads h
 *
 * Memory layout:
 *   pij:    (num_heads, block_size)  contiguous
 *   vj:     (block_size, kv_head_num, head_dim)  strided
 *   oi_new: (num_heads, head_dim)  contiguous output
 */
#include <cstdint>

extern "C" void aic_pv_matmul(int64_t* args) {
    float* pij    = reinterpret_cast<float*>(args[0]);  // (num_heads, block_size)
    float* vj     = reinterpret_cast<float*>(args[1]);  // (block_size, kv_head_num, head_dim)
    float* oi_new = reinterpret_cast<float*>(args[2]);  // (num_heads, head_dim)
    int num_heads   = static_cast<int>(args[3]);
    int kv_head_num = static_cast<int>(args[4]);
    int block_size  = static_cast<int>(args[5]);
    int head_dim    = static_cast<int>(args[6]);

    int heads_per_kv = num_heads / kv_head_num;
    int kv_stride = kv_head_num * head_dim;  // stride between tokens in V

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_kv;
        float* pij_h = pij + h * block_size;
        float* oi_new_h = oi_new + h * head_dim;

        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int k = 0; k < block_size; k++) {
                float* vj_token = vj + k * kv_stride + kv_h * head_dim;
                sum += pij_h[k] * vj_token[d];
            }
            oi_new_h[d] = sum;
        }
    }
}
