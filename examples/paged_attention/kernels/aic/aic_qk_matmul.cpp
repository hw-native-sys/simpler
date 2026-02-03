/**
 * QK MatMul Kernel (AIC) - Multi-Head
 *
 * Computes: sij[h] = qi[h] @ kj[kv_h].T  for all heads h
 *
 * Memory layout:
 *   qi:  (num_heads, head_dim)  contiguous
 *   kj:  (block_size, kv_head_num, head_dim)  strided
 *   sij: (num_heads, block_size)  contiguous output
 */
#include <cstdint>

extern "C" void aic_qk_matmul(int64_t* args) {
    float* qi  = reinterpret_cast<float*>(args[0]);   // (num_heads, head_dim)
    float* kj  = reinterpret_cast<float*>(args[1]);   // (block_size, kv_head_num, head_dim)
    float* sij = reinterpret_cast<float*>(args[2]);   // (num_heads, block_size)
    int num_heads   = static_cast<int>(args[3]);
    int kv_head_num = static_cast<int>(args[4]);
    int block_size  = static_cast<int>(args[5]);
    int head_dim    = static_cast<int>(args[6]);

    int heads_per_kv = num_heads / kv_head_num;
    int kv_stride = kv_head_num * head_dim;  // stride between tokens in K

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_kv;
        float* qi_h = qi + h * head_dim;

        for (int j = 0; j < block_size; j++) {
            float* kj_token = kj + j * kv_stride + kv_h * head_dim;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += qi_h[d] * kj_token[d];
            }
            sij[h * block_size + j] = sum;
        }
    }
}
