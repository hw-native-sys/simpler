/**
 * QK MatMul Kernel (AIC) - No external library calls
 * Computes: sij = qi @ kj^T
 */
#include <cstdint>

extern "C" void aic_qk_matmul(int64_t* args) {
    float* qi = reinterpret_cast<float*>(args[0]);
    float* kj = reinterpret_cast<float*>(args[1]);
    float* sij = reinterpret_cast<float*>(args[2]);
    int q_tile = static_cast<int>(args[3]);
    int block_size = static_cast<int>(args[4]);
    int head_dim = static_cast<int>(args[5]);

    for (int i = 0; i < q_tile; i++) {
        for (int j = 0; j < block_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                sum += qi[i * head_dim + k] * kj[j * head_dim + k];
            }
            sij[i * block_size + j] = sum;
        }
    }
}
