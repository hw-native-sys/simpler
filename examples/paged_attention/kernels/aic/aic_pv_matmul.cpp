/**
 * PV MatMul Kernel (AIC) - No external library calls
 * Computes: oi_new = pij @ vj
 */
#include <cstdint>

extern "C" void aic_pv_matmul(int64_t* args) {
    float* pij = reinterpret_cast<float*>(args[0]);
    float* vj = reinterpret_cast<float*>(args[1]);
    float* oi_new = reinterpret_cast<float*>(args[2]);
    int q_tile = static_cast<int>(args[3]);
    int block_size = static_cast<int>(args[4]);
    int head_dim = static_cast<int>(args[5]);

    for (int i = 0; i < q_tile; i++) {
        for (int j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < block_size; k++) {
                sum += pij[i * block_size + k] * vj[k * head_dim + j];
            }
            oi_new[i * head_dim + j] = sum;
        }
    }
}
