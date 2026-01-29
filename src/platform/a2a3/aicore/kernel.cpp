/**
 * Minimal AICore Kernel
 */
#include <cstdint>

#include "aicore.h"

class Runtime;

#ifdef __AIV__
#define KERNEL_ENTRY(x) \
    x##_0_mix_aiv  // 动态生成函数名 KERNEL_ENTRY(my_kernel) ->
                   // my_kernel_0_mix_aiv
#define block_idx block_idx_aiv
#define core_type core_type_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#define block_idx block_idx_aic
#define core_type core_type_aic
#endif

[[block_local]] int block_idx;
[[block_local]] int core_type;

extern __aicore__ void aicore_execute(__gm__ Runtime* runtime, int block_idx, int core_type);

/**
 * Kernel entry point with control loop
 *
 * This function implements the AICore-side task execution protocol:
 * 1. Wait for AICPU ready signal (handshake initialization)
 * 2. Signal AICore is ready (aicore_done = core_id + 1)
 * 3. Enter polling loop:
 *    - Check control flag (1 = quit, 0 = continue)
 *    - If task pointer is non-zero, execute task and mark as complete
 *    - Use DCCI to ensure cache coherency with AICPU
 *
 * Each core (AIC or AIV) gets its own handshake buffer indexed by block_idx.
 *
 * @param runtime Address of Runtime structure in device memory
 */
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(__gm__ Runtime* runtime) {
    // Calculate block_idx for this core
#ifdef __AIV__
    block_idx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    core_type = 1;
#else
    block_idx = get_block_idx();
    core_type = 0;
#endif
    aicore_execute(runtime, block_idx, core_type);
}
