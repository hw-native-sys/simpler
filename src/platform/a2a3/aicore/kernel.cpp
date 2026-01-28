/**
* Minimal AICore Kernel
*/
#include <cstdint>
#include "aicore.h"

class Runtime;

#ifdef __AIV__
#define KERNEL_ENTRY(x) x##_0_mix_aiv   // 动态生成函数名 KERNEL_ENTRY(my_kernel) -> my_kernel_0_mix_aiv
#define blockIdx blockIdx_aiv
#define coreType coreType_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#define blockIdx blockIdx_aic
#define coreType coreType_aic
#endif

[[block_local]] int blockIdx;
[[block_local]] int coreType;

extern __aicore__ void AicoreExecute(__gm__ Runtime* runtime, int blockIdx, int coreType);

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
* Each core (AIC or AIV) gets its own handshake buffer indexed by blockIdx.
*
* @param runtime Address of Runtime structure in device memory
*/
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(__gm__ Runtime* runtime) {
    // Calculate blockIdx for this core
#ifdef __AIV__
    blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    coreType = 1;
#else
    blockIdx = get_block_idx();
    coreType = 0;
#endif
    AicoreExecute(runtime, blockIdx, coreType);
}
