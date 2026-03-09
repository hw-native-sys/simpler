/**
 * WindowMemCopyIn: Copy device buffer to HCCL window.
 * Used before TGATHER so remote ranks can read.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* win_dst = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* dev_src = reinterpret_cast<__gm__ float*>(args[1]);
    int count = static_cast<int>(args[2]);

    for (int i = 0; i < count; ++i) {
        win_dst[i] = dev_src[i];
    }
    pipe_barrier(PIPE_ALL);
}
