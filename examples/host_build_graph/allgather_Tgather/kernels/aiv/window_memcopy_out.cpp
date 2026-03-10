/**
 * WindowMemCopyOut: Copy HCCL window to device buffer.
 * After AllGather, every rank copies gathered result to device.
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
    __gm__ float* dev_dst = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* win_src = reinterpret_cast<__gm__ float*>(args[1]);
    int count = static_cast<int>(args[2]);

    for (int i = 0; i < count; ++i) {
        dev_dst[i] = win_src[i];
    }
    pipe_barrier(PIPE_ALL);
}
