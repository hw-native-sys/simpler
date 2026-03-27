/**
 * Async Completion Demo - Simulated Producer Kernel (func_id=0)
 *
 * Implements: out[i] = in[i] * 2.0
 *
 * After storing the output, writes 1 to a GM completion flag, then registers
 * the completion via the CQ. The scheduler reads the CQ after this kernel
 * returns and polls the flag address.
 *
 * Kernel args layout (packed by scheduler):
 *   args[0] = &Tensor(in)              — input tensor struct pointer
 *   args[1] = &Tensor(out)             — output tensor struct pointer
 *   args[2] = event_flag_gm_addr       — GM flag addr (pre-allocated by golden.py)
 *   args[3] = cq_addr                  — completion queue (appended by submit_deferred)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "pto_cq_types.h"
#include "pto_cq_kernel_api.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* in_tensor  = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    uint64_t event_flag_addr  = static_cast<uint64_t>(args[2]);
    uint64_t cq_addr          = static_cast<uint64_t>(args[3]);

    __gm__ float* in_data  = reinterpret_cast<__gm__ float*>(in_tensor->buffer.addr)  + in_tensor->start_offset;
    __gm__ float* out_data = reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kTRows_ = 128;
    constexpr int kTCols_ = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;

    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData inTile(vRows, vCols);
    TileData outTile(vRows, vCols);
    TASSIGN(inTile, 0x0);
    TASSIGN(outTile, 0x10000);

    GlobalData inGlobal(in_data);
    GlobalData outGlobal(out_data);

    TLOAD(inTile, inGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // out = in + in = in * 2.0
    TADD(outTile, inTile, inTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, outTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);

    // Signal async completion: write non-zero flag to GM
    volatile __gm__ int32_t* flag = reinterpret_cast<volatile __gm__ int32_t*>(
        static_cast<uintptr_t>(event_flag_addr));
#if defined(SINGLE_CACHE_LINE) && defined(DSB_DDR)
    dcci((__gm__ int32_t*)flag, SINGLE_CACHE_LINE);
    *flag = 1;
    dcci((__gm__ int32_t*)flag, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    *flag = 1;
#endif

    volatile __gm__ PTO2CompletionQueue* cq = pto2_cq_get(cq_addr);
    pto2_cq_reset(cq);
    pto2_save_expected_completion(PTO2_ENGINE_SDMA, cq, event_flag_addr);
    pto2_cq_flush(cq);
}
