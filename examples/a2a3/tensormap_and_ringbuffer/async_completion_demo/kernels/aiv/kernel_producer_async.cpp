/**
 * Async Completion Demo - Hardware SDMA Producer Kernel (func_id=2)
 *
 * Implements: out[i] = in[i] * 2.0 via TLOAD/TADD/TSTORE, then issues
 * TPUT_ASYNC to exercise the SDMA path and writes the resulting AsyncEvent
 * handle to a GM buffer for the scheduler to poll.
 *
 * This kernel is only compiled for real hardware (a2a3), not for simulation.
 *
 * Kernel args layout (packed by scheduler):
 *   args[0] = &Tensor(in)            — input tensor struct pointer
 *   args[1] = &Tensor(out)           — output tensor struct pointer
 *   args[2] = sdma_workspace_addr    — SDMA workspace (from SdmaWorkspaceManager)
 *   args[3] = event_handle_output    — GM buffer where we write AsyncEvent.handle
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* in_tensor  = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    uint64_t sdma_workspace   = static_cast<uint64_t>(args[2]);
    uint64_t event_handle_out = static_cast<uint64_t>(args[3]);

    __gm__ float* in_data  = reinterpret_cast<__gm__ float*>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float* out_data = reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kTRows = 128;
    constexpr int kTCols = 128;
    constexpr int kTotalElems = kTRows * kTCols;

    using DynShapeDim5 = Shape<1, 1, 1, kTRows, kTCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, kTRows, kTCols, BLayout::RowMajor, -1, -1>;

    // TPUT_ASYNC requires isLogical1D (shape0..3 == 1) and contiguous strides
    using FlatShape = Shape<1, 1, 1, 1, kTotalElems>;
    using FlatStride = Stride<kTotalElems, kTotalElems, kTotalElems, kTotalElems, 1>;
    using FlatGlobalData = GlobalTensor<float, FlatShape, FlatStride>;

    TileData inTile(kTRows, kTCols);
    TileData outTile(kTRows, kTCols);
    TASSIGN(inTile, 0x0);
    TASSIGN(outTile, 0x10000);

    GlobalData inGlobal(in_data);
    GlobalData outGlobal(out_data);
    FlatGlobalData outGlobalFlat(out_data);

    // Compute out = in + in = in * 2.0
    TLOAD(inTile, inGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TADD(outTile, inTile, inTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, outTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);

    // Build SDMA async session
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x20000);

    __gm__ uint8_t* workspace = reinterpret_cast<__gm__ uint8_t*>(static_cast<uintptr_t>(sdma_workspace));
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, workspace, session)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    // Issue TPUT_ASYNC: self-copy out→out to exercise SDMA path.
    // Must use flat 1D GlobalTensor — TPUT_ASYNC requires isLogical1D (shape0..3 == 1).
    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC(outGlobalFlat, outGlobalFlat, session);

    // Write event handle at offset 0 (scheduler reads this as uint64_t).
    // Uses dcci+dsb pattern (same as TNotify) so the scalar write is visible to AICPU.
    volatile __gm__ uint64_t* handle_out = reinterpret_cast<volatile __gm__ uint64_t*>(
        static_cast<uintptr_t>(event_handle_out));
    dcci((__gm__ int32_t*)handle_out, SINGLE_CACHE_LINE);
    *handle_out = event.handle;
    dcci((__gm__ int32_t*)handle_out, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}
