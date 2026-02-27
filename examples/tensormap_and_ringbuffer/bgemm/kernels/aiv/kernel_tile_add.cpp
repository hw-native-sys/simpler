/**
 * Tile-based Element-wise Addition Kernel (Vector Core) - INOUT Pattern
 *
 * Computes: C_tile = C_tile + P (64x64 tile accumulation)
 * Uses TADD instruction
 *
 * Args (TensorData*):
 *   args[0] = C_tile (INOUT: read + write accumulator)
 *   args[1] = P      (INPUT: matmul result to accumulate)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* c_tensor = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* p_tensor = reinterpret_cast<__gm__ TensorData*>(args[1]);

    __gm__ float* c_ptr = reinterpret_cast<__gm__ float*>(c_tensor->buffer.addr) + c_tensor->start_offset;
    __gm__ float* p_ptr = reinterpret_cast<__gm__ float*>(p_tensor->buffer.addr) + p_tensor->start_offset;

    constexpr int TILE = 64;

    using DynShapeDim5 = Shape<1, 1, 1, TILE, TILE>;
    using DynStridDim5 = Stride<1, 1, 1, TILE, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, TILE, TILE, BLayout::RowMajor, -1, -1>;

    TileData cTile(TILE, TILE);
    TileData pTile(TILE, TILE);
    TileData outTile(TILE, TILE);
    TASSIGN(cTile, 0x0);
    TASSIGN(pTile, 0x10000);
    TASSIGN(outTile, 0x20000);

    GlobalData cGlobal(c_ptr);
    GlobalData pGlobal(p_ptr);
    GlobalData outGlobal(c_ptr);  // write back to same C location

    TLOAD(cTile, cGlobal);
    TLOAD(pTile, pGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(outTile, cTile, pTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, outTile);
}
