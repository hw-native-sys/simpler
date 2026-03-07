/**
 * Tile-based Element-wise Addition Kernel (Vector Core) - INOUT Pattern
 *
 * Computes: C_tile = C_tile + P (tile_size x tile_size tile accumulation)
 * Uses TADD instruction
 *
 * Tile size is determined by golden.py configuration and passed through
 * tensor shapes from orchestration.
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

template <int TILE>
static __aicore__ void tile_add_impl(
    __gm__ TensorData* c_tensor,
    __gm__ TensorData* p_tensor) {

    __gm__ float* c_ptr = reinterpret_cast<__gm__ float*>(c_tensor->buffer.addr) + c_tensor->start_offset;
    __gm__ float* p_ptr = reinterpret_cast<__gm__ float*>(p_tensor->buffer.addr) + p_tensor->start_offset;

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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* c_tensor = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* p_tensor = reinterpret_cast<__gm__ TensorData*>(args[1]);
    uint64_t total_elems = static_cast<uint64_t>(p_tensor->shapes[0]);
    uint64_t tile_size = 1;
    while (tile_size * tile_size < total_elems) {
        tile_size <<= 1;
    }
    switch (tile_size) {
        case 16:  tile_add_impl<16>(c_tensor, p_tensor);  break;
        case 32:  tile_add_impl<32>(c_tensor, p_tensor);  break;
        case 64:  tile_add_impl<64>(c_tensor, p_tensor);  break;
        case 128: tile_add_impl<128>(c_tensor, p_tensor); break;
        default: break;
    }
}
