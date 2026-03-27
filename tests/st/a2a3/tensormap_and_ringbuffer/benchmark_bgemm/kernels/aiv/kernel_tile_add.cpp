/**
 * Tile-based Element-wise Addition Kernel (Vector Core) - INOUT Pattern
 *
 * Computes: C_tile = C_tile + P (tile_size x tile_size tile accumulation)
 * Uses TADD instruction
 *
 * Tile size is determined by golden.py configuration and passed through
 * tensor shapes from orchestration.
 *
 * Args (Tensor*):
 *   args[0] = C_tile (INOUT: read + write accumulator)
 *   args[1] = P      (INPUT: matmul result to accumulate)
 *   args[2] = config (INPUT) - int64_t[4]: [tile_size, grid_k, num_groups, incore_loop]
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
    __gm__ float* c_ptr,
    __gm__ float* p_ptr) {

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
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* c_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* p_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* config   = reinterpret_cast<__gm__ Tensor*>(args[2]);

    __gm__ int64_t* cfg = reinterpret_cast<__gm__ int64_t*>(config->buffer.addr);
    uint64_t tile_size = static_cast<uint64_t>(cfg[0]);
    uint64_t tile_elems = tile_size * tile_size;
    int num_tiles = static_cast<int>(cfg[3]);

    __gm__ float* base_c = reinterpret_cast<__gm__ float*>(c_tensor->buffer.addr) + c_tensor->start_offset;
    __gm__ float* base_p = reinterpret_cast<__gm__ float*>(p_tensor->buffer.addr) + p_tensor->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        __gm__ float* c_ptr = base_c + (tile_idx * tile_elems);
        __gm__ float* p_ptr = base_p + (tile_idx * tile_elems);

        switch (tile_size) {
            case 16:  tile_add_impl<16>(c_ptr, p_ptr);  break;
            case 32:  tile_add_impl<32>(c_ptr, p_ptr);  break;
            case 64:  tile_add_impl<64>(c_ptr, p_ptr);  break;
            case 128: tile_add_impl<128>(c_ptr, p_ptr); break;
            default: break;
        }
    }
}
