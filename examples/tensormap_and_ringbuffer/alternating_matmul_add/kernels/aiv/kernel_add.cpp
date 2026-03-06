/**
 * Element-wise Tensor Addition Kernel
 *
 * Implements: out[i] = src0[i] + src1[i]
 * Tile size: ROWS x COLS
 *
 * Args (TensorData*):
 *   args[0] = src0 (INPUT)  - ROWS x COLS
 *   args[1] = src1 (INPUT)  - ROWS x COLS
 *   args[2] = out (OUTPUT)  - ROWS x COLS
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

template <int ROWS, int COLS>
static __aicore__ void add_impl(
    __gm__ TensorData* src0_tensor,
    __gm__ TensorData* src1_tensor,
    __gm__ TensorData* out_tensor) {

    __gm__ float* src0 = reinterpret_cast<__gm__ float*>(src0_tensor->buffer.addr) + src0_tensor->start_offset;
    __gm__ float* src1 = reinterpret_cast<__gm__ float*>(src1_tensor->buffer.addr) + src1_tensor->start_offset;
    __gm__ float* out = reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;

    using DynShapeDim5 = Shape<1, 1, 1, ROWS, COLS>;
    using DynStridDim5 = Stride<1, 1, 1, COLS, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, ROWS, COLS, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(ROWS, COLS);
    TileData src1Tile(ROWS, COLS);
    TileData dstTile(ROWS, COLS);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* src0_tensor = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* src1_tensor = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* out_tensor = reinterpret_cast<__gm__ TensorData*>(args[2]);

    add_impl<64, 128>(src0_tensor, src1_tensor, out_tensor);
}
