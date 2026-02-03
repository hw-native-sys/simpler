/**
 * Matrix Multiplication Kernel (AIC)
 *
 * Implements: out = src0 @ src1 (matrix multiplication)
 *
 * This kernel performs matrix multiplication on AIC (AI Cube) core.
 * Uses half precision input and float output for compatibility with both sim and NPU.
 * Simplified flow: TLOAD -> TMOV -> TMATMUL -> TSTORE
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// Matrix dimensions
constexpr int validM = 128;
constexpr int validK = 128;
constexpr int validN = 128;

// Aligned dimensions (align to 16 for half type)
constexpr int blockAlign = 16;
constexpr int M = 128;
constexpr int K = 128;
constexpr int N = 128;

/**
 * Matrix multiplication kernel implementation
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src0 pointer (left matrix, MxK, half)
 *              args[1] = src1 pointer (right matrix, KxN, half)
 *              args[2] = out pointer (output matrix, MxN, float)
 *              args[3] = size (number of elements, unused for matmul)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    // Unpack arguments - half input, float output
    __gm__ half* src0 = reinterpret_cast<__gm__ half*>(args[0]);
    __gm__ half* src1 = reinterpret_cast<__gm__ half*>(args[1]);
    __gm__ float* out = reinterpret_cast<__gm__ float*>(args[2]);

    // Global tensor types
    using GlobalDataSrc0 = GlobalTensor<half, Shape<1, 1, 1, validM, validK>,
        Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, Shape<1, 1, 1, validK, validN>,
        Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<float, Shape<1, 1, 1, validM, validN>,
        Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    // L1 buffer tiles for loading data (half precision)
    using TileMatAData = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    // Cube tiles for matmul - half * half -> float
    using LeftTile = TileLeft<half, M, K, validM, validK>;
    using RightTile = TileRight<half, K, N, validK, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    // TLOAD: Load from GM to L1
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    // TMOV: Move from L1 to Cube
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // TMATMUL: Matrix multiplication (half * half -> float)
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    // TSTORE: Store result to GM
    TSTORE(dstGlobal, cTile);
}
