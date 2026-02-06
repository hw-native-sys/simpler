// Kernel Function: pv_matmul
// Fixed for PTO-ISA matmul layout requirements on a2a3 hardware

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

constexpr int M = 16;
constexpr int K = 16;
constexpr int N = 16;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    __gm__ float* pij = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* vj = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* oi_new = reinterpret_cast<__gm__ float*>(args[2]);

    // GlobalTensor definitions
    using GlobalDataA = GlobalTensor<float, Shape<1, 1, 1, M, K>, Stride<M*K, M*K, M*K, K, 1>>;
    using GlobalDataB = GlobalTensor<float, Shape<1, 1, 1, K, N>, Stride<K*N, K*N, K*N, N, 1>>;
    using GlobalDataOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M*N, M*N, M*N, N, 1>>;
    
    GlobalDataA pijGlobal(pij);
    GlobalDataB vjGlobal(vj);
    GlobalDataOut oiGlobal(oi_new);

    // L1 tiles: ColMajor + SLayout::RowMajor (required for matmul)
    using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    // L0 tiles: Use the standard TileLeft/TileRight/TileAcc aliases
    using LeftTile = TileLeft<float, M, K, M, K>;
    using RightTile = TileRight<float, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    // Load A and B to L1
    TLOAD(aMatTile, pijGlobal);
    TLOAD(bMatTile, vjGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    // Move from L1 to L0A/L0B
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // Matmul
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    // Store result
    TSTORE(oiGlobal, cTile);
}
