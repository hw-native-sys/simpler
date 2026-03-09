// PV Matmul Kernel: pij(M, K) @ vj(K, N) -> oi_new(M, N)
//
// Fixed tile size: (16, 16) @ (16, 16) -> (16, 16)
//
// pij is float16 (converted from fp32 in softmax_prepare via TCVT).
// vj is stored as (K, N) = (block_size, head_dim) in row-major (ND) layout.
// Standard non-transposed B pattern: ND GlobalB + ColMajor/RowMajor TileMatB.

#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static __aicore__ void pv_matmul_impl(__gm__ uint8_t* pij_raw, __gm__ uint8_t* vj_raw, __gm__ uint8_t* oi_raw)
{
    constexpr int M = 16, K = 16, N = 16;

    __gm__ half* pij = reinterpret_cast<__gm__ half*>(pij_raw);
    __gm__ half* vj  = reinterpret_cast<__gm__ half*>(vj_raw);
    __gm__ float*      oi  = reinterpret_cast<__gm__ float*>(oi_raw);

    // pij (M, K) fp16, vj (K, N) fp16 in ND (row-major), oi_new (M, N) fp32
    using GlobalA   = GlobalTensor<half, Shape<1, 1, 1, M, K>, Stride<M*K, M*K, M*K, K, 1>>;
    using GlobalB   = GlobalTensor<half, Shape<1, 1, 1, K, N>, Stride<K*N, K*N, K*N, N, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M*N, M*N, M*N, N, 1>>;

    GlobalA   pijGlobal(pij);
    GlobalB   vjGlobal(vj);
    GlobalOut oiGlobal(oi);

    // L1 Mat tiles: standard ND pattern for both A and B
    using TileMatA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    // L0 tiles
    using LeftTile  = TileLeft<half, M, K, M, K>;
    using RightTile = TileRight<half, K, N, K, N>;
    using AccTile   = TileAcc<float, M, N, M, N>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile  aTile;
    RightTile bTile;
    AccTile   cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    // Load pij and vj to L1
    TLOAD(aMatTile, pijGlobal);
    TLOAD(bMatTile, vjGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    // Move to L0A/L0B
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // Single matmul: (M,K) x (K,N) -> (M,N)
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(oiGlobal, cTile);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    __gm__ uint8_t* pij    = reinterpret_cast<__gm__ uint8_t*>(args[0]);
    __gm__ uint8_t* vj     = reinterpret_cast<__gm__ uint8_t*>(args[1]);
    __gm__ uint8_t* oi_new = reinterpret_cast<__gm__ uint8_t*>(args[2]);

    pv_matmul_impl(pij, vj, oi_new);
}
