/**
 * Tile-based BGEMM Kernel — Combined Cube + Vector (TPUSH/TPOP)
 *
 * Computes one tile iteration: P = A[m,k] @ B[k,n], then C[m,n] += P
 *
 * Single source compiled twice:
 *   - AIC (Cube):   __DAV_CUBE__ defined → TLOAD, TMATMUL, TPUSH
 *   - AIV (Vector):  __DAV_VEC__ defined → TPOP, TADD, TSTORE
 *
 * Intermediate result P is transferred via VEC_FIFO (TPUSH/TPOP),
 * bypassing GM. The accumulator C is still read/written via GM.
 *
 * Simulation fallback (__CPU_SIM):
 *   Uses separate AIC/AIV tasks with GM intermediary (no TPUSH/TPOP).
 *   AIC args: [A, B, P_output]    AIV args: [C_inout, P_input]
 *
 * Hardware args (MixedKernels):
 *   args[0] = input_a  (INPUT)
 *   args[1] = input_b  (INPUT)
 *   args[2] = C_tile   (INOUT: read + write accumulator)
 */

 #include <cstdint>
 #include <pto/pto-inst.hpp>
 #ifndef __CPU_SIM
 #include <pto/common/fifo.hpp>
 #endif
 
 #include "tensor.h"
 
 using namespace pto;
 
 #ifndef __gm__
 #define __gm__
 #endif
 
 #ifndef __aicore__
 #define __aicore__ [aicore]
 #endif
 
 #ifdef __DAV_CUBE__
 constexpr bool DAV_CUBE = true;
 #else
 constexpr bool DAV_CUBE = false;
 #endif
 
 #ifdef __DAV_VEC__
 constexpr bool DAV_VEC = true;
 #else
 constexpr bool DAV_VEC = false;
 #endif
 
 // Tile dimensions (must match golden.py)
 constexpr int TILE = 64;
 constexpr int M = TILE;
 constexpr int K = TILE;
 constexpr int N = TILE;
 
 // =============================================================================
 // Simulation: separate AIC/AIV tasks with GM intermediate (no TPUSH/TPOP)
 // =============================================================================
 #ifdef __CPU_SIM
 
 extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
     // AIC path: args = [A (input), B (input), P (output)]
     if constexpr (DAV_CUBE) {
         __gm__ Tensor* input_a_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
         __gm__ Tensor* input_b_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
         __gm__ Tensor* output_tensor  = reinterpret_cast<__gm__ Tensor*>(args[2]);
 
         __gm__ float* input_a = reinterpret_cast<__gm__ float*>(input_a_tensor->buffer.addr) + input_a_tensor->start_offset;
         __gm__ float* input_b = reinterpret_cast<__gm__ float*>(input_b_tensor->buffer.addr) + input_b_tensor->start_offset;
         __gm__ float* output  = reinterpret_cast<__gm__ float*>(output_tensor->buffer.addr)  + output_tensor->start_offset;
 
         using GlobalDataA = GlobalTensor<float, Shape<1, 1, 1, M, K>,
             pto::Stride<M * K, M * K, M * K, K, 1>>;
         using GlobalDataB = GlobalTensor<float, Shape<1, 1, 1, K, N>,
             pto::Stride<K * N, K * N, K * N, N, 1>>;
         using GlobalDataC = GlobalTensor<float, Shape<1, 1, 1, M, N>,
             pto::Stride<M * N, M * N, M * N, N, 1>>;
 
         GlobalDataA src0Global(input_a);
         GlobalDataB src1Global(input_b);
         GlobalDataC dstGlobal(output);
 
         using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
         using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
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
 
         TLOAD(aMatTile, src0Global);
         TLOAD(bMatTile, src1Global);
 
         set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
         wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
 
         TMOV(aTile, aMatTile);
         TMOV(bTile, bMatTile);
 
         set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
         wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
 
         TMATMUL(cTile, aTile, bTile);
 
         set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
         wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
 
         TSTORE(dstGlobal, cTile);
 
         set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
         wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
     }
 
     // AIV path: args = [C (inout), P (input)]
     if constexpr (DAV_VEC) {
         __gm__ Tensor* c_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
         __gm__ Tensor* p_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
 
         __gm__ float* c_ptr = reinterpret_cast<__gm__ float*>(c_tensor->buffer.addr) + c_tensor->start_offset;
         __gm__ float* p_ptr = reinterpret_cast<__gm__ float*>(p_tensor->buffer.addr) + p_tensor->start_offset;
 
         using DynShapeDim5 = Shape<1, 1, 1, TILE, TILE>;
         using DynStridDim5 = pto::Stride<1, 1, 1, TILE, 1>;
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
 
         set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
         wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
     }
 }
 
 // =============================================================================
 // Hardware: MixedKernels with TPUSH/TPOP via VEC_FIFO
 // =============================================================================
 #else  // !__CPU_SIM
 
 #define VEC_CORES 2
 constexpr int VEC_M = M / VEC_CORES;  // each vector sub-core handles half the rows
 
 // TPUSH/TPOP pipe configuration
 constexpr uint16_t PP_FLAG_ID = 0;
 constexpr uint8_t PP_FIFO_DEPTH = 2;
 
 // Cube accumulator (full M×N tile in L0C)
 using AccTileT = TileAcc<float, M, N, M, N>;
 // Vector consumer tile (half tile: VEC_M×N in UB, split across 2 vector sub-cores)
 using VecFifoTileT = Tile<TileType::Vec, float, VEC_M, N, BLayout::RowMajor, VEC_M, N>;
 
 // Cube→Vector pipe via on-chip VEC_FIFO (bypasses global memory)
 using PipeT = TPipe<PP_FLAG_ID, Direction::DIR_C2V, sizeof(float) * VEC_M * N, PP_FIFO_DEPTH>;
 
 extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
     __gm__ Tensor* input_a_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
     __gm__ Tensor* input_b_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
     __gm__ Tensor* c_tensor       = reinterpret_cast<__gm__ Tensor*>(args[2]);
 
     // Pipe and FIFO tile are declared in common scope (both sides reference the type)
     VecFifoTileT vecFifoTile;
     PipeT mPipe((__gm__ void *)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x0);
 
     // =========================================================================
     // Cube side: TLOAD A,B → TMATMUL → TPUSH result to vector via VEC_FIFO
     // =========================================================================
     if constexpr (DAV_CUBE) {
         __gm__ float* input_a = reinterpret_cast<__gm__ float*>(input_a_tensor->buffer.addr)
                                 + input_a_tensor->start_offset;
         __gm__ float* input_b = reinterpret_cast<__gm__ float*>(input_b_tensor->buffer.addr)
                                 + input_b_tensor->start_offset;
 
         using GlobalDataA = GlobalTensor<float, Shape<1, 1, 1, M, K>,
             pto::Stride<M * K, M * K, M * K, K, 1>>;
         using GlobalDataB = GlobalTensor<float, Shape<1, 1, 1, K, N>,
             pto::Stride<K * N, K * N, K * N, N, 1>>;
 
         GlobalDataA src0Global(input_a);
         GlobalDataB src1Global(input_b);
 
         using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
         using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
         using LeftTile = TileLeft<float, M, K, M, K>;
         using RightTile = TileRight<float, K, N, K, N>;
 
         TileMatA aMatTile;
         TileMatB bMatTile;
         TASSIGN(aMatTile, 0x0);
         TASSIGN(bMatTile, 0x20000);
 
         LeftTile aTile;
         RightTile bTile;
         AccTileT accTile;
         TASSIGN(aTile, 0x0);
         TASSIGN(bTile, 0x0);
         TASSIGN(accTile, 0x0);
 
         // Load A and B from GM to L1
         TLOAD(aMatTile, src0Global);
         TLOAD(bMatTile, src1Global);
 
         set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
         wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
 
         // Move from L1 to L0A/L0B
         TMOV(aTile, aMatTile);
         TMOV(bTile, bMatTile);
 
         set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
         wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
 
         // Matrix multiply
         TMATMUL(accTile, aTile, bTile);
 
         set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
         wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
 
         // Push result directly to vector core's UB (replaces TSTORE to GM)
         TPUSH<PipeT, AccTileT, TileSplitAxis::TILE_UP_DOWN>(mPipe, accTile);
     }
 
     // =========================================================================
     // Vector side: TPOP result from cube → TLOAD C from GM → TADD → TSTORE
     // =========================================================================
     if constexpr (DAV_VEC) {
         uint32_t subBlockIdx = get_subblockid();
 
         __gm__ float* c_ptr = reinterpret_cast<__gm__ float*>(c_tensor->buffer.addr)
                               + c_tensor->start_offset;
         // Each vector sub-core handles its half: sub-core 0 → rows [0, VEC_M),
         //                                       sub-core 1 → rows [VEC_M, M)
         __gm__ float* c_sub = c_ptr + static_cast<size_t>(subBlockIdx) * VEC_M * N;
 
         using GlobalC = GlobalTensor<float, Shape<1, 1, 1, VEC_M, N>,
             pto::Stride<VEC_M * N, VEC_M * N, VEC_M * N, N, 1>>;
 
         GlobalC cGlobal(c_sub);
         GlobalC outGlobal(c_sub);  // write back to same location
 
         using VecTile = Tile<TileType::Vec, float, VEC_M, N, BLayout::RowMajor, VEC_M, N>;
 
         VecTile cTile;
         VecTile outTile;
         // Place after FIFO buffer: FIFO uses [0x0, FIFO_DEPTH * VEC_M * N * 4)
         // = [0x0, 2 * 32 * 64 * 4) = [0x0, 0x4000)
         TASSIGN(cTile, 0x4000);
         TASSIGN(outTile, 0x6000);
 
         // Pop matmul result from cube via VEC_FIFO (replaces TLOAD from GM)
         TPOP<PipeT, VecFifoTileT, TileSplitAxis::TILE_UP_DOWN>(mPipe, vecFifoTile);
 
         // Load current C tile from GM
         TLOAD(cTile, cGlobal);
 
         set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
         wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
 
         // Accumulate: C += P
         TADD(outTile, cTile, vecFifoTile);
         TFREE<PipeT, TileSplitAxis::TILE_UP_DOWN>(mPipe);
 
         set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
         wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
 
         // Store result back to GM
         TSTORE(outGlobal, outTile);
     }
 }
 
 #endif  // __CPU_SIM
 