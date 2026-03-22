/**
 * AscendC AddCustom Operator — Reference Source
 *
 * This file shows the AscendC operator source that would be compiled
 * externally using the AscendC toolchain (CMake + npu_op_kernel_options).
 * It is provided for documentation purposes — the actual compilation is
 * done outside of simpler using the AscendC build system.
 *
 * AscendC compilation produces:
 *   1. A compiled .o file (the AICore kernel binary)
 *   2. A tiling data blob (for static tiling, baked into the kernel)
 *
 * These artifacts are then consumed by simpler's AscendCCompiler, which
 * wraps them with a PTO-compatible kernel_entry and registers the
 * combined binary with the PTO runtime.
 *
 * To build this operator:
 *   1. Place it in a tikcpp_smoke-style project structure
 *   2. Configure CMakePresets.json for target SoC (e.g. Ascend910B)
 *   3. Add to CMakeLists.txt:
 *        npu_op_kernel_options(ascendc_kernels ALL OPTIONS --save-temp-files)
 *   4. Build with: bash run.sh --is-dynamic=0
 *   5. Find artifacts in: build_out/op_kernel/AddCustom_<soc>/kernel_<x>/kernel_meta/
 *   6. Copy .o to this directory as add_custom.o
 *
 * The kernel entry symbol exported by AscendC compilation is typically
 * the operator class name in snake_case — here "add_custom".
 *
 * Calling convention (after static tiling degeneration):
 *   void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z,
 *                   __gm__ uint8_t* workspace, __gm__ uint8_t* tiling);
 *
 * With static tiling, the tiling parameter points to compile-time-fixed data
 * that is embedded in the wrapper kernel by AscendCCompiler.
 */

// ============================================================================
// The following is a simplified pseudo-code representation of what AscendC
// generates.  It is NOT directly compilable with PTO's ccec — it requires the
// full AscendC header set (kernel_operator.h, etc.) from the CANN SDK.
// ============================================================================

#if 0  // Not compiled — reference only

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y,
                                __gm__ uint8_t* z, uint32_t totalLength,
                                uint32_t tileNum) {
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = totalLength / tileNum;

        xGm.SetGlobalBuffer((__gm__ float*)x, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// Generated entry point (after static tiling degeneration)
extern "C" __global__ __aicore__ void add_custom(
    __gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    // Static tiling: totalLength and tileNum are fixed at compile time
    KernelAdd op;
    op.Init(x, y, z, /*totalLength=*/16384, /*tileNum=*/8);
    op.Process();
}

#endif
