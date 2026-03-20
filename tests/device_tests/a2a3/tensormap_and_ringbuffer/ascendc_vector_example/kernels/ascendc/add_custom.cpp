/**
 * AscendC AddCustom Operator
 *
 * Element-wise addition: z[i] = x[i] + y[i]
 *
 * Uses AscendC double-buffered pipeline with CopyIn/Compute/CopyOut stages.
 * Static tiling: totalLength=16384 elements, tileNum=8 tiles.
 *
 * Calling convention (extern "C"):
 *   void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z,
 *                   __gm__ uint8_t* workspace, __gm__ uint8_t* tiling);
 */

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

extern "C" __global__ __aicore__ void add_custom(
    __gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    KernelAdd op;
    op.Init(x, y, z, 16384, 8);
    op.Process();
}
