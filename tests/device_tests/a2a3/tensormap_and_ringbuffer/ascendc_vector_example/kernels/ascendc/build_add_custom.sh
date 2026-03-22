#!/bin/bash
# Compile AddCustom AscendC kernel for PTO runtime integration
#
# Based on: https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom
#
# Prerequisites:
#   - CANN toolkit installed (ASCEND_HOME_PATH set)
#   - ccec compiler available
#
# PTO integration requirements (compared to framework-launched AscendC):
#   1. NO __global__ attribute — PTO calls kernels as subroutines from its
#      dispatch loop; __global__ generates an incompatible prologue/epilogue
#   2. NO GetBlockNum()/GetBlockIdx() partitioning — PTO dispatches each task
#      to a single core, so the kernel must process all data
#   3. Static tiling — tiling values hardcoded as constexpr (no runtime tiling)
#
# Usage:
#   bash build_add_custom.sh [--dtype float|half] [--output add_custom.o]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}"
CCEC="${ASCEND_HOME}/bin/ccec"
BUILD_DIR="${SCRIPT_DIR}/_build"

# Defaults
DTYPE="float"
OUTPUT="${SCRIPT_DIR}/add_custom.o"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dtype) DTYPE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$CCEC" ]; then
    echo "Error: ccec not found at $CCEC"
    echo "Set ASCEND_HOME_PATH to your CANN installation"
    exit 1
fi

mkdir -p "$BUILD_DIR"

# Generate PTO-compatible kernel source
# Key differences from the Gitee original:
#   - extern "C" __aicore__ (no __global__)
#   - blockLength = totalLength (no GetBlockNum() division)
#   - no GetBlockIdx() offset on GlobalTensor
cat > "$BUILD_DIR/add_custom.cpp" << 'SRC_EOF'
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// No __global__ — called as subroutine from PTO wrapper's kernel_entry
extern "C" __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr uint32_t totalLength = 16384;
    constexpr uint32_t tileNum = 8;
    KernelAdd op;
    op.Init(x, y, z, totalLength, tileNum);
    op.Process();
}
SRC_EOF

# AscendC include directories
ASC_ROOT="${ASCEND_HOME}/aarch64-linux/asc"
TIKCPP_ROOT="${ASCEND_HOME}/aarch64-linux/tikcpp"

echo "Compiling add_custom.cpp (dtype=$DTYPE)..."
"$CCEC" \
    -c -O3 -std=c++17 \
    --cce-aicore-lang \
    -DTILING_KEY_VAR=0 \
    --cce-aicore-only \
    --cce-aicore-arch=dav-c220-vec \
    --cce-auto-sync \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DDTYPE_X=$DTYPE -DDTYPE_Y=$DTYPE -DDTYPE_Z=$DTYPE \
    -I"${ASC_ROOT}/include" \
    -I"${ASC_ROOT}/include/basic_api" \
    -I"${ASC_ROOT}/include/adv_api" \
    -I"${ASC_ROOT}/include/c_api" \
    -I"${ASC_ROOT}/include/utils" \
    -I"${ASC_ROOT}/impl/adv_api" \
    -I"${ASC_ROOT}/impl/basic_api" \
    -I"${ASC_ROOT}/impl/c_api" \
    -I"${ASC_ROOT}/impl/micro_api" \
    -I"${ASC_ROOT}/impl/simt_api" \
    -I"${ASC_ROOT}/impl/utils" \
    -I"${TIKCPP_ROOT}/tikcfw" \
    -I"${TIKCPP_ROOT}/tikcfw/interface" \
    -I"${TIKCPP_ROOT}/tikcfw/impl" \
    -include "${ASCEND_HOME}/include/ascendc/asc_devkit_version.h" \
    -o "$OUTPUT" \
    "$BUILD_DIR/add_custom.cpp"

echo "Output: $OUTPUT ($(wc -c < "$OUTPUT") bytes)"
echo "Done."
