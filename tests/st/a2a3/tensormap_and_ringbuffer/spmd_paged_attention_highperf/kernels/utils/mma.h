#ifndef INCLUDE_MMA_H
#define INCLUDE_MMA_H

#include "hardware.h"
#include "mem.h"
#include "kernel_operator.h"

// Primary template (no-op / unsupported configuration).
template <ArchType ArchTag, typename ElementA, typename ElementB, typename AccDTypeC, bool IsTransposeA = false>
struct mmad {
    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0) {}

    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0) {}
};

// Partial specialization for IsTransposeA = false.
template <ArchType ArchTag, typename AccDTypeC, typename ElementA, typename ElementB>
struct mmad<ArchTag, ElementA, ElementB, AccDTypeC, false> {

    // Constructor without bias.
    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0)
    {
        mad((__cc__ AccDTypeC*)l0cTensor.GetPhyAddr(),
            (__ca__ ElementA*)l0aTensor.GetPhyAddr(),
            (__cb__ ElementB*)l0bTensor.GetPhyAddr(),
            mTileActual,   // m
            kPartActual,   // k
            nTileActual,   // n
            unitFlag,
            false,         // kDirectionAlign
            false,         // cmatrixSource
            initC);        // cmatrixInitVal
    }

    // Constructor with bias address.
    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0)
    {
        mad((__cc__ AccDTypeC*)l0cTensor.GetPhyAddr(),
            (__ca__ ElementA*)l0aTensor.GetPhyAddr(),
            (__cb__ ElementB*)l0bTensor.GetPhyAddr(),
            biasBt,
            mTileActual,   // m
            kPartActual,   // k
            nTileActual,   // n
            unitFlag,
            false,         // kDirectionAlign
            true,          // cmatrixSource = true (bias present)
            false);        // cmatrixInitVal = false (bias path)
    }
};

#endif
