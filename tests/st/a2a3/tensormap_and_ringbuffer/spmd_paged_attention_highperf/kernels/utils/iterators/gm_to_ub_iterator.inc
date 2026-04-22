#include "../iterator.h"

template <ArchType ArchTag, typename DType> struct gm_to_ub {
    __aicore__ inline gm_to_ub(LocalTensorView<DType> dstTensor, RawAddrTensorView<DType> srcTensor,
                               uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        copy_gm_to_ubuf((__ubuf__ DType*)dstTensor.GetPhyAddr(),
                        (__gm__ DType*)srcTensor.GetPhyAddr(),
                        0, nBurst, lenBurst, srcStride, dstStride);
    };
};

template <ArchType ArchTag, typename DType> struct gm_to_ub_align {
    __aicore__ inline gm_to_ub_align(LocalTensorView<DType> dstTensor, RawAddrTensorView<DType> srcTensor,
                                     uint8_t sid, uint16_t nBurst, uint32_t lenBurst, uint8_t leftPaddingNum,
                                     uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
    {
        if constexpr (sizeof(DType) == 1) {
            copy_gm_to_ubuf_align_b8((__ubuf__ DType*)dstTensor.GetPhyAddr(),
                                     (__gm__ DType*)srcTensor.GetPhyAddr(),
                                     0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
        } else if constexpr (sizeof(DType) == 2) {
            copy_gm_to_ubuf_align_b16((__ubuf__ DType*)dstTensor.GetPhyAddr(),
                                      (__gm__ DType*)srcTensor.GetPhyAddr(),
                                      0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
        } else {
            copy_gm_to_ubuf_align_b32((__ubuf__ DType*)dstTensor.GetPhyAddr(),
                                      (__gm__ DType*)srcTensor.GetPhyAddr(),
                                      0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
        }
    };
};

template <ArchType ArchTag, typename DType> struct ub_to_ub {
    __aicore__ inline ub_to_ub(LocalTensorView<DType> dstTensor, LocalTensorView<DType> srcTensor,
                               uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        copy_ubuf_to_ubuf((__ubuf__ DType*)dstTensor.GetPhyAddr(),
                          (__ubuf__ DType*)srcTensor.GetPhyAddr(),
                          0, nBurst, lenBurst, srcStride, dstStride);
    };
};

template <ArchType ArchTag, typename DataType, DataFormat InDataFormat = DataFormat::ND,
          DataFormat OutDataFormat = DataFormat::ND>
struct ub_to_gm {
    __aicore__ inline ub_to_gm(RawAddrTensorView<DataType> dstTensor, LocalTensorView<DataType> srcTensor,
                               uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        copy_ubuf_to_gm((__gm__ DataType*)dstTensor.GetPhyAddr(),
                        (__ubuf__ DataType*)srcTensor.GetPhyAddr(),
                        0, nBurst, lenBurst, srcStride, dstStride);
    };
};

template <ArchType ArchTag, typename DType> struct ub_to_gm_align {
    __aicore__ inline ub_to_gm_align(RawAddrTensorView<DType> dstTensor, LocalTensorView<DType> srcTensor,
                                     uint8_t sid, uint16_t nBurst, uint32_t lenBurst, uint8_t leftPaddingNum,
                                     uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
    {
        // Match AscendC::DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams) →
        // DataCopyPadUB2GMImpl → copy_ubuf_to_gm_align_b* (blockLen in bytes).
        if constexpr (sizeof(DType) == 1) {
            copy_ubuf_to_gm_align_b8((__gm__ DType*)dstTensor.GetPhyAddr(),
                                     (__ubuf__ DType*)srcTensor.GetPhyAddr(),
                                     sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
        } else if constexpr (sizeof(DType) == 2) {
            copy_ubuf_to_gm_align_b16((__gm__ DType*)dstTensor.GetPhyAddr(),
                                      (__ubuf__ DType*)srcTensor.GetPhyAddr(),
                                      sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
        } else {
            copy_ubuf_to_gm_align_b32((__gm__ DType*)dstTensor.GetPhyAddr(),
                                      (__ubuf__ DType*)srcTensor.GetPhyAddr(),
                                      sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
        }
    };
};
