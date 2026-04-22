#include "../iterator.h"

template <ArchType ArchTag, typename DataType, bool IsTransPose>
struct l1_to_l0_a<ArchTag, DataType, IsTransPose, DataFormat::VECTOR, DataFormat::VECTOR> {
    __aicore__ l1_to_l0_a(LocalTensorView<DataType> l0Tensor,
                          LocalTensorView<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        (void)mTileCeil;
        (void)mSrcStride;
        (void)mDstStride;
        if constexpr (IsTransPose) {
            load_cbuf_to_ca((__ca__ DataType*)l0Tensor.GetPhyAddr(),
                            (__cbuf__ DataType*)l1Tensor.GetPhyAddr(),
                            0, kPartCeil, kSrcStride, kDstStride, 0, 1, 0);
        } else {
            load_cbuf_to_ca((__ca__ DataType*)l0Tensor.GetPhyAddr(),
                            (__cbuf__ DataType*)l1Tensor.GetPhyAddr(),
                            0, kPartCeil, kSrcStride, kDstStride, 0, 0, 0);
        }
    };
};

template <ArchType ArchTag, typename DataType, bool IsTransPose>
struct l1_to_l0_b<ArchTag, DataType, IsTransPose, DataFormat::VECTOR, DataFormat::VECTOR> {
    __aicore__ l1_to_l0_b(LocalTensorView<DataType> l0Tensor,
                          LocalTensorView<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        (void)nTileCeil;
        (void)nSrcStride;
        (void)nDstStride;
        if constexpr (IsTransPose) {
            load_cbuf_to_cb((__cb__ DataType*)l0Tensor.GetPhyAddr(),
                            (__cbuf__ DataType*)l1Tensor.GetPhyAddr(),
                            0, kPartCeil, kSrcStride, kDstStride, 0, 1, 0);
        } else {
            load_cbuf_to_cb((__cb__ DataType*)l0Tensor.GetPhyAddr(),
                            (__cbuf__ DataType*)l1Tensor.GetPhyAddr(),
                            0, kPartCeil, kSrcStride, kDstStride, 0, 0, 0);
        }
    };
};
