#include "../iterator.h"

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float> {
    __aicore__ l0c_to_gm(RawAddrTensorView<float> gmTensor,
                         LocalTensorView<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
        set_nd_para((uint64_t)1);
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm((__gm__ float*)gmTensor.GetPhyAddr(),
                             (__cc__ float*)l0cTensor.GetPhyAddr(),
                             0, nTileActual, mTileActual, dstStride, srcStride, unitFlag,
                             QuantMode_t::NoQuant, 0, false, true);
    };
};
