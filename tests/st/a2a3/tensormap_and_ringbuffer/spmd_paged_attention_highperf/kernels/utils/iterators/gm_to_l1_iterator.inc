#include "../iterator.h"

template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::ND> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ gm_to_l1(LocalTensorView<DataType> l1Tensor,
                        RawAddrTensorView<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        (void)nVal;
        (void)dTileCeil;
        copy_gm_to_cbuf((__cbuf__ DataType*)l1Tensor.GetPhyAddr(),
                        (__gm__ DataType*)gmTensor.GetPhyAddr(),
                        0,
                        1,
                        CeilDiv<BLOCK_SIZE>(nTileActual * dTileActual),
                        0,
                        0,
                        PAD_NONE);
    };
};

template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    __aicore__ gm_to_l1(LocalTensorView<DataType> l1Tensor,
                        RawAddrTensorView<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        (void)nVal;
        (void)dTileCeil;
        if (dVal < STRIDE_LIMIT) {
            if constexpr (sizeof(DataType) == 4) {
                copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ DataType*)l1Tensor.GetPhyAddr(),
                                                  (__gm__ DataType*)gmTensor.GetPhyAddr(),
                                                  0,
                                                  1,
                                                  nTileActual,
                                                  dTileActual,
                                                  0,
                                                  dVal,
                                                  nTileCeil,
                                                  1,
                                                  0);
            } else {
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ DataType*)l1Tensor.GetPhyAddr(),
                                                 (__gm__ DataType*)gmTensor.GetPhyAddr(),
                                                 0,
                                                 1,
                                                 nTileActual,
                                                 dTileActual,
                                                 0,
                                                 dVal,
                                                 nTileCeil,
                                                 1,
                                                 0);
            }
        } else {
            for (uint32_t i = 0; i < nTileActual; i++) {
                if constexpr (sizeof(DataType) == 4) {
                    copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ DataType*)l1Tensor[i * BLOCK_SIZE].GetPhyAddr(),
                                                      (__gm__ DataType*)gmTensor[i * dVal].GetPhyAddr(),
                                                      0,
                                                      1,
                                                      1,
                                                      dTileActual,
                                                      0,
                                                      0,
                                                      nTileCeil,
                                                      0,
                                                      0);
                } else {
                    copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ DataType*)l1Tensor[i * BLOCK_SIZE].GetPhyAddr(),
                                                     (__gm__ DataType*)gmTensor[i * dVal].GetPhyAddr(),
                                                     0,
                                                     1,
                                                     1,
                                                     dTileActual,
                                                     0,
                                                     0,
                                                     nTileCeil,
                                                     0,
                                                     0);
                }
            }
        }
    };
};
