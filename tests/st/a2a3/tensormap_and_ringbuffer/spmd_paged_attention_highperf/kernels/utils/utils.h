#ifndef INCLUDE_UTILS_H
#define INCLUDE_UTILS_H

#include "kernel_operator.h"

// Reinterpret the bit pattern of scalarValue as uint64_t.
template <typename T>
constexpr __aicore__ inline uint64_t GetScalarBitcodeValue(T scalarValue)
{
    union ScalarBitcode {
        __aicore__ ScalarBitcode() {}
        T input;
        uint64_t output;
    } data;
    data.input = scalarValue;
    return data.output;
}

// SetFftsBaseAddr: set FFTS base address register.
// Replaces AscendC::SetSyncBaseAddr.
__aicore__ inline void SetFftsBaseAddr(uint64_t config)
{
    set_ffts_base_addr(config);
}

// SetPadding: set load-data padding value register.
// Replaces AscendC::SetLoadDataPaddingValue<T>.
template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue)
{
    uint64_t paddingValue = 0;
    if constexpr (sizeof(IN_DTYPE) == 2 || sizeof(IN_DTYPE) == 4 || sizeof(IN_DTYPE) == 8) {
        paddingValue = static_cast<uint64_t>(GetScalarBitcodeValue(padValue));
    } else {
        // 1-byte types: replicate into both bytes of a 16-bit word
        paddingValue = ((static_cast<uint64_t>(padValue)) << 8) | (static_cast<uint64_t>(padValue) & 0xFF);
    }
    set_padding(paddingValue);
}

// SetAtomicnone: disable atomic operations.
// Replaces AscendC::SetAtomicNone.
__aicore__ inline void SetAtomicnone()
{
    set_atomic_none();
}

// SetMasknorm: set vector mask mode to normal (bit-mask).
// Replaces AscendC::SetMaskNorm.
__aicore__ inline void SetMasknorm()
{
#if __CCE_AICORE__ == 100
    return;
#endif
    set_mask_norm();
}

// SetNdpara: configure NZ-to-ND fixpipe parameters.
// Replaces AscendC::SetFixpipeNz2ndFlag.
__aicore__ inline void SetNdpara(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    uint64_t ndPara = static_cast<uint64_t>(ndNum) |
                      (static_cast<uint64_t>(srcNdStride) << 16) |
                      (static_cast<uint64_t>(dstNdStride) << 32);
    set_nd_para(ndPara);
}

// SetVectorMask: set vector mask registers (high and low 64-bit words).
// Replaces AscendC::SetVectorMask<T>.
template <typename IN_DTYPE>
__aicore__ inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
{
    set_vector_mask(maskHigh, maskLow);
}

// GetSubBlockidx: return the sub-block (sub-core) index within a block.
// Replaces AscendC::GetSubBlockIdx.
__aicore__ inline int64_t GetSubBlockidx()
{
    return get_subblockid();
}

// WaitFlagDev: wait on a cross-core device-side flag.
// Replaces AscendC::WaitEvent.
__aicore__ inline void WaitFlagDev(uint16_t flagId)
{
    wait_flag_dev(flagId);
}

// FftsCrossCoreSync: issue a cross-core FFTS synchronization.
// Replaces AscendC::CrossCoreSetFlag<mode, pipe>.
template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    uint64_t config = 1ULL | (static_cast<uint64_t>(mode) << 4) | (static_cast<uint64_t>(flagId) << 8);
    ffts_cross_core_sync(pipe, config);
}

#endif
