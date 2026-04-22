#ifndef INCLUDE_MEM_H
#define INCLUDE_MEM_H

#include "hardware.h"
#include "kernel_operator.h"

// LocalTensorView<T>: lightweight wrapper around a raw local-memory address.
// Works for UB (__ubuf__), L1 (__cbuf__), L0A (__ca__), L0B (__cb__), L0C (__cc__).
template <typename T>
struct LocalTensorView {
    uint64_t addr{0};
    __aicore__ LocalTensorView() = default;
    __aicore__ explicit LocalTensorView(uint64_t in) : addr(in) {}
    __aicore__ inline uint64_t GetPhyAddr() const { return addr; }
    __aicore__ inline LocalTensorView<T> operator[](uint64_t offset) const
    {
        return LocalTensorView<T>(addr + offset * sizeof(T));
    }
    template <typename U>
    __aicore__ inline LocalTensorView<U> ReinterpretCast() const
    {
        return LocalTensorView<U>(addr);
    }
    __aicore__ inline operator __ubuf__ T*() const { return (__ubuf__ T*)addr; }
};

template <typename T>
__aicore__ inline __ubuf__ T *Ub(const LocalTensorView<T> &v)
{
    return (__ubuf__ T *)v.GetPhyAddr();
}

// RawAddrTensorView<T>: lightweight wrapper around a GM pointer.
// Replaces AscendC::GlobalTensor<T>.
template <typename T>
struct RawAddrTensorView {
    __gm__ T* ptr{nullptr};
    __aicore__ RawAddrTensorView() = default;
    __aicore__ explicit RawAddrTensorView(__gm__ T* p) : ptr(p) {}
    __aicore__ inline void SetGlobalBuffer(__gm__ T* p) { ptr = p; }
    __aicore__ inline RawAddrTensorView<T> operator[](uint64_t offset) const
    {
        return RawAddrTensorView<T>(ptr + offset);
    }
    __aicore__ inline __gm__ T* GetPhyAddr() const { return ptr; }
    template <typename U>
    __aicore__ inline RawAddrTensorView<U> ReinterpretCast() const
    {
        return RawAddrTensorView<U>((__gm__ U*)ptr);
    }
};

enum class BufferType { ASCEND_UB, ASCEND_CB, ASCEND_L0A, ASCEND_L0B, ASCEND_L0C, ASCEND_MAX };

// AsdopsBuffer: provides GetBuffer<BufferType_, DstDataType>(offset) returning
// a LocalTensorView<DstDataType> pointing into the appropriate on-chip memory.
template <ArchType ArchTag>
struct AsdopsBuffer {
public:
    __aicore__ AsdopsBuffer() {};

    template <BufferType BufferType_, typename DstDataType = half>
    __aicore__ LocalTensorView<DstDataType> GetBuffer(const uint32_t offset) const
    {
        if constexpr (BufferType_ == BufferType::ASCEND_UB) {
            return LocalTensorView<DstDataType>(
                (uint64_t)reinterpret_cast<__ubuf__ uint8_t*>((uintptr_t)offset));
        } else if constexpr (BufferType_ == BufferType::ASCEND_CB) {
            return LocalTensorView<DstDataType>(
                (uint64_t)reinterpret_cast<__cbuf__ uint8_t*>((uintptr_t)offset));
        } else if constexpr (BufferType_ == BufferType::ASCEND_L0A) {
            return LocalTensorView<DstDataType>(
                (uint64_t)reinterpret_cast<__ca__ uint8_t*>((uintptr_t)offset));
        } else if constexpr (BufferType_ == BufferType::ASCEND_L0B) {
            return LocalTensorView<DstDataType>(
                (uint64_t)reinterpret_cast<__cb__ uint8_t*>((uintptr_t)offset));
        } else if constexpr (BufferType_ == BufferType::ASCEND_L0C) {
            return LocalTensorView<DstDataType>(
                (uint64_t)reinterpret_cast<__cc__ uint8_t*>((uintptr_t)offset));
        } else {
            return LocalTensorView<DstDataType>(0);
        }
    }
};

#endif
