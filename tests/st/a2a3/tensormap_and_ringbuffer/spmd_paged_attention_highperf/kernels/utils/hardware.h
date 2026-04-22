#ifndef INCLUDE_HARDWARE_H
#define INCLUDE_HARDWARE_H

enum class ArchType { ASCEND_V220, ASCEND_V200, ASCEND_M200 };

enum class DataFormat { ND = 0, NZ, ZN, ZZ, NN, VECTOR };

template <ArchType ArchTag>
struct HardwareInfo {
    static uint32_t const l2BW = 5;
    static uint32_t const hbmBW = 1;
    static uint32_t const supportMix = 0;
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 128 * 1024;
    static uint32_t const l2Size = 192 * 1024 * 1024;
    static uint32_t const biasSize = 1024;
    static uint32_t const fixBufSize = 7 * 1024;
    static uint32_t const ubSize = 192 * 1024;
    static uint32_t const fractalSize = 512;
    static uint32_t const l1l0BlockSize = 32;
    static uint32_t const btBlockSize = 64;
    static uint32_t const fbBlockSize = 128;
};

#endif
