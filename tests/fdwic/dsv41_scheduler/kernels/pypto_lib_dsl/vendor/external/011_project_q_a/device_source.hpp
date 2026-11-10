// CANNBOTDSL_HOST_ABI_VERSION=9
#include "c_api/asc_simd.h"
#include "acl/acl_rt.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace cannbotdsl_92 {

__aicore__ inline int8_t cannbotdsl_load_dev(__gm__ int8_t* addr)
{
    return ld_dev(reinterpret_cast<__gm__ uint8_t*>(addr), 0);
}

__aicore__ inline uint8_t cannbotdsl_load_dev(__gm__ uint8_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int16_t cannbotdsl_load_dev(__gm__ int16_t* addr)
{
    return ld_dev(reinterpret_cast<__gm__ uint16_t*>(addr), 0);
}

__aicore__ inline uint16_t cannbotdsl_load_dev(__gm__ uint16_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int32_t cannbotdsl_load_dev(__gm__ int32_t* addr)
{
    return ld_dev(reinterpret_cast<__gm__ uint32_t*>(addr), 0);
}

__aicore__ inline uint32_t cannbotdsl_load_dev(__gm__ uint32_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int64_t cannbotdsl_load_dev(__gm__ int64_t* addr)
{
    return static_cast<int64_t>(ld_dev(reinterpret_cast<__gm__ uint64_t*>(addr), 0));
}

__aicore__ inline uint64_t cannbotdsl_load_dev(__gm__ uint64_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int64_t cannbotdsl_get_squeeze_status()
{
    return get_ar();
}

__aicore__ inline bool cannbotdsl_is_aic()
{
    if ASC_IS_AIC { return true; }
    return false;
}

__aicore__ inline bool cannbotdsl_is_aiv()
{
    if ASC_IS_AIV { return true; }
    return false;
}

__aicore__ inline void cannbotdsl_set_atomic_add_int32()
{
    set_atomic_add();
    set_atomic_s32();
}

__aicore__ inline void cannbotdsl_set_l0c_copy_nz_para(
    uint16_t matrix_num, uint16_t src_nz_matrix_stride, uint32_t dst_matrix_stride)
{
    if ASC_IS_AIC {
        uint64_t config = (static_cast<uint64_t>(dst_matrix_stride) << 32) |
                          (static_cast<uint64_t>(src_nz_matrix_stride) << 16) | matrix_num;
        set_loop3_para(config);
    }
}

__aicore__ inline void cannbotdsl_set_l0c_copy_channel_para(uint16_t src_nz_fractal_stride)
{
    if ASC_IS_AIC {
        set_channel_para(static_cast<uint64_t>(src_nz_fractal_stride) << 48);
    }
}

__aicore__ inline void cannbotdsl_set_l12l0a_3d_padding(uint64_t config)
{
    if ASC_IS_AIC {
        set_padding(config);
    }
}

#define CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(BXX)                                                        \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(uint32_t offset)                    \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset); }                                                   \
        return decltype(vag_##BXX(offset)){};                                                         \
    }                                                                                                 \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(                                    \
        uint32_t offset0, uint32_t offset1)                                                           \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset0, offset1); }                                         \
        return decltype(vag_##BXX(offset0, offset1)){};                                               \
    }                                                                                                 \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(                                    \
        uint32_t offset0, uint32_t offset1, uint32_t offset2)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset0, offset1, offset2); }                                \
        return decltype(vag_##BXX(offset0, offset1, offset2)){};                                      \
    }                                                                                                 \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(                                    \
        uint32_t offset0, uint32_t offset1, uint32_t offset2, uint32_t offset3)                       \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset0, offset1, offset2, offset3); }                       \
        return decltype(vag_##BXX(offset0, offset1, offset2, offset3)){};                             \
    }

CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(b8)
CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(b16)
CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(b32)
#undef CANNBOTDSL_DEFINE_UPDATE_ADDR_REG

#define CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(VECTOR_TYPE)                                               \
    __simd_callee__ inline void cannbotdsl_duplicate_highest(                                         \
        VECTOR_TYPE& dst, VECTOR_TYPE src, vector_bool mask)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, src, mask, POS_HIGHEST, MODE_ZEROING); }                            \
    }                                                                                                 \
    __simd_callee__ inline void cannbotdsl_duplicate_merge(                                           \
        VECTOR_TYPE& dst, VECTOR_TYPE src, vector_bool mask)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, src, mask, POS_LOWEST, MODE_MERGING); }                             \
    }                                                                                                 \
    __simd_callee__ inline void cannbotdsl_duplicate_highest_merge(                                   \
        VECTOR_TYPE& dst, VECTOR_TYPE src, vector_bool mask)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, src, mask, POS_HIGHEST, MODE_MERGING); }                            \
    }

CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_uint8_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_int8_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_fp8_e4m3fn_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_fp8_e5m2_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_fp8_e8m0_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_uint16_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_int16_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_half)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_bfloat16_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_uint32_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_int32_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_float)
#undef CANNBOTDSL_DEFINE_DUPLICATE_VECTOR

#define CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(VECTOR_TYPE, SCALAR_TYPE)                                  \
    __simd_callee__ inline void cannbotdsl_duplicate_scalar_merge(                                    \
        VECTOR_TYPE& dst, SCALAR_TYPE value, vector_bool mask)                                       \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, value, mask, MODE_MERGING); }                                       \
    }

CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_uint8_t, uint8_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_int8_t, int8_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_fp8_e4m3fn_t, fp8_e4m3fn_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_fp8_e5m2_t, fp8_e5m2_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_fp8_e8m0_t, fp8_e8m0_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_uint16_t, uint16_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_int16_t, int16_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_half, half)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_bfloat16_t, bfloat16_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_uint32_t, uint32_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_int32_t, int32_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_float, float)
#undef CANNBOTDSL_DEFINE_DUPLICATE_SCALAR

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint8_t& dst, vector_uint16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint8_t& dst, vector_int16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint16_t& dst, vector_uint32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint16_t& dst, vector_int32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_bool& dst, vector_bool src)
{
    if ASC_IS_AIV { ppack(dst, src, LOWER); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint8_t& dst, vector_uint16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint8_t& dst, vector_int16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint16_t& dst, vector_uint32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint16_t& dst, vector_int32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_bool& dst, vector_bool src)
{
    if ASC_IS_AIV { ppack(dst, src, HIGHER); }
}

// Stable bfloat16 multiply-add. CANN 9.1 and 9.2 declare asc_madd only for
// half/float while newer SDKs add the bfloat16 overload. All versions lower
// to the same dav-3510 vmadd primitive, so generated kernels call this
// compatibility layer instead of the version-specific public ABI.
__simd_callee__ inline void cannbotdsl_madd_bfloat16(
    vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask)
{
    if ASC_IS_AIV { vmadd(dst, src0, src1, mask, MODE_ZEROING); }
}

__simd_callee__ inline void cannbotdsl_loadalign_unpack4(
    vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    if ASC_IS_AIV { vlds(dst, src, 0, UNPK4_B8); }
}

__simd_callee__ inline void cannbotdsl_storealign_pack_quarter(
    __ubuf__ float* dst, vector_float src, vector_bool mask)
{
    if ASC_IS_AIV { vsts(src, dst, 0, PK4_B32, mask); }
}

__simd_callee__ inline void cannbotdsl_storealign_squeeze_status_postupdate(
    __ubuf__ uint32_t*& dst, int32_t offset)
{
    if ASC_IS_AIV { sprsts(SPR_AR, dst, offset, POST_UPDATE); }
}

__simd_callee__ inline void cannbotdsl_squeeze_store_init()
{
    if ASC_IS_AIV { sprclr(SPR_AR); }
}

#define CANNBOTDSL_DEFINE_SQUEEZE_STORE(SCALAR_TYPE, VECTOR_TYPE)                                     \
    __simd_callee__ inline void cannbotdsl_squeeze_store(                                            \
        __ubuf__ SCALAR_TYPE* dst_addr, VECTOR_TYPE& dst, vector_store_unalign& cursor,               \
        VECTOR_TYPE src, vector_bool mask)                                                            \
    {                                                                                                 \
        if ASC_IS_AIV {                                                                               \
            vsqz(dst, src, mask, MODE_STORED);                                                        \
            vstur(cursor, dst, dst_addr, POST_UPDATE);                                                \
        }                                                                                             \
    }                                                                                                 \
    __simd_callee__ inline void cannbotdsl_squeeze_store_finalize(                                   \
        __ubuf__ SCALAR_TYPE* dst, vector_store_unalign cursor)                                       \
    {                                                                                                 \
        if ASC_IS_AIV { vstar(cursor, dst); }                                                         \
    }

CANNBOTDSL_DEFINE_SQUEEZE_STORE(int8_t, vector_int8_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(uint8_t, vector_uint8_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(int16_t, vector_int16_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(uint16_t, vector_uint16_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(int32_t, vector_int32_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(uint32_t, vector_uint32_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(half, vector_half)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(float, vector_float)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(bfloat16_t, vector_bfloat16_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(fp8_e4m3fn_t, vector_fp8_e4m3fn_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(fp8_e5m2_t, vector_fp8_e5m2_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(fp8_e8m0_t, vector_fp8_e8m0_t)
#undef CANNBOTDSL_DEFINE_SQUEEZE_STORE

__simd_callee__ inline void cannbotdsl_squeeze_store(
    __ubuf__ fp4x2_e2m1_t* dst_addr, vector_fp4x2_e2m1_t& dst,
    vector_store_unalign& cursor, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsqz(reinterpret_cast<vector_int8_t&>(dst), reinterpret_cast<vector_int8_t&>(src),
             mask, MODE_STORED);
        vstur(cursor, dst, dst_addr, POST_UPDATE);
    }
}

__simd_callee__ inline void cannbotdsl_squeeze_store_finalize(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign cursor)
{
    if ASC_IS_AIV { vstar(cursor, dst); }
}

template <typename T>
__aicore__ inline void cannbotdsl_copy_gm2ub_align(
    __ubuf__ T* dst, __gm__ T* src, uint32_t burst_count, uint32_t burst_len,
    uint8_t left_padding_num, uint8_t right_padding_num, bool enable_constant_pad,
    uint8_t l2_cache_mode, uint64_t src_stride, uint32_t dst_stride)
{
    if ASC_IS_AIV {
        if constexpr (sizeof(T) == 8) {
            copy_gm_to_ubuf_align_v2(
                (__ubuf__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, burst_count, burst_len,
                left_padding_num * 8, right_padding_num * 8,
                enable_constant_pad, l2_cache_mode, src_stride, dst_stride);
        } else {
            copy_gm_to_ubuf_align_v2(
                dst, src, 0, burst_count, burst_len, left_padding_num, right_padding_num,
                enable_constant_pad, l2_cache_mode, src_stride, dst_stride);
        }
    }
}

template <typename T>
__aicore__ inline void cannbotdsl_copy_ub2gm_align(
    __gm__ T* dst, __ubuf__ T* src, uint32_t burst_count, uint32_t burst_len,
    uint8_t l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_v2(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, burst_count, burst_len,
            l2_cache_mode, dst_stride, src_stride);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e2m1_t* dst, __gm__ fp4x2_e2m1_t* src, uint32_t loop0_size, uint32_t loop1_size,
    uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count,
    bool padding_mode, uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e1m2_t* dst, __gm__ fp4x2_e1m2_t* src, uint32_t loop0_size, uint32_t loop1_size,
    uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count,
    bool padding_mode, uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp8_e8m0_t* dst, __gm__ fp8_e8m0_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint32_t loop0_size, uint32_t loop1_size,
    uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count,
    bool padding_mode, uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ half* dst, __gm__ half* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b32(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b32(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ float* dst, __gm__ float* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b32(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

template <typename DstT, typename SrcT>
__aicore__ inline void cannbotdsl_copy_l0c2l1(
    __cbuf__ DstT* dst, __cc__ SrcT* src, uint16_t n_size, uint16_t m_size,
    uint32_t dst_stride, uint16_t src_stride, uint8_t enable_clip_relu_pre,
    uint8_t unit_flag_mode, uint64_t quant_pre_mode, uint8_t relu_pre_mode,
    bool enable_channel_split, bool enable_nz2nd, uint64_t quant_post,
    uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op,
    uint8_t eltwise_antq_en, bool c0_pad_en)
{
    if ASC_IS_AIC {
        copy_matrix_cc_to_cbuf(
            dst, src, 0, n_size, m_size, dst_stride, src_stride, 0,
            enable_clip_relu_pre, unit_flag_mode,
            static_cast<QuantMode_t>(quant_pre_mode), relu_pre_mode,
            enable_channel_split, enable_nz2nd,
            static_cast<QuantMode_post>(quant_post), relu_post, clip_relu_post,
            false, eltwise_op, eltwise_antq_en, false, c0_pad_en, false, false,
            false);
    }
}

} // namespace cannbotdsl_92
#endif

struct KernelArgTensor_fp8_e4m3fn_t_2d {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_fp8_e8m0_t_2d {
    __gm__ fp8_e8m0_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_bfloat16_t_2d {
    __gm__ bfloat16_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds(const KernelArgTensor_fp8_e4m3fn_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

struct KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds {
    __gm__ fp8_e8m0_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds(const KernelArgTensor_fp8_e8m0_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

struct KernelArgTensor_bfloat16_t_2d_Device_ss_ds {
    __gm__ bfloat16_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ss_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__cube__ __global__ __aicore__ void kernel_project_q_a(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm0_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm1_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm2_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm3_arg, KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm4_arg, int64_t arg5, int64_t arg6, int64_t arg7)
{
    __gm__ fp8_e4m3fn_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm1 = gm1_arg.ptr;
    __gm__ fp8_e8m0_t* gm2 = gm2_arg.ptr;
    __gm__ fp8_e8m0_t* gm3 = gm3_arg.ptr;
    __gm__ bfloat16_t* gm4 = gm4_arg.ptr;
    asc_init();
    asc_set_ctrl(0x1000000000000008ULL | (asc_get_ctrl() & (1ULL << 48)));  // reset CTRL like AscendC InitSocState
    asc_set_gm2l1_loop_size(1, 1);
    asc_set_gm2ub_loop_size(1, 1);
    asc_set_ub2gm_loop_size(1, 1);
    asc_set_l0c2gm_config(0, 0, false);
    asc_set_l0c2gm_nz2nd(0, 0, 0);
    asc_set_l0c_copy_prequant(0);
    asc_set_l0c2gm_quant_post(0);
    asc_set_l0c2gm_relu_alpha(0);
    float cannbotdsl_lrelu_alpha = 0.0f;
    asc_set_l0c2gm_lrelu_alpha(cannbotdsl_lrelu_alpha);
    asc_set_l0c2gm_channel_para(0);
    asc_set_copy_pad_val(static_cast<uint32_t>(0));
    asc_set_ndim_pad_value(static_cast<uint32_t>(0));
    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
    asc_set_l13d_padding(static_cast<uint64_t>(0));
    asc_set_l12l0_padding_val(static_cast<uint64_t>(0));

    asc_set_ctrl(asc_get_ctrl() | (1ULL << 51));  // MM_LAYOUT_MODE_BIT

    __cc__ float* buf8 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ fp8_e4m3fn_t* buf9 = (__ca__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cb__ fp8_e4m3fn_t* buf10 = (__cb__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ fp8_e8m0_t* buf11 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(27136);
    __cbuf__ fp8_e4m3fn_t* buf12 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(10240);
    __cbuf__ fp8_e8m0_t* buf13 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(26624);
    __cbuf__ fp8_e4m3fn_t* buf14 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);

    for (int64_t i15 = arg6; i15 < 10; i15 = (int64_t)((uint64_t)i15 + (uint64_t)arg7)) {
        int64_t t16 = (int64_t)((uint64_t)i15 * (uint64_t)128);
        for (int64_t i17 = 0; i17 < 40; i17 = (int64_t)((uint64_t)i17 + (uint64_t)1)) {
            __cbuf__ fp8_e4m3fn_t* view_root18 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf12) + 0);
            __cbuf__ fp8_e8m0_t* view_root19 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf13) + 0);
            __cbuf__ fp8_e8m0_t* view_root20 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf11) + 0);
            __cbuf__ fp8_e4m3fn_t* view_root21 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf14) + 0);
            int64_t t22 = i17 * 128;
            int64_t dim23 = gm0_arg.stride[0];
            asc_lock(PIPE_MTE2, 0);
            int64_t t24 = t22;
            set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(80) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
            asc_copy_gm2l1_nd2nz((__cbuf__ int8_t*)(view_root21 + 0), (__gm__ int8_t*)(gm0 + t24), dim23, 0, 80, 128, 0, false);
            asc_unlock(PIPE_MTE2, 0);
            int64_t t25 = (i15 < 10 ? i15 : 10);
            int64_t t26 = (int64_t)((uint64_t)t25 * (uint64_t)4);
            int64_t t27 = (int64_t)((uint64_t)40 - (uint64_t)t26);
            int64_t t28 = (t27 < 4 ? t27 : 4);
            bool cmp29 = i15 <= 10;
            int64_t sel30 = cmp29 ? t28 : 0;
            int64_t t31 = (int64_t)((uint64_t)i15 * (uint64_t)655360);
            int64_t t32 = i17 * 4096;
            int64_t t33 = (int64_t)((uint64_t)t31 + (uint64_t)t32);
            asc_lock(PIPE_MTE2, 1);
            bool cmp34 = (uint64_t)sel30 > (uint64_t)0;
            if (cmp34) {
                {
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    uint8_t dma_pad = static_cast<uint8_t>(0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                    asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root18 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm1 + t33), sel30, 4096, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)163840, 4096);
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                }
            }
            asc_unlock(PIPE_MTE2, 1);
            int64_t t35 = i17 * 64;
            asc_lock(PIPE_MTE2, 2);
            {
                asc_set_gm2l1_loop_size(1, 1);
                asc_set_gm2l1_loop1_stride(0, 0);
                asc_set_gm2l1_loop2_stride(0, 0);
                uint8_t dma_pad = static_cast<uint8_t>(0);
                asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root19 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm2 + t35), 5, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)2560, 64);
                asc_set_gm2l1_loop_size(1, 1);
                asc_set_gm2l1_loop1_stride(0, 0);
                asc_set_gm2l1_loop2_stride(0, 0);
                asc_set_gm2l1_pad(static_cast<uint32_t>(0));
            }
            asc_unlock(PIPE_MTE2, 2);
            int64_t t36 = (int64_t)((uint64_t)t25 * (uint64_t)8);
            int64_t t37 = (int64_t)((uint64_t)80 - (uint64_t)t36);
            int64_t t38 = (t37 < 8 ? t37 : 8);
            int64_t sel39 = cmp29 ? t38 : 0;
            int64_t t40 = (int64_t)((uint64_t)i15 * (uint64_t)20480);
            int64_t t41 = (int64_t)((uint64_t)t40 + (uint64_t)t35);
            asc_lock(PIPE_MTE2, 3);
            bool cmp42 = (uint64_t)sel39 > (uint64_t)0;
            if (cmp42) {
                {
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    uint8_t dma_pad = static_cast<uint8_t>(0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                    asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root20 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm3 + t41), sel39, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)2560, 64);
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                }
            }
            asc_unlock(PIPE_MTE2, 3);
            __ca__ fp8_e4m3fn_t* view_root43 = reinterpret_cast<__ca__ fp8_e4m3fn_t*>(reinterpret_cast<__ca__ uint8_t*>(buf9) + 0);
            __cbuf__ fp8_e4m3fn_t* view_root44 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf14) + 0);
            __cbuf__ fp8_e8m0_t* view_root45 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf13) + 0);
            __ca__ fp8_e4m3fn_t* view_root46 = reinterpret_cast<__ca__ fp8_e4m3fn_t*>(reinterpret_cast<__ca__ uint8_t*>(buf9) + 0);
            uint64_t selected_root47 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root46)) / 16;
            asc_lock(PIPE_MTE1, 4);
            asc_lock(PIPE_MTE1, 0);
            asc_lock(PIPE_MTE1, 2);
            asc_copy_l12l0a(view_root43 + 0, view_root44 + 0, 0, 0, 5, 4, 5, 5);
            asc_copy_l12l0a_mx(selected_root47, (__cbuf__ fp8_e8m0_t*)(view_root45 + 0), 0, 0, 5, 2, 2, 2);
            asc_unlock(PIPE_MTE1, 2);
            asc_unlock(PIPE_MTE1, 0);
            asc_unlock(PIPE_MTE1, 4);
            __cb__ fp8_e4m3fn_t* view_root48 = reinterpret_cast<__cb__ fp8_e4m3fn_t*>(reinterpret_cast<__cb__ uint8_t*>(buf10) + 0);
            __cbuf__ fp8_e4m3fn_t* view_root49 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf12) + 0);
            __cbuf__ fp8_e8m0_t* view_root50 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf11) + 0);
            __cb__ fp8_e4m3fn_t* view_root51 = reinterpret_cast<__cb__ fp8_e4m3fn_t*>(reinterpret_cast<__cb__ uint8_t*>(buf10) + 0);
            uint64_t selected_root52 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root51)) / 16;
            asc_lock(PIPE_MTE1, 5);
            asc_lock(PIPE_MTE1, 1);
            asc_lock(PIPE_MTE1, 3);
            asc_copy_l12l0b_transpose(view_root48 + 0, view_root49 + 0, 0, 0, 8, 4, 8, 8);
            asc_copy_l12l0b_mx(selected_root52, (__cbuf__ fp8_e8m0_t*)(view_root50 + 0), 0, 0, 8, 2, 2, 2);
            asc_unlock(PIPE_MTE1, 3);
            asc_unlock(PIPE_MTE1, 1);
            asc_unlock(PIPE_MTE1, 5);
            bool cmp53 = i17 == 0;
            __cc__ float* view_root54 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf8) + 0);
            __ca__ fp8_e4m3fn_t* view_root55 = reinterpret_cast<__ca__ fp8_e4m3fn_t*>(reinterpret_cast<__ca__ uint8_t*>(buf9) + 0);
            uint64_t selected_root56 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root55)) / 16;
            __cb__ fp8_e4m3fn_t* view_root57 = reinterpret_cast<__cb__ fp8_e4m3fn_t*>(reinterpret_cast<__cb__ uint8_t*>(buf10) + 0);
            uint64_t selected_root58 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root57)) / 16;
            asc_lock(PIPE_M, 6);
            asc_lock(PIPE_M, 4);
            asc_lock(PIPE_M, 5);
            if (cmp53) {
                asc_mmad_mx(view_root54 + 0, view_root55, view_root57, 80, 128, 128, 0, true, false, true);
            } else {
                asc_mmad_mx(view_root54 + 0, view_root55, view_root57, 80, 128, 128, 0, true, false, false);
            }
            asc_unlock(PIPE_M, 5);
            asc_unlock(PIPE_M, 4);
            asc_unlock(PIPE_M, 6);
        }
        int64_t t59 = (int64_t)((uint64_t)t16 / (uint64_t)16);
        int64_t t60 = (int64_t)((uint64_t)t59 * (uint64_t)16);
        int64_t dim61 = gm4_arg.stride[0];
        __cc__ float* view_root62 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf8) + 0);
        bool cmp63 = (uint64_t)arg5 > (uint64_t)0;
        if (cmp63) {
            asc_lock(PIPE_FIX, 6);
            int32_t t64 = (int32_t)arg5;
            int32_t t65 = (int32_t)dim61;
            int64_t t66 = t60;
            asc_set_l0c2gm_nz2nd(1, 0, 0);
            asc_copy_l0c2gm(gm4 + t66, view_root62, 128, t64, t65, 80, 0, 0, 0, QuantMode_t::F322BF16, 0, false, true, 0, 0, false, 0, false, false, false, false);
            asc_unlock(PIPE_FIX, 6);
        }
    }
    asc_set_ctrl(asc_get_ctrl() & ~(1ULL << 51));  // clear MM_LAYOUT_MODE_BIT
}
