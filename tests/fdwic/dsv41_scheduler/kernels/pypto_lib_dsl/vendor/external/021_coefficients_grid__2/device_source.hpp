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

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_float_1d {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_float_1d_Device_s_s {
    __gm__ float* ptr;
};
static inline KernelArgTensor_float_1d_Device_s_s cannbotdsl_pack_KernelArgTensor_float_1d_Device_s_s(const KernelArgTensor_float_1d& src) {
    return {src.ptr};
}

__simd_vf__ inline void kernel_coefficients_grid_vf0(__ubuf__ float* buf0, __ubuf__ float* buf1, __ubuf__ float* buf2, __ubuf__ float* buf3, float arg4, __ubuf__ float* buf5, float arg6, float arg7, float arg8, __ubuf__ float* buf9, __ubuf__ float* buf10, __ubuf__ float* buf11)
{
    vector_bool vmask;
    float c12 = 2.0f;
    float c13 = 9.99999997E-7f;
    float c14 = -1.0f;
    float c15 = 1.0f;
    vector_bool mask16 = asc_create_mask_b32(PAT_ALL);
    uint32_t rem18 = 24;
    vector_bool mask17 = asc_update_mask_b32(rem18);
    uint32_t rem20 = 4;
    vector_bool mask19 = asc_update_mask_b32(rem20);
    vector_float v21;
    asc_loadalign(v21, buf0 + 0);
    vector_float v22;
    asc_loadalign(v22, buf1 + 0);
    vector_float v23;
    asc_add(v23, v21, v22, mask17);
    vector_float v24;
    asc_loadalign(v24, buf2 + 0);
    vector_float v25;
    asc_add(v25, v23, v24, mask17);
    vector_float v26;
    asc_loadalign(v26, buf3 + 0);
    vector_float v27;
    asc_add(v27, v25, v26, mask17);
    vector_float v28;
    asc_mul_scalar(v28, v27, arg4, mask17);
    vector_uint32_t v29;
    asc_arange(*(vector_int32_t*)&v29, static_cast<int32_t>(0));
    vector_uint32_t v30;
    asc_duplicate_scalar(v30, 3);
    vector_uint32_t v31;
    asc_and(v31, v29, v30, mask16);
    vector_uint32_t v32;
    asc_add_scalar(v32, v31, 4, mask16);
    vector_uint32_t v33;
    asc_add_scalar(v33, v31, 8, mask16);
    vector_uint32_t v34;
    asc_add_scalar(v34, v31, 12, mask16);
    vector_uint32_t v35;
    asc_add_scalar(v35, v31, 16, mask16);
    vector_uint32_t v36;
    asc_add_scalar(v36, v31, 20, mask16);
    vector_float v37;
    asc_loadalign(v37, buf5 + 0);
    vector_float v38;
    asc_gather(v38, v28, v31);
    vector_float v39;
    asc_mul_scalar(v39, v38, arg6, mask19);
    vector_float v40;
    asc_gather(v40, v37, v31);
    vector_float v41;
    asc_add(v41, v39, v40, mask19);
    vector_float v42;
    asc_gather(v42, v28, v32);
    vector_float v43;
    asc_mul_scalar(v43, v42, arg7, mask19);
    vector_float v44;
    asc_gather(v44, v37, v32);
    vector_float v45;
    asc_add(v45, v43, v44, mask19);
    vector_float v46;
    asc_gather(v46, v28, v33);
    vector_float v47;
    asc_mul_scalar(v47, v46, arg8, mask19);
    vector_float v48;
    asc_gather(v48, v37, v33);
    vector_float v49;
    asc_add(v49, v47, v48, mask19);
    vector_float v50;
    asc_gather(v50, v28, v34);
    vector_float v51;
    asc_mul_scalar(v51, v50, arg8, mask19);
    vector_float v52;
    asc_gather(v52, v37, v34);
    vector_float v53;
    asc_add(v53, v51, v52, mask19);
    vector_float v54;
    asc_gather(v54, v28, v35);
    vector_float v55;
    asc_mul_scalar(v55, v54, arg8, mask19);
    vector_float v56;
    asc_gather(v56, v37, v35);
    vector_float v57;
    asc_add(v57, v55, v56, mask19);
    vector_float v58;
    asc_gather(v58, v28, v36);
    vector_float v59;
    asc_mul_scalar(v59, v58, arg8, mask19);
    vector_float v60;
    asc_gather(v60, v37, v36);
    vector_float v61;
    asc_add(v61, v59, v60, mask19);
    vector_float v62;
    asc_duplicate_scalar(v62, c15, mask19);
    vector_float v63;
    asc_mul_scalar(v63, v41, c14, mask19);
    vector_float v64;
    asc_exp(v64, v63, mask19);
    vector_float v65;
    asc_add_scalar(v65, v64, c15, mask19);
    vector_float v66;
    asc_div(v66, v62, v65, mask19);
    vector_float v67;
    asc_mul_scalar(v67, v45, c14, mask19);
    vector_float v68;
    asc_exp(v68, v67, mask19);
    vector_float v69;
    asc_add_scalar(v69, v68, c15, mask19);
    vector_float v70;
    asc_div(v70, v62, v69, mask19);
    vector_float v71;
    asc_add_scalar(v71, v66, c13, mask19);
    asc_storealign(buf9 + 0, v71, mask19);
    vector_float v72;
    asc_mul_scalar(v72, v70, c12, mask19);
    asc_storealign(buf10 + 0, v72, mask19);
    vector_float v73;
    asc_reduce_max(v73, v49, mask19);
    vector_float v74;
    asc_duplicate(v74, v73, mask19);
    vector_float v75;
    asc_sub(v75, v49, v74, mask19);
    vector_float v76;
    asc_exp(v76, v75, mask19);
    vector_uint32_t v77;
    asc_duplicate_scalar(v77, 0);
    vector_float v78;
    asc_gather(v78, v76, v77);
    vector_uint32_t v79;
    asc_duplicate_scalar(v79, 1);
    vector_float v80;
    asc_gather(v80, v76, v79);
    vector_uint32_t v81;
    asc_duplicate_scalar(v81, 2);
    vector_float v82;
    asc_gather(v82, v76, v81);
    vector_float v83;
    asc_gather(v83, v76, v30);
    vector_float v84;
    asc_add(v84, v78, v80, mask19);
    vector_float v85;
    asc_add(v85, v82, v83, mask19);
    vector_float v86;
    asc_add(v86, v84, v85, mask19);
    vector_float v87;
    asc_div(v87, v76, v86, mask19);
    vector_float v88;
    asc_add_scalar(v88, v87, c13, mask19);
    vector_float v89;
    asc_reduce_max(v89, v53, mask19);
    vector_float v90;
    asc_duplicate(v90, v89, mask19);
    vector_float v91;
    asc_sub(v91, v53, v90, mask19);
    vector_float v92;
    asc_exp(v92, v91, mask19);
    vector_float v93;
    asc_gather(v93, v92, v77);
    vector_float v94;
    asc_gather(v94, v92, v79);
    vector_float v95;
    asc_gather(v95, v92, v81);
    vector_float v96;
    asc_gather(v96, v92, v30);
    vector_float v97;
    asc_add(v97, v93, v94, mask19);
    vector_float v98;
    asc_add(v98, v95, v96, mask19);
    vector_float v99;
    asc_add(v99, v97, v98, mask19);
    vector_float v100;
    asc_div(v100, v92, v99, mask19);
    vector_float v101;
    asc_add_scalar(v101, v100, c13, mask19);
    vector_float v102;
    asc_reduce_max(v102, v57, mask19);
    vector_float v103;
    asc_duplicate(v103, v102, mask19);
    vector_float v104;
    asc_sub(v104, v57, v103, mask19);
    vector_float v105;
    asc_exp(v105, v104, mask19);
    vector_float v106;
    asc_gather(v106, v105, v77);
    vector_float v107;
    asc_gather(v107, v105, v79);
    vector_float v108;
    asc_gather(v108, v105, v81);
    vector_float v109;
    asc_gather(v109, v105, v30);
    vector_float v110;
    asc_add(v110, v106, v107, mask19);
    vector_float v111;
    asc_add(v111, v108, v109, mask19);
    vector_float v112;
    asc_add(v112, v110, v111, mask19);
    vector_float v113;
    asc_div(v113, v105, v112, mask19);
    vector_float v114;
    asc_add_scalar(v114, v113, c13, mask19);
    vector_float v115;
    asc_reduce_max(v115, v61, mask19);
    vector_float v116;
    asc_duplicate(v116, v115, mask19);
    vector_float v117;
    asc_sub(v117, v61, v116, mask19);
    vector_float v118;
    asc_exp(v118, v117, mask19);
    vector_float v119;
    asc_gather(v119, v118, v77);
    vector_float v120;
    asc_gather(v120, v118, v79);
    vector_float v121;
    asc_gather(v121, v118, v81);
    vector_float v122;
    asc_gather(v122, v118, v30);
    vector_float v123;
    asc_add(v123, v119, v120, mask19);
    vector_float v124;
    asc_add(v124, v121, v122, mask19);
    vector_float v125;
    asc_add(v125, v123, v124, mask19);
    vector_float v126;
    asc_div(v126, v118, v125, mask19);
    vector_float v127;
    asc_add_scalar(v127, v126, c13, mask19);
    vector_float iter128 = v88;
    vector_float iter129 = v101;
    vector_float iter130 = v114;
    vector_float iter131 = v127;
    for (int32_t i132 = 0; i132 < 20; i132 = (int32_t)((uint32_t)i132 + (uint32_t)1)) {
        bool cmp133 = i132 > 0;
        vector_float ifr134;
        vector_float ifr135;
        vector_float ifr136;
        vector_float ifr137;
        if (cmp133) {
            vector_float v138;
            asc_gather(v138, iter128, v77);
            vector_float v139;
            asc_gather(v139, iter128, v79);
            vector_float v140;
            asc_gather(v140, iter128, v81);
            vector_float v141;
            asc_gather(v141, iter128, v30);
            vector_float v142;
            asc_add(v142, v138, v139, mask19);
            vector_float v143;
            asc_add(v143, v140, v141, mask19);
            vector_float v144;
            asc_add(v144, v142, v143, mask19);
            vector_float v145;
            asc_add_scalar(v145, v144, c13, mask19);
            vector_float v146;
            asc_div(v146, iter128, v145, mask19);
            vector_float v147;
            asc_gather(v147, iter129, v77);
            vector_float v148;
            asc_gather(v148, iter129, v79);
            vector_float v149;
            asc_gather(v149, iter129, v81);
            vector_float v150;
            asc_gather(v150, iter129, v30);
            vector_float v151;
            asc_add(v151, v147, v148, mask19);
            vector_float v152;
            asc_add(v152, v149, v150, mask19);
            vector_float v153;
            asc_add(v153, v151, v152, mask19);
            vector_float v154;
            asc_add_scalar(v154, v153, c13, mask19);
            vector_float v155;
            asc_div(v155, iter129, v154, mask19);
            vector_float v156;
            asc_gather(v156, iter130, v77);
            vector_float v157;
            asc_gather(v157, iter130, v79);
            vector_float v158;
            asc_gather(v158, iter130, v81);
            vector_float v159;
            asc_gather(v159, iter130, v30);
            vector_float v160;
            asc_add(v160, v156, v157, mask19);
            vector_float v161;
            asc_add(v161, v158, v159, mask19);
            vector_float v162;
            asc_add(v162, v160, v161, mask19);
            vector_float v163;
            asc_add_scalar(v163, v162, c13, mask19);
            vector_float v164;
            asc_div(v164, iter130, v163, mask19);
            vector_float v165;
            asc_gather(v165, iter131, v77);
            vector_float v166;
            asc_gather(v166, iter131, v79);
            vector_float v167;
            asc_gather(v167, iter131, v81);
            vector_float v168;
            asc_gather(v168, iter131, v30);
            vector_float v169;
            asc_add(v169, v165, v166, mask19);
            vector_float v170;
            asc_add(v170, v167, v168, mask19);
            vector_float v171;
            asc_add(v171, v169, v170, mask19);
            vector_float v172;
            asc_add_scalar(v172, v171, c13, mask19);
            vector_float v173;
            asc_div(v173, iter131, v172, mask19);
            ifr134 = v146;
            ifr135 = v155;
            ifr136 = v164;
            ifr137 = v173;
        } else {
            ifr134 = iter128;
            ifr135 = iter129;
            ifr136 = iter130;
            ifr137 = iter131;
        }
        vector_float v174;
        asc_add(v174, ifr134, ifr135, mask19);
        vector_float v175;
        asc_add(v175, v174, ifr136, mask19);
        vector_float v176;
        asc_add(v176, v175, ifr137, mask19);
        vector_float v177;
        asc_add_scalar(v177, v176, c13, mask19);
        vector_float v178;
        asc_div(v178, ifr134, v177, mask19);
        vector_float v179;
        asc_div(v179, ifr135, v177, mask19);
        vector_float v180;
        asc_div(v180, ifr136, v177, mask19);
        vector_float v181;
        asc_div(v181, ifr137, v177, mask19);
        iter128 = v178;
        iter129 = v179;
        iter130 = v180;
        iter131 = v181;
    }
    for (int32_t i182 = 0; i182 < 1; i182 = (int32_t)((uint32_t)i182 + (uint32_t)1)) {
        asc_scatter((__ubuf__ float*)(buf11 + 0), iter128, v31, mask19);
        asc_scatter((__ubuf__ float*)(buf11 + 0), iter129, v32, mask19);
        asc_scatter((__ubuf__ float*)(buf11 + 0), iter130, v33, mask19);
        asc_scatter((__ubuf__ float*)(buf11 + 0), iter131, v34, mask19);
    }
}

static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_1d_Device_s_s) == 8 && alignof(KernelArgTensor_float_1d_Device_s_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_1d_Device_s_s) == 8 && alignof(KernelArgTensor_float_1d_Device_s_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_coefficients_grid(KernelArgTensor_float_2d_Device_ds_ds gm183_arg, KernelArgTensor_float_2d_Device_ds_ds gm184_arg, KernelArgTensor_float_1d_Device_s_s gm185_arg, KernelArgTensor_float_1d_Device_s_s gm186_arg, KernelArgTensor_float_2d_Device_ds_ds gm187_arg, KernelArgTensor_float_2d_Device_ds_ds gm188_arg, KernelArgTensor_float_2d_Device_ds_ds gm189_arg, int64_t arg190, int64_t arg191)
{
    __gm__ float* gm183 = gm183_arg.ptr;
    __gm__ float* gm184 = gm184_arg.ptr;
    __gm__ float* gm185 = gm185_arg.ptr;
    __gm__ float* gm186 = gm186_arg.ptr;
    __gm__ float* gm187 = gm187_arg.ptr;
    __gm__ float* gm188 = gm188_arg.ptr;
    __gm__ float* gm189 = gm189_arg.ptr;
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

    __ubuf__ float* buf192 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf193 = (__ubuf__ float*)(uintptr_t)(uint64_t)(256);
    __ubuf__ float* buf194 = (__ubuf__ float*)(uintptr_t)(uint64_t)(512);
    __ubuf__ float* buf195 = (__ubuf__ float*)(uintptr_t)(uint64_t)(768);
    __ubuf__ float* buf196 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1024);
    __ubuf__ float* buf197 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1280);
    __ubuf__ float* buf198 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1536);
    __ubuf__ float* buf199 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1792);

    vector_bool vmask;
    int64_t dim200 = gm183_arg.shape[0];
    int64_t dim201 = gm187_arg.shape[0];
    for (int64_t i202 = arg190; i202 < dim201; i202 = (int64_t)((uint64_t)i202 + (uint64_t)arg191)) {
        int64_t t203 = (int64_t)((uint64_t)dim200 + (uint64_t)i202);
        int64_t t204 = (int64_t)((uint64_t)dim200 * (uint64_t)2);
        int64_t t205 = (int64_t)((uint64_t)t204 + (uint64_t)i202);
        int64_t t206 = (int64_t)((uint64_t)dim200 * (uint64_t)3);
        int64_t t207 = (int64_t)((uint64_t)t206 + (uint64_t)i202);
        int64_t dim208 = gm184_arg.stride[0];
        int64_t t209 = (int64_t)((uint64_t)i202 * (uint64_t)dim208);
        int64_t t210 = (int64_t)((uint64_t)t209 * (uint64_t)4);
        asc_lock(PIPE_MTE2, 0);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf192 + 0), (__gm__ float*)((__gm__ uint8_t*)gm184 + t210), 1, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 0);
        int64_t t211 = (int64_t)((uint64_t)t203 * (uint64_t)dim208);
        int64_t t212 = (int64_t)((uint64_t)t211 * (uint64_t)4);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf193 + 0), (__gm__ float*)((__gm__ uint8_t*)gm184 + t212), 1, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        int64_t t213 = (int64_t)((uint64_t)t205 * (uint64_t)dim208);
        int64_t t214 = (int64_t)((uint64_t)t213 * (uint64_t)4);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf194 + 0), (__gm__ float*)((__gm__ uint8_t*)gm184 + t214), 1, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        int64_t t215 = (int64_t)((uint64_t)t207 * (uint64_t)dim208);
        int64_t t216 = (int64_t)((uint64_t)t215 * (uint64_t)4);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf195 + 0), (__gm__ float*)((__gm__ uint8_t*)gm184 + t216), 1, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_lock(PIPE_MTE2, 4);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf196 + 0), (__gm__ float*)((__gm__ uint8_t*)gm186 + 0), 1, 96, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 4);
        int64_t dim217 = gm183_arg.stride[0];
        int64_t t218 = (int64_t)((uint64_t)i202 * (uint64_t)dim217);
        int64_t t219 = t218;
        float ld220 = gm183[t219];
        float ld221 = gm185[0];
        float ld222 = gm185[1];
        float ld223 = gm185[2];
        asc_lock(PIPE_V, 0);
        asc_lock(PIPE_V, 4);
        asc_lock(PIPE_V, 5);
        asc_lock(PIPE_V, 7);
        kernel_coefficients_grid_vf0((__ubuf__ float*)buf192, (__ubuf__ float*)buf193, (__ubuf__ float*)buf194, (__ubuf__ float*)buf195, ld220, (__ubuf__ float*)buf196, ld221, ld222, ld223, (__ubuf__ float*)buf197, (__ubuf__ float*)buf198, (__ubuf__ float*)buf199);
        asc_unlock(PIPE_V, 7);
        asc_unlock(PIPE_V, 5);
        asc_unlock(PIPE_V, 4);
        asc_unlock(PIPE_V, 0);
        int64_t dim224 = gm187_arg.stride[0];
        int64_t t225 = (int64_t)((uint64_t)i202 * (uint64_t)dim224);
        int64_t t226 = (int64_t)((uint64_t)t225 * (uint64_t)32);
        int64_t t227 = (int64_t)((uint64_t)t226 / (uint64_t)8);
        int64_t t228 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm187) + (uint64_t)t227);
        __gm__ float* root229 = (__gm__ float*)(uintptr_t)(uint64_t)(t228);
        __ubuf__ float* view_root230 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf197) + 0);
        int64_t dim231 = gm187_arg.stride[0];
        int64_t t232 = dim231;
        int64_t t233 = t232 * 4;
        int64_t t234 = t233;
        asc_lock(PIPE_MTE3, 5);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root229 + 0, (__ubuf__ float*)view_root230 + 0, 1, 16, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 5);
        int64_t dim235 = gm188_arg.stride[0];
        int64_t t236 = (int64_t)((uint64_t)i202 * (uint64_t)dim235);
        int64_t t237 = (int64_t)((uint64_t)t236 * (uint64_t)32);
        int64_t t238 = (int64_t)((uint64_t)t237 / (uint64_t)8);
        int64_t t239 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm188) + (uint64_t)t238);
        __gm__ float* root240 = (__gm__ float*)(uintptr_t)(uint64_t)(t239);
        __ubuf__ float* view_root241 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf198) + 0);
        int64_t dim242 = gm188_arg.stride[0];
        int64_t t243 = dim242;
        int64_t t244 = t243 * 4;
        int64_t t245 = t244;
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root240 + 0, (__ubuf__ float*)view_root241 + 0, 1, 16, static_cast<uint8_t>(0), (uint64_t)0, 0);
        int64_t dim246 = gm189_arg.stride[0];
        int64_t t247 = (int64_t)((uint64_t)i202 * (uint64_t)dim246);
        int64_t t248 = (int64_t)((uint64_t)t247 * (uint64_t)32);
        int64_t t249 = (int64_t)((uint64_t)t248 / (uint64_t)8);
        int64_t t250 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm189) + (uint64_t)t249);
        __gm__ float* root251 = (__gm__ float*)(uintptr_t)(uint64_t)(t250);
        __ubuf__ float* view_root252 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf199) + 0);
        int64_t dim253 = gm189_arg.stride[0];
        int64_t t254 = dim253;
        int64_t t255 = t254 * 4;
        int64_t t256 = t255;
        asc_lock(PIPE_MTE3, 7);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root251 + 0, (__ubuf__ float*)view_root252 + 0, 1, 64, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 7);
    }
}
