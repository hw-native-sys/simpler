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

struct KernelArgTensor_bfloat16_t_2d {
    __gm__ bfloat16_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

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

struct KernelArgTensor_bfloat16_t_2d_Device_ds_ds {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ds_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

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

__simd_vf__ inline void kernel_rope_quant_grid_vf0(__ubuf__ float* buf0)
{
    vector_bool vmask;
    float c1 = 0.0f;
    vector_bool mask2 = asc_create_mask_b32(PAT_ALL);
    vector_float v3;
    asc_duplicate_scalar(v3, c1, mask2);
    for (int32_t i4 = 0; i4 < 576; i4 = (int32_t)((uint32_t)i4 + (uint32_t)64)) {
        int32_t t5 = i4;
        asc_storealign(buf0 + t5, v3, mask2);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_rope_quant_grid_vf1(__ubuf__ bfloat16_t* buf6, __ubuf__ float* buf7, __ubuf__ float* buf8, __ubuf__ float* buf9, __ubuf__ fp8_e4m3fn_t* buf10, int32_t arg11, __ubuf__ uint8_t* buf12)
{
    vector_bool vmask;
    float c13 = 448.0f;
    float c14 = -448.0f;
    vector_bool mask15 = asc_create_mask_b32(PAT_ALL);
    uint32_t rem17 = 32;
    vector_bool mask16 = asc_update_mask_b32(rem17);
    for (int32_t i18 = 0; i18 < 512; i18 = (int32_t)((uint32_t)i18 + (uint32_t)64)) {
        int32_t t19 = i18;
        vector_float v20;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v20, (__ubuf__ bfloat16_t*)(buf6 + t19));
        vector_float v21;
        asc_bfloat162float(v21, *(vector_bfloat16_t*)&v20, mask15);
        asc_storealign(buf7 + t19, v21, mask15);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i22 = 0; i22 < 1; i22 = (int32_t)((uint32_t)i22 + (uint32_t)1)) {
        vector_float v23;
        vector_float v24;
        asc_loadalign_deintlv(v23, v24, (__ubuf__ float*)(buf7 + 448));
        vector_float v25;
        asc_loadalign(v25, buf8 + 0);
        vector_float v26;
        asc_loadalign(v26, buf9 + 0);
        vector_float v27;
        asc_mul(v27, v23, v25, mask16);
        vector_float v28;
        asc_mul(v28, v24, v26, mask16);
        vector_float v29;
        asc_mul(v29, v24, v25, mask16);
        vector_float v30;
        asc_mul(v30, v23, v26, mask16);
        vector_float v31;
        asc_add(v31, v27, v28, mask16);
        vector_float v32;
        asc_sub(v32, v29, v30, mask16);
        vector_bfloat16_t v33;
        asc_float2bfloat16_rn(v33, v31, mask16, ASC_POSITION_EVEN);
        vector_bfloat16_t v34;
        asc_float2bfloat16_rn(v34, v32, mask16, ASC_POSITION_EVEN);
        vector_float v35;
        asc_bfloat162float(v35, *(vector_bfloat16_t*)&v33, mask16);
        vector_float v36;
        asc_bfloat162float(v36, *(vector_bfloat16_t*)&v34, mask16);
        vector_uint32_t &r37 = (vector_uint32_t &)v35;
        vector_uint32_t &r38 = (vector_uint32_t &)v36;
        asc_storealign_intlv((__ubuf__ uint32_t*)(buf7 + 448), r37, r38);
    }
    asc_mem_bar(VST_VLD);
    uint32_t rem40 = 32;
    vector_bool mask39 = asc_update_mask_b32(rem40);
    for (int32_t i41 = 0; i41 < 16; i41 = (int32_t)((uint32_t)i41 + (uint32_t)1)) {
        int32_t t42 = i41 * 32;
        int32_t t43 = t42;
        vector_float v44;
        asc_loadalign(v44, buf7 + t43);
        vector_float v45;
        asc_abs(v45, v44, mask39);
        vector_float v46;
        asc_reduce_max(v46, v45, mask39);
        vector_int32_t &r47 = (vector_int32_t &)v46;
        vector_int32_t v48;
        asc_shiftright_scalar(v48, r47, 23, mask39);
        vector_int32_t v49;
        asc_add_scalar(v49, v48, -8, mask39);
        vector_int32_t v50;
        asc_max_scalar(v50, v49, 0, mask39);
        vector_int32_t v51;
        asc_duplicate(v51, v50, mask39);
        vector_int32_t v52;
        asc_duplicate_scalar(v52, 254, mask39);
        vector_int32_t v53;
        asc_sub(v53, v52, v51, mask39);
        vector_int32_t v54;
        asc_min_scalar(v54, v51, 1, mask39);
        vector_int32_t v55;
        asc_mul(v55, v53, v54, mask39);
        vector_int32_t v56;
        asc_shiftleft_scalar(v56, v55, 23, mask39);
        vector_float &r57 = (vector_float &)v56;
        vector_float v58;
        asc_mul(v58, v44, r57, mask39);
        vector_float v59;
        asc_max_scalar(v59, v58, c14, mask39);
        vector_float v60;
        asc_min_scalar(v60, v59, c13, mask39);
        vector_fp8_e4m3fn_t v61;
        asc_float2e4m3_rn(v61, v60, mask39);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf10 + t43), *(vector_float*)&v61, mask39);
        vector_uint8_t v62;
        asc_int322uint8(v62, v51, mask39);
        int32_t t63 = i41 / 2;
        int32_t t64 = t63 * 32;
        int32_t t65 = (int32_t)((uint32_t)arg11 * (uint32_t)2);
        int32_t t66 = (int32_t)((uint32_t)t64 + (uint32_t)t65);
        int32_t t67 = i41 % 2;
        int32_t t68 = (int32_t)((uint32_t)t66 + (uint32_t)t67);
        int32_t t69 = t68;
        asc_storealign_1st(buf12 + t69, v62);
    }
}

__simd_vf__ inline void kernel_rope_quant_grid_vf2(__ubuf__ float* buf70, __ubuf__ fp8_e4m3fn_t* buf71, int32_t arg72, __ubuf__ uint8_t* buf73)
{
    vector_bool vmask;
    float c74 = 448.0f;
    float c75 = -448.0f;
    uint32_t rem77 = 32;
    vector_bool mask76 = asc_update_mask_b32(rem77);
    for (int32_t i78 = 0; i78 < 16; i78 = (int32_t)((uint32_t)i78 + (uint32_t)1)) {
        int32_t t79 = i78 * 32;
        int32_t t80 = t79;
        vector_float v81;
        asc_loadalign(v81, buf70 + t80);
        vector_float v82;
        asc_abs(v82, v81, mask76);
        vector_float v83;
        asc_reduce_max(v83, v82, mask76);
        vector_int32_t &r84 = (vector_int32_t &)v83;
        vector_int32_t v85;
        asc_shiftright_scalar(v85, r84, 23, mask76);
        vector_int32_t v86;
        asc_add_scalar(v86, v85, -8, mask76);
        vector_int32_t v87;
        asc_max_scalar(v87, v86, 0, mask76);
        vector_int32_t v88;
        asc_duplicate(v88, v87, mask76);
        vector_int32_t v89;
        asc_duplicate_scalar(v89, 254, mask76);
        vector_int32_t v90;
        asc_sub(v90, v89, v88, mask76);
        vector_int32_t v91;
        asc_min_scalar(v91, v88, 1, mask76);
        vector_int32_t v92;
        asc_mul(v92, v90, v91, mask76);
        vector_int32_t v93;
        asc_shiftleft_scalar(v93, v92, 23, mask76);
        vector_float &r94 = (vector_float &)v93;
        vector_float v95;
        asc_mul(v95, v81, r94, mask76);
        vector_float v96;
        asc_max_scalar(v96, v95, c75, mask76);
        vector_float v97;
        asc_min_scalar(v97, v96, c74, mask76);
        vector_fp8_e4m3fn_t v98;
        asc_float2e4m3_rn(v98, v97, mask76);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf71 + t80), *(vector_float*)&v98, mask76);
        vector_uint8_t v99;
        asc_int322uint8(v99, v88, mask76);
        int32_t t100 = i78 / 2;
        int32_t t101 = t100 * 32;
        int32_t t102 = (int32_t)((uint32_t)arg72 * (uint32_t)2);
        int32_t t103 = (int32_t)((uint32_t)t101 + (uint32_t)t102);
        int32_t t104 = i78 % 2;
        int32_t t105 = (int32_t)((uint32_t)t103 + (uint32_t)t104);
        int32_t t106 = t105;
        asc_storealign_1st(buf73 + t106, v99);
    }
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_rope_quant_grid(KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm107_arg, KernelArgTensor_float_2d_Device_ds_ds gm108_arg, KernelArgTensor_float_2d_Device_ds_ds gm109_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm110_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm111_arg, int64_t arg112, int64_t arg113, int32_t arg114)
{
    __gm__ bfloat16_t* gm107 = gm107_arg.ptr;
    __gm__ float* gm108 = gm108_arg.ptr;
    __gm__ float* gm109 = gm109_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm110 = gm110_arg.ptr;
    __gm__ fp8_e8m0_t* gm111 = gm111_arg.ptr;
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

    __ubuf__ bfloat16_t* buf115 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf116 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1024);
    __ubuf__ float* buf117 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1280);
    __ubuf__ float* buf118 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1536);
    __ubuf__ fp8_e4m3fn_t* buf119 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(3840);
    __ubuf__ uint8_t* buf120 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(4352);

    vector_bool vmask;
    for (int64_t i121 = arg112; i121 < 320; i121 = (int64_t)((uint64_t)i121 + (uint64_t)arg113)) {
        int64_t t122 = i121 % 64;
        bool cmp123 = t122 != 0;
        bool cmp124 = t122 < 0;
        bool t125 = cmp123 & cmp124;
        int64_t sel126 = t125 ? 64 : 0;
        int64_t t127 = t122 + sel126;
        int64_t t128 = i121 / 64;
        int64_t t129 = (int64_t)(uint8_t)t125;
        int64_t t130 = (int64_t)((uint64_t)t128 - (uint64_t)t129);
        int64_t t131 = (int64_t)((uint64_t)t130 * (uint64_t)16);
        int64_t t132 = (int64_t)arg114;
        int64_t t133 = (int64_t)((uint64_t)t132 - (uint64_t)t131);
        int64_t t134 = (t133 < 16 ? t133 : 16);
        int64_t t135 = (t134 > 0 ? t134 : 0);
        int64_t dim136 = gm107_arg.shape[0];
        int64_t t137 = (t131 < dim136 ? t131 : dim136);
        int64_t t138 = t127 % 8;
        bool cmp139 = t138 != 0;
        bool cmp140 = t138 < 0;
        bool t141 = cmp139 & cmp140;
        int64_t t142 = t127 / 8;
        int64_t t143 = (int64_t)(uint8_t)t141;
        int64_t t144 = (int64_t)((uint64_t)t142 - (uint64_t)t143);
        int64_t sel145 = t141 ? 8 : 0;
        int64_t t146 = t138 + sel145;
        int64_t t147 = (int64_t)((uint64_t)t144 * (uint64_t)80);
        int64_t t148 = (int64_t)((uint64_t)t147 + (uint64_t)t131);
        int64_t t149 = t146 * 512;
        int64_t t150 = (int64_t)((uint64_t)t130 * (uint64_t)2048);
        int64_t t151 = t146 * 256;
        int64_t t152 = (int64_t)((uint64_t)t150 + (uint64_t)t151);
        for (int64_t i153 = 0; i153 < 16; i153 = (int64_t)((uint64_t)i153 + (uint64_t)1)) {
            kernel_rope_quant_grid_vf0((__ubuf__ float*)buf118);
            bool cmp154 = i153 < t135;
            if (cmp154) {
                int64_t t155 = (i153 < t135 ? i153 : t135);
                int64_t t156 = (int64_t)((uint64_t)t135 - (uint64_t)t155);
                int64_t t157 = (t156 < 1 ? t156 : 1);
                bool cmp158 = i153 <= t135;
                int64_t sel159 = cmp158 ? t157 : 0;
                int64_t dim160 = gm107_arg.stride[0];
                int64_t t161 = (int64_t)((uint64_t)dim160 * (uint64_t)2);
                int64_t t162 = t127 * 1024;
                int64_t t163 = (int64_t)((uint64_t)i153 + (uint64_t)t137);
                int64_t t164 = (int64_t)((uint64_t)t163 * (uint64_t)dim160);
                int64_t t165 = (int64_t)((uint64_t)t164 * (uint64_t)2);
                int64_t t166 = (int64_t)((uint64_t)t162 + (uint64_t)t165);
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf115 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm107 + t166), sel159, 1024, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t161, 1024);
                }
                asc_unlock(PIPE_MTE2, 0);
                int64_t dim167 = gm108_arg.stride[0];
                int64_t t168 = (int64_t)((uint64_t)dim167 * (uint64_t)4);
                int64_t t169 = (int64_t)((uint64_t)t163 * (uint64_t)dim167);
                int64_t t170 = (int64_t)((uint64_t)t169 * (uint64_t)4);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf116 + 0), (__gm__ float*)((__gm__ uint8_t*)gm108 + t170), sel159, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t168, 256);
                }
                int64_t dim171 = gm109_arg.stride[0];
                int64_t t172 = (int64_t)((uint64_t)dim171 * (uint64_t)4);
                int64_t t173 = (int64_t)((uint64_t)t163 * (uint64_t)dim171);
                int64_t t174 = (int64_t)((uint64_t)t173 * (uint64_t)4);
                asc_lock(PIPE_MTE2, 2);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf117 + 0), (__gm__ float*)((__gm__ uint8_t*)gm109 + t174), sel159, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t172, 256);
                }
                asc_unlock(PIPE_MTE2, 2);
                int32_t t175 = (int32_t)i153;
                asc_lock(PIPE_V, 0);
                asc_lock(PIPE_V, 2);
                asc_lock(PIPE_V, 3);
                asc_lock(PIPE_V, 4);
                kernel_rope_quant_grid_vf1((__ubuf__ bfloat16_t*)buf115, (__ubuf__ float*)buf118, (__ubuf__ float*)buf116, (__ubuf__ float*)buf117, (__ubuf__ fp8_e4m3fn_t*)buf119, t175, (__ubuf__ uint8_t*)buf120);
                asc_unlock(PIPE_V, 4);
                asc_unlock(PIPE_V, 3);
                asc_unlock(PIPE_V, 2);
                asc_unlock(PIPE_V, 0);
            } else {
                int32_t t176 = (int32_t)i153;
                asc_lock(PIPE_V, 3);
                asc_lock(PIPE_V, 4);
                kernel_rope_quant_grid_vf2((__ubuf__ float*)buf118, (__ubuf__ fp8_e4m3fn_t*)buf119, t176, (__ubuf__ uint8_t*)buf120);
                asc_unlock(PIPE_V, 4);
                asc_unlock(PIPE_V, 3);
            }
            int64_t t177 = (int64_t)((uint64_t)i153 + (uint64_t)t148);
            int64_t dim178 = gm110_arg.stride[0];
            int64_t t179 = (int64_t)((uint64_t)t177 * (uint64_t)dim178);
            int64_t t180 = (int64_t)((uint64_t)t179 + (uint64_t)t149);
            int64_t t181 = (int64_t)((uint64_t)t180 * (uint64_t)8);
            int64_t t182 = (int64_t)((uint64_t)t181 / (uint64_t)8);
            int64_t t183 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm110) + (uint64_t)t182);
            __gm__ fp8_e4m3fn_t* root184 = (__gm__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(t183);
            __ubuf__ fp8_e4m3fn_t* view_root185 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf119) + 0);
            int64_t dim186 = gm110_arg.stride[0];
            asc_lock(PIPE_MTE3, 3);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ fp8_e4m3fn_t*)root184 + 0, (__ubuf__ fp8_e4m3fn_t*)view_root185 + 0, 1, 512, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 3);
        }
        int64_t dim187 = gm111_arg.stride[0];
        int64_t t188 = (int64_t)((uint64_t)dim187 * (uint64_t)8);
        int64_t t189 = t188 / 8;
        int64_t dim190 = gm111_arg.stride[0];
        int64_t t191 = (int64_t)((uint64_t)t144 * (uint64_t)dim190);
        int64_t t192 = (int64_t)((uint64_t)t191 + (uint64_t)t152);
        int64_t t193 = (int64_t)((uint64_t)t192 * (uint64_t)8);
        int64_t t194 = (int64_t)((uint64_t)t193 / (uint64_t)8);
        int64_t t195 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm111) + (uint64_t)t194);
        __gm__ uint8_t* root196 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t195);
        __ubuf__ uint8_t* view_root197 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf120) + 0);
        asc_lock(PIPE_MTE3, 4);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root196 + 0, (__ubuf__ uint8_t*)view_root197 + 0, 1, 256, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 4);
    }
}
