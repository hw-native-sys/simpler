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

struct KernelArgTensor_bfloat16_t_2d_Device_ds_ds {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ds_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_bfloat16_t_2d_Device_ss_ds {
    __gm__ bfloat16_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ss_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__cube__ __global__ __aicore__ void kernel_project_c2a_grid(KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm0_arg, KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm1_arg, KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm2_arg, KernelArgTensor_float_2d_Device_ds_ds gm3_arg, KernelArgTensor_float_2d_Device_ds_ds gm4_arg, int64_t arg5, int64_t arg6, int32_t arg7)
{
    __gm__ bfloat16_t* gm0 = gm0_arg.ptr;
    __gm__ bfloat16_t* gm1 = gm1_arg.ptr;
    __gm__ bfloat16_t* gm2 = gm2_arg.ptr;
    __gm__ float* gm3 = gm3_arg.ptr;
    __gm__ float* gm4 = gm4_arg.ptr;
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

    __cc__ float* buf8 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ bfloat16_t* buf9 = (__ca__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cb__ bfloat16_t* buf10 = (__cb__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cc__ float* buf11 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ bfloat16_t* buf12 = (__ca__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cb__ bfloat16_t* buf13 = (__cb__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ bfloat16_t* buf14 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(40960);
    __cbuf__ bfloat16_t* buf15 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ bfloat16_t* buf16 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(40960);
    __cbuf__ bfloat16_t* buf17 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);

    int64_t dim18 = gm0_arg.shape[0];
    int32_t t19 = (arg7 > 0 ? arg7 : 0);
    int64_t t20 = (int64_t)t19;
    int64_t t21 = (dim18 < t20 ? dim18 : t20);
    int64_t t22 = (int64_t)((uint64_t)t21 + (uint64_t)79);
    int64_t t23 = t22 % 80;
    bool cmp24 = t23 != 0;
    bool cmp25 = t23 < 0;
    bool t26 = cmp24 & cmp25;
    int64_t t27 = t22 / 80;
    int64_t t28 = (int64_t)(uint8_t)t26;
    int64_t t29 = (int64_t)((uint64_t)t27 - (uint64_t)t28);
    int64_t t30 = (int64_t)((uint64_t)t29 * (uint64_t)16);
    for (int64_t i31 = arg5; i31 < t30; i31 = (int64_t)((uint64_t)i31 + (uint64_t)arg6)) {
        int64_t t32 = i31 % 16;
        bool cmp33 = t32 != 0;
        bool cmp34 = t32 < 0;
        bool t35 = cmp33 & cmp34;
        int64_t t36 = i31 / 16;
        int64_t t37 = (int64_t)(uint8_t)t35;
        int64_t t38 = (int64_t)((uint64_t)t36 - (uint64_t)t37);
        int64_t t39 = (int64_t)((uint64_t)t38 * (uint64_t)80);
        int64_t t40 = i31 % 2;
        bool cmp41 = t40 != 0;
        bool cmp42 = t40 < 0;
        bool t43 = cmp41 & cmp42;
        int64_t t44 = i31 / 2;
        int64_t t45 = (int64_t)(uint8_t)t43;
        int64_t t46 = (int64_t)((uint64_t)t44 - (uint64_t)t45);
        int64_t t47 = t46 % 8;
        bool cmp48 = t47 != 0;
        bool cmp49 = t47 < 0;
        bool t50 = cmp48 & cmp49;
        int64_t sel51 = t50 ? 8 : 0;
        int64_t t52 = t47 + sel51;
        int64_t t53 = (int64_t)((uint64_t)t21 - (uint64_t)t39);
        int64_t t54 = (t53 < 80 ? t53 : 80);
        int64_t t55 = t52 * 64;
        int64_t sel56 = t43 ? 2 : 0;
        int64_t t57 = t40 + sel56;
        bool cmp58 = t57 == 0;
        if (cmp58) {
            for (int64_t i59 = 0; i59 < 20; i59 = (int64_t)((uint64_t)i59 + (uint64_t)1)) {
                __cbuf__ bfloat16_t* view_root60 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf16) + 0);
                __cbuf__ bfloat16_t* view_root61 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf17) + 0);
                int64_t t62 = (int64_t)((uint64_t)t39 / (uint64_t)16);
                int64_t dim63 = gm0_arg.stride[0];
                int64_t t64 = (int64_t)((uint64_t)t62 * (uint64_t)dim63);
                int64_t t65 = i59 * 16;
                int64_t t66 = (int64_t)((uint64_t)t64 + (uint64_t)t65);
                int64_t t67 = (int64_t)((uint64_t)t66 * (uint64_t)16);
                int64_t t68 = t54 / 80;
                int64_t t69 = (t68 < 0 ? t68 : 0);
                int64_t t70 = (int64_t)((uint64_t)t69 * (uint64_t)80);
                int64_t t71 = (int64_t)((uint64_t)t54 - (uint64_t)t70);
                int64_t t72 = (t71 < 80 ? t71 : 80);
                bool cmp73 = t68 >= 0;
                int64_t sel74 = cmp73 ? t72 : 0;
                int64_t dim75 = gm0_arg.stride[0];
                int64_t t76 = dim75;
                int64_t t77 = t76 * 2;
                int64_t t78 = t77;
                asc_lock(PIPE_MTE2, 0);
                bool cmp79 = (uint64_t)sel74 > (uint64_t)0;
                if (cmp79) {
                    int64_t t80 = t67;
                    int32_t t81 = (int32_t)sel74;
                    set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(80) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                    asc_copy_gm2l1_nd2nz(view_root61 + 0, gm0 + t80, t78, 0, t81, 256, 0, false);
                }
                asc_unlock(PIPE_MTE2, 0);
                int64_t dim82 = gm1_arg.stride[0];
                int64_t t83 = (int64_t)((uint64_t)dim82 * (uint64_t)2);
                int64_t t84 = (int64_t)((uint64_t)t52 * (uint64_t)dim82);
                int64_t t85 = (int64_t)((uint64_t)t84 * (uint64_t)8);
                int64_t t86 = i59 * 8192;
                int64_t t87 = (int64_t)((uint64_t)t85 + (uint64_t)t86);
                asc_lock(PIPE_MTE2, 1);
                {
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    union { float f; uint32_t u; } padding_f32_bits88;
                    padding_f32_bits88.f = static_cast<float>(0);
                    uint16_t padding_bf16_bits89 = static_cast<uint16_t>((padding_f32_bits88.u & 0x7fffffffU) > 0x7f800000U ? ((padding_f32_bits88.u >> 16) | 0x40U) : ((padding_f32_bits88.u + 0x7fffU + ((padding_f32_bits88.u >> 16) & 1U)) >> 16));
                    asc_set_gm2l1_pad(static_cast<uint32_t>(padding_bf16_bits89));
                    asc_copy_gm2l1_align((__cbuf__ bfloat16_t*)((__cbuf__ uint8_t*)view_root60 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm1 + t87), 4, 8192, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t83, 8192);
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                }
                asc_unlock(PIPE_MTE2, 1);
                __ca__ bfloat16_t* view_root90 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf9) + 0);
                __cbuf__ bfloat16_t* view_root91 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf17) + 0);
                asc_lock(PIPE_MTE1, 2);
                asc_lock(PIPE_MTE1, 0);
                asc_copy_l12l0a(view_root90 + 0, view_root91 + 0, 0, 0, 5, 16, 5, 5);
                asc_unlock(PIPE_MTE1, 0);
                asc_unlock(PIPE_MTE1, 2);
                __cb__ bfloat16_t* view_root92 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf10) + 0);
                __cbuf__ bfloat16_t* view_root93 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf16) + 0);
                asc_lock(PIPE_MTE1, 3);
                asc_lock(PIPE_MTE1, 1);
                asc_copy_l12l0b_transpose(view_root92 + 0, view_root93 + 0, 0, 0, 16, 4, 16, 4);
                asc_unlock(PIPE_MTE1, 1);
                asc_unlock(PIPE_MTE1, 3);
                bool cmp94 = i59 == 0;
                __cc__ float* view_root95 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf8) + 0);
                __ca__ bfloat16_t* view_root96 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf9) + 0);
                __cb__ bfloat16_t* view_root97 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf10) + 0);
                asc_lock(PIPE_M, 4);
                asc_lock(PIPE_M, 2);
                asc_lock(PIPE_M, 3);
                if (cmp94) {
                    asc_mmad(view_root95 + 0, view_root96, view_root97, 80, 256, 64, 0, true, false, true);
                } else {
                    asc_mmad(view_root95 + 0, view_root96, view_root97, 80, 256, 64, 0, true, false, false);
                }
                asc_unlock(PIPE_M, 3);
                asc_unlock(PIPE_M, 2);
                asc_unlock(PIPE_M, 4);
            }
            int64_t dim98 = gm3_arg.stride[0];
            int64_t t99 = (int64_t)((uint64_t)t39 * (uint64_t)dim98);
            int64_t t100 = t99;
            int64_t t101 = t55;
            int64_t t102 = t100 + t101;
            int64_t dim103 = gm3_arg.stride[0];
            __cc__ float* view_root104 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf8) + 0);
            bool cmp105 = (uint64_t)t54 > (uint64_t)0;
            if (cmp105) {
                int64_t t106 = t102;
                asc_lock(PIPE_FIX, 4);
                int32_t t107 = (int32_t)t54;
                int32_t t108 = (int32_t)dim103;
                int64_t t109 = t106;
                asc_set_l0c2gm_nz2nd(1, 0, 0);
                asc_copy_l0c2gm(gm3 + t109, view_root104, 64, t107, t108, 80, 0, 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                asc_unlock(PIPE_FIX, 4);
            }
        } else {
            for (int64_t i110 = 0; i110 < 20; i110 = (int64_t)((uint64_t)i110 + (uint64_t)1)) {
                __cbuf__ bfloat16_t* view_root111 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf14) + 0);
                __cbuf__ bfloat16_t* view_root112 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf15) + 0);
                int64_t t113 = (int64_t)((uint64_t)t39 / (uint64_t)16);
                int64_t dim114 = gm0_arg.stride[0];
                int64_t t115 = (int64_t)((uint64_t)t113 * (uint64_t)dim114);
                int64_t t116 = i110 * 16;
                int64_t t117 = (int64_t)((uint64_t)t115 + (uint64_t)t116);
                int64_t t118 = (int64_t)((uint64_t)t117 * (uint64_t)16);
                int64_t t119 = t54 / 80;
                int64_t t120 = (t119 < 0 ? t119 : 0);
                int64_t t121 = (int64_t)((uint64_t)t120 * (uint64_t)80);
                int64_t t122 = (int64_t)((uint64_t)t54 - (uint64_t)t121);
                int64_t t123 = (t122 < 80 ? t122 : 80);
                bool cmp124 = t119 >= 0;
                int64_t sel125 = cmp124 ? t123 : 0;
                int64_t dim126 = gm0_arg.stride[0];
                int64_t t127 = dim126;
                int64_t t128 = t127 * 2;
                int64_t t129 = t128;
                asc_lock(PIPE_MTE2, 5);
                bool cmp130 = (uint64_t)sel125 > (uint64_t)0;
                if (cmp130) {
                    int64_t t131 = t118;
                    int32_t t132 = (int32_t)sel125;
                    set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(80) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                    asc_copy_gm2l1_nd2nz(view_root112 + 0, gm0 + t131, t129, 0, t132, 256, 0, false);
                }
                asc_unlock(PIPE_MTE2, 5);
                int64_t dim133 = gm2_arg.stride[0];
                int64_t t134 = (int64_t)((uint64_t)dim133 * (uint64_t)2);
                int64_t t135 = (int64_t)((uint64_t)t52 * (uint64_t)dim133);
                int64_t t136 = (int64_t)((uint64_t)t135 * (uint64_t)8);
                int64_t t137 = i110 * 8192;
                int64_t t138 = (int64_t)((uint64_t)t136 + (uint64_t)t137);
                asc_lock(PIPE_MTE2, 6);
                {
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    union { float f; uint32_t u; } padding_f32_bits139;
                    padding_f32_bits139.f = static_cast<float>(0);
                    uint16_t padding_bf16_bits140 = static_cast<uint16_t>((padding_f32_bits139.u & 0x7fffffffU) > 0x7f800000U ? ((padding_f32_bits139.u >> 16) | 0x40U) : ((padding_f32_bits139.u + 0x7fffU + ((padding_f32_bits139.u >> 16) & 1U)) >> 16));
                    asc_set_gm2l1_pad(static_cast<uint32_t>(padding_bf16_bits140));
                    asc_copy_gm2l1_align((__cbuf__ bfloat16_t*)((__cbuf__ uint8_t*)view_root111 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm2 + t138), 4, 8192, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t134, 8192);
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                }
                asc_unlock(PIPE_MTE2, 6);
                __ca__ bfloat16_t* view_root141 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf12) + 0);
                __cbuf__ bfloat16_t* view_root142 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf15) + 0);
                asc_lock(PIPE_MTE1, 7);
                asc_lock(PIPE_MTE1, 5);
                asc_copy_l12l0a(view_root141 + 0, view_root142 + 0, 0, 0, 5, 16, 5, 5);
                asc_unlock(PIPE_MTE1, 5);
                asc_unlock(PIPE_MTE1, 7);
                __cb__ bfloat16_t* view_root143 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf13) + 0);
                __cbuf__ bfloat16_t* view_root144 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf14) + 0);
                asc_lock(PIPE_MTE1, 8);
                asc_lock(PIPE_MTE1, 6);
                asc_copy_l12l0b_transpose(view_root143 + 0, view_root144 + 0, 0, 0, 16, 4, 16, 4);
                asc_unlock(PIPE_MTE1, 6);
                asc_unlock(PIPE_MTE1, 8);
                bool cmp145 = i110 == 0;
                __cc__ float* view_root146 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf11) + 0);
                __ca__ bfloat16_t* view_root147 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf12) + 0);
                __cb__ bfloat16_t* view_root148 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf13) + 0);
                asc_lock(PIPE_M, 9);
                asc_lock(PIPE_M, 7);
                asc_lock(PIPE_M, 8);
                if (cmp145) {
                    asc_mmad(view_root146 + 0, view_root147, view_root148, 80, 256, 64, 0, true, false, true);
                } else {
                    asc_mmad(view_root146 + 0, view_root147, view_root148, 80, 256, 64, 0, true, false, false);
                }
                asc_unlock(PIPE_M, 8);
                asc_unlock(PIPE_M, 7);
                asc_unlock(PIPE_M, 9);
            }
            int64_t dim149 = gm4_arg.stride[0];
            int64_t t150 = (int64_t)((uint64_t)t39 * (uint64_t)dim149);
            int64_t t151 = t150;
            int64_t t152 = t55;
            int64_t t153 = t151 + t152;
            int64_t dim154 = gm4_arg.stride[0];
            __cc__ float* view_root155 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf11) + 0);
            bool cmp156 = (uint64_t)t54 > (uint64_t)0;
            if (cmp156) {
                int64_t t157 = t153;
                asc_lock(PIPE_FIX, 9);
                int32_t t158 = (int32_t)t54;
                int32_t t159 = (int32_t)dim154;
                int64_t t160 = t157;
                asc_set_l0c2gm_nz2nd(1, 0, 0);
                asc_copy_l0c2gm(gm4 + t160, view_root155, 64, t158, t159, 80, 0, 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                asc_unlock(PIPE_FIX, 9);
            }
        }
    }
}
