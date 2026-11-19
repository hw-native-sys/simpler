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

struct KernelArgTensor_float_3d {
    __gm__ float* ptr;
    int64_t shape[3];
    int64_t stride[3];
};

struct KernelArgTensor_int32_t_1d {
    __gm__ int32_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_bfloat16_t_1d {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_bfloat16_t_2d {
    __gm__ bfloat16_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_int32_t_2d {
    __gm__ int32_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_float_3d_Device_dss_dds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[2];
};
static inline KernelArgTensor_float_3d_Device_dss_dds cannbotdsl_pack_KernelArgTensor_float_3d_Device_dss_dds(const KernelArgTensor_float_3d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0], src.stride[1]}};
}

struct KernelArgTensor_int32_t_1d_Device_d_s {
    __gm__ int32_t* ptr;
    int64_t shape[1];
};
static inline KernelArgTensor_int32_t_1d_Device_d_s cannbotdsl_pack_KernelArgTensor_int32_t_1d_Device_d_s(const KernelArgTensor_int32_t_1d& src) {
    return {src.ptr, {src.shape[0]}};
}

struct KernelArgTensor_bfloat16_t_1d_Device_s_s {
    __gm__ bfloat16_t* ptr;
};
static inline KernelArgTensor_bfloat16_t_1d_Device_s_s cannbotdsl_pack_KernelArgTensor_bfloat16_t_1d_Device_s_s(const KernelArgTensor_bfloat16_t_1d& src) {
    return {src.ptr};
}

struct KernelArgTensor_bfloat16_t_2d_Device_ds_ds {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ds_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_int32_t_2d_Device_ds_ds {
    __gm__ int32_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_ds_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

__simd_vf__ inline void kernel_pool_state_grid_vf0(__ubuf__ float* buf0, __ubuf__ float* buf1, __ubuf__ float* buf2, __ubuf__ float* buf3, __ubuf__ bfloat16_t* buf4)
{
    vector_bool vmask;
    vector_bool mask5 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i6 = 0; i6 < 512; i6 = (int32_t)((uint32_t)i6 + (uint32_t)64)) {
        int32_t t7 = i6;
        vector_float v8;
        asc_loadalign(v8, buf0 + t7);
        vector_float v9;
        asc_loadalign(v9, buf1 + t7);
        vector_float v10;
        asc_max(v10, v8, v9, mask5);
        vector_float v11;
        asc_sub(v11, v8, v10, mask5);
        vector_float v12;
        asc_exp(v12, v11, mask5);
        vector_float v13;
        asc_sub(v13, v9, v10, mask5);
        vector_float v14;
        asc_exp(v14, v13, mask5);
        vector_float v15;
        asc_add(v15, v12, v14, mask5);
        vector_float v16;
        asc_div(v16, v12, v15, mask5);
        vector_float v17;
        asc_div(v17, v14, v15, mask5);
        vector_float v18;
        asc_loadalign(v18, buf2 + t7);
        vector_float v19;
        asc_mul(v19, v18, v16, mask5);
        vector_float v20;
        asc_loadalign(v20, buf3 + t7);
        vector_float v21;
        asc_mul(v21, v20, v17, mask5);
        vector_float v22;
        asc_add(v22, v19, v21, mask5);
        vector_bfloat16_t v23;
        asc_float2bfloat16_rn(v23, v22, mask5, ASC_POSITION_EVEN);
        asc_storealign_pack((__ubuf__ uint32_t*)(buf4 + t7), *(vector_uint32_t*)&v23, mask5);
    }
}

__simd_vf__ inline void kernel_pool_state_grid_vf1(__ubuf__ bfloat16_t* buf24, __ubuf__ bfloat16_t* buf25, __ubuf__ bfloat16_t* buf26)
{
    vector_bool vmask;
    float c27 = 1.0f;
    float c28 = 9.99999997E-7f;
    float c29 = 0.001953125f;
    float c30 = 0.0f;
    vector_bool mask31 = asc_create_mask_b32(PAT_ALL);
    vector_float v32;
    asc_duplicate_scalar(v32, c30, mask31);
    vector_float iter33 = v32;
    for (int32_t i34 = 0; i34 < 512; i34 = (int32_t)((uint32_t)i34 + (uint32_t)64)) {
        int32_t t35 = i34;
        vector_float v36;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v36, (__ubuf__ bfloat16_t*)(buf24 + t35));
        vector_float v37;
        asc_bfloat162float(v37, *(vector_bfloat16_t*)&v36, mask31);
        vector_float v38;
        asc_mul(v38, v37, v37, mask31);
        vector_float v39;
        asc_add(v39, iter33, v38, mask31);
        iter33 = v39;
    }
    vector_float v40;
    asc_reduce_sum(v40, iter33, mask31);
    vector_float v41;
    asc_mul_scalar(v41, v40, c29, mask31);
    vector_float v42;
    asc_add_scalar(v42, v41, c28, mask31);
    vector_float v43;
    asc_sqrt(v43, v42, mask31);
    vector_float v44;
    asc_duplicate_scalar(v44, c27, mask31);
    vector_float v45;
    asc_div(v45, v44, v43, mask31);
    vector_float v46;
    asc_duplicate(v46, v45, mask31);
    for (int32_t i47 = 0; i47 < 512; i47 = (int32_t)((uint32_t)i47 + (uint32_t)64)) {
        int32_t t48 = i47;
        vector_float v49;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v49, (__ubuf__ bfloat16_t*)(buf24 + t48));
        vector_float v50;
        asc_bfloat162float(v50, *(vector_bfloat16_t*)&v49, mask31);
        vector_float v51;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v51, (__ubuf__ bfloat16_t*)(buf25 + t48));
        vector_float v52;
        asc_bfloat162float(v52, *(vector_bfloat16_t*)&v51, mask31);
        vector_float v53;
        asc_mul(v53, v50, v46, mask31);
        vector_float v54;
        asc_mul(v54, v53, v52, mask31);
        vector_bfloat16_t v55;
        asc_float2bfloat16_rn(v55, v54, mask31, ASC_POSITION_EVEN);
        asc_storealign_pack((__ubuf__ uint32_t*)(buf26 + t48), *(vector_uint32_t*)&v55, mask31);
    }
}

static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_3d_Device_dss_dds) == 32 && alignof(KernelArgTensor_float_3d_Device_dss_dds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8 && alignof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_pool_state_grid(KernelArgTensor_float_2d_Device_ds_ds gm56_arg, KernelArgTensor_float_2d_Device_ds_ds gm57_arg, KernelArgTensor_float_3d_Device_dss_dds gm58_arg, KernelArgTensor_int32_t_1d_Device_d_s gm59_arg, KernelArgTensor_bfloat16_t_1d_Device_s_s gm60_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm61_arg, KernelArgTensor_int32_t_1d_Device_d_s gm62_arg, KernelArgTensor_int32_t_1d_Device_d_s gm63_arg, KernelArgTensor_int32_t_2d_Device_ds_ds gm64_arg, int64_t arg65, int64_t arg66, int32_t arg67)
{
    __gm__ float* gm56 = gm56_arg.ptr;
    __gm__ float* gm57 = gm57_arg.ptr;
    __gm__ float* gm58 = gm58_arg.ptr;
    __gm__ int32_t* gm59 = gm59_arg.ptr;
    __gm__ bfloat16_t* gm60 = gm60_arg.ptr;
    __gm__ bfloat16_t* gm61 = gm61_arg.ptr;
    __gm__ int32_t* gm62 = gm62_arg.ptr;
    __gm__ int32_t* gm63 = gm63_arg.ptr;
    __gm__ int32_t* gm64 = gm64_arg.ptr;
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

    __ubuf__ float* buf68 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf69 = (__ubuf__ float*)(uintptr_t)(uint64_t)(2048);
    __ubuf__ float* buf70 = (__ubuf__ float*)(uintptr_t)(uint64_t)(4096);
    __ubuf__ float* buf71 = (__ubuf__ float*)(uintptr_t)(uint64_t)(6144);
    __ubuf__ bfloat16_t* buf72 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(8192);
    __ubuf__ bfloat16_t* buf73 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(9216);
    __ubuf__ bfloat16_t* buf74 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(10240);

    vector_bool vmask;
    int64_t dim75 = gm56_arg.shape[0];
    int32_t t76 = (arg67 > 0 ? arg67 : 0);
    int64_t t77 = (int64_t)t76;
    int64_t t78 = (dim75 < t77 ? dim75 : t77);
    int64_t dim79 = gm58_arg.shape[0];
    for (int64_t i80 = arg65; i80 < dim79; i80 = (int64_t)((uint64_t)i80 + (uint64_t)arg66)) {
        asc_lock(PIPE_MTE2, 4);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf72 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm60 + 0), 1, 1024, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 4);
        int64_t dim81 = gm64_arg.shape[0];
        for (int64_t i82 = 0; i82 < dim81; i82 = (int64_t)((uint64_t)i82 + (uint64_t)1)) {
            int64_t dim83 = gm64_arg.stride[0];
            int64_t t84 = dim83;
            int64_t t85 = i82;
            int64_t t86 = t85 * t84;
            int32_t ld87 = gm64[t86];
            int64_t t88 = (int64_t)ld87;
            bool cmp89 = t88 == i80;
            if (cmp89) {
                int32_t ld90 = gm62[t85];
                int32_t t91 = (ld90 > 0 ? ld90 : 0);
                int64_t t92 = (int64_t)((uint64_t)i82 + (uint64_t)1);
                int64_t t93 = t92;
                int32_t ld94 = gm62[t93];
                int64_t t95 = (int64_t)ld94;
                int64_t t96 = (t95 < t78 ? t95 : t78);
                int64_t t97 = (int64_t)t91;
                for (int64_t i98 = t97; i98 < t96; i98 = (int64_t)((uint64_t)i98 + (uint64_t)1)) {
                    int64_t t99 = i98;
                    int32_t ld100 = gm59[t99];
                    int32_t ld101 = gm63[t99];
                    int64_t t102 = (int64_t)ld101;
                    bool cmp103 = t102 == i82;
                    bool cmp104 = ld100 >= 0;
                    int32_t t105 = ld100 % 2;
                    bool cmp106 = t105 == 1;
                    bool sel107 = cmp104 ? cmp106 : false;
                    bool sel108 = cmp103 ? sel107 : false;
                    if (sel108) {
                        int64_t t109 = (i98 < dim75 ? i98 : dim75);
                        int64_t t110 = (int64_t)((uint64_t)dim75 - (uint64_t)t109);
                        int64_t t111 = (t110 < 1 ? t110 : 1);
                        bool cmp112 = i98 <= dim75;
                        int64_t sel113 = cmp112 ? t111 : 0;
                        int64_t dim114 = gm56_arg.stride[0];
                        int64_t t115 = (int64_t)((uint64_t)dim114 * (uint64_t)4);
                        int64_t t116 = (int64_t)((uint64_t)i98 * (uint64_t)dim114);
                        int64_t t117 = (int64_t)((uint64_t)t116 * (uint64_t)4);
                        asc_lock(PIPE_MTE2, 0);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf68 + 0), (__gm__ float*)((__gm__ uint8_t*)gm56 + t117), sel113, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t115, 2048);
                        }
                        asc_unlock(PIPE_MTE2, 0);
                        int64_t dim118 = gm57_arg.shape[0];
                        int64_t t119 = (i98 < dim118 ? i98 : dim118);
                        int64_t t120 = (int64_t)((uint64_t)dim118 - (uint64_t)t119);
                        int64_t t121 = (t120 < 1 ? t120 : 1);
                        bool cmp122 = i98 <= dim118;
                        int64_t sel123 = cmp122 ? t121 : 0;
                        int64_t dim124 = gm57_arg.stride[0];
                        int64_t t125 = (int64_t)((uint64_t)dim124 * (uint64_t)4);
                        int64_t t126 = (int64_t)((uint64_t)i98 * (uint64_t)dim124);
                        int64_t t127 = (int64_t)((uint64_t)t126 * (uint64_t)4);
                        asc_lock(PIPE_MTE2, 1);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf69 + 0), (__gm__ float*)((__gm__ uint8_t*)gm57 + t127), sel123, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t125, 2048);
                        }
                        asc_unlock(PIPE_MTE2, 1);
                        bool cmp128 = i98 > t97;
                        if (cmp128) {
                            int64_t t129 = (int64_t)((uint64_t)i98 - (uint64_t)1);
                            int64_t t130 = (t129 < dim75 ? t129 : dim75);
                            int64_t t131 = (int64_t)((uint64_t)dim75 - (uint64_t)t130);
                            int64_t t132 = (t131 < 1 ? t131 : 1);
                            bool cmp133 = t129 <= dim75;
                            int64_t sel134 = cmp133 ? t132 : 0;
                            int64_t t135 = (int64_t)((uint64_t)t129 * (uint64_t)dim114);
                            int64_t t136 = (int64_t)((uint64_t)t135 * (uint64_t)4);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf70 + 0), (__gm__ float*)((__gm__ uint8_t*)gm56 + t136), sel134, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t115, 2048);
                            }
                            int64_t t137 = (t129 < dim118 ? t129 : dim118);
                            int64_t t138 = (int64_t)((uint64_t)dim118 - (uint64_t)t137);
                            int64_t t139 = (t138 < 1 ? t138 : 1);
                            bool cmp140 = t129 <= dim118;
                            int64_t sel141 = cmp140 ? t139 : 0;
                            int64_t t142 = (int64_t)((uint64_t)t129 * (uint64_t)dim124);
                            int64_t t143 = (int64_t)((uint64_t)t142 * (uint64_t)4);
                            asc_lock(PIPE_MTE2, 3);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf71 + 0), (__gm__ float*)((__gm__ uint8_t*)gm57 + t143), sel141, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t125, 2048);
                            }
                            asc_unlock(PIPE_MTE2, 3);
                        } else {
                            int32_t t144 = (int32_t)((uint32_t)ld100 - (uint32_t)1);
                            int32_t t145 = t144 % 8;
                            bool cmp146 = t145 != 0;
                            bool cmp147 = t145 < 0;
                            bool t148 = cmp146 & cmp147;
                            int32_t sel149 = t148 ? 8 : 0;
                            int32_t t150 = t145 + sel149;
                            int64_t t151 = (int64_t)t150;
                            int64_t dim152 = gm58_arg.stride[0];
                            int64_t t153 = (int64_t)((uint64_t)i80 * (uint64_t)dim152);
                            int64_t t154 = (int64_t)((uint64_t)t153 * (uint64_t)4);
                            int64_t t155 = t151 * 4096;
                            int64_t t156 = (int64_t)((uint64_t)t154 + (uint64_t)t155);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf70 + 0), (__gm__ float*)((__gm__ uint8_t*)gm58 + t156), 1, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                            }
                            int64_t t157 = (int64_t)((uint64_t)t156 + (uint64_t)2048);
                            asc_lock(PIPE_MTE2, 3);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf71 + 0), (__gm__ float*)((__gm__ uint8_t*)gm58 + t157), 1, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                            }
                            asc_unlock(PIPE_MTE2, 3);
                        }
                        asc_lock(PIPE_V, 3);
                        asc_lock(PIPE_V, 0);
                        kernel_pool_state_grid_vf0((__ubuf__ float*)buf71, (__ubuf__ float*)buf69, (__ubuf__ float*)buf70, (__ubuf__ float*)buf68, (__ubuf__ bfloat16_t*)buf73);
                        asc_unlock(PIPE_V, 0);
                        asc_unlock(PIPE_V, 3);
                        asc_lock(PIPE_V, 4);
                        asc_lock(PIPE_V, 5);
                        kernel_pool_state_grid_vf1((__ubuf__ bfloat16_t*)buf73, (__ubuf__ bfloat16_t*)buf72, (__ubuf__ bfloat16_t*)buf74);
                        asc_unlock(PIPE_V, 5);
                        asc_unlock(PIPE_V, 4);
                        int64_t dim158 = gm61_arg.stride[0];
                        int64_t t159 = (int64_t)((uint64_t)i98 * (uint64_t)dim158);
                        int64_t t160 = (int64_t)((uint64_t)t159 * (uint64_t)16);
                        int64_t t161 = (int64_t)((uint64_t)t160 / (uint64_t)8);
                        int64_t t162 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm61) + (uint64_t)t161);
                        __gm__ bfloat16_t* root163 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t162);
                        __ubuf__ bfloat16_t* view_root164 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf74) + 0);
                        int64_t dim165 = gm61_arg.stride[0];
                        int64_t t166 = dim165;
                        int64_t t167 = t166 * 2;
                        int64_t t168 = t167;
                        asc_lock(PIPE_MTE3, 5);
                        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root163 + 0, (__ubuf__ bfloat16_t*)view_root164 + 0, 1, 1024, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        asc_unlock(PIPE_MTE3, 5);
                    }
                }
            }
        }
        for (int64_t i169 = 0; i169 < dim81; i169 = (int64_t)((uint64_t)i169 + (uint64_t)1)) {
            int64_t dim170 = gm64_arg.stride[0];
            int64_t t171 = dim170;
            int64_t t172 = i169;
            int64_t t173 = t172 * t171;
            int32_t ld174 = gm64[t173];
            int64_t t175 = (int64_t)ld174;
            bool cmp176 = t175 == i80;
            if (cmp176) {
                int64_t t177 = (int64_t)((uint64_t)i169 + (uint64_t)1);
                int64_t t178 = t177;
                int32_t ld179 = gm62[t178];
                int64_t t180 = (int64_t)ld179;
                int64_t t181 = (t180 < t78 ? t180 : t78);
                int32_t ld182 = gm62[t172];
                int32_t t183 = (ld182 > 0 ? ld182 : 0);
                int64_t t184 = (int64_t)((uint64_t)t181 - (uint64_t)8);
                int64_t t185 = (int64_t)t183;
                int64_t t186 = (t185 > t184 ? t185 : t184);
                for (int64_t i187 = t186; i187 < t181; i187 = (int64_t)((uint64_t)i187 + (uint64_t)1)) {
                    int64_t t188 = i187;
                    int32_t ld189 = gm59[t188];
                    int32_t ld190 = gm63[t188];
                    int64_t t191 = (int64_t)ld190;
                    bool cmp192 = t191 == i169;
                    bool cmp193 = ld189 >= 0;
                    bool sel194 = cmp192 ? cmp193 : false;
                    if (sel194) {
                        int64_t t195 = (i187 < dim75 ? i187 : dim75);
                        int64_t t196 = (int64_t)((uint64_t)dim75 - (uint64_t)t195);
                        int64_t t197 = (t196 < 1 ? t196 : 1);
                        bool cmp198 = i187 <= dim75;
                        int64_t sel199 = cmp198 ? t197 : 0;
                        int64_t dim200 = gm56_arg.stride[0];
                        int64_t t201 = (int64_t)((uint64_t)dim200 * (uint64_t)4);
                        int64_t t202 = (int64_t)((uint64_t)i187 * (uint64_t)dim200);
                        int64_t t203 = (int64_t)((uint64_t)t202 * (uint64_t)4);
                        asc_lock(PIPE_MTE2, 0);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf68 + 0), (__gm__ float*)((__gm__ uint8_t*)gm56 + t203), sel199, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t201, 2048);
                        }
                        asc_unlock(PIPE_MTE2, 0);
                        int64_t dim204 = gm57_arg.shape[0];
                        int64_t t205 = (i187 < dim204 ? i187 : dim204);
                        int64_t t206 = (int64_t)((uint64_t)dim204 - (uint64_t)t205);
                        int64_t t207 = (t206 < 1 ? t206 : 1);
                        bool cmp208 = i187 <= dim204;
                        int64_t sel209 = cmp208 ? t207 : 0;
                        int64_t dim210 = gm57_arg.stride[0];
                        int64_t t211 = (int64_t)((uint64_t)dim210 * (uint64_t)4);
                        int64_t t212 = (int64_t)((uint64_t)i187 * (uint64_t)dim210);
                        int64_t t213 = (int64_t)((uint64_t)t212 * (uint64_t)4);
                        asc_lock(PIPE_MTE2, 1);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf69 + 0), (__gm__ float*)((__gm__ uint8_t*)gm57 + t213), sel209, 2048, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t211, 2048);
                        }
                        asc_unlock(PIPE_MTE2, 1);
                        int32_t t214 = ld189 % 8;
                        int64_t t215 = (int64_t)t214;
                        int64_t dim216 = gm58_arg.stride[0];
                        int64_t t217 = (int64_t)((uint64_t)i80 * (uint64_t)dim216);
                        int64_t t218 = (int64_t)((uint64_t)t217 * (uint64_t)32);
                        int64_t t219 = t215 * 32768;
                        int64_t t220 = (int64_t)((uint64_t)t218 + (uint64_t)t219);
                        int64_t t221 = (int64_t)((uint64_t)t220 / (uint64_t)8);
                        int64_t t222 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm58) + (uint64_t)t221);
                        __gm__ float* root223 = (__gm__ float*)(uintptr_t)(uint64_t)(t222);
                        __ubuf__ float* view_root224 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf68) + 0);
                        asc_lock(PIPE_MTE3, 0);
                        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root223 + 0, (__ubuf__ float*)view_root224 + 0, 1, 2048, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        asc_unlock(PIPE_MTE3, 0);
                        int64_t t225 = t215 * 1024;
                        int64_t t226 = t225 + 512;
                        int64_t t227 = t226 * 32;
                        int64_t t228 = (int64_t)((uint64_t)t218 + (uint64_t)t227);
                        int64_t t229 = (int64_t)((uint64_t)t228 / (uint64_t)8);
                        int64_t t230 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm58) + (uint64_t)t229);
                        __gm__ float* root231 = (__gm__ float*)(uintptr_t)(uint64_t)(t230);
                        __ubuf__ float* view_root232 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf69) + 0);
                        asc_lock(PIPE_MTE3, 1);
                        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root231 + 0, (__ubuf__ float*)view_root232 + 0, 1, 2048, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        asc_unlock(PIPE_MTE3, 1);
                    }
                }
            }
        }
    }
}
