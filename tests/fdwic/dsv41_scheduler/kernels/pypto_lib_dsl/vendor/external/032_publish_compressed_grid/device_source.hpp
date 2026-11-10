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

struct KernelArgTensor_int32_t_1d {
    __gm__ int32_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_int32_t_2d {
    __gm__ int32_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_int64_t_1d {
    __gm__ int64_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_uint8_t_2d {
    __gm__ uint8_t* ptr;
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

struct KernelArgTensor_int32_t_1d_Device_d_s {
    __gm__ int32_t* ptr;
    int64_t shape[1];
};
static inline KernelArgTensor_int32_t_1d_Device_d_s cannbotdsl_pack_KernelArgTensor_int32_t_1d_Device_d_s(const KernelArgTensor_int32_t_1d& src) {
    return {src.ptr, {src.shape[0]}};
}

struct KernelArgTensor_int32_t_2d_Device_ds_ds {
    __gm__ int32_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_ds_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_int64_t_1d_Device_d_s {
    __gm__ int64_t* ptr;
    int64_t shape[1];
};
static inline KernelArgTensor_int64_t_1d_Device_d_s cannbotdsl_pack_KernelArgTensor_int64_t_1d_Device_d_s(const KernelArgTensor_int64_t_1d& src) {
    return {src.ptr, {src.shape[0]}};
}

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_uint8_t_2d_Device_ds_ds {
    __gm__ uint8_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_uint8_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_uint8_t_2d_Device_ds_ds(const KernelArgTensor_uint8_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

__simd_vf__ inline void kernel_publish_compressed_grid_vf0(__ubuf__ bfloat16_t* buf0, __ubuf__ float* buf1, __ubuf__ float* buf2, __ubuf__ float* buf3, __ubuf__ bfloat16_t* buf4, __ubuf__ float* buf5, __ubuf__ uint8_t* buf6)
{
    vector_bool vmask;
    float c7 = 5.0f;
    float c8 = 3.5f;
    float c9 = 2.5f;
    float c10 = 1.75f;
    float c11 = 1.25f;
    float c12 = 0.75f;
    float c13 = 0.25f;
    float c14 = 0.001953125f;
    float c15 = 0.166666672f;
    vector_bool mask16 = asc_create_mask_b32(PAT_ALL);
    uint32_t rem18 = 32;
    vector_bool mask17 = asc_update_mask_b32(rem18);
    uint32_t rem20 = 16;
    vector_bool mask19 = asc_update_mask_b32(rem20);
    for (int32_t i21 = 0; i21 < 512; i21 = (int32_t)((uint32_t)i21 + (uint32_t)64)) {
        int32_t t22 = i21;
        vector_float v23;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v23, (__ubuf__ bfloat16_t*)(buf0 + t22));
        vector_float v24;
        asc_bfloat162float(v24, *(vector_bfloat16_t*)&v23, mask16);
        asc_storealign(buf1 + t22, v24, mask16);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i25 = 0; i25 < 1; i25 = (int32_t)((uint32_t)i25 + (uint32_t)1)) {
        vector_float v26;
        vector_float v27;
        asc_loadalign_deintlv(v26, v27, (__ubuf__ float*)(buf1 + 448));
        vector_float v28;
        asc_loadalign(v28, buf2 + 0);
        vector_float v29;
        asc_loadalign(v29, buf3 + 0);
        vector_float v30;
        asc_mul(v30, v26, v28, mask17);
        vector_float v31;
        asc_mul(v31, v27, v29, mask17);
        vector_float v32;
        asc_mul(v32, v26, v29, mask17);
        vector_float v33;
        asc_mul(v33, v27, v28, mask17);
        vector_float v34;
        asc_sub(v34, v30, v31, mask17);
        vector_float v35;
        asc_add(v35, v32, v33, mask17);
        vector_bfloat16_t v36;
        asc_float2bfloat16_rn(v36, v34, mask17, ASC_POSITION_EVEN);
        vector_bfloat16_t v37;
        asc_float2bfloat16_rn(v37, v35, mask17, ASC_POSITION_EVEN);
        vector_float v38;
        asc_bfloat162float(v38, *(vector_bfloat16_t*)&v36, mask17);
        vector_float v39;
        asc_bfloat162float(v39, *(vector_bfloat16_t*)&v37, mask17);
        vector_uint32_t &r40 = (vector_uint32_t &)v38;
        vector_uint32_t &r41 = (vector_uint32_t &)v39;
        asc_storealign_intlv((__ubuf__ uint32_t*)(buf1 + 448), r40, r41);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i42 = 0; i42 < 32; i42 = (int32_t)((uint32_t)i42 + (uint32_t)1)) {
        int32_t t43 = i42 * 16;
        int32_t t44 = t43;
        vector_float v45;
        asc_loadalign(v45, buf1 + t44);
        vector_float v46;
        asc_abs(v46, v45, mask19);
        vector_float v47;
        asc_reduce_max(v47, v46, mask19);
        vector_float v48;
        asc_mul_scalar(v48, v47, c15, mask19);
        vector_float v49;
        asc_max_scalar(v49, v48, c14, mask19);
        vector_bfloat16_t v50;
        asc_float2bfloat16_rn(v50, v49, mask19, ASC_POSITION_EVEN);
        int32_t t51 = i42;
        asc_storealign_1st(buf4 + t51, v50);
        vector_float v52;
        asc_bfloat162float(v52, *(vector_bfloat16_t*)&v50, mask19);
        vector_float v53;
        asc_duplicate(v53, v52, mask19);
        vector_float v54;
        asc_div(v54, v45, v53, mask19);
        asc_storealign(buf5 + t44, v54, mask19);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i55 = 0; i55 < 512; i55 = (int32_t)((uint32_t)i55 + (uint32_t)64)) {
        int32_t t56 = i55;
        vector_float v57;
        asc_loadalign(v57, buf5 + t56);
        vector_float v58;
        asc_abs(v58, v57, mask16);
        vector_uint32_t v59;
        asc_duplicate_scalar(v59, 0, mask16);
        vector_bool m60;
        asc_gt_scalar(m60, v58, c13, mask16);
        vector_uint32_t v61;
        asc_duplicate_scalar(v61, 1, mask16);
        vector_uint32_t v62;
        asc_select(v62, v61, v59, m60);
        vector_bool m63;
        asc_gt_scalar(m63, v58, c12, mask16);
        vector_uint32_t v64;
        asc_duplicate_scalar(v64, 2, mask16);
        vector_uint32_t v65;
        asc_select(v65, v64, v62, m63);
        vector_bool m66;
        asc_gt_scalar(m66, v58, c11, mask16);
        vector_uint32_t v67;
        asc_duplicate_scalar(v67, 3, mask16);
        vector_uint32_t v68;
        asc_select(v68, v67, v65, m66);
        vector_bool m69;
        asc_gt_scalar(m69, v58, c10, mask16);
        vector_uint32_t v70;
        asc_duplicate_scalar(v70, 4, mask16);
        vector_uint32_t v71;
        asc_select(v71, v70, v68, m69);
        vector_bool m72;
        asc_gt_scalar(m72, v58, c9, mask16);
        vector_uint32_t v73;
        asc_duplicate_scalar(v73, 5, mask16);
        vector_uint32_t v74;
        asc_select(v74, v73, v71, m72);
        vector_bool m75;
        asc_gt_scalar(m75, v58, c8, mask16);
        vector_uint32_t v76;
        asc_duplicate_scalar(v76, 6, mask16);
        vector_uint32_t v77;
        asc_select(v77, v76, v74, m75);
        vector_bool m78;
        asc_gt_scalar(m78, v58, c7, mask16);
        vector_uint32_t v79;
        asc_duplicate_scalar(v79, 7, mask16);
        vector_uint32_t v80;
        asc_select(v80, v79, v77, m78);
        vector_uint32_t &r81 = (vector_uint32_t &)v57;
        vector_uint32_t v82;
        asc_shiftright_scalar(v82, r81, 28, mask16);
        vector_uint32_t v83;
        asc_duplicate_scalar(v83, 8, mask16);
        vector_uint32_t v84;
        asc_and(v84, v82, v83, mask16);
        vector_uint32_t v85;
        asc_or(v85, v80, v84, mask16);
        vector_uint32_t v86;
        vector_uint32_t v87;
        asc_deintlv(v86, v87, v85, v85);
        vector_uint32_t v88;
        asc_shiftleft_scalar(v88, v87, 4, mask17);
        vector_uint32_t v89;
        asc_or(v89, v86, v88, mask17);
        int32_t t90 = i55 / 2;
        int32_t t91 = t90;
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf6 + t91), *(vector_float*)&v89, mask17);
    }
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int64_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int64_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_publish_compressed_grid(KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm92_arg, KernelArgTensor_int32_t_1d_Device_d_s gm93_arg, KernelArgTensor_int32_t_1d_Device_d_s gm94_arg, KernelArgTensor_int32_t_1d_Device_d_s gm95_arg, KernelArgTensor_int32_t_2d_Device_ds_ds gm96_arg, KernelArgTensor_int64_t_1d_Device_d_s gm97_arg, KernelArgTensor_float_2d_Device_ds_ds gm98_arg, KernelArgTensor_float_2d_Device_ds_ds gm99_arg, KernelArgTensor_uint8_t_2d_Device_ds_ds gm100_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm101_arg, int64_t arg102, int64_t arg103, int64_t arg104, int64_t arg105)
{
    __gm__ bfloat16_t* gm92 = gm92_arg.ptr;
    __gm__ int32_t* gm93 = gm93_arg.ptr;
    __gm__ int32_t* gm94 = gm94_arg.ptr;
    __gm__ int32_t* gm95 = gm95_arg.ptr;
    __gm__ int32_t* gm96 = gm96_arg.ptr;
    __gm__ int64_t* gm97 = gm97_arg.ptr;
    __gm__ float* gm98 = gm98_arg.ptr;
    __gm__ float* gm99 = gm99_arg.ptr;
    __gm__ uint8_t* gm100 = gm100_arg.ptr;
    __gm__ bfloat16_t* gm101 = gm101_arg.ptr;
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

    __ubuf__ bfloat16_t* buf106 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf107 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1024);
    __ubuf__ float* buf108 = (__ubuf__ float*)(uintptr_t)(uint64_t)(3328);
    __ubuf__ float* buf109 = (__ubuf__ float*)(uintptr_t)(uint64_t)(5376);
    __ubuf__ float* buf110 = (__ubuf__ float*)(uintptr_t)(uint64_t)(5632);
    __ubuf__ uint8_t* buf111 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(5888);
    __ubuf__ bfloat16_t* buf112 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(6144);

    vector_bool vmask;
    int64_t dim113 = gm100_arg.shape[0];
    int64_t t114 = dim113 % 128;
    bool cmp115 = t114 != 0;
    bool cmp116 = t114 < 0;
    bool t117 = cmp115 & cmp116;
    int64_t t118 = dim113 / 128;
    int64_t t119 = (int64_t)(uint8_t)t117;
    int64_t t120 = (int64_t)((uint64_t)t118 - (uint64_t)t119);
    for (int64_t i121 = arg104; i121 < t120; i121 = (int64_t)((uint64_t)i121 + (uint64_t)arg105)) {
        int64_t t122 = (int64_t)((uint64_t)i121 * (uint64_t)128);
        for (int64_t i123 = 0; i123 < arg102; i123 = (int64_t)((uint64_t)i123 + (uint64_t)1)) {
            int64_t t124 = i123;
            int64_t ld125 = gm97[t124];
            int32_t ld126 = gm95[t124];
            int32_t ld127 = gm93[t124];
            bool cmp128 = ld125 >= 0;
            bool ifr129;
            if (cmp128) {
                int64_t t130 = ld125 / 128;
                bool cmp131 = t130 == i121;
                bool ifr132;
                if (cmp131) {
                    bool cmp133 = ld126 >= 0;
                    bool ifr134;
                    if (cmp133) {
                        int64_t dim135 = gm96_arg.shape[0];
                        int64_t t136 = (int64_t)ld126;
                        bool cmp137 = t136 < dim135;
                        ifr134 = cmp137;
                    } else {
                        ifr134 = false;
                    }
                    ifr132 = ifr134;
                } else {
                    ifr132 = false;
                }
                ifr129 = ifr132;
            } else {
                ifr129 = false;
            }
            if (ifr129) {
                int64_t dim138 = gm96_arg.stride[0];
                int64_t t139 = dim138;
                int64_t t140 = ld126;
                int64_t t141 = t140 * t139;
                int32_t ld142 = gm96[t141];
                int32_t ld143 = gm94[t140];
                int32_t t144 = (int32_t)((uint32_t)ld126 + (uint32_t)1);
                int64_t t145 = t144;
                int32_t ld146 = gm94[t145];
                bool cmp147 = ld142 >= 0;
                int64_t t148 = (int64_t)ld142;
                bool cmp149 = t148 < arg103;
                int64_t t150 = (int64_t)ld143;
                bool cmp151 = i123 >= t150;
                int64_t t152 = (int64_t)ld146;
                bool cmp153 = i123 < t152;
                bool sel154 = cmp151 ? cmp153 : false;
                bool sel155 = cmp149 ? sel154 : false;
                bool sel156 = cmp147 ? sel155 : false;
                if (sel156) {
                    bool cmp157 = ld127 >= 0;
                    int32_t t158 = ld127 % 2;
                    bool cmp159 = t158 == 1;
                    bool sel160 = cmp157 ? cmp159 : false;
                    if (sel160) {
                        int64_t dim161 = gm92_arg.shape[0];
                        int64_t t162 = (i123 < dim161 ? i123 : dim161);
                        int64_t t163 = (int64_t)((uint64_t)dim161 - (uint64_t)t162);
                        int64_t t164 = (t163 < 1 ? t163 : 1);
                        bool cmp165 = i123 <= dim161;
                        int64_t sel166 = cmp165 ? t164 : 0;
                        int64_t dim167 = gm92_arg.stride[0];
                        int64_t t168 = (int64_t)((uint64_t)dim167 * (uint64_t)2);
                        int64_t t169 = (int64_t)((uint64_t)i123 * (uint64_t)dim167);
                        int64_t t170 = (int64_t)((uint64_t)t169 * (uint64_t)2);
                        asc_lock(PIPE_MTE2, 0);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf106 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm92 + t170), sel166, 1024, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t168, 1024);
                        }
                        asc_unlock(PIPE_MTE2, 0);
                        int64_t dim171 = gm98_arg.shape[0];
                        int64_t t172 = (i123 < dim171 ? i123 : dim171);
                        int64_t t173 = (int64_t)((uint64_t)dim171 - (uint64_t)t172);
                        int64_t t174 = (t173 < 1 ? t173 : 1);
                        bool cmp175 = i123 <= dim171;
                        int64_t sel176 = cmp175 ? t174 : 0;
                        int64_t dim177 = gm98_arg.stride[0];
                        int64_t t178 = (int64_t)((uint64_t)dim177 * (uint64_t)4);
                        int64_t t179 = (int64_t)((uint64_t)i123 * (uint64_t)dim177);
                        int64_t t180 = (int64_t)((uint64_t)t179 * (uint64_t)4);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf109 + 0), (__gm__ float*)((__gm__ uint8_t*)gm98 + t180), sel176, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t178, 256);
                        }
                        int64_t dim181 = gm99_arg.shape[0];
                        int64_t t182 = (i123 < dim181 ? i123 : dim181);
                        int64_t t183 = (int64_t)((uint64_t)dim181 - (uint64_t)t182);
                        int64_t t184 = (t183 < 1 ? t183 : 1);
                        bool cmp185 = i123 <= dim181;
                        int64_t sel186 = cmp185 ? t184 : 0;
                        int64_t dim187 = gm99_arg.stride[0];
                        int64_t t188 = (int64_t)((uint64_t)dim187 * (uint64_t)4);
                        int64_t t189 = (int64_t)((uint64_t)i123 * (uint64_t)dim187);
                        int64_t t190 = (int64_t)((uint64_t)t189 * (uint64_t)4);
                        asc_lock(PIPE_MTE2, 2);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf110 + 0), (__gm__ float*)((__gm__ uint8_t*)gm99 + t190), sel186, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t188, 256);
                        }
                        asc_unlock(PIPE_MTE2, 2);
                        asc_lock(PIPE_V, 0);
                        asc_lock(PIPE_V, 2);
                        asc_lock(PIPE_V, 4);
                        asc_lock(PIPE_V, 3);
                        kernel_publish_compressed_grid_vf0((__ubuf__ bfloat16_t*)buf106, (__ubuf__ float*)buf107, (__ubuf__ float*)buf109, (__ubuf__ float*)buf110, (__ubuf__ bfloat16_t*)buf112, (__ubuf__ float*)buf108, (__ubuf__ uint8_t*)buf111);
                        asc_unlock(PIPE_V, 3);
                        asc_unlock(PIPE_V, 4);
                        asc_unlock(PIPE_V, 2);
                        asc_unlock(PIPE_V, 0);
                        int64_t t191 = ld125 % 128;
                        int64_t t192 = (int64_t)((uint64_t)t191 + (uint64_t)t122);
                        int64_t dim193 = gm100_arg.stride[0];
                        int64_t t194 = (int64_t)((uint64_t)t192 * (uint64_t)dim193);
                        int64_t t195 = (int64_t)((uint64_t)t194 * (uint64_t)8);
                        int64_t t196 = (int64_t)((uint64_t)t195 / (uint64_t)8);
                        int64_t t197 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm100) + (uint64_t)t196);
                        __gm__ uint8_t* root198 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t197);
                        __ubuf__ uint8_t* view_root199 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf111) + 0);
                        int64_t dim200 = gm100_arg.stride[0];
                        asc_lock(PIPE_MTE3, 3);
                        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root198 + 0, (__ubuf__ uint8_t*)view_root199 + 0, 1, 256, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        asc_unlock(PIPE_MTE3, 3);
                        int64_t dim201 = gm101_arg.stride[0];
                        int64_t t202 = (int64_t)((uint64_t)t192 * (uint64_t)dim201);
                        int64_t t203 = (int64_t)((uint64_t)t202 * (uint64_t)16);
                        int64_t t204 = (int64_t)((uint64_t)t203 / (uint64_t)8);
                        int64_t t205 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm101) + (uint64_t)t204);
                        __gm__ bfloat16_t* root206 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t205);
                        __ubuf__ bfloat16_t* view_root207 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf112) + 0);
                        int64_t dim208 = gm101_arg.stride[0];
                        int64_t t209 = dim208;
                        int64_t t210 = t209 * 2;
                        int64_t t211 = t210;
                        asc_lock(PIPE_MTE3, 4);
                        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root206 + 0, (__ubuf__ bfloat16_t*)view_root207 + 0, 1, 64, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        asc_unlock(PIPE_MTE3, 4);
                    }
                }
            }
        }
    }
}
