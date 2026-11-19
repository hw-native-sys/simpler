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

struct KernelArgTensor_bfloat16_t_1d {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_int64_t_1d {
    __gm__ int64_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_uint8_t_2d {
    __gm__ uint8_t* ptr;
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

struct KernelArgTensor_bfloat16_t_1d_Device_s_s {
    __gm__ bfloat16_t* ptr;
};
static inline KernelArgTensor_bfloat16_t_1d_Device_s_s cannbotdsl_pack_KernelArgTensor_bfloat16_t_1d_Device_s_s(const KernelArgTensor_bfloat16_t_1d& src) {
    return {src.ptr};
}

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_int64_t_1d_Device_d_s {
    __gm__ int64_t* ptr;
    int64_t shape[1];
};
static inline KernelArgTensor_int64_t_1d_Device_d_s cannbotdsl_pack_KernelArgTensor_int64_t_1d_Device_d_s(const KernelArgTensor_int64_t_1d& src) {
    return {src.ptr, {src.shape[0]}};
}

struct KernelArgTensor_uint8_t_2d_Device_ds_ds {
    __gm__ uint8_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_uint8_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_uint8_t_2d_Device_ds_ds(const KernelArgTensor_uint8_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds {
    __gm__ fp8_e8m0_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds(const KernelArgTensor_fp8_e8m0_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

__simd_vf__ inline void kernel_publish_index_grid_vf0(__ubuf__ bfloat16_t* buf0, __ubuf__ bfloat16_t* buf1, __ubuf__ float* buf2, __ubuf__ float* buf3, __ubuf__ float* buf4, int64_t arg5, __ubuf__ uint8_t* buf6, __ubuf__ uint8_t* buf7)
{
    vector_bool vmask;
    float c8 = 5.0f;
    float c9 = 3.5f;
    float c10 = 2.5f;
    float c11 = 1.75f;
    float c12 = 1.25f;
    float c13 = 0.75f;
    float c14 = 0.25f;
    float c15 = 0.166666672f;
    float c16 = 7.0529661E-38f;
    float c17 = 1.0f;
    float c18 = 9.99999997E-7f;
    float c19 = 0.0078125f;
    vector_bool mask20 = asc_create_mask_b32(PAT_ALL);
    vector_float v21;
    asc_loadalign_unpack(*(vector_bfloat16_t*)&v21, (__ubuf__ bfloat16_t*)(buf0 + 0));
    vector_float v22;
    asc_loadalign_unpack(*(vector_bfloat16_t*)&v22, (__ubuf__ bfloat16_t*)(buf0 + 64));
    vector_float v23;
    asc_bfloat162float(v23, *(vector_bfloat16_t*)&v21, mask20);
    vector_float v24;
    asc_bfloat162float(v24, *(vector_bfloat16_t*)&v22, mask20);
    vector_float v25;
    asc_mul(v25, v23, v23, mask20);
    vector_float v26;
    asc_mul(v26, v24, v24, mask20);
    vector_float v27;
    asc_reduce_sum(v27, v25, mask20);
    vector_float v28;
    asc_reduce_sum(v28, v26, mask20);
    vector_float v29;
    asc_add(v29, v27, v28, mask20);
    vector_float v30;
    asc_mul_scalar(v30, v29, c19, mask20);
    vector_float v31;
    asc_add_scalar(v31, v30, c18, mask20);
    vector_float v32;
    asc_sqrt(v32, v31, mask20);
    vector_float v33;
    asc_duplicate_scalar(v33, c17, mask20);
    vector_float v34;
    asc_div(v34, v33, v32, mask20);
    vector_float v35;
    asc_duplicate(v35, v34, mask20);
    for (int32_t i36 = 0; i36 < 128; i36 = (int32_t)((uint32_t)i36 + (uint32_t)64)) {
        int32_t t37 = i36;
        vector_float v38;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v38, (__ubuf__ bfloat16_t*)(buf0 + t37));
        vector_float v39;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v39, (__ubuf__ bfloat16_t*)(buf1 + t37));
        vector_float v40;
        asc_bfloat162float(v40, *(vector_bfloat16_t*)&v38, mask20);
        vector_float v41;
        asc_bfloat162float(v41, *(vector_bfloat16_t*)&v39, mask20);
        vector_float v42;
        asc_mul(v42, v40, v35, mask20);
        vector_float v43;
        asc_mul(v43, v42, v41, mask20);
        vector_bfloat16_t v44;
        asc_float2bfloat16_rn(v44, v43, mask20, ASC_POSITION_EVEN);
        vector_float v45;
        asc_bfloat162float(v45, *(vector_bfloat16_t*)&v44, mask20);
        asc_storealign(buf2 + t37, v45, mask20);
    }
    asc_mem_bar(VST_VLD);
    uint32_t rem47 = 32;
    vector_bool mask46 = asc_update_mask_b32(rem47);
    uint32_t rem49 = 32;
    vector_bool mask48 = asc_update_mask_b32(rem49);
    vector_float v50;
    vector_float v51;
    asc_loadalign_deintlv(v50, v51, (__ubuf__ float*)(buf2 + 64));
    vector_float v52;
    asc_loadalign(v52, buf3 + 0);
    vector_float v53;
    asc_loadalign(v53, buf4 + 0);
    vector_float v54;
    asc_mul(v54, v50, v52, mask46);
    vector_float v55;
    asc_mul(v55, v51, v53, mask46);
    vector_float v56;
    asc_mul(v56, v50, v53, mask46);
    vector_float v57;
    asc_mul(v57, v51, v52, mask46);
    vector_float v58;
    asc_sub(v58, v54, v55, mask46);
    vector_float v59;
    asc_add(v59, v56, v57, mask46);
    vector_bfloat16_t v60;
    asc_float2bfloat16_rn(v60, v58, mask46, ASC_POSITION_EVEN);
    vector_bfloat16_t v61;
    asc_float2bfloat16_rn(v61, v59, mask46, ASC_POSITION_EVEN);
    vector_float v62;
    asc_bfloat162float(v62, *(vector_bfloat16_t*)&v60, mask46);
    vector_float v63;
    asc_bfloat162float(v63, *(vector_bfloat16_t*)&v61, mask46);
    vector_uint32_t &r64 = (vector_uint32_t &)v62;
    vector_uint32_t &r65 = (vector_uint32_t &)v63;
    asc_storealign_intlv((__ubuf__ uint32_t*)(buf2 + 64), r64, r65);
    asc_mem_bar(VST_VLD);
    for (int32_t i66 = 0; i66 < 4; i66 = (int32_t)((uint32_t)i66 + (uint32_t)1)) {
        int32_t t67 = i66 * 32;
        int32_t t68 = t67;
        vector_float v69;
        asc_loadalign(v69, buf2 + t68);
        vector_float v70;
        asc_abs(v70, v69, mask46);
        vector_float v71;
        asc_reduce_max(v71, v70, mask46);
        vector_float v72;
        asc_max_scalar(v72, v71, c16, mask46);
        vector_float v73;
        asc_mul_scalar(v73, v72, c15, mask46);
        vector_int32_t &r74 = (vector_int32_t &)v73;
        vector_int32_t v75;
        asc_add_scalar(v75, r74, 8388607, mask46);
        vector_int32_t v76;
        asc_shiftright_scalar(v76, v75, 23, mask46);
        vector_int32_t v77;
        asc_duplicate(v77, v76, mask46);
        vector_int32_t v78;
        asc_shiftleft_scalar(v78, v77, 23, mask46);
        vector_float &r79 = (vector_float &)v78;
        vector_uint8_t v80;
        asc_int322uint8(v80, v77, mask46);
        int64_t t81 = (int64_t)i66;
        int64_t t82 = (int64_t)((uint64_t)arg5 + (uint64_t)t81);
        int32_t t83 = t82;
        asc_storealign_1st(buf6 + t83, v80);
        vector_float v84;
        asc_div(v84, v69, r79, mask46);
        asc_storealign(buf2 + t68, v84, mask46);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i85 = 0; i85 < 128; i85 = (int32_t)((uint32_t)i85 + (uint32_t)64)) {
        int32_t t86 = i85;
        vector_float v87;
        asc_loadalign(v87, buf2 + t86);
        vector_float v88;
        asc_abs(v88, v87, mask20);
        vector_uint32_t v89;
        asc_duplicate_scalar(v89, 0, mask20);
        vector_bool m90;
        asc_gt_scalar(m90, v88, c14, mask20);
        vector_uint32_t v91;
        asc_duplicate_scalar(v91, 1, mask20);
        vector_uint32_t v92;
        asc_select(v92, v91, v89, m90);
        vector_bool m93;
        asc_ge_scalar(m93, v88, c13, mask20);
        vector_uint32_t v94;
        asc_duplicate_scalar(v94, 2, mask20);
        vector_uint32_t v95;
        asc_select(v95, v94, v92, m93);
        vector_bool m96;
        asc_gt_scalar(m96, v88, c12, mask20);
        vector_uint32_t v97;
        asc_duplicate_scalar(v97, 3, mask20);
        vector_uint32_t v98;
        asc_select(v98, v97, v95, m96);
        vector_bool m99;
        asc_ge_scalar(m99, v88, c11, mask20);
        vector_uint32_t v100;
        asc_duplicate_scalar(v100, 4, mask20);
        vector_uint32_t v101;
        asc_select(v101, v100, v98, m99);
        vector_bool m102;
        asc_gt_scalar(m102, v88, c10, mask20);
        vector_uint32_t v103;
        asc_duplicate_scalar(v103, 5, mask20);
        vector_uint32_t v104;
        asc_select(v104, v103, v101, m102);
        vector_bool m105;
        asc_ge_scalar(m105, v88, c9, mask20);
        vector_uint32_t v106;
        asc_duplicate_scalar(v106, 6, mask20);
        vector_uint32_t v107;
        asc_select(v107, v106, v104, m105);
        vector_bool m108;
        asc_gt_scalar(m108, v88, c8, mask20);
        vector_uint32_t v109;
        asc_duplicate_scalar(v109, 7, mask20);
        vector_uint32_t v110;
        asc_select(v110, v109, v107, m108);
        vector_uint32_t &r111 = (vector_uint32_t &)v87;
        vector_uint32_t v112;
        asc_shiftright_scalar(v112, r111, 28, mask20);
        vector_uint32_t v113;
        asc_duplicate_scalar(v113, 8);
        vector_uint32_t v114;
        asc_and(v114, v112, v113, mask20);
        vector_uint32_t v115;
        asc_or(v115, v110, v114, mask20);
        vector_uint32_t v116;
        vector_uint32_t v117;
        asc_deintlv(v116, v117, v115, v115);
        vector_uint32_t v118;
        asc_shiftleft_scalar(v118, v117, 4, mask48);
        vector_uint32_t v119;
        asc_or(v119, v116, v118, mask48);
        int32_t t120 = i85 / 2;
        int32_t t121 = t120;
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf7 + t121), *(vector_float*)&v119, mask48);
    }
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8 && alignof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int64_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int64_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_publish_index_grid(KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm122_arg, KernelArgTensor_bfloat16_t_1d_Device_s_s gm123_arg, KernelArgTensor_float_2d_Device_ds_ds gm124_arg, KernelArgTensor_float_2d_Device_ds_ds gm125_arg, KernelArgTensor_int64_t_1d_Device_d_s gm126_arg, KernelArgTensor_uint8_t_2d_Device_ds_ds gm127_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds gm128_arg, int64_t arg129, int64_t arg130, int32_t arg131)
{
    __gm__ bfloat16_t* gm122 = gm122_arg.ptr;
    __gm__ bfloat16_t* gm123 = gm123_arg.ptr;
    __gm__ float* gm124 = gm124_arg.ptr;
    __gm__ float* gm125 = gm125_arg.ptr;
    __gm__ int64_t* gm126 = gm126_arg.ptr;
    __gm__ uint8_t* gm127 = gm127_arg.ptr;
    __gm__ fp8_e8m0_t* gm128 = gm128_arg.ptr;
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

    __ubuf__ bfloat16_t* buf132 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ bfloat16_t* buf133 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(256);
    __ubuf__ float* buf134 = (__ubuf__ float*)(uintptr_t)(uint64_t)(512);
    __ubuf__ float* buf135 = (__ubuf__ float*)(uintptr_t)(uint64_t)(768);
    __ubuf__ float* buf136 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1024);
    __ubuf__ uint8_t* buf137 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(1792);
    __ubuf__ uint8_t* buf138 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(1856);

    vector_bool vmask;
    int64_t dim139 = gm122_arg.shape[0];
    int64_t dim140 = gm126_arg.shape[0];
    int32_t t141 = (arg131 > 0 ? arg131 : 0);
    int64_t t142 = (int64_t)t141;
    int64_t t143 = (dim140 < t142 ? dim140 : t142);
    int64_t t144 = (dim139 < t143 ? dim139 : t143);
    int64_t dim145 = gm127_arg.shape[0];
    int64_t t146 = dim145 % 128;
    bool cmp147 = t146 != 0;
    bool cmp148 = t146 < 0;
    bool t149 = cmp147 & cmp148;
    int64_t t150 = dim145 / 128;
    int64_t t151 = (int64_t)(uint8_t)t149;
    int64_t t152 = (int64_t)((uint64_t)t150 - (uint64_t)t151);
    for (int64_t i153 = arg129; i153 < t152; i153 = (int64_t)((uint64_t)i153 + (uint64_t)arg130)) {
        int64_t t154 = (int64_t)((uint64_t)i153 * (uint64_t)128);
        asc_lock(PIPE_MTE2, 1);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf133 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm123 + 0), 1, 256, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 1);
        int64_t dim155 = gm128_arg.stride[0];
        int64_t t156 = (int64_t)((uint64_t)t154 * (uint64_t)dim155);
        asc_lock(PIPE_MTE2, 5);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf138 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm128 + t156), 1, 512, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 5);
        for (int64_t i157 = 0; i157 < t144; i157 = (int64_t)((uint64_t)i157 + (uint64_t)1)) {
            int64_t t158 = i157;
            int64_t ld159 = gm126[t158];
            bool cmp160 = ld159 >= 0;
            int64_t t161 = ld159 / 128;
            bool cmp162 = t161 == i153;
            bool sel163 = cmp160 ? cmp162 : false;
            if (sel163) {
                int64_t t164 = (i157 < dim139 ? i157 : dim139);
                int64_t t165 = (int64_t)((uint64_t)dim139 - (uint64_t)t164);
                int64_t t166 = (t165 < 1 ? t165 : 1);
                bool cmp167 = i157 <= dim139;
                int64_t sel168 = cmp167 ? t166 : 0;
                int64_t dim169 = gm122_arg.stride[0];
                int64_t t170 = (int64_t)((uint64_t)dim169 * (uint64_t)2);
                int64_t t171 = (int64_t)((uint64_t)i157 * (uint64_t)dim169);
                int64_t t172 = (int64_t)((uint64_t)t171 * (uint64_t)2);
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf132 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm122 + t172), sel168, 256, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t170, 256);
                }
                asc_unlock(PIPE_MTE2, 0);
                int64_t dim173 = gm124_arg.shape[0];
                int64_t t174 = (i157 < dim173 ? i157 : dim173);
                int64_t t175 = (int64_t)((uint64_t)dim173 - (uint64_t)t174);
                int64_t t176 = (t175 < 1 ? t175 : 1);
                bool cmp177 = i157 <= dim173;
                int64_t sel178 = cmp177 ? t176 : 0;
                int64_t dim179 = gm124_arg.stride[0];
                int64_t t180 = (int64_t)((uint64_t)dim179 * (uint64_t)4);
                int64_t t181 = (int64_t)((uint64_t)i157 * (uint64_t)dim179);
                int64_t t182 = (int64_t)((uint64_t)t181 * (uint64_t)4);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf134 + 0), (__gm__ float*)((__gm__ uint8_t*)gm124 + t182), sel178, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t180, 256);
                }
                int64_t dim183 = gm125_arg.shape[0];
                int64_t t184 = (i157 < dim183 ? i157 : dim183);
                int64_t t185 = (int64_t)((uint64_t)dim183 - (uint64_t)t184);
                int64_t t186 = (t185 < 1 ? t185 : 1);
                bool cmp187 = i157 <= dim183;
                int64_t sel188 = cmp187 ? t186 : 0;
                int64_t dim189 = gm125_arg.stride[0];
                int64_t t190 = (int64_t)((uint64_t)dim189 * (uint64_t)4);
                int64_t t191 = (int64_t)((uint64_t)i157 * (uint64_t)dim189);
                int64_t t192 = (int64_t)((uint64_t)t191 * (uint64_t)4);
                asc_lock(PIPE_MTE2, 3);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf135 + 0), (__gm__ float*)((__gm__ uint8_t*)gm125 + t192), sel188, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t190, 256);
                }
                asc_unlock(PIPE_MTE2, 3);
                int64_t t193 = ld159 % 128;
                int64_t t194 = t193 * 4;
                asc_lock(PIPE_V, 0);
                asc_lock(PIPE_V, 1);
                asc_lock(PIPE_V, 3);
                asc_lock(PIPE_V, 4);
                kernel_publish_index_grid_vf0((__ubuf__ bfloat16_t*)buf132, (__ubuf__ bfloat16_t*)buf133, (__ubuf__ float*)buf136, (__ubuf__ float*)buf134, (__ubuf__ float*)buf135, t194, (__ubuf__ uint8_t*)buf138, (__ubuf__ uint8_t*)buf137);
                asc_unlock(PIPE_V, 4);
                asc_unlock(PIPE_V, 3);
                asc_unlock(PIPE_V, 1);
                asc_unlock(PIPE_V, 0);
                int64_t t195 = (int64_t)((uint64_t)t193 + (uint64_t)t154);
                int64_t dim196 = gm127_arg.stride[0];
                int64_t t197 = (int64_t)((uint64_t)t195 * (uint64_t)dim196);
                int64_t t198 = (int64_t)((uint64_t)t197 * (uint64_t)8);
                int64_t t199 = (int64_t)((uint64_t)t198 / (uint64_t)8);
                int64_t t200 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm127) + (uint64_t)t199);
                __gm__ uint8_t* root201 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t200);
                __ubuf__ uint8_t* view_root202 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf137) + 0);
                int64_t dim203 = gm127_arg.stride[0];
                asc_lock(PIPE_MTE3, 4);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root201 + 0, (__ubuf__ uint8_t*)view_root202 + 0, 1, 64, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 4);
            }
        }
        int64_t t204 = (int64_t)((uint64_t)t156 * (uint64_t)8);
        int64_t t205 = (int64_t)((uint64_t)t204 / (uint64_t)8);
        int64_t t206 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm128) + (uint64_t)t205);
        __gm__ uint8_t* root207 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t206);
        __ubuf__ uint8_t* view_root208 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf138) + 0);
        asc_lock(PIPE_MTE3, 5);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root207 + 0, (__ubuf__ uint8_t*)view_root208 + 0, 1, 512, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 5);
    }
}
