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

struct KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds(const KernelArgTensor_fp8_e4m3fn_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds(const KernelArgTensor_fp8_e4m3fn_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

struct KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds {
    __gm__ fp8_e8m0_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds cannbotdsl_pack_KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds(const KernelArgTensor_fp8_e8m0_t_2d& src) {
    return {src.ptr, {src.shape[1]}, {src.stride[0]}};
}

struct KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds {
    __gm__ fp8_e8m0_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds(const KernelArgTensor_fp8_e8m0_t_2d& src) {
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

static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds) == 24 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__simd_vf__ inline void kernel_project_query_grid_vf0(int64_t arg12, __ubuf__ float* buf13, __ubuf__ float* buf14, __ubuf__ float* buf15, __ubuf__ float* buf16, __ubuf__ uint8_t* buf17, __ubuf__ uint8_t* buf18)
{
    vector_bool vmask;
    float c19 = 5.0f;
    float c20 = 3.5f;
    float c21 = 2.5f;
    float c22 = 1.75f;
    float c23 = 1.25f;
    float c24 = 0.75f;
    float c25 = 0.25f;
    float c26 = 0.166666672f;
    float c27 = 7.0529661E-38f;
    vector_bool mask28 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i29 = 0; i29 < 128; i29 = (int32_t)((uint32_t)i29 + (uint32_t)64)) {
        int64_t t30 = (int64_t)((uint64_t)arg12 * (uint64_t)128);
        int64_t t31 = (int64_t)i29;
        int64_t t32 = (int64_t)((uint64_t)t30 + (uint64_t)t31);
        int32_t t33 = t32;
        vector_float v34;
        asc_loadalign(v34, buf13 + t33);
        vector_bfloat16_t v35;
        asc_float2bfloat16_rn(v35, v34, mask28, ASC_POSITION_EVEN);
        vector_float v36;
        asc_bfloat162float(v36, *(vector_bfloat16_t*)&v35, mask28);
        int32_t t37 = i29;
        asc_storealign(buf14 + t37, v36, mask28);
    }
    asc_mem_bar(VST_VLD);
    uint32_t rem39 = 32;
    vector_bool mask38 = asc_update_mask_b32(rem39);
    uint32_t rem41 = 32;
    vector_bool mask40 = asc_update_mask_b32(rem41);
    vector_float v42;
    vector_float v43;
    asc_loadalign_deintlv(v42, v43, (__ubuf__ float*)(buf14 + 64));
    vector_float v44;
    asc_loadalign(v44, buf15 + 0);
    vector_float v45;
    asc_loadalign(v45, buf16 + 0);
    vector_float v46;
    asc_mul(v46, v42, v44, mask38);
    vector_float v47;
    asc_mul(v47, v43, v45, mask38);
    vector_float v48;
    asc_mul(v48, v42, v45, mask38);
    vector_float v49;
    asc_mul(v49, v43, v44, mask38);
    vector_float v50;
    asc_sub(v50, v46, v47, mask38);
    vector_float v51;
    asc_add(v51, v48, v49, mask38);
    vector_bfloat16_t v52;
    asc_float2bfloat16_rn(v52, v50, mask38, ASC_POSITION_EVEN);
    vector_bfloat16_t v53;
    asc_float2bfloat16_rn(v53, v51, mask38, ASC_POSITION_EVEN);
    vector_float v54;
    asc_bfloat162float(v54, *(vector_bfloat16_t*)&v52, mask38);
    vector_float v55;
    asc_bfloat162float(v55, *(vector_bfloat16_t*)&v53, mask38);
    vector_uint32_t &r56 = (vector_uint32_t &)v54;
    vector_uint32_t &r57 = (vector_uint32_t &)v55;
    asc_storealign_intlv((__ubuf__ uint32_t*)(buf14 + 64), r56, r57);
    asc_mem_bar(VST_VLD);
    for (int32_t i58 = 0; i58 < 4; i58 = (int32_t)((uint32_t)i58 + (uint32_t)1)) {
        int32_t t59 = i58 * 32;
        int32_t t60 = t59;
        vector_float v61;
        asc_loadalign(v61, buf14 + t60);
        vector_float v62;
        asc_abs(v62, v61, mask38);
        vector_float v63;
        asc_reduce_max(v63, v62, mask38);
        vector_float v64;
        asc_max_scalar(v64, v63, c27, mask38);
        vector_float v65;
        asc_mul_scalar(v65, v64, c26, mask38);
        vector_int32_t &r66 = (vector_int32_t &)v65;
        vector_int32_t v67;
        asc_add_scalar(v67, r66, 8388607, mask38);
        vector_int32_t v68;
        asc_shiftright_scalar(v68, v67, 23, mask38);
        vector_int32_t v69;
        asc_duplicate(v69, v68, mask38);
        vector_int32_t v70;
        asc_shiftleft_scalar(v70, v69, 23, mask38);
        vector_float &r71 = (vector_float &)v70;
        vector_uint8_t v72;
        asc_int322uint8(v72, v69, mask38);
        int32_t t73 = i58;
        asc_storealign_1st(buf17 + t73, v72);
        vector_float v74;
        asc_div(v74, v61, r71, mask38);
        asc_storealign(buf14 + t60, v74, mask38);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i75 = 0; i75 < 128; i75 = (int32_t)((uint32_t)i75 + (uint32_t)64)) {
        int32_t t76 = i75;
        vector_float v77;
        asc_loadalign(v77, buf14 + t76);
        vector_float v78;
        asc_abs(v78, v77, mask28);
        vector_uint32_t v79;
        asc_duplicate_scalar(v79, 0, mask28);
        vector_bool m80;
        asc_gt_scalar(m80, v78, c25, mask28);
        vector_uint32_t v81;
        asc_duplicate_scalar(v81, 1, mask28);
        vector_uint32_t v82;
        asc_select(v82, v81, v79, m80);
        vector_bool m83;
        asc_ge_scalar(m83, v78, c24, mask28);
        vector_uint32_t v84;
        asc_duplicate_scalar(v84, 2, mask28);
        vector_uint32_t v85;
        asc_select(v85, v84, v82, m83);
        vector_bool m86;
        asc_gt_scalar(m86, v78, c23, mask28);
        vector_uint32_t v87;
        asc_duplicate_scalar(v87, 3, mask28);
        vector_uint32_t v88;
        asc_select(v88, v87, v85, m86);
        vector_bool m89;
        asc_ge_scalar(m89, v78, c22, mask28);
        vector_uint32_t v90;
        asc_duplicate_scalar(v90, 4, mask28);
        vector_uint32_t v91;
        asc_select(v91, v90, v88, m89);
        vector_bool m92;
        asc_gt_scalar(m92, v78, c21, mask28);
        vector_uint32_t v93;
        asc_duplicate_scalar(v93, 5, mask28);
        vector_uint32_t v94;
        asc_select(v94, v93, v91, m92);
        vector_bool m95;
        asc_ge_scalar(m95, v78, c20, mask28);
        vector_uint32_t v96;
        asc_duplicate_scalar(v96, 6, mask28);
        vector_uint32_t v97;
        asc_select(v97, v96, v94, m95);
        vector_bool m98;
        asc_gt_scalar(m98, v78, c19, mask28);
        vector_uint32_t v99;
        asc_duplicate_scalar(v99, 7, mask28);
        vector_uint32_t v100;
        asc_select(v100, v99, v97, m98);
        vector_uint32_t &r101 = (vector_uint32_t &)v77;
        vector_uint32_t v102;
        asc_shiftright_scalar(v102, r101, 28, mask28);
        vector_uint32_t v103;
        asc_duplicate_scalar(v103, 8);
        vector_uint32_t v104;
        asc_and(v104, v102, v103, mask28);
        vector_uint32_t v105;
        asc_or(v105, v100, v104, mask28);
        vector_uint32_t v106;
        vector_uint32_t v107;
        asc_deintlv(v106, v107, v105, v105);
        vector_uint32_t v108;
        asc_shiftleft_scalar(v108, v107, 4, mask40);
        vector_uint32_t v109;
        asc_or(v109, v106, v108, mask40);
        int32_t t110 = i75 / 2;
        int32_t t111 = t110;
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf18 + t111), *(vector_float*)&v109, mask40);
    }
}

__global__ __aicore__ void kernel_project_query_grid(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds gm0_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm1_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds gm2_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm3_arg, KernelArgTensor_float_2d_Device_ds_ds gm4_arg, KernelArgTensor_float_2d_Device_ds_ds gm5_arg, KernelArgTensor_uint8_t_2d_Device_ds_ds gm6_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds gm7_arg, int64_t arg8, int64_t arg9, int64_t arg10, int32_t arg11)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ fp8_e4m3fn_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm1 = gm1_arg.ptr;
    __gm__ fp8_e8m0_t* gm2 = gm2_arg.ptr;
    __gm__ fp8_e8m0_t* gm3 = gm3_arg.ptr;
    __gm__ float* gm4 = gm4_arg.ptr;
    __gm__ float* gm5 = gm5_arg.ptr;
    __gm__ uint8_t* gm6 = gm6_arg.ptr;
    __gm__ fp8_e8m0_t* gm7 = gm7_arg.ptr;
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

    __ubuf__ float* buf112 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __cc__ float* buf113 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ fp8_e4m3fn_t* buf114 = (__ca__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cb__ fp8_e4m3fn_t* buf115 = (__cb__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ fp8_e8m0_t* buf116 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(27136);
    __cbuf__ fp8_e4m3fn_t* buf117 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(10240);
    __cbuf__ fp8_e8m0_t* buf118 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(26624);
    __cbuf__ fp8_e4m3fn_t* buf119 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf120 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf121 = (__ubuf__ float*)(uintptr_t)(uint64_t)(20480);
    __ubuf__ float* buf122 = (__ubuf__ float*)(uintptr_t)(uint64_t)(20736);
    __ubuf__ float* buf123 = (__ubuf__ float*)(uintptr_t)(uint64_t)(20992);
    __ubuf__ uint8_t* buf124 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(21760);
    __ubuf__ uint8_t* buf125 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(21824);

    if ASCEND_IS_AIC {
        int64_t dim126 = gm6_arg.shape[0];
        int32_t t127 = (arg11 > 0 ? arg11 : 0);
        int64_t t128 = (int64_t)t127;
        int64_t t129 = (dim126 < t128 ? dim126 : t128);
        int64_t t130 = (int64_t)((uint64_t)t129 + (uint64_t)79);
        int64_t t131 = t130 % 80;
        bool cmp132 = t131 != 0;
        bool cmp133 = t131 < 0;
        bool t134 = cmp132 & cmp133;
        int64_t t135 = t130 / 80;
        int64_t t136 = (int64_t)(uint8_t)t134;
        int64_t t137 = (int64_t)((uint64_t)t135 - (uint64_t)t136);
        int64_t t138 = (int64_t)((uint64_t)t137 * (uint64_t)32);
        for (int64_t i139 = arg8; i139 < t138; i139 = (int64_t)((uint64_t)i139 + (uint64_t)arg9)) {
            int64_t t140 = i139 % 32;
            bool cmp141 = t140 != 0;
            bool cmp142 = t140 < 0;
            bool t143 = cmp141 & cmp142;
            int64_t t144 = i139 / 32;
            int64_t t145 = (int64_t)(uint8_t)t143;
            int64_t t146 = (int64_t)((uint64_t)t144 - (uint64_t)t145);
            int64_t t147 = (int64_t)((uint64_t)t146 * (uint64_t)80);
            int64_t sel148 = t143 ? 32 : 0;
            int64_t t149 = t140 + sel148;
            int64_t t150 = (int64_t)((uint64_t)t146 * (uint64_t)3200);
            for (int64_t i151 = 0; i151 < 10; i151 = (int64_t)((uint64_t)i151 + (uint64_t)1)) {
                __cbuf__ fp8_e4m3fn_t* view_root152 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf117) + 0);
                __cbuf__ fp8_e8m0_t* view_root153 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf118) + 0);
                __cbuf__ fp8_e8m0_t* view_root154 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf116) + 0);
                __cbuf__ fp8_e4m3fn_t* view_root155 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf119) + 0);
                int64_t t156 = (int64_t)((uint64_t)t147 / (uint64_t)8);
                int64_t dim157 = gm0_arg.stride[0];
                int64_t t158 = (int64_t)((uint64_t)t156 * (uint64_t)dim157);
                int64_t t159 = i151 * 16;
                int64_t t160 = (int64_t)((uint64_t)t158 + (uint64_t)t159);
                int64_t t161 = (int64_t)((uint64_t)t160 * (uint64_t)8);
                int64_t dim162 = gm0_arg.stride[0];
                asc_lock(PIPE_MTE2, 0);
                int64_t t163 = t161;
                set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(80) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                asc_copy_gm2l1_nd2nz((__cbuf__ int8_t*)(view_root155 + 0), (__gm__ int8_t*)(gm0 + t163), dim162, 0, 80, 128, 0, false);
                asc_unlock(PIPE_MTE2, 0);
                int64_t t164 = t149 * 163840;
                int64_t t165 = i151 * 4096;
                int64_t t166 = t164 + t165;
                asc_lock(PIPE_MTE2, 1);
                {
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    uint8_t dma_pad = static_cast<uint8_t>(0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                    asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root152 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm1 + t166), 4, 4096, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)40960, 4096);
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                }
                asc_unlock(PIPE_MTE2, 1);
                int64_t t167 = i151 * 64;
                int64_t t168 = (int64_t)((uint64_t)t150 + (uint64_t)t167);
                asc_lock(PIPE_MTE2, 2);
                {
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    uint8_t dma_pad = static_cast<uint8_t>(0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                    asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root153 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm2 + t168), 5, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)640, 64);
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                }
                asc_unlock(PIPE_MTE2, 2);
                int64_t t169 = t149 * 5120;
                int64_t t170 = t169 + t167;
                asc_lock(PIPE_MTE2, 3);
                {
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    uint8_t dma_pad = static_cast<uint8_t>(0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                    asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root154 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm3 + t170), 8, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)640, 64);
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_set_gm2l1_loop1_stride(0, 0);
                    asc_set_gm2l1_loop2_stride(0, 0);
                    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                }
                asc_unlock(PIPE_MTE2, 3);
                __ca__ fp8_e4m3fn_t* view_root171 = reinterpret_cast<__ca__ fp8_e4m3fn_t*>(reinterpret_cast<__ca__ uint8_t*>(buf114) + 0);
                uint64_t selected_root172 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root171)) / 16;
                asc_lock(PIPE_MTE1, 4);
                asc_lock(PIPE_MTE1, 0);
                asc_lock(PIPE_MTE1, 2);
                asc_copy_l12l0a(view_root171 + 0, view_root155 + 0, 0, 0, 5, 4, 5, 5);
                asc_copy_l12l0a_mx(selected_root172, (__cbuf__ fp8_e8m0_t*)(view_root153 + 0), 0, 0, 5, 2, 2, 2);
                asc_unlock(PIPE_MTE1, 2);
                asc_unlock(PIPE_MTE1, 0);
                asc_unlock(PIPE_MTE1, 4);
                __cb__ fp8_e4m3fn_t* view_root173 = reinterpret_cast<__cb__ fp8_e4m3fn_t*>(reinterpret_cast<__cb__ uint8_t*>(buf115) + 0);
                uint64_t selected_root174 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root173)) / 16;
                asc_lock(PIPE_MTE1, 5);
                asc_lock(PIPE_MTE1, 1);
                asc_lock(PIPE_MTE1, 3);
                asc_copy_l12l0b_transpose(view_root173 + 0, view_root152 + 0, 0, 0, 8, 4, 8, 8);
                asc_copy_l12l0b_mx(selected_root174, (__cbuf__ fp8_e8m0_t*)(view_root154 + 0), 0, 0, 8, 2, 2, 2);
                asc_unlock(PIPE_MTE1, 3);
                asc_unlock(PIPE_MTE1, 1);
                asc_unlock(PIPE_MTE1, 5);
                bool cmp175 = i151 == 0;
                __cc__ float* view_root176 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf113) + 0);
                asc_lock(PIPE_M, 6);
                asc_lock(PIPE_M, 4);
                asc_lock(PIPE_M, 5);
                if (cmp175) {
                    asc_mmad_mx(view_root176 + 0, view_root171, view_root173, 80, 128, 128, 0, true, false, true);
                } else {
                    asc_mmad_mx(view_root176 + 0, view_root171, view_root173, 80, 128, 128, 0, true, false, false);
                }
                asc_unlock(PIPE_M, 5);
                asc_unlock(PIPE_M, 4);
                asc_unlock(PIPE_M, 6);
            }
            __ubuf__ float* view_root177 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf112) + 0);
            __cc__ float* view_root178 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf113) + 0);
            asc_sync_intra_wait(PIPE_FIX, 0);
            asc_sync_intra_wait(PIPE_FIX, 16);
            asc_lock(PIPE_FIX, 6);
            asc_set_l0c2gm_nz2nd(1, 0, 0);
            asc_copy_l0c2ub((__ubuf__ float*)view_root177 + 0, view_root178, 128, 80, 128, 80, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
            asc_unlock(PIPE_FIX, 6);
            asc_sync_intra_arrive(PIPE_FIX, 0);
            asc_sync_intra_arrive(PIPE_FIX, 16);
        }
        asc_sync_intra_wait(PIPE_FIX, 0);
        asc_sync_intra_wait(PIPE_FIX, 16);
    }

    if ASCEND_IS_AIV {
        asc_sync_intra_arrive(PIPE_V, 0);
        int64_t dim179 = gm6_arg.shape[0];
        int32_t t180 = (arg11 > 0 ? arg11 : 0);
        int64_t t181 = (int64_t)t180;
        int64_t t182 = (dim179 < t181 ? dim179 : t181);
        int64_t t183 = (int64_t)((uint64_t)t182 + (uint64_t)79);
        int64_t t184 = t183 % 80;
        bool cmp185 = t184 != 0;
        bool cmp186 = t184 < 0;
        bool t187 = cmp185 & cmp186;
        int64_t t188 = t183 / 80;
        int64_t t189 = (int64_t)(uint8_t)t187;
        int64_t t190 = (int64_t)((uint64_t)t188 - (uint64_t)t189);
        int64_t t191 = (int64_t)((uint64_t)t190 * (uint64_t)32);
        for (int64_t i192 = arg8; i192 < t191; i192 = (int64_t)((uint64_t)i192 + (uint64_t)arg9)) {
            int64_t t193 = i192 % 32;
            bool cmp194 = t193 != 0;
            bool cmp195 = t193 < 0;
            bool t196 = cmp194 & cmp195;
            int64_t t197 = i192 / 32;
            int64_t t198 = (int64_t)(uint8_t)t196;
            int64_t t199 = (int64_t)((uint64_t)t197 - (uint64_t)t198);
            int64_t t200 = (int64_t)((uint64_t)t199 * (uint64_t)80);
            int64_t sel201 = t196 ? 32 : 0;
            int64_t t202 = t193 + sel201;
            int64_t t203 = (int64_t)((uint64_t)t182 - (uint64_t)t200);
            int64_t t204 = (t203 < 80 ? t203 : 80);
            int64_t t205 = t202 * 64;
            int64_t t206 = t202 * 4;
            int64_t t207 = (int64_t)((uint64_t)arg10 * (uint64_t)40);
            int64_t t208 = (int64_t)((uint64_t)t204 - (uint64_t)t207);
            int64_t t209 = (t208 < 40 ? t208 : 40);
            int64_t t210 = (t209 > 0 ? t209 : 0);
            asc_sync_intra_wait(PIPE_V, 0);
            for (int64_t i211 = 0; i211 < t210; i211 = (int64_t)((uint64_t)i211 + (uint64_t)1)) {
                int64_t t212 = (int64_t)((uint64_t)t207 + (uint64_t)i211);
                int64_t t213 = (t212 < t204 ? t212 : t204);
                int64_t t214 = (int64_t)((uint64_t)t204 - (uint64_t)t213);
                int64_t t215 = (t214 < 1 ? t214 : 1);
                bool cmp216 = t212 <= t204;
                int64_t sel217 = cmp216 ? t215 : 0;
                int64_t dim218 = gm4_arg.stride[0];
                int64_t t219 = (int64_t)((uint64_t)dim218 * (uint64_t)4);
                int64_t t220 = (int64_t)((uint64_t)t212 + (uint64_t)t200);
                int64_t t221 = (int64_t)((uint64_t)t220 * (uint64_t)dim218);
                int64_t t222 = (int64_t)((uint64_t)t221 * (uint64_t)4);
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf121 + 0), (__gm__ float*)((__gm__ uint8_t*)gm4 + t222), sel217, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t219, 256);
                }
                asc_unlock(PIPE_MTE2, 0);
                int64_t dim223 = gm5_arg.stride[0];
                int64_t t224 = (int64_t)((uint64_t)dim223 * (uint64_t)4);
                int64_t t225 = (int64_t)((uint64_t)t220 * (uint64_t)dim223);
                int64_t t226 = (int64_t)((uint64_t)t225 * (uint64_t)4);
                asc_lock(PIPE_MTE2, 1);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf122 + 0), (__gm__ float*)((__gm__ uint8_t*)gm5 + t226), sel217, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t224, 256);
                }
                asc_unlock(PIPE_MTE2, 1);
                asc_lock(PIPE_V, 0);
                asc_lock(PIPE_V, 1);
                asc_lock(PIPE_V, 3);
                asc_lock(PIPE_V, 2);
                kernel_project_query_grid_vf0(i211, (__ubuf__ float*)buf120, (__ubuf__ float*)buf123, (__ubuf__ float*)buf121, (__ubuf__ float*)buf122, (__ubuf__ uint8_t*)buf125, (__ubuf__ uint8_t*)buf124);
                asc_unlock(PIPE_V, 2);
                asc_unlock(PIPE_V, 3);
                asc_unlock(PIPE_V, 1);
                asc_unlock(PIPE_V, 0);
                int64_t dim227 = gm6_arg.stride[0];
                int64_t t228 = (int64_t)((uint64_t)t220 * (uint64_t)dim227);
                int64_t t229 = (int64_t)((uint64_t)t228 + (uint64_t)t205);
                int64_t t230 = (int64_t)((uint64_t)t229 * (uint64_t)8);
                int64_t t231 = (int64_t)((uint64_t)t230 / (uint64_t)8);
                int64_t t232 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm6) + (uint64_t)t231);
                __gm__ uint8_t* root233 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t232);
                __ubuf__ uint8_t* view_root234 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf124) + 0);
                int64_t dim235 = gm6_arg.stride[0];
                asc_lock(PIPE_MTE3, 2);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root233 + 0, (__ubuf__ uint8_t*)view_root234 + 0, 1, 64, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 2);
                int64_t dim236 = gm7_arg.stride[0];
                int64_t t237 = (int64_t)((uint64_t)dim236 * (uint64_t)8);
                int64_t t238 = t237 / 8;
                int64_t dim239 = gm7_arg.stride[0];
                int64_t t240 = (int64_t)((uint64_t)t220 * (uint64_t)dim239);
                int64_t t241 = (int64_t)((uint64_t)t240 + (uint64_t)t206);
                int64_t t242 = (int64_t)((uint64_t)t241 * (uint64_t)8);
                int64_t t243 = (int64_t)((uint64_t)t242 / (uint64_t)8);
                int64_t t244 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm7) + (uint64_t)t243);
                __gm__ uint8_t* root245 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t244);
                __ubuf__ uint8_t* view_root246 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf125) + 0);
                asc_lock(PIPE_MTE3, 3);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root245 + 0, (__ubuf__ uint8_t*)view_root246 + 0, 1, 4, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 3);
            }
            asc_sync_intra_arrive(PIPE_V, 0);
        }
    }
}
