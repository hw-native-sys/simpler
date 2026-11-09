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

__aicore__ inline uint16_t cannbotdsl_get_ffts_config(uint16_t flagId, uint16_t mode)
{
    constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
    constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;
    return 0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE);
}

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

struct KernelArgTensor_bfloat16_t_2d {
    __gm__ bfloat16_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_uint8_t_4d {
    __gm__ uint8_t* ptr;
    int64_t shape[4];
    int64_t stride[4];
};

struct KernelArgTensor_fp8_e8m0_t_4d {
    __gm__ fp8_e8m0_t* ptr;
    int64_t shape[4];
    int64_t stride[4];
};

struct KernelArgTensor_int32_t_2d {
    __gm__ int32_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_int32_t_1d {
    __gm__ int32_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_uint16_t_2d {
    __gm__ uint16_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

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

struct KernelArgTensor_bfloat16_t_2d_Device_ds_ds {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ds_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_uint8_t_4d_Device_dsss_ddds {
    __gm__ uint8_t* ptr;
    int64_t shape[1];
    int64_t stride[3];
};
static inline KernelArgTensor_uint8_t_4d_Device_dsss_ddds cannbotdsl_pack_KernelArgTensor_uint8_t_4d_Device_dsss_ddds(const KernelArgTensor_uint8_t_4d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0], src.stride[1], src.stride[2]}};
}

struct KernelArgTensor_fp8_e8m0_t_4d_Device_dsss_ddds {
    __gm__ fp8_e8m0_t* ptr;
    int64_t shape[1];
    int64_t stride[3];
};
static inline KernelArgTensor_fp8_e8m0_t_4d_Device_dsss_ddds cannbotdsl_pack_KernelArgTensor_fp8_e8m0_t_4d_Device_dsss_ddds(const KernelArgTensor_fp8_e8m0_t_4d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0], src.stride[1], src.stride[2]}};
}

struct KernelArgTensor_int32_t_2d_Device_dd_ds {
    __gm__ int32_t* ptr;
    int64_t shape[2];
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_dd_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_dd_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.shape[0], src.shape[1]}, {src.stride[0]}};
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

struct KernelArgTensor_int32_t_2d_Device_sd_ds {
    __gm__ int32_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_sd_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_sd_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.shape[1]}, {src.stride[0]}};
}

struct KernelArgTensor_uint16_t_2d_Device_dd_ds {
    __gm__ uint16_t* ptr;
    int64_t shape[2];
    int64_t stride[1];
};
static inline KernelArgTensor_uint16_t_2d_Device_dd_ds cannbotdsl_pack_KernelArgTensor_uint16_t_2d_Device_dd_ds(const KernelArgTensor_uint16_t_2d& src) {
    return {src.ptr, {src.shape[0], src.shape[1]}, {src.stride[0]}};
}

struct KernelArgTensor_float_2d_Device_dd_ds {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_dd_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_dd_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0], src.shape[1]}, {src.stride[0]}};
}

static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_4d_Device_dsss_ddds) == 40 && alignof(KernelArgTensor_uint8_t_4d_Device_dsss_ddds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_4d_Device_dsss_ddds) == 40 && alignof(KernelArgTensor_fp8_e8m0_t_4d_Device_dsss_ddds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_sd_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_sd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint16_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_uint16_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_float_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__simd_vf__ inline void kernel_score_topk_mix_vf0(__ubuf__ bfloat16_t* buf23)
{
    vector_bool vmask;
    bfloat16_t c24 = __builtin_bit_cast(bfloat16_t, (uint16_t)0);
    for (int32_t i25 = 0; i25 < 1; i25 = (int32_t)((uint32_t)i25 + (uint32_t)1)) {
        uint32_t rem27 = 128;
        vector_bool mask26 = asc_update_mask_b16(rem27);
        vector_bfloat16_t v28;
        asc_duplicate_scalar(v28, c24, mask26);
        asc_storealign(buf23 + 110080, v28, mask26);
        asc_storealign(buf23 + 110208, v28, mask26);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf1(__ubuf__ bfloat16_t* buf29, __ubuf__ bfloat16_t* buf30)
{
    vector_bool vmask;
    float c31 = 0.015625f;
    uint32_t rem33 = 32;
    vector_bool mask32 = asc_update_mask_b32(rem33);
    uint32_t rem35 = 32;
    vector_bool mask34 = asc_update_mask_b16(rem35);
    for (int32_t i36 = 0; i36 < 6; i36 = (int32_t)((uint32_t)i36 + (uint32_t)1)) {
        int32_t t37 = i36 * 32;
        int32_t t38 = t37;
        int32_t t39 = (int32_t)((uint32_t)t38 + (uint32_t)110080);
        vector_float v40;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v40, (__ubuf__ bfloat16_t*)(buf29 + t39));
        vector_float v41;
        asc_bfloat162float(v41, *(vector_bfloat16_t*)&v40, mask32);
        vector_float v42;
        asc_mul_scalar(v42, v41, c31, mask32);
        vector_bfloat16_t v43;
        asc_float2bfloat16_rn(v43, v42, mask32, ASC_POSITION_EVEN);
        vector_uint32_t &r44 = (vector_uint32_t &)v43;
        vector_uint16_t v45;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v45, *(vector_uint32_t*)&r44);
        int32_t t46 = i36 * 128;
        vector_bfloat16_t &r47 = (vector_bfloat16_t &)v45;
        int32_t t48 = t46;
        int32_t t49 = (int32_t)((uint32_t)t48 + (uint32_t)110336);
        asc_storealign(buf30 + t49, r47, mask34);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf2(__ubuf__ int32_t* buf50, __ubuf__ uint16_t* buf51)
{
    vector_bool vmask;
    uint32_t rem53 = 64;
    vector_bool mask52 = asc_update_mask_b16(rem53);
    for (int32_t i54 = 0; i54 < 4; i54 = (int32_t)((uint32_t)i54 + (uint32_t)1)) {
        int32_t t55 = i54 * 64;
        int32_t t56 = t55;
        int32_t t57 = (int32_t)((uint32_t)t56 + (uint32_t)55808);
        vector_int32_t v58;
        asc_loadalign(v58, buf50 + t57);
        vector_uint32_t &r59 = (vector_uint32_t &)v58;
        vector_uint16_t v60;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v60, *(vector_uint32_t*)&r59);
        int32_t t61 = (int32_t)((uint32_t)t56 + (uint32_t)112128);
        asc_storealign(buf51 + t61, v60, mask52);
    }
}

__simd_vf__ inline void kernel_score_topk_mix_vf3(__ubuf__ bfloat16_t* buf62, int32_t arg63, int64_t arg64, int64_t arg65, __ubuf__ bfloat16_t* buf66, __ubuf__ uint16_t* buf67, bool arg68, int32_t arg69, __ubuf__ uint16_t* buf70, int32_t arg71)
{
    vector_bool vmask;
    bfloat16_t c72 = __builtin_bit_cast(bfloat16_t, (uint16_t)0);
    uint32_t rem74 = 128;
    vector_bool mask73 = asc_update_mask_b16(rem74);
    vector_uint16_t v75;
    asc_duplicate_scalar(v75, 127, mask73);
    vector_bfloat16_t v76;
    asc_duplicate_scalar(v76, c72, mask73);
    int32_t t77 = (int32_t)((uint32_t)arg63 + (uint32_t)110336);
    vector_bfloat16_t v78;
    asc_loadalign(v78, buf62 + t77);
    vector_bfloat16_t iter79 = v76;
    #pragma unroll(1)
    for (int32_t i80 = 0; i80 < 32; i80 = (int32_t)((uint32_t)i80 + (uint32_t)1)) {
        int16_t t81 = (int16_t)i80;
        vector_uint16_t v82;
        asc_duplicate_scalar(v82, t81, mask73);
        vector_bfloat16_t v83;
        asc_gather(v83, v78, v82);
        int32_t t84 = (int32_t)arg64;
        int32_t t85 = (int32_t)((uint32_t)t84 * (uint32_t)32);
        int32_t t86 = (int32_t)((uint32_t)t85 + (uint32_t)i80);
        int32_t t87 = (int32_t)((uint32_t)t86 * (uint32_t)256);
        int32_t t88 = (int32_t)arg65;
        int32_t t89 = (int32_t)((uint32_t)t88 * (uint32_t)128);
        int32_t t90 = (int32_t)((uint32_t)t87 + (uint32_t)t89);
        int32_t t91 = t90;
        vector_bfloat16_t v92;
        asc_loadalign(v92, buf66 + t91);
        vector_bfloat16_t v93;
        asc_max_scalar(v93, v92, c72, mask73);
        cannbotdsl_92::cannbotdsl_madd_bfloat16(v93, v83, iter79, mask73);
        iter79 = v93;
    }
    vector_uint16_t v94;
    asc_duplicate_scalar(v94, 1, mask73);
    vector_uint16_t ifr95;
    if (arg68) {
        int32_t t96 = (int32_t)arg65;
        int32_t t97 = (int32_t)((uint32_t)t96 * (uint32_t)128);
        int32_t t98 = t97;
        int32_t t99 = (int32_t)((uint32_t)t98 + (uint32_t)112128);
        vector_uint16_t v100;
        asc_loadalign(v100, buf67 + t99);
        ifr95 = v100;
    } else {
        ifr95 = v94;
    }
    uint32_t rem102 = arg69;
    vector_bool mask101 = asc_update_mask_b16(rem102);
    vector_bool m103;
    asc_eq_scalar(m103, ifr95, 1, mask73);
    vector_bool m104;
    asc_and(m104, mask101, m103, mask73);
    vector_bfloat16_t v105;
    asc_add_scalar(v105, iter79, c72, mask73);
    vector_uint16_t &r106 = (vector_uint16_t &)v105;
    vector_uint16_t v107;
    asc_duplicate_scalar(v107, -32768, mask73);
    vector_uint16_t v108;
    asc_xor(v108, r106, v107, mask73);
    vector_uint16_t v109;
    asc_duplicate_scalar(v109, -1, mask73);
    vector_uint16_t v110;
    asc_xor(v110, r106, v109, mask73);
    vector_int16_t &r111 = (vector_int16_t &)r106;
    vector_bool m112;
    asc_lt_scalar(m112, r111, 0, mask73);
    vector_uint16_t v113;
    asc_select(v113, v110, v108, m112);
    vector_uint16_t v114;
    asc_select(v114, v113, v75, m104);
    int32_t t115 = (int32_t)((uint32_t)arg71 + (uint32_t)32768);
    for (int32_t i116 = 0; i116 < 1; i116 = (int32_t)((uint32_t)i116 + (uint32_t)1)) {
        asc_storealign(buf70 + t115, v114, mask73);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf4(int64_t arg117, __ubuf__ uint16_t* buf118, bool arg119, int32_t arg120, __ubuf__ uint16_t* buf121, int32_t arg122)
{
    vector_bool vmask;
    bfloat16_t c123 = __builtin_bit_cast(bfloat16_t, (uint16_t)0);
    uint32_t rem125 = 128;
    vector_bool mask124 = asc_update_mask_b16(rem125);
    vector_uint16_t v126;
    asc_duplicate_scalar(v126, 127, mask124);
    vector_bfloat16_t v127;
    asc_duplicate_scalar(v127, c123, mask124);
    vector_uint16_t v128;
    asc_duplicate_scalar(v128, 1, mask124);
    vector_uint16_t ifr129;
    if (arg119) {
        int32_t t130 = (int32_t)arg117;
        int32_t t131 = (int32_t)((uint32_t)t130 * (uint32_t)128);
        int32_t t132 = t131;
        int32_t t133 = (int32_t)((uint32_t)t132 + (uint32_t)112128);
        vector_uint16_t v134;
        asc_loadalign(v134, buf118 + t133);
        ifr129 = v134;
    } else {
        ifr129 = v128;
    }
    uint32_t rem136 = arg120;
    vector_bool mask135 = asc_update_mask_b16(rem136);
    vector_bool m137;
    asc_eq_scalar(m137, ifr129, 1, mask124);
    vector_bool m138;
    asc_and(m138, mask135, m137, mask124);
    vector_bfloat16_t v139;
    asc_add_scalar(v139, v127, c123, mask124);
    vector_uint16_t &r140 = (vector_uint16_t &)v139;
    vector_uint16_t v141;
    asc_duplicate_scalar(v141, -32768, mask124);
    vector_uint16_t v142;
    asc_xor(v142, r140, v141, mask124);
    vector_uint16_t v143;
    asc_duplicate_scalar(v143, -1, mask124);
    vector_uint16_t v144;
    asc_xor(v144, r140, v143, mask124);
    vector_int16_t &r145 = (vector_int16_t &)r140;
    vector_bool m146;
    asc_lt_scalar(m146, r145, 0, mask124);
    vector_uint16_t v147;
    asc_select(v147, v144, v142, m146);
    vector_uint16_t v148;
    asc_select(v148, v147, v126, m138);
    int32_t t149 = (int32_t)((uint32_t)arg122 + (uint32_t)32768);
    for (int32_t i150 = 0; i150 < 1; i150 = (int32_t)((uint32_t)i150 + (uint32_t)1)) {
        asc_storealign(buf121 + t149, v148, mask124);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf5(int64_t arg151, __ubuf__ uint16_t* buf152, int32_t arg153, int64_t arg154, __ubuf__ uint16_t* buf155)
{
    vector_bool vmask;
    uint32_t rem157 = 128;
    vector_bool mask156 = asc_update_mask_b16(rem157);
    vector_uint16_t v158;
    asc_duplicate_scalar(v158, 0, mask156);
    vector_uint16_t iter159 = v158;
    vector_uint16_t iter160 = v158;
    #pragma unroll(1)
    for (int64_t i161 = 0; i161 < arg154; i161 = (int64_t)((uint64_t)i161 + (uint64_t)1)) {
        int64_t t162 = (int64_t)((uint64_t)i161 * (uint64_t)256);
        int64_t t163 = (int64_t)((uint64_t)arg151 - (uint64_t)t162);
        int32_t t164 = (int32_t)t163;
        uint32_t rem166 = t164;
        vector_bool mask165 = asc_update_mask_b8(rem166);
        int32_t t167 = t162;
        int32_t t168 = (int32_t)((uint32_t)arg153 + (uint32_t)t167);
        vector_uint8_t v169;
        vector_uint8_t v170;
        asc_loadalign_deintlv(v169, v170, (__ubuf__ uint8_t*)(buf152 + t168));
        vector_uint16_t v171 = iter159;
        asc_cumulative_histogram_bin0(v171, v170, mask165);
        vector_uint16_t v172 = iter160;
        asc_cumulative_histogram_bin1(v172, v170, mask165);
        iter159 = v171;
        iter160 = v172;
    }
    for (int32_t i173 = 0; i173 < 1; i173 = (int32_t)((uint32_t)i173 + (uint32_t)1)) {
        asc_storealign(buf155 + 28160, iter159, mask156);
        asc_storealign(buf155 + 28288, iter160, mask156);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf6(int64_t arg174, __ubuf__ uint16_t* buf175, int32_t arg176, int32_t arg177, int64_t arg178, __ubuf__ uint16_t* buf179)
{
    vector_bool vmask;
    uint32_t rem181 = 128;
    vector_bool mask180 = asc_update_mask_b16(rem181);
    vector_uint16_t v182;
    asc_duplicate_scalar(v182, 0, mask180);
    vector_uint16_t iter183 = v182;
    vector_uint16_t iter184 = v182;
    #pragma unroll(1)
    for (int64_t i185 = 0; i185 < arg178; i185 = (int64_t)((uint64_t)i185 + (uint64_t)1)) {
        int64_t t186 = (int64_t)((uint64_t)i185 * (uint64_t)256);
        int64_t t187 = (int64_t)((uint64_t)arg174 - (uint64_t)t186);
        int32_t t188 = (int32_t)t187;
        uint32_t rem190 = t188;
        vector_bool mask189 = asc_update_mask_b8(rem190);
        int32_t t191 = t186;
        int32_t t192 = (int32_t)((uint32_t)arg176 + (uint32_t)t191);
        vector_uint8_t v193;
        vector_uint8_t v194;
        asc_loadalign_deintlv(v193, v194, (__ubuf__ uint8_t*)(buf175 + t192));
        bool cmp195 = arg177 < 0;
        vector_uint16_t ifr196;
        vector_uint16_t ifr197;
        if (cmp195) {
            vector_uint16_t v198 = iter183;
            asc_cumulative_histogram_bin0(v198, v194, mask189);
            vector_uint16_t v199 = iter184;
            asc_cumulative_histogram_bin1(v199, v194, mask189);
            ifr196 = v198;
            ifr197 = v199;
        } else {
            int8_t t200 = (int8_t)arg177;
            vector_bool m201;
            asc_eq_scalar(m201, v194, t200, mask189);
            vector_uint16_t v202 = iter183;
            asc_cumulative_histogram_bin0(v202, v193, m201);
            vector_uint16_t v203 = iter184;
            asc_cumulative_histogram_bin1(v203, v193, m201);
            ifr196 = v202;
            ifr197 = v203;
        }
        iter183 = ifr196;
        iter184 = ifr197;
    }
    for (int32_t i204 = 0; i204 < 1; i204 = (int32_t)((uint32_t)i204 + (uint32_t)1)) {
        asc_storealign(buf179 + 28160, iter183, mask180);
        asc_storealign(buf179 + 28288, iter184, mask180);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf7(__ubuf__ uint16_t* buf205, int64_t arg206, __ubuf__ uint16_t* buf207, int32_t arg208, int32_t arg209, int64_t arg210)
{
    vector_bool vmask;
    vector_store_unalign _sua_buf205;
    for (int32_t i211 = 0; i211 < 1; i211 = (int32_t)((uint32_t)i211 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_init();
    }
    #pragma unroll(1)
    for (int64_t i212 = 0; i212 < arg210; i212 = (int64_t)((uint64_t)i212 + (uint64_t)1)) {
        int64_t t213 = (int64_t)((uint64_t)i212 * (uint64_t)128);
        int64_t t214 = (int64_t)((uint64_t)arg206 - (uint64_t)t213);
        int32_t t215 = (int32_t)t214;
        uint32_t rem217 = t215;
        vector_bool mask216 = asc_update_mask_b16(rem217);
        int32_t t218 = t213;
        int32_t t219 = (int32_t)((uint32_t)arg208 + (uint32_t)t218);
        vector_uint16_t v220;
        asc_loadalign(v220, buf207 + t219);
        int16_t t221 = (int16_t)t213;
        vector_uint16_t v222;
        asc_arange(*(vector_int16_t*)&v222, static_cast<int16_t>(t221));
        int16_t t223 = (int16_t)arg209;
        vector_bool m224;
        asc_eq_scalar(m224, v220, t223, mask216);
        vector_uint16_t v225;
        cannbotdsl_92::cannbotdsl_squeeze_store(buf205 + 0, v225, _sua_buf205, v222, m224);
    }
    for (int32_t i226 = 0; i226 < 1; i226 = (int32_t)((uint32_t)i226 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_finalize(buf205 + 0, _sua_buf205);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf8(__ubuf__ int32_t* buf227, int64_t arg228, __ubuf__ uint16_t* buf229, int32_t arg230, int32_t arg231, int32_t arg232, int64_t arg233)
{
    vector_bool vmask;
    vector_store_unalign _sua_buf227;
    uint32_t rem235 = 64;
    vector_bool mask234 = asc_update_mask_b32(rem235);
    vector_int32_t v236;
    asc_duplicate_scalar(v236, -1, mask234);
    for (int32_t i237 = 0; i237 < 8; i237 = (int32_t)((uint32_t)i237 + (uint32_t)1)) {
        int32_t t238 = i237 * 64;
        int32_t t239 = t238;
        int32_t t240 = (int32_t)((uint32_t)t239 + (uint32_t)13568);
        asc_storealign(buf227 + t240, v236, mask234);
    }
    asc_mem_bar(VST_VST);
    for (int32_t i241 = 0; i241 < 1; i241 = (int32_t)((uint32_t)i241 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_init();
    }
    #pragma unroll(1)
    for (int64_t i242 = 0; i242 < arg233; i242 = (int64_t)((uint64_t)i242 + (uint64_t)1)) {
        int64_t t243 = (int64_t)((uint64_t)i242 * (uint64_t)64);
        int64_t t244 = (int64_t)((uint64_t)arg228 - (uint64_t)t243);
        int32_t t245 = (int32_t)t244;
        uint32_t rem247 = t245;
        vector_bool mask246 = asc_update_mask_b32(rem247);
        int32_t t248 = t243;
        int32_t t249 = (int32_t)((uint32_t)arg230 + (uint32_t)t248);
        vector_uint16_t v250;
        asc_loadalign(v250, buf229 + t249);
        vector_uint32_t v251;
        asc_unpack_lower(v251, v250);
        int32_t t252 = (int32_t)t243;
        vector_int32_t v253;
        asc_arange(v253, t252);
        vector_bool m254;
        asc_gt_scalar(m254, v251, arg231, mask246);
        vector_bool m255;
        asc_eq_scalar(m255, v251, arg231, mask246);
        vector_bool m256;
        asc_le_scalar(m256, v253, arg232, mask246);
        vector_bool m257;
        asc_and(m257, m255, m256, mask246);
        vector_bool m258;
        asc_or(m258, m254, m257, mask246);
        vector_bool m259;
        asc_gt_scalar(m259, v251, 127, mask246);
        vector_bool m260;
        asc_and(m260, m258, m259, mask246);
        vector_int32_t v261;
        cannbotdsl_92::cannbotdsl_squeeze_store(buf227 + 13568, v261, _sua_buf227, v253, m260);
    }
    for (int32_t i262 = 0; i262 < 1; i262 = (int32_t)((uint32_t)i262 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_finalize(buf227 + 13568, _sua_buf227);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf9(__ubuf__ int32_t* buf263, __ubuf__ uint16_t* buf264, int32_t arg265, __ubuf__ uint16_t* buf266, int64_t arg267, __ubuf__ int32_t* buf268)
{
    vector_bool vmask;
    uint32_t rem270 = 64;
    vector_bool mask269 = asc_update_mask_b32(rem270);
    uint32_t rem272 = 64;
    vector_bool mask271 = asc_update_mask_b16(rem272);
    for (int32_t i273 = 0; i273 < 8; i273 = (int32_t)((uint32_t)i273 + (uint32_t)1)) {
        int32_t t274 = i273 * 64;
        int32_t t275 = t274;
        int32_t t276 = (int32_t)((uint32_t)t275 + (uint32_t)13568);
        vector_int32_t v277;
        asc_loadalign(v277, buf263 + t276);
        vector_bool m278;
        asc_ge_scalar(m278, v277, 0, mask269);
        vector_int32_t v279;
        asc_max_scalar(v279, v277, 0, mask269);
        vector_uint32_t &r280 = (vector_uint32_t &)v279;
        vector_uint16_t v281;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v281, *(vector_uint32_t*)&r280);
        vector_uint16_t v282;
        asc_gather(v282, (__ubuf__ uint16_t*)(buf264 + arg265), v281, mask271);
        vector_int32_t v283;
        asc_duplicate_scalar(v283, 0, mask269);
        vector_int32_t v284;
        asc_duplicate_scalar(v284, 1, mask269);
        vector_int32_t v285;
        asc_select(v285, v284, v283, m278);
        vector_uint32_t &r286 = (vector_uint32_t &)v285;
        vector_uint16_t v287;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v287, *(vector_uint32_t*)&r286);
        vector_bool m288;
        asc_eq_scalar(m288, v287, 1, mask271);
        vector_uint16_t v289;
        asc_duplicate_scalar(v289, 0, mask271);
        vector_uint16_t v290;
        asc_select(v290, v282, v289, m288);
        int32_t t291 = (int32_t)((uint32_t)t275 + (uint32_t)25600);
        asc_storealign(buf266 + t291, v290, mask271);
        int32_t t292 = (int32_t)arg267;
        vector_int32_t v293;
        asc_add_scalar(v293, v277, t292, mask269);
        vector_int32_t v294;
        asc_duplicate_scalar(v294, -1, mask269);
        vector_int32_t v295;
        asc_select(v295, v293, v294, m278);
        int32_t t296 = (int32_t)((uint32_t)t275 + (uint32_t)13056);
        asc_storealign(buf268 + t296, v295, mask269);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf10(__ubuf__ uint16_t* buf297, int32_t arg298, __ubuf__ float* buf299)
{
    vector_bool vmask;
    uint32_t rem301 = 128;
    vector_bool mask300 = asc_update_mask_b16(rem301);
    int32_t t302 = (int32_t)((uint32_t)arg298 + (uint32_t)32768);
    for (int32_t i303 = 0; i303 < 1; i303 = (int32_t)((uint32_t)i303 + (uint32_t)1)) {
        vector_uint16_t v304;
        asc_loadalign(v304, buf297 + t302);
        vector_uint16_t v305;
        asc_duplicate_scalar(v305, -32768, mask300);
        vector_uint16_t v306;
        asc_xor(v306, v304, v305, mask300);
        vector_uint16_t v307;
        asc_duplicate_scalar(v307, -1, mask300);
        vector_uint16_t v308;
        asc_xor(v308, v304, v307, mask300);
        vector_bool m309;
        asc_ge_scalar(m309, v304, -32768, mask300);
        vector_uint16_t v310;
        asc_select(v310, v306, v308, m309);
        vector_bfloat16_t &r311 = (vector_bfloat16_t &)v310;
        uint32_t rem313 = 64;
        vector_bool mask312 = asc_update_mask_b32(rem313);
        vector_uint16_t &r314 = (vector_uint16_t &)r311;
        vector_uint32_t v315;
        asc_unpack_lower(v315, r314);
        vector_uint32_t v316;
        asc_shiftleft_scalar(v316, v315, 16, mask312);
        vector_float &r317 = (vector_float &)v316;
        asc_storealign(buf299 + 56320, r317, mask312);
        uint32_t rem319 = 64;
        vector_bool mask318 = asc_update_mask_b32(rem319);
        vector_uint32_t v320;
        asc_unpack_upper(v320, r314);
        vector_uint32_t v321;
        asc_shiftleft_scalar(v321, v320, 16, mask318);
        vector_float &r322 = (vector_float &)v321;
        asc_storealign(buf299 + 56384, r322, mask318);
    }
}

__simd_vf__ inline void kernel_score_topk_mix_vf11(int32_t arg323, int64_t arg324, int64_t arg325, __ubuf__ int32_t* buf326)
{
    vector_bool vmask;
    uint32_t rem328 = 64;
    vector_bool mask327 = asc_update_mask_b32(rem328);
    for (int32_t i329 = 0; i329 < 2; i329 = (int32_t)((uint32_t)i329 + (uint32_t)1)) {
        int32_t t330 = i329 * 64;
        vector_int32_t v331;
        asc_arange(v331, t330);
        int32_t t332 = (int32_t)((uint32_t)arg323 * (uint32_t)128);
        vector_int32_t v333;
        asc_add_scalar(v333, v331, t332, mask327);
        int32_t t334 = (int32_t)arg324;
        vector_bool m335;
        asc_lt_scalar(m335, v331, t334, mask327);
        vector_int32_t v336;
        asc_duplicate_scalar(v336, -1, mask327);
        vector_int32_t v337;
        asc_select(v337, v333, v336, m335);
        int32_t t338 = (int32_t)arg325;
        int32_t t339 = (int32_t)((uint32_t)t338 * (uint32_t)128);
        int32_t t340 = (int32_t)((uint32_t)t339 + (uint32_t)t330);
        int32_t t341 = t340;
        int32_t t342 = (int32_t)((uint32_t)t341 + (uint32_t)33152);
        asc_storealign(buf326 + t342, v337, mask327);
    }
}

__simd_vf__ inline void kernel_score_topk_mix_vf12(__ubuf__ int32_t* buf343)
{
    vector_bool vmask;
    uint32_t rem345 = 64;
    vector_bool mask344 = asc_update_mask_b32(rem345);
    for (int32_t i346 = 0; i346 < 8; i346 = (int32_t)((uint32_t)i346 + (uint32_t)1)) {
        int32_t t347 = i346 * 64;
        vector_int32_t v348;
        asc_duplicate_scalar(v348, -1, mask344);
        int32_t t349 = t347;
        int32_t t350 = (int32_t)((uint32_t)t349 + (uint32_t)33152);
        asc_storealign(buf343 + t350, v348, mask344);
    }
}

__simd_vf__ inline void kernel_score_topk_mix_vf13(int64_t arg351, __ubuf__ uint16_t* buf352, int64_t arg353, __ubuf__ uint16_t* buf354)
{
    vector_bool vmask;
    uint32_t rem356 = 128;
    vector_bool mask355 = asc_update_mask_b16(rem356);
    vector_uint16_t v357;
    asc_duplicate_scalar(v357, 0, mask355);
    vector_uint16_t iter358 = v357;
    vector_uint16_t iter359 = v357;
    #pragma unroll(1)
    for (int64_t i360 = 0; i360 < arg353; i360 = (int64_t)((uint64_t)i360 + (uint64_t)1)) {
        int64_t t361 = (int64_t)((uint64_t)i360 * (uint64_t)256);
        int64_t t362 = (int64_t)((uint64_t)arg351 - (uint64_t)t361);
        int32_t t363 = (int32_t)t362;
        uint32_t rem365 = t363;
        vector_bool mask364 = asc_update_mask_b8(rem365);
        int32_t t366 = t361;
        vector_uint8_t v367;
        vector_uint8_t v368;
        asc_loadalign_deintlv(v367, v368, (__ubuf__ uint8_t*)(buf352 + t366));
        vector_uint16_t v369 = iter358;
        asc_cumulative_histogram_bin0(v369, v368, mask364);
        vector_uint16_t v370 = iter359;
        asc_cumulative_histogram_bin1(v370, v368, mask364);
        iter358 = v369;
        iter359 = v370;
    }
    for (int32_t i371 = 0; i371 < 1; i371 = (int32_t)((uint32_t)i371 + (uint32_t)1)) {
        asc_storealign(buf354 + 65536, iter358, mask355);
        asc_storealign(buf354 + 65664, iter359, mask355);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf14(int64_t arg372, __ubuf__ uint16_t* buf373, int32_t arg374, int64_t arg375, __ubuf__ uint16_t* buf376)
{
    vector_bool vmask;
    uint32_t rem378 = 128;
    vector_bool mask377 = asc_update_mask_b16(rem378);
    vector_uint16_t v379;
    asc_duplicate_scalar(v379, 0, mask377);
    vector_uint16_t iter380 = v379;
    vector_uint16_t iter381 = v379;
    #pragma unroll(1)
    for (int64_t i382 = 0; i382 < arg375; i382 = (int64_t)((uint64_t)i382 + (uint64_t)1)) {
        int64_t t383 = (int64_t)((uint64_t)i382 * (uint64_t)256);
        int64_t t384 = (int64_t)((uint64_t)arg372 - (uint64_t)t383);
        int32_t t385 = (int32_t)t384;
        uint32_t rem387 = t385;
        vector_bool mask386 = asc_update_mask_b8(rem387);
        int32_t t388 = t383;
        vector_uint8_t v389;
        vector_uint8_t v390;
        asc_loadalign_deintlv(v389, v390, (__ubuf__ uint8_t*)(buf373 + t388));
        bool cmp391 = arg374 < 0;
        vector_uint16_t ifr392;
        vector_uint16_t ifr393;
        if (cmp391) {
            vector_uint16_t v394 = iter380;
            asc_cumulative_histogram_bin0(v394, v390, mask386);
            vector_uint16_t v395 = iter381;
            asc_cumulative_histogram_bin1(v395, v390, mask386);
            ifr392 = v394;
            ifr393 = v395;
        } else {
            int8_t t396 = (int8_t)arg374;
            vector_bool m397;
            asc_eq_scalar(m397, v390, t396, mask386);
            vector_uint16_t v398 = iter380;
            asc_cumulative_histogram_bin0(v398, v389, m397);
            vector_uint16_t v399 = iter381;
            asc_cumulative_histogram_bin1(v399, v389, m397);
            ifr392 = v398;
            ifr393 = v399;
        }
        iter380 = ifr392;
        iter381 = ifr393;
    }
    for (int32_t i400 = 0; i400 < 1; i400 = (int32_t)((uint32_t)i400 + (uint32_t)1)) {
        asc_storealign(buf376 + 65536, iter380, mask377);
        asc_storealign(buf376 + 65664, iter381, mask377);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf15(__ubuf__ uint16_t* buf401, int64_t arg402, __ubuf__ uint16_t* buf403, int32_t arg404, int64_t arg405)
{
    vector_bool vmask;
    vector_store_unalign _sua_buf401;
    for (int32_t i406 = 0; i406 < 1; i406 = (int32_t)((uint32_t)i406 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_init();
    }
    #pragma unroll(1)
    for (int64_t i407 = 0; i407 < arg405; i407 = (int64_t)((uint64_t)i407 + (uint64_t)1)) {
        int64_t t408 = (int64_t)((uint64_t)i407 * (uint64_t)128);
        int64_t t409 = (int64_t)((uint64_t)arg402 - (uint64_t)t408);
        int32_t t410 = (int32_t)t409;
        uint32_t rem412 = t410;
        vector_bool mask411 = asc_update_mask_b16(rem412);
        int32_t t413 = t408;
        vector_uint16_t v414;
        asc_loadalign(v414, buf403 + t413);
        int16_t t415 = (int16_t)t408;
        vector_uint16_t v416;
        asc_arange(*(vector_int16_t*)&v416, static_cast<int16_t>(t415));
        int16_t t417 = (int16_t)arg404;
        vector_bool m418;
        asc_eq_scalar(m418, v414, t417, mask411);
        vector_uint16_t v419;
        cannbotdsl_92::cannbotdsl_squeeze_store(buf401 + 16384, v419, _sua_buf401, v416, m418);
    }
    for (int32_t i420 = 0; i420 < 1; i420 = (int32_t)((uint32_t)i420 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_finalize(buf401 + 16384, _sua_buf401);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf16(__ubuf__ int32_t* buf421, int64_t arg422, __ubuf__ uint16_t* buf423, int32_t arg424, int32_t arg425, int64_t arg426)
{
    vector_bool vmask;
    vector_store_unalign _sua_buf421;
    uint32_t rem428 = 64;
    vector_bool mask427 = asc_update_mask_b32(rem428);
    vector_int32_t v429;
    asc_duplicate_scalar(v429, -1, mask427);
    for (int32_t i430 = 0; i430 < 8; i430 = (int32_t)((uint32_t)i430 + (uint32_t)1)) {
        int32_t t431 = i430 * 64;
        int32_t t432 = t431;
        int32_t t433 = (int32_t)((uint32_t)t432 + (uint32_t)55296);
        asc_storealign(buf421 + t433, v429, mask427);
    }
    asc_mem_bar(VST_VST);
    for (int32_t i434 = 0; i434 < 1; i434 = (int32_t)((uint32_t)i434 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_init();
    }
    #pragma unroll(1)
    for (int64_t i435 = 0; i435 < arg426; i435 = (int64_t)((uint64_t)i435 + (uint64_t)1)) {
        int64_t t436 = (int64_t)((uint64_t)i435 * (uint64_t)64);
        int64_t t437 = (int64_t)((uint64_t)arg422 - (uint64_t)t436);
        int32_t t438 = (int32_t)t437;
        uint32_t rem440 = t438;
        vector_bool mask439 = asc_update_mask_b32(rem440);
        int32_t t441 = t436;
        vector_uint16_t v442;
        asc_loadalign(v442, buf423 + t441);
        vector_uint32_t v443;
        asc_unpack_lower(v443, v442);
        int32_t t444 = (int32_t)t436;
        vector_int32_t v445;
        asc_arange(v445, t444);
        vector_bool m446;
        asc_gt_scalar(m446, v443, arg424, mask439);
        vector_bool m447;
        asc_eq_scalar(m447, v443, arg424, mask439);
        vector_bool m448;
        asc_le_scalar(m448, v445, arg425, mask439);
        vector_bool m449;
        asc_and(m449, m447, m448, mask439);
        vector_bool m450;
        asc_or(m450, m446, m449, mask439);
        vector_bool m451;
        asc_gt_scalar(m451, v443, 127, mask439);
        vector_bool m452;
        asc_and(m452, m450, m451, mask439);
        vector_int32_t v453;
        cannbotdsl_92::cannbotdsl_squeeze_store(buf421 + 55296, v453, _sua_buf421, v445, m452);
    }
    for (int32_t i454 = 0; i454 < 1; i454 = (int32_t)((uint32_t)i454 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_finalize(buf421 + 55296, _sua_buf421);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf17(__ubuf__ int32_t* buf455, __ubuf__ uint16_t* buf456, __ubuf__ uint16_t* buf457, __ubuf__ int32_t* buf458, __ubuf__ int32_t* buf459)
{
    vector_bool vmask;
    uint32_t rem461 = 64;
    vector_bool mask460 = asc_update_mask_b32(rem461);
    uint32_t rem463 = 64;
    vector_bool mask462 = asc_update_mask_b16(rem463);
    for (int32_t i464 = 0; i464 < 8; i464 = (int32_t)((uint32_t)i464 + (uint32_t)1)) {
        int32_t t465 = i464 * 64;
        int32_t t466 = t465;
        int32_t t467 = (int32_t)((uint32_t)t466 + (uint32_t)55296);
        vector_int32_t v468;
        asc_loadalign(v468, buf455 + t467);
        vector_bool m469;
        asc_ge_scalar(m469, v468, 0, mask460);
        vector_int32_t v470;
        asc_max_scalar(v470, v468, 0, mask460);
        vector_uint32_t &r471 = (vector_uint32_t &)v470;
        vector_uint16_t v472;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v472, *(vector_uint32_t*)&r471);
        vector_uint16_t v473;
        asc_gather(v473, (__ubuf__ uint16_t*)(buf456 + 0), v472, mask462);
        vector_int32_t v474;
        asc_duplicate_scalar(v474, 0, mask460);
        vector_int32_t v475;
        asc_duplicate_scalar(v475, 1, mask460);
        vector_int32_t v476;
        asc_select(v476, v475, v474, m469);
        vector_uint32_t &r477 = (vector_uint32_t &)v476;
        vector_uint16_t v478;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v478, *(vector_uint32_t*)&r477);
        vector_bool m479;
        asc_eq_scalar(m479, v478, 1, mask462);
        vector_uint16_t v480;
        asc_duplicate_scalar(v480, 0, mask462);
        vector_uint16_t v481;
        asc_select(v481, v473, v480, m479);
        int32_t t482 = (int32_t)((uint32_t)t466 + (uint32_t)65792);
        asc_storealign(buf457 + t482, v481, mask462);
        vector_int32_t v483;
        asc_gather(v483, (__ubuf__ int32_t*)(buf458 + 16384), r471, mask460);
        vector_int32_t v484;
        asc_duplicate_scalar(v484, -1, mask460);
        vector_int32_t v485;
        asc_select(v485, v483, v484, m469);
        int32_t t486 = (int32_t)((uint32_t)t466 + (uint32_t)33152);
        asc_storealign(buf459 + t486, v485, mask460);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf18(__ubuf__ int32_t* buf487, __ubuf__ int32_t* buf488, bool arg489)
{
    vector_bool vmask;
    uint32_t rem491 = 64;
    vector_bool mask490 = asc_update_mask_b32(rem491);
    for (int32_t i492 = 0; i492 < 8; i492 = (int32_t)((uint32_t)i492 + (uint32_t)1)) {
        int32_t t493 = i492 * 64;
        int32_t t494 = t493;
        int32_t t495 = (int32_t)((uint32_t)t494 + (uint32_t)33152);
        vector_int32_t v496;
        asc_loadalign(v496, buf487 + t495);
        vector_bool m497;
        asc_ge_scalar(m497, v496, 0, mask490);
        vector_int32_t v498;
        asc_max_scalar(v498, v496, 0, mask490);
        vector_int32_t ifr499;
        if (arg489) {
            vector_uint32_t &r500 = (vector_uint32_t &)v498;
            vector_int32_t v501;
            asc_gather(v501, (__ubuf__ int32_t*)(buf488 + 33664), r500, mask490);
            ifr499 = v501;
        } else {
            vector_int32_t v502;
            asc_shiftright_scalar(v502, v498, 7, mask490);
            vector_uint32_t &r503 = (vector_uint32_t &)v502;
            vector_int32_t v504;
            asc_gather(v504, (__ubuf__ int32_t*)(buf488 + 33664), r503, mask490);
            vector_int32_t v505;
            asc_duplicate_scalar(v505, 127, mask490);
            vector_int32_t v506;
            asc_and(v506, v498, v505, mask490);
            vector_int32_t v507;
            asc_shiftleft_scalar(v507, v504, 7, mask490);
            vector_int32_t v508;
            asc_add(v508, v507, v506, mask490);
            ifr499 = v508;
        }
        vector_int32_t v509;
        asc_duplicate_scalar(v509, -1, mask490);
        vector_int32_t v510;
        asc_select(v510, ifr499, v509, m497);
        asc_storealign(buf487 + t495, v510, mask490);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_score_topk_mix_vf19(__ubuf__ uint16_t* buf511, __ubuf__ uint16_t* buf512)
{
    vector_bool vmask;
}

__simd_vf__ inline void kernel_score_topk_mix_vf20(__ubuf__ int32_t* buf513, __ubuf__ int32_t* buf514)
{
    vector_bool vmask;
}

__global__ __aicore__ void kernel_score_topk_mix(KernelArgTensor_uint8_t_2d_Device_ds_ds gm0_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds gm1_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm7_arg, KernelArgTensor_uint8_t_4d_Device_dsss_ddds gm2_arg, KernelArgTensor_fp8_e8m0_t_4d_Device_dsss_ddds gm3_arg, KernelArgTensor_int32_t_2d_Device_dd_ds gm4_arg, KernelArgTensor_int32_t_1d_Device_d_s gm5_arg, KernelArgTensor_int32_t_1d_Device_d_s gm8_arg, KernelArgTensor_int32_t_2d_Device_ds_ds gm9_arg, KernelArgTensor_int32_t_2d_Device_sd_ds gm6_arg, KernelArgTensor_uint16_t_2d_Device_dd_ds gm10_arg, KernelArgTensor_int32_t_2d_Device_dd_ds gm11_arg, KernelArgTensor_int32_t_2d_Device_dd_ds gm12_arg, KernelArgTensor_int32_t_2d_Device_dd_ds gm13_arg, KernelArgTensor_float_2d_Device_dd_ds gm14_arg, int64_t arg15, int64_t arg16, int64_t arg17, int64_t arg18, int64_t arg19, int32_t arg20, int32_t arg21, int32_t arg22)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ uint8_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e8m0_t* gm1 = gm1_arg.ptr;
    __gm__ bfloat16_t* gm7 = gm7_arg.ptr;
    __gm__ uint8_t* gm2 = gm2_arg.ptr;
    __gm__ fp8_e8m0_t* gm3 = gm3_arg.ptr;
    __gm__ int32_t* gm4 = gm4_arg.ptr;
    __gm__ int32_t* gm5 = gm5_arg.ptr;
    __gm__ int32_t* gm8 = gm8_arg.ptr;
    __gm__ int32_t* gm9 = gm9_arg.ptr;
    __gm__ int32_t* gm6 = gm6_arg.ptr;
    __gm__ uint16_t* gm10 = gm10_arg.ptr;
    __gm__ int32_t* gm11 = gm11_arg.ptr;
    __gm__ int32_t* gm12 = gm12_arg.ptr;
    __gm__ int32_t* gm13 = gm13_arg.ptr;
    __gm__ float* gm14 = gm14_arg.ptr;
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

    __ubuf__ uint8_t* buf515 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(0);
    __cc__ float* buf516 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ fp4x2_e2m1_t* buf517 = (__ca__ fp4x2_e2m1_t*)(uintptr_t)(uint64_t)(0);
    __cb__ fp4x2_e2m1_t* buf518 = (__cb__ fp4x2_e2m1_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ fp8_e8m0_t* buf519 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(29696);
    __cbuf__ fp4x2_e2m1_t* buf520 = (__cbuf__ fp4x2_e2m1_t*)(uintptr_t)(uint64_t)(12288);
    __cbuf__ fp8_e8m0_t* buf521 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(28672);
    __cbuf__ fp4x2_e2m1_t* buf522 = (__cbuf__ fp4x2_e2m1_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ uint8_t* buf523 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(0);

    if ASCEND_IS_AIC {
        int64_t dim524 = gm14_arg.shape[1];
        int64_t t525 = (int64_t)((uint64_t)dim524 + (uint64_t)255);
        int64_t t526 = t525 % 256;
        bool cmp527 = t526 != 0;
        bool cmp528 = t526 < 0;
        bool t529 = cmp527 & cmp528;
        int64_t t530 = t525 / 256;
        int64_t t531 = (int64_t)(uint8_t)t529;
        int64_t t532 = (int64_t)((uint64_t)t530 - (uint64_t)t531);
        int64_t dim533 = gm6_arg.stride[0];
        int64_t t534 = dim533;
        int64_t t535 = t534 * 3;
        int32_t ld536 = gm6[t535];
        int64_t t537 = (int64_t)ld536;
        int64_t t538 = (int64_t)((uint64_t)t537 * (uint64_t)t532);
        int64_t t539 = (arg16 < t538 ? arg16 : t538);
        int64_t dim540 = gm0_arg.shape[0];
        int64_t t541 = (int64_t)((uint64_t)dim540 * (uint64_t)32);
        int64_t dim542 = gm2_arg.shape[0];
        for (int64_t i543 = arg17; i543 < t539; i543 = (int64_t)((uint64_t)i543 + (uint64_t)arg18)) {
            int64_t t544 = (int64_t)((uint64_t)t538 * (uint64_t)i543);
            int64_t t545 = t544 % t539;
            bool cmp546 = t545 != 0;
            bool cmp547 = t545 < 0;
            bool cmp548 = t539 < 0;
            bool t549 = cmp547 ^ cmp548;
            bool t550 = cmp546 & t549;
            int64_t t551 = t544 / t539;
            int64_t t552 = (int64_t)(uint8_t)t550;
            int64_t t553 = (int64_t)((uint64_t)t551 - (uint64_t)t552);
            int64_t t554 = (int64_t)((uint64_t)i543 + (uint64_t)1);
            int64_t t555 = (int64_t)((uint64_t)t538 * (uint64_t)t554);
            int64_t t556 = t555 % t539;
            bool cmp557 = t556 != 0;
            bool cmp558 = t556 < 0;
            bool t559 = cmp558 ^ cmp548;
            bool t560 = cmp557 & t559;
            int64_t t561 = t555 / t539;
            int64_t t562 = (int64_t)(uint8_t)t560;
            int64_t t563 = (int64_t)((uint64_t)t561 - (uint64_t)t562);
            int64_t wh564 = t553;
            int64_t wh_result565;
            while (true) {
                bool cmp566 = wh564 < t563;
                wh_result565 = wh564;
                if (!cmp566) break;
                int64_t t567 = wh_result565 % t532;
                bool cmp568 = t567 != 0;
                bool cmp569 = t567 < 0;
                bool cmp570 = t532 < 0;
                bool t571 = cmp569 ^ cmp570;
                bool t572 = cmp568 & t571;
                int64_t t573 = wh_result565 / t532;
                int64_t t574 = (int64_t)(uint8_t)t572;
                int64_t t575 = (int64_t)((uint64_t)t573 - (uint64_t)t574);
                int64_t t576 = (int64_t)((uint64_t)t575 + (uint64_t)1);
                int64_t t577 = (int64_t)((uint64_t)t576 * (uint64_t)t532);
                int64_t t578 = (t563 < t577 ? t563 : t577);
                int64_t sel579 = t572 ? t532 : 0;
                int64_t t580 = (int64_t)((uint64_t)t567 + (uint64_t)sel579);
                int64_t t581 = (int64_t)((uint64_t)t578 - (uint64_t)wh_result565);
                int64_t t582 = t575;
                int32_t ld583 = gm6[t582];
                int64_t t584 = (int64_t)ld583;
                int64_t t585 = t534 * 2;
                int64_t t586 = t585 + t582;
                int32_t ld587 = gm6[t586];
                int64_t t588 = (int64_t)ld587;
                int64_t t589 = ld583;
                int32_t ld590 = gm5[t589];
                int64_t t591 = (int64_t)ld590;
                int64_t t592 = (int64_t)((uint64_t)t584 * (uint64_t)32);
                int64_t t593 = (int64_t)((uint64_t)t592 + (uint64_t)192);
                int64_t t594 = (t593 < t541 ? t593 : t541);
                int64_t t595 = (int64_t)((uint64_t)t594 - (uint64_t)t592);
                __cbuf__ fp4x2_e2m1_t* view_root596 = reinterpret_cast<__cbuf__ fp4x2_e2m1_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf522) + 0);
                int64_t t597 = (int64_t)((uint64_t)t592 / (uint64_t)4);
                int64_t t598 = (int64_t)((uint64_t)t597 * (uint64_t)512);
                __gm__ fp4x2_e2m1_t* view_root599 = reinterpret_cast<__gm__ fp4x2_e2m1_t*>(reinterpret_cast<__gm__ uint8_t*>(gm0) + 0);
                bool cmp600 = (uint64_t)t595 > (uint64_t)0;
                if (cmp600) {
                    int64_t t601 = t598;
                    int32_t t602 = (int32_t)t595;
                    set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(192) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                    asc_copy_gm2l1_nd2nz((__cbuf__ int8_t*)(view_root596 + (0) / 2), (__gm__ int8_t*)(view_root599 + (t601) / 2), 64, 0, t602, 64, 0, false);
                }
                int64_t t603 = (int64_t)((uint64_t)t584 * (uint64_t)128);
                __cbuf__ fp8_e8m0_t* view_root604 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf521) + 0);
                asc_lock(PIPE_MTE2, 2);
                if (cmp600) {
                    int64_t t605 = t603;
                    int32_t t606 = (int32_t)t595;
                    set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(2) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                    asc_copy_gm2l1_dn2nz((__cbuf__ int16_t*)(view_root604 + 0), (__gm__ int16_t*)(gm1 + t605), 4, 0, 2, t606, 0, false);
                }
                asc_unlock(PIPE_MTE2, 2);
                __ca__ fp4x2_e2m1_t* view_root607 = reinterpret_cast<__ca__ fp4x2_e2m1_t*>(reinterpret_cast<__ca__ uint8_t*>(buf517) + 0);
                uint64_t selected_root608 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root607)) / 16;
                asc_lock(PIPE_MTE1, 2);
                asc_copy_l12l0a(view_root607 + (0) / 2, view_root596 + (0) / 2, 0, 0, 12, 2, 12, 12);
                asc_copy_l12l0a_mx(selected_root608, (__cbuf__ fp8_e8m0_t*)(view_root604 + 0), 0, 0, 12, 2, 2, 2);
                asc_unlock(PIPE_MTE1, 2);
                __ubuf__ bfloat16_t* view_root609 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf515) + 0);
                for (int64_t i610 = 0; i610 < t581; i610 = (int64_t)((uint64_t)i610 + (uint64_t)1)) {
                    int64_t t611 = (int64_t)((uint64_t)t580 + (uint64_t)i610);
                    int64_t t612 = (int64_t)((uint64_t)t611 * (uint64_t)256);
                    bool cmp613 = arg15 > 0;
                    int64_t ifr614;
                    int64_t ifr615;
                    if (cmp613) {
                        bool cmp616 = t612 < t588;
                        int64_t ifr617;
                        if (cmp616) {
                            int64_t t618 = (int64_t)((uint64_t)t584 * (uint64_t)arg15);
                            int64_t t619 = (int64_t)((uint64_t)t618 + (uint64_t)t612);
                            int64_t t620 = t619 % 128;
                            bool cmp621 = t620 != 0;
                            bool cmp622 = t620 < 0;
                            bool t623 = cmp621 & cmp622;
                            int64_t t624 = t619 / 128;
                            int64_t t625 = (int64_t)(uint8_t)t623;
                            int64_t t626 = (int64_t)((uint64_t)t624 - (uint64_t)t625);
                            ifr617 = t626;
                        } else {
                            ifr617 = -1;
                        }
                        int64_t t627 = (int64_t)((uint64_t)t612 + (uint64_t)128);
                        bool cmp628 = t627 < t588;
                        int64_t ifr629;
                        if (cmp628) {
                            int64_t t630 = (int64_t)((uint64_t)t584 * (uint64_t)arg15);
                            int64_t t631 = (int64_t)((uint64_t)t630 + (uint64_t)t612);
                            int64_t t632 = t631 % 128;
                            bool cmp633 = t632 != 0;
                            bool cmp634 = t632 < 0;
                            bool t635 = cmp633 & cmp634;
                            int64_t t636 = t631 / 128;
                            int64_t t637 = (int64_t)(uint8_t)t635;
                            int64_t t638 = (int64_t)((uint64_t)t636 - (uint64_t)t637);
                            int64_t t639 = (int64_t)((uint64_t)t638 + (uint64_t)1);
                            ifr629 = t639;
                        } else {
                            ifr629 = -1;
                        }
                        ifr614 = ifr617;
                        ifr615 = ifr629;
                    } else {
                        bool cmp640 = t591 >= 0;
                        bool ifr641;
                        if (cmp640) {
                            int64_t dim642 = gm4_arg.shape[0];
                            bool cmp643 = t591 < dim642;
                            ifr641 = cmp643;
                        } else {
                            ifr641 = false;
                        }
                        int64_t ifr644;
                        int64_t ifr645;
                        if (ifr641) {
                            bool cmp646 = t612 < t588;
                            bool ifr647;
                            if (cmp646) {
                                int64_t t648 = t612 % 128;
                                bool cmp649 = t648 != 0;
                                bool cmp650 = t648 < 0;
                                bool t651 = cmp649 & cmp650;
                                int64_t t652 = t612 / 128;
                                int64_t t653 = (int64_t)(uint8_t)t651;
                                int64_t t654 = (int64_t)((uint64_t)t652 - (uint64_t)t653);
                                int64_t dim655 = gm4_arg.shape[1];
                                bool cmp656 = t654 < dim655;
                                ifr647 = cmp656;
                            } else {
                                ifr647 = false;
                            }
                            int64_t ifr657;
                            if (ifr647) {
                                int64_t t658 = t612 % 128;
                                bool cmp659 = t658 != 0;
                                bool cmp660 = t658 < 0;
                                bool t661 = cmp659 & cmp660;
                                int64_t t662 = t612 / 128;
                                int64_t t663 = (int64_t)(uint8_t)t661;
                                int64_t t664 = (int64_t)((uint64_t)t662 - (uint64_t)t663);
                                int64_t dim665 = gm4_arg.stride[0];
                                int64_t t666 = dim665;
                                int64_t t667 = ld590;
                                int64_t t668 = t664;
                                int64_t t669 = t667 * t666;
                                int64_t t670 = t669 + t668;
                                int32_t ld671 = gm4[t670];
                                int64_t t672 = (int64_t)ld671;
                                ifr657 = t672;
                            } else {
                                ifr657 = -1;
                            }
                            int64_t t673 = (int64_t)((uint64_t)t612 + (uint64_t)128);
                            bool cmp674 = t673 < t588;
                            bool ifr675;
                            if (cmp674) {
                                int64_t t676 = t612 % 128;
                                bool cmp677 = t676 != 0;
                                bool cmp678 = t676 < 0;
                                bool t679 = cmp677 & cmp678;
                                int64_t t680 = t612 / 128;
                                int64_t t681 = (int64_t)(uint8_t)t679;
                                int64_t t682 = (int64_t)((uint64_t)t680 - (uint64_t)t681);
                                int64_t t683 = (int64_t)((uint64_t)t682 + (uint64_t)1);
                                int64_t dim684 = gm4_arg.shape[1];
                                bool cmp685 = t683 < dim684;
                                ifr675 = cmp685;
                            } else {
                                ifr675 = false;
                            }
                            int64_t ifr686;
                            if (ifr675) {
                                int64_t t687 = t612 % 128;
                                bool cmp688 = t687 != 0;
                                bool cmp689 = t687 < 0;
                                bool t690 = cmp688 & cmp689;
                                int64_t t691 = t612 / 128;
                                int64_t t692 = (int64_t)(uint8_t)t690;
                                int64_t t693 = (int64_t)((uint64_t)t691 - (uint64_t)t692);
                                int64_t t694 = (int64_t)((uint64_t)t693 + (uint64_t)1);
                                int64_t dim695 = gm4_arg.stride[0];
                                int64_t t696 = dim695;
                                int64_t t697 = ld590;
                                int64_t t698 = t694;
                                int64_t t699 = t697 * t696;
                                int64_t t700 = t699 + t698;
                                int32_t ld701 = gm4[t700];
                                int64_t t702 = (int64_t)ld701;
                                ifr686 = t702;
                            } else {
                                ifr686 = -1;
                            }
                            ifr644 = ifr657;
                            ifr645 = ifr686;
                        } else {
                            ifr644 = -1;
                            ifr645 = -1;
                        }
                        bool cmp703 = ifr644 >= dim542;
                        int64_t sel704 = cmp703 ? -1 : ifr644;
                        bool cmp705 = ifr645 >= dim542;
                        int64_t sel706 = cmp705 ? -1 : ifr645;
                        ifr614 = sel704;
                        ifr615 = sel706;
                    }
                    bool cmp707 = ifr614 >= 0;
                    bool cmp708 = ifr615 >= 0;
                    bool sel709 = cmp707 ? true : cmp708;
                    if (sel709) {
                        for (int64_t i710 = 0; i710 < 2; i710 = (int64_t)((uint64_t)i710 + (uint64_t)1)) {
                            bool cmp711 = i710 == 0;
                            int64_t sel712 = cmp711 ? ifr614 : ifr615;
                            bool cmp713 = sel712 < 0;
                            int64_t sel714 = cmp707 ? ifr614 : ifr615;
                            int64_t sel715 = cmp713 ? sel714 : sel712;
                            int64_t t716 = (int64_t)((uint64_t)sel715 * (uint64_t)128);
                            __cbuf__ fp4x2_e2m1_t* view_root717 = reinterpret_cast<__cbuf__ fp4x2_e2m1_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf520) + 0);
                            int64_t t718 = (int64_t)((uint64_t)t716 / (uint64_t)4);
                            int64_t t719 = (int64_t)((uint64_t)t718 * (uint64_t)512);
                            int64_t t720 = i710 * 8192;
                            asc_lock(PIPE_MTE2, 1);
                            __gm__ fp4x2_e2m1_t* view_root721 = reinterpret_cast<__gm__ fp4x2_e2m1_t*>(reinterpret_cast<__gm__ uint8_t*>(gm2) + 0);
                            int64_t t722 = t720;
                            int64_t t723 = t719;
                            set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(256) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                            asc_copy_gm2l1_nd2nz((__cbuf__ int8_t*)(view_root717 + (t722) / 2), (__gm__ int8_t*)(view_root721 + (t723) / 2), 64, 0, 128, 64, 0, false);
                            asc_unlock(PIPE_MTE2, 1);
                            int64_t t724 = (int64_t)((uint64_t)sel715 * (uint64_t)512);
                            int64_t t725 = i710 * 128;
                            int64_t t726 = t725 / 16;
                            int64_t t727 = t726 * 64;
                            int64_t t728 = t725 % 16;
                            int64_t t729 = t728 * 2;
                            int64_t t730 = t727 + t729;
                            int64_t t731 = t730 * 8;
                            int64_t t732 = (int64_t)((uint64_t)t731 / (uint64_t)8);
                            int64_t t733 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(buf519) + (uint64_t)t732);
                            __cbuf__ fp8_e8m0_t* selected_root734 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(t733);
                            asc_lock(PIPE_MTE2, 3);
                            int64_t t735 = t724;
                            set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(2) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                            asc_copy_gm2l1_dn2nz((__cbuf__ int16_t*)(selected_root734 + 0), (__gm__ int16_t*)(gm3 + t735), 4, 0, 2, 128, 0, false);
                            asc_unlock(PIPE_MTE2, 3);
                        }
                        __cb__ fp4x2_e2m1_t* view_root736 = reinterpret_cast<__cb__ fp4x2_e2m1_t*>(reinterpret_cast<__cb__ uint8_t*>(buf518) + 0);
                        __cbuf__ fp4x2_e2m1_t* view_root737 = reinterpret_cast<__cbuf__ fp4x2_e2m1_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf520) + 0);
                        __cbuf__ fp8_e8m0_t* view_root738 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf519) + 0);
                        uint64_t selected_root739 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root736)) / 16;
                        asc_lock(PIPE_MTE1, 5);
                        asc_lock(PIPE_MTE1, 1);
                        asc_lock(PIPE_MTE1, 3);
                        asc_copy_l12l0b(view_root736 + (0) / 2, view_root737 + (0) / 2, 0, 0, 16, 2, 16, 16);
                        asc_copy_l12l0b_mx(selected_root739, (__cbuf__ fp8_e8m0_t*)(view_root738 + 0), 0, 0, 16, 2, 2, 2);
                        asc_unlock(PIPE_MTE1, 3);
                        asc_unlock(PIPE_MTE1, 1);
                        asc_unlock(PIPE_MTE1, 5);
                        __cc__ float* view_root740 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf516) + 0);
                        asc_lock(PIPE_M, 6);
                        asc_lock(PIPE_M, 5);
                        asc_mmad_mx(view_root740 + 0, view_root607, view_root736, 192, 128, 256, 0, true, false, true);
                        asc_unlock(PIPE_M, 5);
                        asc_unlock(PIPE_M, 6);
                        asc_sync_intra_wait(PIPE_FIX, 0);
                        asc_sync_intra_wait(PIPE_FIX, 16);
                        asc_lock(PIPE_FIX, 6);
                        asc_set_l0c2gm_nz2nd(1, 0, 0);
                        asc_copy_l0c2ub((__ubuf__ bfloat16_t*)view_root609 + 0, view_root740, 256, 96, 256, 192, 0, static_cast<bool>(0), 0, 0, QuantMode_t::F322BF16, 0, false, true, 0, 0, false, 0, false, false, false, false);
                        asc_set_l0c2gm_nz2nd(1, 0, 0);
                        asc_copy_l0c2ub((__ubuf__ bfloat16_t*)view_root609 + 0, view_root740 + 1536, 256, 96, 256, 192, 0, static_cast<bool>(1), 0, 0, QuantMode_t::F322BF16, 0, false, true, 0, 0, false, 0, false, false, false, false);
                        asc_unlock(PIPE_FIX, 6);
                        asc_sync_intra_arrive(PIPE_FIX, 0);
                        asc_sync_intra_arrive(PIPE_FIX, 16);
                    }
                }
                asc_sync();
                asc_sync_block_wait(PIPE_S, 5);
                ffts_cross_core_sync(PIPE_MTE3, cannbotdsl_get_ffts_config(6, 2));
                wh564 = t578;
            }
        }
        asc_sync_intra_wait(PIPE_FIX, 0);
        asc_sync_intra_wait(PIPE_FIX, 16);
        asc_sync();
        asc_sync_block_wait(PIPE_S, 0);
        ffts_cross_core_sync(PIPE_FIX, cannbotdsl_get_ffts_config(1, 0));
        asc_sync_block_wait(PIPE_S, 1);
        ffts_cross_core_sync(PIPE_MTE3, cannbotdsl_get_ffts_config(2, 2));
    }

    if ASCEND_IS_AIV {
        asc_sync_intra_arrive(PIPE_V, 0);
        int64_t dim741 = gm14_arg.shape[1];
        int64_t t742 = (int64_t)((uint64_t)dim741 + (uint64_t)255);
        int64_t t743 = t742 % 256;
        bool cmp744 = t743 != 0;
        bool cmp745 = t743 < 0;
        bool t746 = cmp744 & cmp745;
        int64_t t747 = t742 / 256;
        int64_t t748 = (int64_t)(uint8_t)t746;
        int64_t t749 = (int64_t)((uint64_t)t747 - (uint64_t)t748);
        int64_t dim750 = gm6_arg.stride[0];
        int64_t t751 = dim750;
        int64_t t752 = t751 * 3;
        int32_t ld753 = gm6[t752];
        int64_t t754 = (int64_t)ld753;
        int64_t t755 = (int64_t)((uint64_t)t754 * (uint64_t)t749);
        int64_t t756 = (arg16 < t755 ? arg16 : t755);
        int64_t dim757 = gm2_arg.shape[0];
        for (int64_t i758 = arg17; i758 < t756; i758 = (int64_t)((uint64_t)i758 + (uint64_t)arg18)) {
            int64_t t759 = (int64_t)((uint64_t)t755 * (uint64_t)i758);
            int64_t t760 = t759 % t756;
            bool cmp761 = t760 != 0;
            bool cmp762 = t760 < 0;
            bool cmp763 = t756 < 0;
            bool t764 = cmp762 ^ cmp763;
            bool t765 = cmp761 & t764;
            int64_t t766 = t759 / t756;
            int64_t t767 = (int64_t)(uint8_t)t765;
            int64_t t768 = (int64_t)((uint64_t)t766 - (uint64_t)t767);
            int64_t t769 = (int64_t)((uint64_t)i758 + (uint64_t)1);
            int64_t t770 = (int64_t)((uint64_t)t755 * (uint64_t)t769);
            int64_t t771 = t770 % t756;
            bool cmp772 = t771 != 0;
            bool cmp773 = t771 < 0;
            bool t774 = cmp773 ^ cmp763;
            bool t775 = cmp772 & t774;
            int64_t t776 = t770 / t756;
            int64_t t777 = (int64_t)(uint8_t)t775;
            int64_t t778 = (int64_t)((uint64_t)t776 - (uint64_t)t777);
            int64_t wh779 = t768;
            int64_t wh_result780;
            while (true) {
                bool cmp781 = wh779 < t778;
                wh_result780 = wh779;
                if (!cmp781) break;
                int64_t t782 = wh_result780 % t749;
                bool cmp783 = t782 != 0;
                bool cmp784 = t782 < 0;
                bool cmp785 = t749 < 0;
                bool t786 = cmp784 ^ cmp785;
                bool t787 = cmp783 & t786;
                int64_t t788 = wh_result780 / t749;
                int64_t t789 = (int64_t)(uint8_t)t787;
                int64_t t790 = (int64_t)((uint64_t)t788 - (uint64_t)t789);
                int64_t t791 = (int64_t)((uint64_t)t790 + (uint64_t)1);
                int64_t t792 = (int64_t)((uint64_t)t791 * (uint64_t)t749);
                int64_t t793 = (t778 < t792 ? t778 : t792);
                int64_t sel794 = t787 ? t749 : 0;
                int64_t t795 = (int64_t)((uint64_t)t782 + (uint64_t)sel794);
                int64_t t796 = (int64_t)((uint64_t)t793 - (uint64_t)wh_result780);
                int64_t t797 = t790;
                int32_t ld798 = gm6[t797];
                int64_t t799 = (int64_t)ld798;
                int64_t t800 = t751 + t797;
                int32_t ld801 = gm6[t800];
                int64_t t802 = (int64_t)ld801;
                int64_t t803 = (int64_t)((uint64_t)t802 - (uint64_t)t799);
                int64_t t804 = t751 * 2;
                int64_t t805 = t804 + t797;
                int32_t ld806 = gm6[t805];
                int64_t t807 = (int64_t)ld806;
                int64_t t808 = ld798;
                int32_t ld809 = gm5[t808];
                int64_t t810 = (int64_t)ld809;
                asc_lock(PIPE_V, 4);
                __ubuf__ bfloat16_t* view_root811 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                kernel_score_topk_mix_vf0((__ubuf__ bfloat16_t*)view_root811);
                asc_unlock(PIPE_V, 4);
                int64_t dim812 = gm7_arg.stride[0];
                int64_t t813 = (int64_t)((uint64_t)dim812 * (uint64_t)2);
                int64_t t814 = (int64_t)((uint64_t)t799 * (uint64_t)dim812);
                int64_t t815 = (int64_t)((uint64_t)t814 * (uint64_t)2);
                asc_lock(PIPE_MTE2, 4);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf523 + 220160), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm7 + t815), t803, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t813, 64);
                }
                asc_unlock(PIPE_MTE2, 4);
                asc_lock(PIPE_V, 4);
                kernel_score_topk_mix_vf1((__ubuf__ bfloat16_t*)view_root811, (__ubuf__ bfloat16_t*)view_root811);
                asc_unlock(PIPE_V, 4);
                for (int64_t i816 = 0; i816 < t796; i816 = (int64_t)((uint64_t)i816 + (uint64_t)1)) {
                    int64_t t817 = (int64_t)((uint64_t)t795 + (uint64_t)i816);
                    int64_t t818 = (int64_t)((uint64_t)t817 * (uint64_t)256);
                    bool cmp819 = arg15 > 0;
                    int64_t ifr820;
                    int64_t ifr821;
                    if (cmp819) {
                        bool cmp822 = t818 < t807;
                        int64_t ifr823;
                        if (cmp822) {
                            int64_t t824 = (int64_t)((uint64_t)t799 * (uint64_t)arg15);
                            int64_t t825 = (int64_t)((uint64_t)t824 + (uint64_t)t818);
                            int64_t t826 = t825 % 128;
                            bool cmp827 = t826 != 0;
                            bool cmp828 = t826 < 0;
                            bool t829 = cmp827 & cmp828;
                            int64_t t830 = t825 / 128;
                            int64_t t831 = (int64_t)(uint8_t)t829;
                            int64_t t832 = (int64_t)((uint64_t)t830 - (uint64_t)t831);
                            ifr823 = t832;
                        } else {
                            ifr823 = -1;
                        }
                        int64_t t833 = (int64_t)((uint64_t)t818 + (uint64_t)128);
                        bool cmp834 = t833 < t807;
                        int64_t ifr835;
                        if (cmp834) {
                            int64_t t836 = (int64_t)((uint64_t)t799 * (uint64_t)arg15);
                            int64_t t837 = (int64_t)((uint64_t)t836 + (uint64_t)t818);
                            int64_t t838 = t837 % 128;
                            bool cmp839 = t838 != 0;
                            bool cmp840 = t838 < 0;
                            bool t841 = cmp839 & cmp840;
                            int64_t t842 = t837 / 128;
                            int64_t t843 = (int64_t)(uint8_t)t841;
                            int64_t t844 = (int64_t)((uint64_t)t842 - (uint64_t)t843);
                            int64_t t845 = (int64_t)((uint64_t)t844 + (uint64_t)1);
                            ifr835 = t845;
                        } else {
                            ifr835 = -1;
                        }
                        ifr820 = ifr823;
                        ifr821 = ifr835;
                    } else {
                        bool cmp846 = t810 >= 0;
                        bool ifr847;
                        if (cmp846) {
                            int64_t dim848 = gm4_arg.shape[0];
                            bool cmp849 = t810 < dim848;
                            ifr847 = cmp849;
                        } else {
                            ifr847 = false;
                        }
                        int64_t ifr850;
                        int64_t ifr851;
                        if (ifr847) {
                            bool cmp852 = t818 < t807;
                            bool ifr853;
                            if (cmp852) {
                                int64_t t854 = t818 % 128;
                                bool cmp855 = t854 != 0;
                                bool cmp856 = t854 < 0;
                                bool t857 = cmp855 & cmp856;
                                int64_t t858 = t818 / 128;
                                int64_t t859 = (int64_t)(uint8_t)t857;
                                int64_t t860 = (int64_t)((uint64_t)t858 - (uint64_t)t859);
                                int64_t dim861 = gm4_arg.shape[1];
                                bool cmp862 = t860 < dim861;
                                ifr853 = cmp862;
                            } else {
                                ifr853 = false;
                            }
                            int64_t ifr863;
                            if (ifr853) {
                                int64_t t864 = t818 % 128;
                                bool cmp865 = t864 != 0;
                                bool cmp866 = t864 < 0;
                                bool t867 = cmp865 & cmp866;
                                int64_t t868 = t818 / 128;
                                int64_t t869 = (int64_t)(uint8_t)t867;
                                int64_t t870 = (int64_t)((uint64_t)t868 - (uint64_t)t869);
                                int64_t dim871 = gm4_arg.stride[0];
                                int64_t t872 = dim871;
                                int64_t t873 = ld809;
                                int64_t t874 = t870;
                                int64_t t875 = t873 * t872;
                                int64_t t876 = t875 + t874;
                                int32_t ld877 = gm4[t876];
                                int64_t t878 = (int64_t)ld877;
                                ifr863 = t878;
                            } else {
                                ifr863 = -1;
                            }
                            int64_t t879 = (int64_t)((uint64_t)t818 + (uint64_t)128);
                            bool cmp880 = t879 < t807;
                            bool ifr881;
                            if (cmp880) {
                                int64_t t882 = t818 % 128;
                                bool cmp883 = t882 != 0;
                                bool cmp884 = t882 < 0;
                                bool t885 = cmp883 & cmp884;
                                int64_t t886 = t818 / 128;
                                int64_t t887 = (int64_t)(uint8_t)t885;
                                int64_t t888 = (int64_t)((uint64_t)t886 - (uint64_t)t887);
                                int64_t t889 = (int64_t)((uint64_t)t888 + (uint64_t)1);
                                int64_t dim890 = gm4_arg.shape[1];
                                bool cmp891 = t889 < dim890;
                                ifr881 = cmp891;
                            } else {
                                ifr881 = false;
                            }
                            int64_t ifr892;
                            if (ifr881) {
                                int64_t t893 = t818 % 128;
                                bool cmp894 = t893 != 0;
                                bool cmp895 = t893 < 0;
                                bool t896 = cmp894 & cmp895;
                                int64_t t897 = t818 / 128;
                                int64_t t898 = (int64_t)(uint8_t)t896;
                                int64_t t899 = (int64_t)((uint64_t)t897 - (uint64_t)t898);
                                int64_t t900 = (int64_t)((uint64_t)t899 + (uint64_t)1);
                                int64_t dim901 = gm4_arg.stride[0];
                                int64_t t902 = dim901;
                                int64_t t903 = ld809;
                                int64_t t904 = t900;
                                int64_t t905 = t903 * t902;
                                int64_t t906 = t905 + t904;
                                int32_t ld907 = gm4[t906];
                                int64_t t908 = (int64_t)ld907;
                                ifr892 = t908;
                            } else {
                                ifr892 = -1;
                            }
                            ifr850 = ifr863;
                            ifr851 = ifr892;
                        } else {
                            ifr850 = -1;
                            ifr851 = -1;
                        }
                        bool cmp909 = ifr850 >= dim757;
                        int64_t sel910 = cmp909 ? -1 : ifr850;
                        bool cmp911 = ifr851 >= dim757;
                        int64_t sel912 = cmp911 ? -1 : ifr851;
                        ifr820 = sel910;
                        ifr821 = sel912;
                    }
                    bool cmp913 = ifr820 >= 0;
                    bool cmp914 = ifr821 >= 0;
                    bool sel915 = cmp913 ? true : cmp914;
                    if (cmp819) {
                        int64_t t916 = (int64_t)((uint64_t)t799 * (uint64_t)arg15);
                        int64_t t917 = (int64_t)((uint64_t)t916 + (uint64_t)t818);
                        int64_t dim918 = gm12_arg.shape[1];
                        int64_t t919 = (int64_t)((uint64_t)dim918 - (uint64_t)t917);
                        int64_t t920 = (t919 < 256 ? t919 : 256);
                        int64_t t921 = (int64_t)((uint64_t)t917 * (uint64_t)4);
                        int64_t t922 = (int64_t)((uint64_t)t920 * (uint64_t)4);
                        bool cmp923 = t922 == 0;
                        bool cmp924 = t922 != 0;
                        int64_t t925 = (int64_t)(uint8_t)cmp924;
                        int64_t sel926 = cmp923 ? 4 : t922;
                        asc_lock(PIPE_MTE2, 5);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf523 + 223232), (__gm__ int32_t*)((__gm__ uint8_t*)gm12 + t921), t925, sel926, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        }
                        asc_unlock(PIPE_MTE2, 5);
                        asc_lock(PIPE_V, 5);
                        __ubuf__ int32_t* view_root927 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                        __ubuf__ uint16_t* view_root928 = reinterpret_cast<__ubuf__ uint16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                        kernel_score_topk_mix_vf2((__ubuf__ int32_t*)view_root927, (__ubuf__ uint16_t*)view_root928);
                        asc_unlock(PIPE_V, 5);
                    }
                    if (sel915) {
                        asc_sync_intra_wait(PIPE_V, 0);
                        #pragma unroll(1)
                        for (int64_t i929 = 0; i929 < 3; i929 = (int64_t)((uint64_t)i929 + (uint64_t)1)) {
                            int64_t t930 = (int64_t)((uint64_t)arg19 * (uint64_t)3);
                            int64_t t931 = (int64_t)((uint64_t)t930 + (uint64_t)i929);
                            bool cmp932 = t931 < t803;
                            int32_t ifr933;
                            if (cmp932) {
                                int64_t ifr934;
                                if (cmp819) {
                                    ifr934 = t807;
                                } else {
                                    int64_t t935 = (int64_t)((uint64_t)t799 + (uint64_t)t931);
                                    int64_t t936 = t935;
                                    int32_t ld937 = gm8[t936];
                                    int64_t t938 = (int64_t)ld937;
                                    int64_t t939 = (t938 < dim741 ? t938 : dim741);
                                    ifr934 = t939;
                                }
                                int32_t t940 = (int32_t)ifr934;
                                ifr933 = t940;
                            } else {
                                ifr933 = 0;
                            }
                            #pragma unroll(1)
                            for (int64_t i941 = 0; i941 < 2; i941 = (int64_t)((uint64_t)i941 + (uint64_t)1)) {
                                int64_t t942 = (int64_t)ifr933;
                                int64_t t943 = (int64_t)((uint64_t)t942 - (uint64_t)t818);
                                int64_t t944 = i941 * 128;
                                int64_t t945 = (int64_t)((uint64_t)t943 - (uint64_t)t944);
                                int64_t t946 = (t945 < 128 ? t945 : 128);
                                int64_t t947 = (t946 > 0 ? t946 : 0);
                                bool cmp948 = i941 == 0;
                                bool cmp949 = ifr820 < 0;
                                bool sel950 = cmp948 ? cmp949 : false;
                                bool cmp951 = i941 == 1;
                                bool cmp952 = ifr821 < 0;
                                bool sel953 = cmp951 ? cmp952 : false;
                                bool sel954 = sel950 ? true : sel953;
                                int64_t sel955 = sel954 ? 0 : t947;
                                int32_t t956 = (int32_t)t931;
                                int32_t t957 = (int32_t)((uint32_t)t956 * (uint32_t)128);
                                int64_t t958 = t957;
                                int32_t t959 = (int32_t)sel955;
                                int32_t t960 = (int32_t)i929;
                                int32_t t961 = t960 * 25600;
                                int64_t t962 = (int64_t)((uint64_t)i816 * (uint64_t)256);
                                int64_t t963 = (int64_t)t961;
                                int64_t t964 = (int64_t)((uint64_t)t963 + (uint64_t)t962);
                                int32_t t965 = (int32_t)i941;
                                int32_t t966 = t965 * 128;
                                int64_t t967 = (int64_t)t966;
                                int64_t t968 = (int64_t)((uint64_t)t964 + (uint64_t)t967);
                                int64_t t969 = t968;
                                __ubuf__ uint16_t* view_root970 = reinterpret_cast<__ubuf__ uint16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                                kernel_score_topk_mix_vf3((__ubuf__ bfloat16_t*)view_root811, (int32_t)t958, i929, i941, (__ubuf__ bfloat16_t*)view_root811, (__ubuf__ uint16_t*)view_root970, cmp819, t959, (__ubuf__ uint16_t*)view_root970, (int32_t)t969);
                            }
                        }
                        asc_sync_intra_arrive(PIPE_V, 0);
                    } else {
                        #pragma unroll(1)
                        for (int64_t i971 = 0; i971 < 3; i971 = (int64_t)((uint64_t)i971 + (uint64_t)1)) {
                            int64_t t972 = (int64_t)((uint64_t)arg19 * (uint64_t)3);
                            int64_t t973 = (int64_t)((uint64_t)t972 + (uint64_t)i971);
                            bool cmp974 = t973 < t803;
                            int32_t ifr975;
                            if (cmp974) {
                                int64_t ifr976;
                                if (cmp819) {
                                    ifr976 = t807;
                                } else {
                                    int64_t t977 = (int64_t)((uint64_t)t799 + (uint64_t)t973);
                                    int64_t t978 = t977;
                                    int32_t ld979 = gm8[t978];
                                    int64_t t980 = (int64_t)ld979;
                                    int64_t t981 = (t980 < dim741 ? t980 : dim741);
                                    ifr976 = t981;
                                }
                                int32_t t982 = (int32_t)ifr976;
                                ifr975 = t982;
                            } else {
                                ifr975 = 0;
                            }
                            #pragma unroll(1)
                            for (int64_t i983 = 0; i983 < 2; i983 = (int64_t)((uint64_t)i983 + (uint64_t)1)) {
                                int64_t t984 = (int64_t)ifr975;
                                int64_t t985 = (int64_t)((uint64_t)t984 - (uint64_t)t818);
                                int64_t t986 = i983 * 128;
                                int64_t t987 = (int64_t)((uint64_t)t985 - (uint64_t)t986);
                                int64_t t988 = (t987 < 128 ? t987 : 128);
                                int64_t t989 = (t988 > 0 ? t988 : 0);
                                bool cmp990 = i983 == 0;
                                bool cmp991 = i983 == 1;
                                bool sel992 = cmp990 ? true : cmp991;
                                int64_t sel993 = sel992 ? 0 : t989;
                                int32_t t994 = (int32_t)sel993;
                                int32_t t995 = (int32_t)i971;
                                int32_t t996 = t995 * 25600;
                                int64_t t997 = (int64_t)((uint64_t)i816 * (uint64_t)256);
                                int64_t t998 = (int64_t)t996;
                                int64_t t999 = (int64_t)((uint64_t)t998 + (uint64_t)t997);
                                int32_t t1000 = (int32_t)i983;
                                int32_t t1001 = t1000 * 128;
                                int64_t t1002 = (int64_t)t1001;
                                int64_t t1003 = (int64_t)((uint64_t)t999 + (uint64_t)t1002);
                                int64_t t1004 = t1003;
                                __ubuf__ uint16_t* view_root1005 = reinterpret_cast<__ubuf__ uint16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                                kernel_score_topk_mix_vf4(i983, (__ubuf__ uint16_t*)view_root1005, cmp819, t994, (__ubuf__ uint16_t*)view_root1005, (int32_t)t1004);
                            }
                        }
                    }
                }
                int64_t t1006 = (int64_t)((uint64_t)t796 * (uint64_t)256);
                int64_t t1007 = (int64_t)((uint64_t)t795 * (uint64_t)256);
                int64_t t1008 = (int64_t)((uint64_t)dim741 - (uint64_t)t1007);
                int64_t t1009 = (t1006 < t1008 ? t1006 : t1008);
                for (int64_t i1010 = 0; i1010 < 3; i1010 = (int64_t)((uint64_t)i1010 + (uint64_t)1)) {
                    int64_t t1011 = (int64_t)((uint64_t)arg19 * (uint64_t)3);
                    int64_t t1012 = (int64_t)((uint64_t)t1011 + (uint64_t)i1010);
                    bool cmp1013 = t1012 < t803;
                    if (cmp1013) {
                        bool cmp1014 = t1009 > 512;
                        int32_t ifr1015;
                        int32_t ifr1016;
                        if (cmp1014) {
                            int64_t t1017 = (int64_t)((uint64_t)t1009 + (uint64_t)255);
                            int64_t t1018 = t1017 % 256;
                            bool cmp1019 = t1018 != 0;
                            bool cmp1020 = t1018 < 0;
                            bool t1021 = cmp1019 & cmp1020;
                            int64_t t1022 = t1017 / 256;
                            int64_t t1023 = (int64_t)(uint8_t)t1021;
                            int64_t t1024 = (int64_t)((uint64_t)t1022 - (uint64_t)t1023);
                            int64_t t1025 = i1010 * 25600;
                            int64_t t1026 = t1025;
                            int64_t t1027 = t1026 + 32768;
                            __ubuf__ uint16_t* view_root1028 = reinterpret_cast<__ubuf__ uint16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                            kernel_score_topk_mix_vf5(t1009, (__ubuf__ uint16_t*)view_root1028, (int32_t)t1027, t1024, (__ubuf__ uint16_t*)view_root1028);
                            asc_sync();
                            int64_t t1029 = (int64_t)((uint64_t)t1009 + (uint64_t)-511);
                            int32_t t1030 = (int32_t)t1029;
                            int32_t iter1031 = 255;
                            int32_t iter1032 = 0;
                            for (int64_t i1033 = 0; i1033 < 8; i1033 = (int64_t)((uint64_t)i1033 + (uint64_t)1)) {
                                int32_t t1034 = (int32_t)((uint32_t)iter1032 + (uint32_t)iter1031);
                                int32_t t1035 = t1034 % 2;
                                bool cmp1036 = t1035 != 0;
                                bool cmp1037 = t1035 < 0;
                                bool t1038 = cmp1036 & cmp1037;
                                int32_t t1039 = t1034 / 2;
                                int32_t t1040 = (int32_t)(uint8_t)t1038;
                                int32_t t1041 = (int32_t)((uint32_t)t1039 - (uint32_t)t1040);
                                asc_lock(PIPE_S, 3);
                                int64_t t1042 = t1041;
                                int64_t t1043 = t1042 + 28160;
                                uint16_t ld1044 = view_root1028[t1043];
                                asc_unlock(PIPE_S, 3);
                                int16_t scalar1045 = (int16_t)ld1044;
                                int32_t t1046 = (int32_t)(uint16_t)scalar1045;
                                bool cmp1047 = t1046 >= t1030;
                                int32_t sel1048 = cmp1047 ? t1041 : iter1031;
                                int32_t t1049 = (int32_t)((uint32_t)t1041 + (uint32_t)1);
                                int32_t sel1050 = cmp1047 ? iter1032 : t1049;
                                iter1031 = sel1048;
                                iter1032 = sel1050;
                            }
                            bool cmp1051 = iter1032 > 0;
                            int32_t ifr1052;
                            if (cmp1051) {
                                int32_t t1053 = (int32_t)((uint32_t)iter1032 - (uint32_t)1);
                                asc_lock(PIPE_S, 3);
                                int64_t t1054 = t1053;
                                int64_t t1055 = t1054 + 28160;
                                uint16_t ld1056 = view_root1028[t1055];
                                asc_unlock(PIPE_S, 3);
                                int16_t scalar1057 = (int16_t)ld1056;
                                int32_t t1058 = (int32_t)(uint16_t)scalar1057;
                                ifr1052 = t1058;
                            } else {
                                ifr1052 = 0;
                            }
                            int32_t t1059 = (int32_t)((uint32_t)t1030 - (uint32_t)ifr1052);
                            asc_lock(PIPE_V, 3);
                            kernel_score_topk_mix_vf6(t1009, (__ubuf__ uint16_t*)view_root1028, (int32_t)t1027, iter1032, t1024, (__ubuf__ uint16_t*)view_root1028);
                            asc_unlock(PIPE_V, 3);
                            asc_sync();
                            int32_t iter1060 = 255;
                            int32_t iter1061 = 0;
                            for (int64_t i1062 = 0; i1062 < 8; i1062 = (int64_t)((uint64_t)i1062 + (uint64_t)1)) {
                                int32_t t1063 = (int32_t)((uint32_t)iter1061 + (uint32_t)iter1060);
                                int32_t t1064 = t1063 % 2;
                                bool cmp1065 = t1064 != 0;
                                bool cmp1066 = t1064 < 0;
                                bool t1067 = cmp1065 & cmp1066;
                                int32_t t1068 = t1063 / 2;
                                int32_t t1069 = (int32_t)(uint8_t)t1067;
                                int32_t t1070 = (int32_t)((uint32_t)t1068 - (uint32_t)t1069);
                                int64_t t1071 = t1070;
                                int64_t t1072 = t1071 + 28160;
                                uint16_t ld1073 = view_root1028[t1072];
                                int16_t scalar1074 = (int16_t)ld1073;
                                int32_t t1075 = (int32_t)(uint16_t)scalar1074;
                                bool cmp1076 = t1075 >= t1059;
                                int32_t sel1077 = cmp1076 ? t1070 : iter1060;
                                int32_t t1078 = (int32_t)((uint32_t)t1070 + (uint32_t)1);
                                int32_t sel1079 = cmp1076 ? iter1061 : t1078;
                                iter1060 = sel1077;
                                iter1061 = sel1079;
                            }
                            bool cmp1080 = iter1061 > 0;
                            int32_t ifr1081;
                            if (cmp1080) {
                                int32_t t1082 = (int32_t)((uint32_t)iter1061 - (uint32_t)1);
                                int64_t t1083 = t1082;
                                int64_t t1084 = t1083 + 28160;
                                uint16_t ld1085 = view_root1028[t1084];
                                int16_t scalar1086 = (int16_t)ld1085;
                                int32_t t1087 = (int32_t)(uint16_t)scalar1086;
                                ifr1081 = t1087;
                            } else {
                                ifr1081 = 0;
                            }
                            int64_t t1088 = iter1061;
                            int64_t t1089 = t1088 + 28160;
                            uint16_t ld1090 = view_root1028[t1089];
                            int16_t scalar1091 = (int16_t)ld1090;
                            int32_t t1092 = (int32_t)(uint16_t)scalar1091;
                            int32_t t1093 = (int32_t)((uint32_t)t1092 - (uint32_t)ifr1081);
                            int32_t t1094 = (int32_t)((uint32_t)t1059 - (uint32_t)ifr1081);
                            int32_t t1095 = (int32_t)((uint32_t)iter1032 * (uint32_t)256);
                            int32_t t1096 = (int32_t)((uint32_t)t1095 + (uint32_t)iter1061);
                            int64_t t1097 = (int64_t)((uint64_t)t1009 + (uint64_t)127);
                            int64_t t1098 = t1097 % 128;
                            bool cmp1099 = t1098 != 0;
                            bool cmp1100 = t1098 < 0;
                            bool t1101 = cmp1099 & cmp1100;
                            int64_t t1102 = t1097 / 128;
                            int64_t t1103 = (int64_t)(uint8_t)t1101;
                            int64_t t1104 = (int64_t)((uint64_t)t1102 - (uint64_t)t1103);
                            kernel_score_topk_mix_vf7((__ubuf__ uint16_t*)view_root1028, t1009, (__ubuf__ uint16_t*)view_root1028, (int32_t)t1027, t1096, t1104);
                            asc_sync();
                            int32_t t1105 = (int32_t)((uint32_t)t1093 - (uint32_t)t1094);
                            int64_t t1106 = t1105;
                            uint16_t ld1107 = view_root1028[t1106];
                            int16_t scalar1108 = (int16_t)ld1107;
                            int32_t t1109 = (int32_t)(uint16_t)scalar1108;
                            ifr1015 = t1096;
                            ifr1016 = t1109;
                        } else {
                            ifr1015 = 127;
                            ifr1016 = -1;
                        }
                        int64_t t1110 = (int64_t)((uint64_t)t1009 + (uint64_t)63);
                        int64_t t1111 = t1110 % 64;
                        bool cmp1112 = t1111 != 0;
                        bool cmp1113 = t1111 < 0;
                        bool t1114 = cmp1112 & cmp1113;
                        int64_t t1115 = t1110 / 64;
                        int64_t t1116 = (int64_t)(uint8_t)t1114;
                        int64_t t1117 = (int64_t)((uint64_t)t1115 - (uint64_t)t1116);
                        __ubuf__ int32_t* view_root1118 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                        int64_t t1119 = i1010 * 25600;
                        int64_t t1120 = t1119;
                        int64_t t1121 = t1120 + 32768;
                        __ubuf__ uint16_t* view_root1122 = reinterpret_cast<__ubuf__ uint16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                        kernel_score_topk_mix_vf8((__ubuf__ int32_t*)view_root1118, t1009, (__ubuf__ uint16_t*)view_root1122, (int32_t)t1121, ifr1015, ifr1016, t1117);
                        asc_lock(PIPE_V, 1);
                        asc_lock(PIPE_V, 2);
                        kernel_score_topk_mix_vf9((__ubuf__ int32_t*)view_root1118, (__ubuf__ uint16_t*)view_root1122, (int32_t)t1121, (__ubuf__ uint16_t*)view_root1122, t1007, (__ubuf__ int32_t*)view_root1118);
                        asc_unlock(PIPE_V, 2);
                        asc_unlock(PIPE_V, 1);
                        int64_t t1123 = (int64_t)((uint64_t)i758 * (uint64_t)512);
                        int64_t t1124 = (int64_t)((uint64_t)t799 + (uint64_t)t1012);
                        int64_t dim1125 = gm10_arg.stride[0];
                        int64_t t1126 = (int64_t)((uint64_t)t1124 * (uint64_t)dim1125);
                        int64_t t1127 = (int64_t)((uint64_t)t1126 + (uint64_t)t1123);
                        int64_t t1128 = (int64_t)((uint64_t)t1127 * (uint64_t)16);
                        int64_t t1129 = (int64_t)((uint64_t)t1128 / (uint64_t)8);
                        int64_t t1130 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm10) + (uint64_t)t1129);
                        __gm__ uint16_t* root1131 = (__gm__ uint16_t*)(uintptr_t)(uint64_t)(t1130);
                        int64_t t1132 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(buf523) + (uint64_t)51200);
                        __ubuf__ uint16_t* selected_root1133 = (__ubuf__ uint16_t*)(uintptr_t)(uint64_t)(t1132);
                        int64_t dim1134 = gm10_arg.stride[0];
                        int64_t t1135 = dim1134;
                        int64_t t1136 = t1135 * 2;
                        int64_t t1137 = t1136;
                        asc_lock(PIPE_MTE3, 1);
                        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint16_t*)root1131 + 0, (__ubuf__ uint16_t*)selected_root1133 + 0, 1, 1024, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        asc_unlock(PIPE_MTE3, 1);
                        int64_t dim1138 = gm11_arg.stride[0];
                        int64_t t1139 = (int64_t)((uint64_t)t1124 * (uint64_t)dim1138);
                        int64_t t1140 = (int64_t)((uint64_t)t1139 + (uint64_t)t1123);
                        int64_t t1141 = (int64_t)((uint64_t)t1140 * (uint64_t)32);
                        int64_t t1142 = (int64_t)((uint64_t)t1141 / (uint64_t)8);
                        int64_t t1143 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm11) + (uint64_t)t1142);
                        __gm__ int32_t* root1144 = (__gm__ int32_t*)(uintptr_t)(uint64_t)(t1143);
                        int64_t t1145 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(buf523) + (uint64_t)52224);
                        __ubuf__ int32_t* selected_root1146 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(t1145);
                        int64_t dim1147 = gm11_arg.stride[0];
                        int64_t t1148 = dim1147;
                        int64_t t1149 = t1148 * 4;
                        int64_t t1150 = t1149;
                        asc_lock(PIPE_MTE3, 2);
                        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int32_t*)root1144 + 0, (__ubuf__ int32_t*)selected_root1146 + 0, 1, 2048, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        asc_unlock(PIPE_MTE3, 2);
                        bool cmp1151 = arg21 != 0;
                        if (cmp1151) {
                            for (int64_t i1152 = 0; i1152 < t1009; i1152 = (int64_t)((uint64_t)i1152 + (uint64_t)128)) {
                                int64_t t1153 = (int64_t)((uint64_t)t1009 - (uint64_t)i1152);
                                int64_t t1154 = (t1153 < 128 ? t1153 : 128);
                                int32_t t1155 = (int32_t)i1010;
                                int32_t t1156 = t1155 * 25600;
                                int64_t t1157 = (int64_t)t1156;
                                int64_t t1158 = (int64_t)((uint64_t)t1157 + (uint64_t)i1152);
                                int64_t t1159 = t1158;
                                asc_lock(PIPE_V, 6);
                                __ubuf__ float* view_root1160 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                                kernel_score_topk_mix_vf10((__ubuf__ uint16_t*)view_root1122, (int32_t)t1159, (__ubuf__ float*)view_root1160);
                                asc_unlock(PIPE_V, 6);
                                int64_t t1161 = (int64_t)((uint64_t)t1007 + (uint64_t)i1152);
                                int64_t dim1162 = gm14_arg.stride[0];
                                int64_t t1163 = (int64_t)((uint64_t)t1124 * (uint64_t)dim1162);
                                int64_t t1164 = (int64_t)((uint64_t)t1163 + (uint64_t)t1161);
                                int64_t t1165 = (int64_t)((uint64_t)t1164 * (uint64_t)32);
                                int64_t t1166 = (int64_t)((uint64_t)t1165 / (uint64_t)8);
                                int64_t t1167 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm14) + (uint64_t)t1166);
                                __gm__ float* root1168 = (__gm__ float*)(uintptr_t)(uint64_t)(t1167);
                                int64_t t1169 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(buf523) + (uint64_t)225280);
                                __ubuf__ float* selected_root1170 = (__ubuf__ float*)(uintptr_t)(uint64_t)(t1169);
                                int64_t t1171 = t1154;
                                int64_t dim1172 = gm14_arg.stride[0];
                                int64_t t1173 = dim1172;
                                int64_t t1174 = t1171 * 4;
                                int64_t t1175 = t1173 * 4;
                                bool cmp1176 = (uint64_t)t1154 > (uint64_t)0;
                                if (cmp1176) {
                                    int64_t t1177 = t1174;
                                    int64_t t1178 = t1175;
                                    asc_lock(PIPE_MTE3, 6);
                                    int32_t t1179 = (int32_t)t1177;
                                    cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root1168 + 0, (__ubuf__ float*)selected_root1170 + 0, 1, t1179, static_cast<uint8_t>(0), (uint64_t)0, 0);
                                    asc_unlock(PIPE_MTE3, 6);
                                }
                            }
                        }
                    }
                }
                asc_sync();
                ffts_cross_core_sync(PIPE_MTE3, cannbotdsl_get_ffts_config(5, 2));
                asc_sync_block_wait(PIPE_S, 6);
                wh779 = t793;
            }
        }
        asc_sync();
        ffts_cross_core_sync(PIPE_MTE3, cannbotdsl_get_ffts_config(0, 2));
        ffts_cross_core_sync(PIPE_MTE3, cannbotdsl_get_ffts_config(1, 0));
        asc_sync_block_wait(PIPE_S, 1);
        asc_sync_block_wait(PIPE_S, 2);
        int64_t t1180 = (int64_t)((uint64_t)arg17 * (uint64_t)2);
        int64_t t1181 = (int64_t)((uint64_t)t1180 + (uint64_t)arg19);
        int64_t t1182 = (int64_t)((uint64_t)arg18 * (uint64_t)2);
        int64_t t1183 = (int64_t)arg20;
        for (int64_t i1184 = t1181; i1184 < t1183; i1184 = (int64_t)((uint64_t)i1184 + (uint64_t)t1182)) {
            bool cmp1185 = arg22 != 0;
            bool ifr1186;
            if (cmp1185) {
                int64_t t1187 = i1184;
                int32_t ld1188 = gm8[t1187];
                bool cmp1189 = ld1188 <= 512;
                ifr1186 = cmp1189;
            } else {
                ifr1186 = false;
            }
            if (ifr1186) {
                int64_t t1190 = i1184;
                int32_t ld1191 = gm8[t1190];
                int64_t t1192 = (int64_t)ld1191;
                int64_t t1193 = (t1192 < 512 ? t1192 : 512);
                int64_t t1194 = (t1193 > 0 ? t1193 : 0);
                int32_t ld1195 = gm5[t1190];
                for (int64_t i1196 = 0; i1196 < 4; i1196 = (int64_t)((uint64_t)i1196 + (uint64_t)1)) {
                    int64_t t1197 = i1196 * 128;
                    int64_t t1198 = (int64_t)((uint64_t)t1194 - (uint64_t)t1197);
                    int64_t t1199 = (t1198 > 0 ? t1198 : 0);
                    int64_t t1200 = (t1199 < 128 ? t1199 : 128);
                    bool cmp1201 = t1200 > 0;
                    bool ifr1202;
                    if (cmp1201) {
                        int64_t dim1203 = gm4_arg.shape[1];
                        bool cmp1204 = i1196 < dim1203;
                        ifr1202 = cmp1204;
                    } else {
                        ifr1202 = false;
                    }
                    int32_t ifr1205;
                    if (ifr1202) {
                        int64_t dim1206 = gm4_arg.stride[0];
                        int64_t t1207 = dim1206;
                        int64_t t1208 = ld1195;
                        int64_t t1209 = i1196;
                        int64_t t1210 = t1208 * t1207;
                        int64_t t1211 = t1210 + t1209;
                        int32_t ld1212 = gm4[t1211];
                        ifr1205 = ld1212;
                    } else {
                        ifr1205 = -1;
                    }
                    bool cmp1213 = ifr1205 < 0;
                    int64_t t1214 = (int64_t)ifr1205;
                    bool cmp1215 = t1214 >= dim757;
                    bool sel1216 = cmp1213 ? true : cmp1215;
                    int32_t sel1217 = sel1216 ? 0 : ifr1205;
                    int64_t sel1218 = sel1216 ? 0 : t1200;
                    asc_lock(PIPE_V, 11);
                    __ubuf__ int32_t* view_root1219 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                    kernel_score_topk_mix_vf11(sel1217, sel1218, i1196, (__ubuf__ int32_t*)view_root1219);
                    asc_unlock(PIPE_V, 11);
                }
                int64_t dim1220 = gm9_arg.stride[0];
                int64_t t1221 = (int64_t)((uint64_t)i1184 * (uint64_t)dim1220);
                int64_t t1222 = (int64_t)((uint64_t)t1221 * (uint64_t)32);
                int64_t t1223 = (int64_t)((uint64_t)t1222 / (uint64_t)8);
                int64_t t1224 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm9) + (uint64_t)t1223);
                __gm__ int32_t* root1225 = (__gm__ int32_t*)(uintptr_t)(uint64_t)(t1224);
                int64_t t1226 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(buf523) + (uint64_t)132608);
                __ubuf__ int32_t* selected_root1227 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(t1226);
                int64_t dim1228 = gm9_arg.stride[0];
                int64_t t1229 = dim1228;
                int64_t t1230 = t1229 * 4;
                int64_t t1231 = t1230;
                asc_lock(PIPE_MTE3, 11);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int32_t*)root1225 + 0, (__ubuf__ int32_t*)selected_root1227 + 0, 1, 2048, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 11);
            } else {
                bool cmp1232 = t756 == 0;
                if (cmp1232) {
                    asc_lock(PIPE_V, 11);
                    __ubuf__ int32_t* view_root1233 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                    kernel_score_topk_mix_vf12((__ubuf__ int32_t*)view_root1233);
                    asc_unlock(PIPE_V, 11);
                    int64_t dim1234 = gm9_arg.stride[0];
                    int64_t t1235 = (int64_t)((uint64_t)i1184 * (uint64_t)dim1234);
                    int64_t t1236 = (int64_t)((uint64_t)t1235 * (uint64_t)32);
                    int64_t t1237 = (int64_t)((uint64_t)t1236 / (uint64_t)8);
                    int64_t t1238 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm9) + (uint64_t)t1237);
                    __gm__ int32_t* root1239 = (__gm__ int32_t*)(uintptr_t)(uint64_t)(t1238);
                    int64_t t1240 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(buf523) + (uint64_t)132608);
                    __ubuf__ int32_t* selected_root1241 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(t1240);
                    int64_t dim1242 = gm9_arg.stride[0];
                    int64_t t1243 = dim1242;
                    int64_t t1244 = t1243 * 4;
                    int64_t t1245 = t1244;
                    asc_lock(PIPE_MTE3, 11);
                    cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int32_t*)root1239 + 0, (__ubuf__ int32_t*)selected_root1241 + 0, 1, 2048, static_cast<uint8_t>(0), (uint64_t)0, 0);
                    asc_unlock(PIPE_MTE3, 11);
                } else {
                    int64_t t1246 = (int64_t)((uint64_t)i1184 + (uint64_t)1);
                    int64_t t1247 = t1246;
                    int64_t t1248 = t752 + t1247;
                    int32_t ld1249 = gm6[t1248];
                    int64_t t1250 = (int64_t)ld1249;
                    int64_t t1251 = (int64_t)((uint64_t)t1250 * (uint64_t)t749);
                    int64_t t1252 = (int64_t)((uint64_t)t1251 + (uint64_t)1);
                    int64_t t1253 = (int64_t)((uint64_t)t1252 * (uint64_t)t756);
                    int64_t t1254 = (int64_t)((uint64_t)t1253 - (uint64_t)1);
                    int64_t t1255 = t1254 % t755;
                    bool cmp1256 = t1255 != 0;
                    bool cmp1257 = t1255 < 0;
                    bool cmp1258 = t755 < 0;
                    bool t1259 = cmp1257 ^ cmp1258;
                    bool t1260 = cmp1256 & t1259;
                    int64_t t1261 = t1254 / t755;
                    int64_t t1262 = (int64_t)(uint8_t)t1260;
                    int64_t t1263 = (int64_t)((uint64_t)t1261 - (uint64_t)t1262);
                    int64_t t1264 = (int64_t)((uint64_t)t1250 + (uint64_t)1);
                    int64_t t1265 = (int64_t)((uint64_t)t1264 * (uint64_t)t749);
                    int64_t t1266 = (int64_t)((uint64_t)t1265 * (uint64_t)t756);
                    int64_t t1267 = (int64_t)((uint64_t)t1266 - (uint64_t)1);
                    int64_t t1268 = t1267 % t755;
                    bool cmp1269 = t1268 != 0;
                    bool cmp1270 = t1268 < 0;
                    bool t1271 = cmp1270 ^ cmp1258;
                    bool t1272 = cmp1269 & t1271;
                    int64_t t1273 = t1267 / t755;
                    int64_t t1274 = (int64_t)(uint8_t)t1272;
                    int64_t t1275 = (int64_t)((uint64_t)t1273 - (uint64_t)t1274);
                    int64_t wh1276 = 0;
                    int64_t wh1277 = t1263;
                    int64_t wh_result1278;
                    int64_t wh_result1279;
                    while (true) {
                        bool cmp1280 = wh1277 <= t1275;
                        wh_result1278 = wh1276;
                        wh_result1279 = wh1277;
                        if (!cmp1280) break;
                        int64_t t1281 = (int64_t)((uint64_t)t1275 - (uint64_t)wh_result1279);
                        int64_t t1282 = (int64_t)((uint64_t)t1281 + (uint64_t)1);
                        int64_t t1283 = wh_result1278 % 512;
                        bool cmp1284 = t1283 != 0;
                        bool cmp1285 = t1283 < 0;
                        bool t1286 = cmp1284 & cmp1285;
                        int64_t t1287 = wh_result1278 / 512;
                        int64_t t1288 = (int64_t)(uint8_t)t1286;
                        int64_t t1289 = (int64_t)((uint64_t)t1287 - (uint64_t)t1288);
                        int64_t t1290 = (int64_t)((uint64_t)32 - (uint64_t)t1289);
                        int64_t t1291 = (t1282 < t1290 ? t1282 : t1290);
                        int64_t t1292 = (int64_t)((uint64_t)t1291 * (uint64_t)512);
                        int64_t dim1293 = gm10_arg.stride[0];
                        int64_t t1294 = (int64_t)((uint64_t)i1184 * (uint64_t)dim1293);
                        int64_t t1295 = (int64_t)((uint64_t)t1294 * (uint64_t)2);
                        int64_t t1296 = (int64_t)((uint64_t)wh_result1279 * (uint64_t)1024);
                        int64_t t1297 = (int64_t)((uint64_t)t1295 + (uint64_t)t1296);
                        int64_t t1298 = wh_result1278 * 2;
                        int64_t t1299 = (int64_t)((uint64_t)t1291 * (uint64_t)1024);
                        bool cmp1300 = t1299 == 0;
                        bool cmp1301 = t1299 != 0;
                        int64_t t1302 = (int64_t)(uint8_t)cmp1301;
                        int64_t sel1303 = cmp1300 ? 2 : t1299;
                        asc_lock(PIPE_MTE2, 7);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint16_t*)((__ubuf__ uint8_t*)buf523 + t1298), (__gm__ uint16_t*)((__gm__ uint8_t*)gm10 + t1297), t1302, sel1303, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        }
                        asc_unlock(PIPE_MTE2, 7);
                        int64_t dim1304 = gm11_arg.stride[0];
                        int64_t t1305 = (int64_t)((uint64_t)i1184 * (uint64_t)dim1304);
                        int64_t t1306 = (int64_t)((uint64_t)t1305 * (uint64_t)4);
                        int64_t t1307 = (int64_t)((uint64_t)wh_result1279 * (uint64_t)2048);
                        int64_t t1308 = (int64_t)((uint64_t)t1306 + (uint64_t)t1307);
                        int64_t t1309 = wh_result1278 * 4;
                        int64_t t1310 = t1309 + 65536;
                        int64_t t1311 = (int64_t)((uint64_t)t1291 * (uint64_t)2048);
                        bool cmp1312 = t1311 == 0;
                        bool cmp1313 = t1311 != 0;
                        int64_t t1314 = (int64_t)(uint8_t)cmp1313;
                        int64_t sel1315 = cmp1312 ? 4 : t1311;
                        asc_lock(PIPE_MTE2, 8);
                        {
                            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf523 + t1310), (__gm__ int32_t*)((__gm__ uint8_t*)gm11 + t1308), t1314, sel1315, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                        }
                        asc_unlock(PIPE_MTE2, 8);
                        int64_t t1316 = (int64_t)((uint64_t)wh_result1278 + (uint64_t)t1292);
                        bool cmp1317 = t1316 > 512;
                        int32_t ifr1318;
                        int32_t ifr1319;
                        if (cmp1317) {
                            int64_t t1320 = (int64_t)((uint64_t)t1316 + (uint64_t)255);
                            int64_t t1321 = t1320 % 256;
                            bool cmp1322 = t1321 != 0;
                            bool cmp1323 = t1321 < 0;
                            bool t1324 = cmp1322 & cmp1323;
                            int64_t t1325 = t1320 / 256;
                            int64_t t1326 = (int64_t)(uint8_t)t1324;
                            int64_t t1327 = (int64_t)((uint64_t)t1325 - (uint64_t)t1326);
                            asc_lock(PIPE_V, 7);
                            __ubuf__ uint16_t* view_root1328 = reinterpret_cast<__ubuf__ uint16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                            kernel_score_topk_mix_vf13(t1316, (__ubuf__ uint16_t*)view_root1328, t1327, (__ubuf__ uint16_t*)view_root1328);
                            asc_unlock(PIPE_V, 7);
                            asc_sync();
                            int64_t t1329 = (int64_t)((uint64_t)t1316 + (uint64_t)-511);
                            int32_t t1330 = (int32_t)t1329;
                            int32_t iter1331 = 255;
                            int32_t iter1332 = 0;
                            for (int64_t i1333 = 0; i1333 < 8; i1333 = (int64_t)((uint64_t)i1333 + (uint64_t)1)) {
                                int32_t t1334 = (int32_t)((uint32_t)iter1332 + (uint32_t)iter1331);
                                int32_t t1335 = t1334 % 2;
                                bool cmp1336 = t1335 != 0;
                                bool cmp1337 = t1335 < 0;
                                bool t1338 = cmp1336 & cmp1337;
                                int32_t t1339 = t1334 / 2;
                                int32_t t1340 = (int32_t)(uint8_t)t1338;
                                int32_t t1341 = (int32_t)((uint32_t)t1339 - (uint32_t)t1340);
                                asc_lock(PIPE_S, 10);
                                int64_t t1342 = t1341;
                                int64_t t1343 = t1342 + 65536;
                                uint16_t ld1344 = view_root1328[t1343];
                                asc_unlock(PIPE_S, 10);
                                int16_t scalar1345 = (int16_t)ld1344;
                                int32_t t1346 = (int32_t)(uint16_t)scalar1345;
                                bool cmp1347 = t1346 >= t1330;
                                int32_t sel1348 = cmp1347 ? t1341 : iter1331;
                                int32_t t1349 = (int32_t)((uint32_t)t1341 + (uint32_t)1);
                                int32_t sel1350 = cmp1347 ? iter1332 : t1349;
                                iter1331 = sel1348;
                                iter1332 = sel1350;
                            }
                            bool cmp1351 = iter1332 > 0;
                            int32_t ifr1352;
                            if (cmp1351) {
                                int32_t t1353 = (int32_t)((uint32_t)iter1332 - (uint32_t)1);
                                asc_lock(PIPE_S, 10);
                                int64_t t1354 = t1353;
                                int64_t t1355 = t1354 + 65536;
                                uint16_t ld1356 = view_root1328[t1355];
                                asc_unlock(PIPE_S, 10);
                                int16_t scalar1357 = (int16_t)ld1356;
                                int32_t t1358 = (int32_t)(uint16_t)scalar1357;
                                ifr1352 = t1358;
                            } else {
                                ifr1352 = 0;
                            }
                            int32_t t1359 = (int32_t)((uint32_t)t1330 - (uint32_t)ifr1352);
                            asc_lock(PIPE_V, 10);
                            kernel_score_topk_mix_vf14(t1316, (__ubuf__ uint16_t*)view_root1328, iter1332, t1327, (__ubuf__ uint16_t*)view_root1328);
                            asc_unlock(PIPE_V, 10);
                            asc_sync();
                            int32_t iter1360 = 255;
                            int32_t iter1361 = 0;
                            for (int64_t i1362 = 0; i1362 < 8; i1362 = (int64_t)((uint64_t)i1362 + (uint64_t)1)) {
                                int32_t t1363 = (int32_t)((uint32_t)iter1361 + (uint32_t)iter1360);
                                int32_t t1364 = t1363 % 2;
                                bool cmp1365 = t1364 != 0;
                                bool cmp1366 = t1364 < 0;
                                bool t1367 = cmp1365 & cmp1366;
                                int32_t t1368 = t1363 / 2;
                                int32_t t1369 = (int32_t)(uint8_t)t1367;
                                int32_t t1370 = (int32_t)((uint32_t)t1368 - (uint32_t)t1369);
                                int64_t t1371 = t1370;
                                int64_t t1372 = t1371 + 65536;
                                uint16_t ld1373 = view_root1328[t1372];
                                int16_t scalar1374 = (int16_t)ld1373;
                                int32_t t1375 = (int32_t)(uint16_t)scalar1374;
                                bool cmp1376 = t1375 >= t1359;
                                int32_t sel1377 = cmp1376 ? t1370 : iter1360;
                                int32_t t1378 = (int32_t)((uint32_t)t1370 + (uint32_t)1);
                                int32_t sel1379 = cmp1376 ? iter1361 : t1378;
                                iter1360 = sel1377;
                                iter1361 = sel1379;
                            }
                            bool cmp1380 = iter1361 > 0;
                            int32_t ifr1381;
                            if (cmp1380) {
                                int32_t t1382 = (int32_t)((uint32_t)iter1361 - (uint32_t)1);
                                int64_t t1383 = t1382;
                                int64_t t1384 = t1383 + 65536;
                                uint16_t ld1385 = view_root1328[t1384];
                                int16_t scalar1386 = (int16_t)ld1385;
                                int32_t t1387 = (int32_t)(uint16_t)scalar1386;
                                ifr1381 = t1387;
                            } else {
                                ifr1381 = 0;
                            }
                            int64_t t1388 = iter1361;
                            int64_t t1389 = t1388 + 65536;
                            uint16_t ld1390 = view_root1328[t1389];
                            int16_t scalar1391 = (int16_t)ld1390;
                            int32_t t1392 = (int32_t)(uint16_t)scalar1391;
                            int32_t t1393 = (int32_t)((uint32_t)t1392 - (uint32_t)ifr1381);
                            int32_t t1394 = (int32_t)((uint32_t)t1359 - (uint32_t)ifr1381);
                            int32_t t1395 = (int32_t)((uint32_t)iter1332 * (uint32_t)256);
                            int32_t t1396 = (int32_t)((uint32_t)t1395 + (uint32_t)iter1361);
                            int64_t t1397 = (int64_t)((uint64_t)t1316 + (uint64_t)127);
                            int64_t t1398 = t1397 % 128;
                            bool cmp1399 = t1398 != 0;
                            bool cmp1400 = t1398 < 0;
                            bool t1401 = cmp1399 & cmp1400;
                            int64_t t1402 = t1397 / 128;
                            int64_t t1403 = (int64_t)(uint8_t)t1401;
                            int64_t t1404 = (int64_t)((uint64_t)t1402 - (uint64_t)t1403);
                            kernel_score_topk_mix_vf15((__ubuf__ uint16_t*)view_root1328, t1316, (__ubuf__ uint16_t*)view_root1328, t1396, t1404);
                            asc_sync();
                            int32_t t1405 = (int32_t)((uint32_t)t1393 - (uint32_t)t1394);
                            int64_t t1406 = t1405;
                            int64_t t1407 = t1406 + 16384;
                            uint16_t ld1408 = view_root1328[t1407];
                            int16_t scalar1409 = (int16_t)ld1408;
                            int32_t t1410 = (int32_t)(uint16_t)scalar1409;
                            ifr1318 = t1396;
                            ifr1319 = t1410;
                        } else {
                            ifr1318 = 127;
                            ifr1319 = -1;
                        }
                        int64_t t1411 = (int64_t)((uint64_t)t1316 + (uint64_t)63);
                        int64_t t1412 = t1411 % 64;
                        bool cmp1413 = t1412 != 0;
                        bool cmp1414 = t1412 < 0;
                        bool t1415 = cmp1413 & cmp1414;
                        int64_t t1416 = t1411 / 64;
                        int64_t t1417 = (int64_t)(uint8_t)t1415;
                        int64_t t1418 = (int64_t)((uint64_t)t1416 - (uint64_t)t1417);
                        asc_lock(PIPE_V, 7);
                        __ubuf__ int32_t* view_root1419 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                        __ubuf__ uint16_t* view_root1420 = reinterpret_cast<__ubuf__ uint16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf523) + 0);
                        kernel_score_topk_mix_vf16((__ubuf__ int32_t*)view_root1419, t1316, (__ubuf__ uint16_t*)view_root1420, ifr1318, ifr1319, t1418);
                        asc_lock(PIPE_V, 8);
                        asc_lock(PIPE_V, 11);
                        kernel_score_topk_mix_vf17((__ubuf__ int32_t*)view_root1419, (__ubuf__ uint16_t*)view_root1420, (__ubuf__ uint16_t*)view_root1420, (__ubuf__ int32_t*)view_root1419, (__ubuf__ int32_t*)view_root1419);
                        asc_unlock(PIPE_V, 11);
                        asc_unlock(PIPE_V, 8);
                        asc_unlock(PIPE_V, 7);
                        int64_t t1421 = (int64_t)((uint64_t)wh_result1279 + (uint64_t)t1291);
                        bool cmp1422 = t1421 > t1275;
                        int64_t sel1423 = cmp1422 ? wh_result1278 : 512;
                        if (cmp1422) {
                            bool cmp1424 = arg15 > 0;
                            if (cmp1424) {
                                int64_t dim1425 = gm13_arg.shape[1];
                                int64_t t1426 = (int64_t)((uint64_t)i1184 * (uint64_t)arg15);
                                int64_t t1427 = (int64_t)((uint64_t)dim1425 - (uint64_t)t1426);
                                int64_t t1428 = (dim741 < t1427 ? dim741 : t1427);
                                int64_t t1429 = (int64_t)((uint64_t)t1426 * (uint64_t)4);
                                int64_t t1430 = (int64_t)((uint64_t)t1428 * (uint64_t)4);
                                bool cmp1431 = t1430 == 0;
                                bool cmp1432 = t1430 != 0;
                                int64_t t1433 = (int64_t)(uint8_t)cmp1432;
                                int64_t sel1434 = cmp1431 ? 4 : t1430;
                                asc_lock(PIPE_MTE2, 12);
                                {
                                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf523 + 134656), (__gm__ int32_t*)((__gm__ uint8_t*)gm13 + t1429), t1433, sel1434, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                                }
                                asc_unlock(PIPE_MTE2, 12);
                            } else {
                                int64_t t1435 = (int64_t)((uint64_t)dim741 + (uint64_t)127);
                                int64_t t1436 = t1435 % 128;
                                bool cmp1437 = t1436 != 0;
                                bool cmp1438 = t1436 < 0;
                                bool t1439 = cmp1437 & cmp1438;
                                int64_t t1440 = t1435 / 128;
                                int64_t t1441 = (int64_t)(uint8_t)t1439;
                                int64_t t1442 = (int64_t)((uint64_t)t1440 - (uint64_t)t1441);
                                int64_t t1443 = i1184;
                                int32_t ld1444 = gm5[t1443];
                                int64_t t1445 = (int64_t)ld1444;
                                int64_t dim1446 = gm4_arg.stride[0];
                                int64_t t1447 = (int64_t)((uint64_t)t1445 * (uint64_t)dim1446);
                                int64_t t1448 = (int64_t)((uint64_t)t1447 * (uint64_t)4);
                                int64_t t1449 = (int64_t)((uint64_t)t1442 * (uint64_t)4);
                                bool cmp1450 = t1449 == 0;
                                bool cmp1451 = t1449 != 0;
                                int64_t t1452 = (int64_t)(uint8_t)cmp1451;
                                int64_t sel1453 = cmp1450 ? 4 : t1449;
                                asc_lock(PIPE_MTE2, 12);
                                {
                                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf523 + 134656), (__gm__ int32_t*)((__gm__ uint8_t*)gm4 + t1448), t1452, sel1453, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                                }
                                asc_unlock(PIPE_MTE2, 12);
                            }
                            asc_lock(PIPE_V, 11);
                            asc_lock(PIPE_V, 12);
                            kernel_score_topk_mix_vf18((__ubuf__ int32_t*)view_root1419, (__ubuf__ int32_t*)view_root1419, cmp1424);
                            asc_unlock(PIPE_V, 12);
                            asc_unlock(PIPE_V, 11);
                            int64_t dim1454 = gm9_arg.stride[0];
                            int64_t t1455 = (int64_t)((uint64_t)i1184 * (uint64_t)dim1454);
                            int64_t t1456 = (int64_t)((uint64_t)t1455 * (uint64_t)32);
                            int64_t t1457 = (int64_t)((uint64_t)t1456 / (uint64_t)8);
                            int64_t t1458 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm9) + (uint64_t)t1457);
                            __gm__ int32_t* root1459 = (__gm__ int32_t*)(uintptr_t)(uint64_t)(t1458);
                            int64_t t1460 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(buf523) + (uint64_t)132608);
                            __ubuf__ int32_t* selected_root1461 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(t1460);
                            int64_t dim1462 = gm9_arg.stride[0];
                            int64_t t1463 = dim1462;
                            int64_t t1464 = t1463 * 4;
                            int64_t t1465 = t1464;
                            asc_lock(PIPE_MTE3, 11);
                            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int32_t*)root1459 + 0, (__ubuf__ int32_t*)selected_root1461 + 0, 1, 2048, static_cast<uint8_t>(0), (uint64_t)0, 0);
                            asc_unlock(PIPE_MTE3, 11);
                        } else {
                            asc_lock(PIPE_V, 7);
                            asc_copy_ub2ub((__ubuf__ void*)(view_root1420 + 0), (__ubuf__ void*)(view_root1420 + 65792), 1024);
                            kernel_score_topk_mix_vf19((__ubuf__ uint16_t*)view_root1420, (__ubuf__ uint16_t*)view_root1420);
                            asc_unlock(PIPE_V, 7);
                            asc_lock(PIPE_V, 8);
                            asc_copy_ub2ub((__ubuf__ void*)(view_root1419 + 16384), (__ubuf__ void*)(view_root1419 + 33152), 2048);
                            kernel_score_topk_mix_vf20((__ubuf__ int32_t*)view_root1419, (__ubuf__ int32_t*)view_root1419);
                            asc_unlock(PIPE_V, 8);
                        }
                        wh1276 = sel1423;
                        wh1277 = t1421;
                    }
                }
            }
        }
    }
}
