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

struct KernelArgTensor_bfloat16_t_2d_Device_ss_ds {
    __gm__ bfloat16_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ss_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

struct KernelArgTensor_float_2d_Device_sd_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_sd_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_sd_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[1]}, {src.stride[0]}};
}

struct KernelArgTensor_float_2d_Device_dd_ds {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_dd_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_dd_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0], src.shape[1]}, {src.stride[0]}};
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_sd_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_sd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_float_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_float_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__simd_vf__ inline void kernel_project_gate_grid_vf0(__ubuf__ float* buf8, __ubuf__ float* buf9, __ubuf__ float* buf10, __ubuf__ float* buf11)
{
    vector_bool vmask;
    float c12 = -10.0f;
    float c13 = 1.0f;
    float c14 = -1.0f;
    float c15 = 0.0f;
    vector_bool mask16 = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t v17;
    asc_arange(*(vector_int32_t*)&v17, static_cast<int32_t>(0));
    vector_uint32_t v18;
    asc_duplicate_scalar(v18, 15, mask16);
    vector_uint32_t v19;
    asc_and(v19, v17, v18, mask16);
    vector_float v20;
    asc_gather(v20, (__ubuf__ float*)(buf8 + 0), v19, mask16);
    for (int32_t i21 = 0; i21 < 128; i21 = (int32_t)((uint32_t)i21 + (uint32_t)64)) {
        int32_t t22 = i21;
        vector_float v23;
        asc_loadalign(v23, buf9 + t22);
        vector_float v24;
        asc_max_scalar(v24, v23, c15, mask16);
        vector_float v25;
        asc_abs(v25, v23, mask16);
        vector_float v26;
        asc_mul_scalar(v26, v25, c14, mask16);
        vector_float v27;
        asc_exp(v27, v26, mask16);
        vector_float v28;
        asc_add_scalar(v28, v27, c13, mask16);
        vector_float v29;
        asc_ln(v29, v28, mask16);
        vector_float v30;
        asc_add(v30, v24, v29, mask16);
        vector_float v31;
        asc_mul_scalar(v31, v23, c14, mask16);
        vector_float v32;
        asc_add_scalar(v32, v31, c12, mask16);
        vector_float v33;
        asc_max_scalar(v33, v32, c15, mask16);
        vector_float v34;
        asc_min_scalar(v34, v33, c13, mask16);
        vector_float v35;
        asc_min_scalar(v35, v23, c15, mask16);
        vector_float v36;
        asc_exp(v36, v35, mask16);
        vector_float v37;
        asc_mul(v37, v34, v36, mask16);
        vector_float v38;
        asc_max(v38, v30, v37, mask16);
        vector_float v39;
        asc_sqrt(v39, v38, mask16);
        asc_storealign(buf10 + t22, v39, mask16);
        vector_float v40;
        asc_add(v40, v39, v20, mask16);
        asc_storealign(buf11 + t22, v40, mask16);
    }
}

__global__ __aicore__ void kernel_project_gate_grid(KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm0_arg, KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm1_arg, KernelArgTensor_float_2d_Device_sd_ds gm2_arg, KernelArgTensor_float_2d_Device_dd_ds gm3_arg, KernelArgTensor_float_2d_Device_dd_ds gm4_arg, int64_t arg5, int64_t arg6, int64_t arg7)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ bfloat16_t* gm0 = gm0_arg.ptr;
    __gm__ bfloat16_t* gm1 = gm1_arg.ptr;
    __gm__ float* gm2 = gm2_arg.ptr;
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

    __ubuf__ float* buf41 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __cc__ float* buf42 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ bfloat16_t* buf43 = (__ca__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cb__ bfloat16_t* buf44 = (__cb__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ bfloat16_t* buf45 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(16384);
    __cbuf__ bfloat16_t* buf46 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf47 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf48 = (__ubuf__ float*)(uintptr_t)(uint64_t)(512);
    __ubuf__ float* buf49 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1024);
    __ubuf__ float* buf50 = (__ubuf__ float*)(uintptr_t)(uint64_t)(1536);

    if ASCEND_IS_AIC {
        int64_t dim51 = gm3_arg.shape[0];
        int64_t t52 = dim51 % 16;
        bool cmp53 = t52 != 0;
        bool cmp54 = t52 < 0;
        bool t55 = cmp53 & cmp54;
        int64_t t56 = dim51 / 16;
        int64_t t57 = (int64_t)(uint8_t)t55;
        int64_t t58 = (int64_t)((uint64_t)t56 - (uint64_t)t57);
        int64_t t59 = (int64_t)((uint64_t)t58 * (uint64_t)8);
        for (int64_t i60 = arg5; i60 < t59; i60 = (int64_t)((uint64_t)i60 + (uint64_t)arg6)) {
            int64_t t61 = i60 % 8;
            bool cmp62 = t61 != 0;
            bool cmp63 = t61 < 0;
            bool t64 = cmp62 & cmp63;
            int64_t t65 = i60 / 8;
            int64_t t66 = (int64_t)(uint8_t)t64;
            int64_t t67 = (int64_t)((uint64_t)t65 - (uint64_t)t66);
            int64_t t68 = (int64_t)((uint64_t)t67 * (uint64_t)16);
            int64_t sel69 = t64 ? 8 : 0;
            int64_t t70 = t61 + sel69;
            int64_t t71 = t70 * 16;
            asc_set_fp32_mode();
            for (int64_t i72 = 0; i72 < 10; i72 = (int64_t)((uint64_t)i72 + (uint64_t)1)) {
                __cbuf__ bfloat16_t* view_root73 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf46) + 0);
                int64_t t74 = (int64_t)((uint64_t)t68 / (uint64_t)16);
                int64_t dim75 = gm0_arg.stride[0];
                int64_t t76 = (int64_t)((uint64_t)t74 * (uint64_t)dim75);
                int64_t t77 = i72 * 32;
                int64_t t78 = (int64_t)((uint64_t)t76 + (uint64_t)t77);
                int64_t t79 = (int64_t)((uint64_t)t78 * (uint64_t)16);
                int64_t dim80 = gm0_arg.stride[0];
                int64_t t81 = dim80;
                int64_t t82 = t81 * 2;
                int64_t t83 = t82;
                asc_lock(PIPE_MTE2, 0);
                int64_t t84 = t79;
                set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(16) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                asc_copy_gm2l1_nd2nz(view_root73 + 0, gm0 + t84, t83, 0, 16, 512, 0, false);
                asc_unlock(PIPE_MTE2, 0);
                __cbuf__ bfloat16_t* view_root85 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf45) + 0);
                int64_t t86 = (int64_t)((uint64_t)t71 / (uint64_t)16);
                int64_t dim87 = gm1_arg.stride[0];
                int64_t t88 = (int64_t)((uint64_t)t86 * (uint64_t)dim87);
                int64_t t89 = (int64_t)((uint64_t)t88 + (uint64_t)t77);
                int64_t t90 = (int64_t)((uint64_t)t89 * (uint64_t)16);
                int64_t dim91 = gm1_arg.stride[0];
                int64_t t92 = dim91;
                int64_t t93 = t92 * 2;
                int64_t t94 = t93;
                asc_lock(PIPE_MTE2, 1);
                int64_t t95 = t90;
                set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(16) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                asc_copy_gm2l1_nd2nz(view_root85 + 0, gm1 + t95, t94, 0, 16, 512, 0, false);
                asc_unlock(PIPE_MTE2, 1);
                __ca__ bfloat16_t* view_root96 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf43) + 0);
                asc_lock(PIPE_MTE1, 2);
                asc_lock(PIPE_MTE1, 0);
                asc_copy_l12l0a(view_root96 + 0, view_root73 + 0, 0, 0, 1, 32, 1, 1);
                asc_unlock(PIPE_MTE1, 0);
                asc_unlock(PIPE_MTE1, 2);
                __cb__ bfloat16_t* view_root97 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf44) + 0);
                asc_lock(PIPE_MTE1, 3);
                asc_lock(PIPE_MTE1, 1);
                asc_copy_l12l0b(view_root97 + 0, view_root85 + 0, 0, 0, 1, 32, 1, 1);
                asc_unlock(PIPE_MTE1, 1);
                asc_unlock(PIPE_MTE1, 3);
                bool cmp98 = i72 == 0;
                __cc__ float* view_root99 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf42) + 0);
                asc_lock(PIPE_M, 4);
                asc_lock(PIPE_M, 2);
                asc_lock(PIPE_M, 3);
                if (cmp98) {
                    asc_mmad(view_root99 + 0, view_root96, view_root97, 16, 512, 16, 0, true, false, true);
                } else {
                    asc_mmad(view_root99 + 0, view_root96, view_root97, 16, 512, 16, 0, true, false, false);
                }
                asc_unlock(PIPE_M, 3);
                asc_unlock(PIPE_M, 2);
                asc_unlock(PIPE_M, 4);
            }
            __ubuf__ float* view_root100 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf41) + 0);
            __cc__ float* view_root101 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf42) + 0);
            asc_sync_intra_wait(PIPE_FIX, 0);
            asc_sync_intra_wait(PIPE_FIX, 16);
            asc_lock(PIPE_FIX, 4);
            asc_set_l0c2gm_nz2nd(1, 0, 0);
            asc_copy_l0c2ub((__ubuf__ float*)view_root100 + 0, view_root101, 16, 16, 16, 16, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
            asc_unlock(PIPE_FIX, 4);
            asc_sync_intra_arrive(PIPE_FIX, 0);
            asc_sync_intra_arrive(PIPE_FIX, 16);
        }
        asc_sync_intra_wait(PIPE_FIX, 0);
        asc_sync_intra_wait(PIPE_FIX, 16);
    }

    if ASCEND_IS_AIV {
        asc_sync_intra_arrive(PIPE_V, 0);
        int64_t dim102 = gm3_arg.shape[0];
        int64_t t103 = dim102 % 16;
        bool cmp104 = t103 != 0;
        bool cmp105 = t103 < 0;
        bool t106 = cmp104 & cmp105;
        int64_t t107 = dim102 / 16;
        int64_t t108 = (int64_t)(uint8_t)t106;
        int64_t t109 = (int64_t)((uint64_t)t107 - (uint64_t)t108);
        int64_t t110 = (int64_t)((uint64_t)t109 * (uint64_t)8);
        for (int64_t i111 = arg5; i111 < t110; i111 = (int64_t)((uint64_t)i111 + (uint64_t)arg6)) {
            int64_t t112 = i111 % 8;
            bool cmp113 = t112 != 0;
            bool cmp114 = t112 < 0;
            bool t115 = cmp113 & cmp114;
            int64_t t116 = i111 / 8;
            int64_t t117 = (int64_t)(uint8_t)t115;
            int64_t t118 = (int64_t)((uint64_t)t116 - (uint64_t)t117);
            int64_t t119 = (int64_t)((uint64_t)t118 * (uint64_t)16);
            int64_t sel120 = t115 ? 8 : 0;
            int64_t t121 = t112 + sel120;
            int64_t t122 = t121 * 16;
            int64_t t123 = t121 * 64;
            asc_lock(PIPE_MTE2, 2);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf50 + 0), (__gm__ float*)((__gm__ uint8_t*)gm2 + t123), 1, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
            }
            asc_unlock(PIPE_MTE2, 2);
            asc_sync_intra_wait(PIPE_V, 0);
            asc_lock(PIPE_V, 2);
            asc_lock(PIPE_V, 0);
            asc_lock(PIPE_V, 1);
            kernel_project_gate_grid_vf0((__ubuf__ float*)buf50, (__ubuf__ float*)buf47, (__ubuf__ float*)buf48, (__ubuf__ float*)buf49);
            asc_unlock(PIPE_V, 1);
            asc_unlock(PIPE_V, 0);
            asc_unlock(PIPE_V, 2);
            asc_sync_intra_arrive(PIPE_V, 0);
            int64_t dim124 = gm3_arg.stride[0];
            int64_t t125 = (int64_t)((uint64_t)t119 * (uint64_t)dim124);
            int64_t t126 = (int64_t)((uint64_t)t125 + (uint64_t)t122);
            int64_t t127 = (int64_t)((uint64_t)t126 * (uint64_t)32);
            int64_t t128 = (int64_t)((uint64_t)t127 / (uint64_t)8);
            int64_t t129 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm3) + (uint64_t)t128);
            __gm__ float* root130 = (__gm__ float*)(uintptr_t)(uint64_t)(t129);
            __ubuf__ float* view_root131 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf48) + 0);
            int64_t dim132 = gm3_arg.stride[0];
            int64_t t133 = dim132;
            int64_t t134 = (int64_t)((uint64_t)arg7 * (uint64_t)8);
            int64_t t135 = t134;
            int64_t t136 = t135 * t133;
            int64_t t137 = t133 * 4;
            int64_t t138 = t137;
            asc_lock(PIPE_MTE3, 0);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root130 + t136, (__ubuf__ float*)view_root131 + 0, 8, 64, static_cast<uint8_t>(0), (uint64_t)t138, 64);
            asc_unlock(PIPE_MTE3, 0);
            int64_t dim139 = gm4_arg.stride[0];
            int64_t t140 = (int64_t)((uint64_t)t119 * (uint64_t)dim139);
            int64_t t141 = (int64_t)((uint64_t)t140 + (uint64_t)t122);
            int64_t t142 = (int64_t)((uint64_t)t141 * (uint64_t)32);
            int64_t t143 = (int64_t)((uint64_t)t142 / (uint64_t)8);
            int64_t t144 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm4) + (uint64_t)t143);
            __gm__ float* root145 = (__gm__ float*)(uintptr_t)(uint64_t)(t144);
            __ubuf__ float* view_root146 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf49) + 0);
            int64_t dim147 = gm4_arg.stride[0];
            int64_t t148 = dim147;
            int64_t t149 = t135 * t148;
            int64_t t150 = t148 * 4;
            int64_t t151 = t150;
            asc_lock(PIPE_MTE3, 1);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ float*)root145 + t149, (__ubuf__ float*)view_root146 + 0, 8, 64, static_cast<uint8_t>(0), (uint64_t)t151, 64);
            asc_unlock(PIPE_MTE3, 1);
        }
    }
}
