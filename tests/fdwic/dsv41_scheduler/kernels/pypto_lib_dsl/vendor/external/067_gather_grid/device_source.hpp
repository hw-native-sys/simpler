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

struct KernelArgTensor_int8_t_2d {
    __gm__ int8_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_uint8_t_2d {
    __gm__ uint8_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_int32_t_2d {
    __gm__ int32_t* ptr;
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

struct KernelArgTensor_int8_t_2d_Device_ds_ds {
    __gm__ int8_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_int8_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_int8_t_2d_Device_ds_ds(const KernelArgTensor_int8_t_2d& src) {
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

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
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

struct KernelArgTensor_int32_t_2d_Device_ss_ds {
    __gm__ int32_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_ss_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
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

struct KernelArgTensor_float_2d_Device_ss_ds {
    __gm__ float* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ss_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

__simd_vf__ inline void kernel_gather_grid_vf0(__ubuf__ uint8_t* buf0)
{
    vector_bool vmask;
    vector_bool mask1 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t v2;
    asc_duplicate_scalar(v2, 0, mask1);
    for (int32_t i3 = 0; i3 < 2560; i3 = (int32_t)((uint32_t)i3 + (uint32_t)256)) {
        int32_t t4 = i3;
        asc_storealign(buf0 + t4, v2, mask1);
    }
}

static_assert(sizeof(KernelArgTensor_int8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_gather_grid(KernelArgTensor_int8_t_2d_Device_ds_ds gm5_arg, KernelArgTensor_uint8_t_2d_Device_ds_ds gm6_arg, KernelArgTensor_float_2d_Device_ds_ds gm7_arg, KernelArgTensor_int32_t_2d_Device_ds_ds gm8_arg, KernelArgTensor_int32_t_2d_Device_ss_ds gm9_arg, KernelArgTensor_int32_t_2d_Device_ss_ds gm10_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm11_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm12_arg, KernelArgTensor_float_2d_Device_ss_ds gm13_arg, KernelArgTensor_int32_t_2d_Device_ss_ds gm14_arg, int64_t arg15, int64_t arg16)
{
    __gm__ int8_t* gm5 = gm5_arg.ptr;
    __gm__ uint8_t* gm6 = gm6_arg.ptr;
    __gm__ float* gm7 = gm7_arg.ptr;
    __gm__ int32_t* gm8 = gm8_arg.ptr;
    __gm__ int32_t* gm9 = gm9_arg.ptr;
    __gm__ int32_t* gm10 = gm10_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm11 = gm11_arg.ptr;
    __gm__ fp8_e8m0_t* gm12 = gm12_arg.ptr;
    __gm__ float* gm13 = gm13_arg.ptr;
    __gm__ int32_t* gm14 = gm14_arg.ptr;
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

    __ubuf__ int8_t* buf17 = (__ubuf__ int8_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ uint8_t* buf18 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(5120);
    __ubuf__ uint8_t* buf19 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(5280);

    vector_bool vmask;
    int64_t iter20 = 0;
    for (int64_t i21 = 0; i21 < 64; i21 = (int64_t)((uint64_t)i21 + (uint64_t)1)) {
        int64_t dim22 = gm10_arg.stride[0];
        int64_t t23 = dim22;
        int64_t t24 = i21;
        int64_t t25 = t24 * t23;
        int32_t ld26 = gm10[t25];
        int32_t t27 = (int32_t)((uint32_t)ld26 + (uint32_t)15);
        int32_t t28 = t27 % 16;
        bool cmp29 = t28 != 0;
        bool cmp30 = t28 < 0;
        bool t31 = cmp29 & cmp30;
        int32_t t32 = t27 / 16;
        int32_t t33 = (int32_t)(uint8_t)t31;
        int32_t t34 = (int32_t)((uint32_t)t32 - (uint32_t)t33);
        int64_t t35 = i21 * 160;
        int64_t t36 = (int64_t)((uint64_t)arg15 + (uint64_t)arg16);
        int64_t t37 = iter20 % arg16;
        bool cmp38 = t37 != 0;
        bool cmp39 = t37 < 0;
        bool cmp40 = arg16 < 0;
        bool t41 = cmp39 ^ cmp40;
        bool t42 = cmp38 & t41;
        int64_t sel43 = t42 ? arg16 : 0;
        int64_t t44 = (int64_t)((uint64_t)t37 + (uint64_t)sel43);
        int64_t t45 = (int64_t)((uint64_t)t36 - (uint64_t)t44);
        int64_t t46 = t45 % arg16;
        bool cmp47 = t46 != 0;
        bool cmp48 = t46 < 0;
        bool t49 = cmp48 ^ cmp40;
        bool t50 = cmp47 & t49;
        int64_t sel51 = t50 ? arg16 : 0;
        int64_t t52 = (int64_t)((uint64_t)t46 + (uint64_t)sel51);
        int64_t t53 = (int64_t)t34;
        for (int64_t i54 = t52; i54 < t53; i54 = (int64_t)((uint64_t)i54 + (uint64_t)arg16)) {
            int64_t t55 = (int64_t)((uint64_t)i54 * (uint64_t)16);
            int64_t t56 = (int64_t)ld26;
            int64_t t57 = (int64_t)((uint64_t)t56 - (uint64_t)t55);
            int64_t t58 = (t57 < 16 ? t57 : 16);
            int64_t t59 = (int64_t)((uint64_t)t35 + (uint64_t)t55);
            asc_lock(PIPE_V, 2);
            kernel_gather_grid_vf0((__ubuf__ uint8_t*)buf19);
            asc_unlock(PIPE_V, 2);
            int64_t iter60 = 0;
            for (int64_t i61 = 0; i61 < 2; i61 = (int64_t)((uint64_t)i61 + (uint64_t)1)) {
                int64_t dim62 = gm9_arg.stride[0];
                int64_t t63 = dim62;
                int64_t t64 = i61;
                int64_t t65 = t64 * t63;
                int64_t t66 = t65 + t24;
                int32_t ld67 = gm9[t66];
                int64_t t68 = (int64_t)ld67;
                int64_t t69 = (int64_t)((uint64_t)iter60 + (uint64_t)t68);
                int64_t t70 = (t55 > iter60 ? t55 : iter60);
                int64_t t71 = (int64_t)((uint64_t)t55 + (uint64_t)t58);
                int64_t t72 = (t71 < t69 ? t71 : t69);
                for (int64_t i73 = t70; i73 < t72; i73 = (int64_t)((uint64_t)i73 + (uint64_t)1)) {
                    int64_t t74 = i61 * 80;
                    int64_t t75 = (int64_t)((uint64_t)t74 + (uint64_t)i73);
                    int64_t t76 = (int64_t)((uint64_t)t75 - (uint64_t)iter60);
                    int64_t t77 = (int64_t)((uint64_t)i73 - (uint64_t)t55);
                    int64_t t78 = (t76 < 160 ? t76 : 160);
                    int64_t t79 = (int64_t)((uint64_t)160 - (uint64_t)t78);
                    int64_t t80 = (t79 < 1 ? t79 : 1);
                    bool cmp81 = t76 <= 160;
                    int64_t sel82 = cmp81 ? t80 : 0;
                    int64_t dim83 = gm5_arg.stride[0];
                    int64_t t84 = (int64_t)((uint64_t)t76 + (uint64_t)t35);
                    int64_t t85 = (int64_t)((uint64_t)t84 * (uint64_t)dim83);
                    asc_lock(PIPE_MTE2, 0);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int8_t*)((__ubuf__ uint8_t*)buf17 + 0), (__gm__ int8_t*)((__gm__ uint8_t*)gm5 + t85), sel82, 5120, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)dim83, 5120);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    int64_t dim86 = gm11_arg.stride[0];
                    int64_t t87 = (int64_t)((uint64_t)dim86 * (uint64_t)8);
                    int64_t t88 = t87 / 8;
                    int64_t dim89 = gm11_arg.stride[0];
                    int64_t t90 = (int64_t)((uint64_t)t59 * (uint64_t)dim89);
                    int64_t t91 = (int64_t)((uint64_t)t90 * (uint64_t)8);
                    int64_t t92 = (int64_t)((uint64_t)t77 * (uint64_t)t88);
                    int64_t t93 = (int64_t)((uint64_t)t92 * (uint64_t)8);
                    int64_t t94 = (int64_t)((uint64_t)t91 + (uint64_t)t93);
                    int64_t t95 = (int64_t)((uint64_t)t94 / (uint64_t)8);
                    int64_t t96 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm11) + (uint64_t)t95);
                    __gm__ int8_t* root97 = (__gm__ int8_t*)(uintptr_t)(uint64_t)(t96);
                    __ubuf__ int8_t* view_root98 = reinterpret_cast<__ubuf__ int8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf17) + 0);
                    asc_lock(PIPE_MTE3, 0);
                    cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int8_t*)root97 + 0, (__ubuf__ int8_t*)view_root98 + 0, 1, 5120, static_cast<uint8_t>(0), (uint64_t)0, 0);
                    asc_unlock(PIPE_MTE3, 0);
                    int64_t dim99 = gm6_arg.stride[0];
                    int64_t t100 = (int64_t)((uint64_t)t84 * (uint64_t)dim99);
                    asc_lock(PIPE_MTE2, 1);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf18 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm6 + t100), sel82, 160, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)dim99, 160);
                    }
                    asc_unlock(PIPE_MTE2, 1);
                    for (int64_t i101 = 0; i101 < 160; i101 = (int64_t)((uint64_t)i101 + (uint64_t)1)) {
                        int64_t t102 = i101 / 2;
                        int64_t t103 = t102 * 32;
                        int64_t t104 = (int64_t)((uint64_t)t77 * (uint64_t)2);
                        int64_t t105 = (int64_t)((uint64_t)t103 + (uint64_t)t104);
                        int64_t t106 = i101 % 2;
                        int64_t t107 = (int64_t)((uint64_t)t105 + (uint64_t)t106);
                        asc_lock(PIPE_S, 1);
                        int64_t t108 = i101;
                        uint8_t ld109 = buf18[t108];
                        asc_unlock(PIPE_S, 1);
                        int8_t scalar110 = (int8_t)ld109;
                        uint8_t scalar111 = (uint8_t)scalar110;
                        asc_lock(PIPE_S, 2);
                        int64_t t112 = t107;
                        buf19[t112] = scalar111;
                        asc_unlock(PIPE_S, 2);
                    }
                    int64_t t113 = (int64_t)((uint64_t)t35 / (uint64_t)32);
                    int64_t dim114 = gm7_arg.stride[0];
                    int64_t t115 = (int64_t)((uint64_t)t113 * (uint64_t)dim114);
                    int64_t t116 = (int64_t)((uint64_t)t115 * (uint64_t)32);
                    int64_t t117 = t116;
                    int64_t dim118 = gm7_arg.stride[0];
                    int64_t t119 = dim118;
                    int64_t t120 = t76;
                    int64_t t121 = t120 * t119;
                    int64_t t122 = t117 + t121;
                    float ld123 = gm7[t122];
                    int64_t dim124 = gm13_arg.stride[0];
                    int64_t t125 = (int64_t)((uint64_t)t59 * (uint64_t)dim124);
                    int64_t t126 = t125;
                    int64_t dim127 = gm13_arg.stride[0];
                    int64_t t128 = dim127;
                    int64_t t129 = t77;
                    int64_t t130 = t129 * t128;
                    int64_t t131 = t126 + t130;
                    gm13[t131] = ld123;
                    int64_t dim132 = gm8_arg.stride[0];
                    int64_t t133 = (int64_t)((uint64_t)t113 * (uint64_t)dim132);
                    int64_t t134 = (int64_t)((uint64_t)t133 * (uint64_t)32);
                    int64_t t135 = t134;
                    int64_t dim136 = gm8_arg.stride[0];
                    int64_t t137 = dim136;
                    int64_t t138 = t120 * t137;
                    int64_t t139 = t135 + t138;
                    int32_t ld140 = gm8[t139];
                    int64_t dim141 = gm14_arg.stride[0];
                    int64_t t142 = (int64_t)((uint64_t)t59 * (uint64_t)dim141);
                    int64_t t143 = t142;
                    int64_t dim144 = gm14_arg.stride[0];
                    int64_t t145 = dim144;
                    int64_t t146 = t129 * t145;
                    int64_t t147 = t143 + t146;
                    gm14[t147] = ld140;
                }
                iter60 = t69;
            }
            int64_t dim148 = gm12_arg.stride[0];
            int64_t t149 = (int64_t)((uint64_t)dim148 * (uint64_t)8);
            int64_t t150 = t149 / 8;
            int64_t t151 = (int64_t)((uint64_t)t59 * (uint64_t)1280);
            int64_t t152 = (int64_t)((uint64_t)t151 / (uint64_t)8);
            int64_t t153 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm12) + (uint64_t)t152);
            __gm__ uint8_t* root154 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t153);
            __ubuf__ uint8_t* view_root155 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf19) + 0);
            asc_lock(PIPE_MTE3, 2);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root154 + 0, (__ubuf__ uint8_t*)view_root155 + 0, 1, 2560, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 2);
        }
        int64_t t156 = (int64_t)((uint64_t)iter20 + (uint64_t)t53);
        iter20 = t156;
    }
}
