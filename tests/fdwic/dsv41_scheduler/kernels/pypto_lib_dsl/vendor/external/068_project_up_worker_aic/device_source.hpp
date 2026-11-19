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

struct KernelArgTensor_int32_t_2d {
    __gm__ int32_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_uint8_t_3d {
    __gm__ uint8_t* ptr;
    int64_t shape[3];
    int64_t stride[3];
};

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

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

struct KernelArgTensor_int32_t_2d_Device_ss_ds {
    __gm__ int32_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_ss_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

struct KernelArgTensor_uint8_t_3d_Device_sss_dds {
    __gm__ uint8_t* ptr;
    int64_t stride[2];
};
static inline KernelArgTensor_uint8_t_3d_Device_sss_dds cannbotdsl_pack_KernelArgTensor_uint8_t_3d_Device_sss_dds(const KernelArgTensor_uint8_t_3d& src) {
    return {src.ptr, {src.stride[0], src.stride[1]}};
}

struct KernelArgTensor_float_2d_Device_ss_ds {
    __gm__ float* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ss_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_3d_Device_sss_dds) == 24 && alignof(KernelArgTensor_uint8_t_3d_Device_sss_dds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_3d_Device_sss_dds) == 24 && alignof(KernelArgTensor_uint8_t_3d_Device_sss_dds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__simd_vf__ inline void kernel_project_up_worker_vf0(__ubuf__ uint8_t* buf12, __ubuf__ fp8_e4m3fn_t* buf13)
{
    vector_bool vmask;
    vector_bool mask14 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t v15;
    asc_arange(*(vector_int8_t*)&v15, static_cast<int8_t>(0));
    vector_uint8_t v16;
    asc_duplicate_scalar(v16, 7, mask14);
    vector_uint8_t v17;
    asc_and(v17, v15, v16, mask14);
    vector_uint8_t v18;
    asc_shiftleft_scalar(v18, v17, 2, mask14);
    vector_uint8_t v19;
    asc_add_scalar(v19, v18, 48, mask14);
    vector_uint8_t v20;
    asc_duplicate_scalar(v20, 0, mask14);
    vector_uint8_t v21;
    asc_duplicate_scalar(v21, 48, mask14);
    vector_bool m22;
    asc_eq_scalar(m22, v17, 0, mask14);
    vector_uint8_t v23;
    asc_select(v23, v20, v19, m22);
    vector_bool m24;
    asc_eq_scalar(m24, v17, 1, mask14);
    vector_uint8_t v25;
    asc_select(v25, v21, v23, m24);
    vector_uint8_t v26;
    asc_duplicate_scalar(v26, 8, mask14);
    vector_uint8_t v27;
    asc_and(v27, v15, v26, mask14);
    vector_uint8_t v28;
    asc_shiftleft_scalar(v28, v27, 4, mask14);
    vector_uint8_t v29;
    asc_or(v29, v25, v28, mask14);
    for (int32_t i30 = 0; i30 < 64; i30 = (int32_t)((uint32_t)i30 + (uint32_t)2)) {
        int32_t t31 = i30 * 128;
        int32_t t32 = t31;
        vector_uint8_t v33;
        asc_loadalign(v33, buf12 + t32);
        vector_uint8_t v34;
        asc_shiftright_scalar(v34, v33, 4, mask14);
        vector_uint8_t v35;
        asc_gather(v35, v29, v33);
        vector_uint8_t v36;
        asc_gather(v36, v29, v34);
        vector_uint8_t v37;
        vector_uint8_t v38;
        asc_intlv(v37, v38, v35, v36);
        int32_t t39 = i30 * 32;
        int32_t t40 = t39;
        asc_storealign((__ubuf__ uint8_t*)(buf13 + t40), v37, (uint16_t)64, (uint16_t)0, mask14);
        int32_t t41 = i30 + 1;
        int32_t t42 = t41 * 32;
        int32_t t43 = t42;
        asc_storealign((__ubuf__ uint8_t*)(buf13 + t43), v38, (uint16_t)64, (uint16_t)0, mask14);
    }
}

__simd_vf__ inline void kernel_project_up_worker_vf1(__ubuf__ uint8_t* buf44, __ubuf__ fp8_e4m3fn_t* buf45)
{
    vector_bool vmask;
    vector_bool mask46 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t v47;
    asc_arange(*(vector_int8_t*)&v47, static_cast<int8_t>(0));
    vector_uint8_t v48;
    asc_duplicate_scalar(v48, 7, mask46);
    vector_uint8_t v49;
    asc_and(v49, v47, v48, mask46);
    vector_uint8_t v50;
    asc_shiftleft_scalar(v50, v49, 2, mask46);
    vector_uint8_t v51;
    asc_add_scalar(v51, v50, 48, mask46);
    vector_uint8_t v52;
    asc_duplicate_scalar(v52, 0, mask46);
    vector_uint8_t v53;
    asc_duplicate_scalar(v53, 48, mask46);
    vector_bool m54;
    asc_eq_scalar(m54, v49, 0, mask46);
    vector_uint8_t v55;
    asc_select(v55, v52, v51, m54);
    vector_bool m56;
    asc_eq_scalar(m56, v49, 1, mask46);
    vector_uint8_t v57;
    asc_select(v57, v53, v55, m56);
    vector_uint8_t v58;
    asc_duplicate_scalar(v58, 8, mask46);
    vector_uint8_t v59;
    asc_and(v59, v47, v58, mask46);
    vector_uint8_t v60;
    asc_shiftleft_scalar(v60, v59, 4, mask46);
    vector_uint8_t v61;
    asc_or(v61, v57, v60, mask46);
    for (int32_t i62 = 0; i62 < 64; i62 = (int32_t)((uint32_t)i62 + (uint32_t)2)) {
        int32_t t63 = i62 * 128;
        int32_t t64 = t63;
        vector_uint8_t v65;
        asc_loadalign(v65, buf44 + t64);
        vector_uint8_t v66;
        asc_shiftright_scalar(v66, v65, 4, mask46);
        vector_uint8_t v67;
        asc_gather(v67, v61, v65);
        vector_uint8_t v68;
        asc_gather(v68, v61, v66);
        vector_uint8_t v69;
        vector_uint8_t v70;
        asc_intlv(v69, v70, v67, v68);
        int32_t t71 = i62 * 32;
        int32_t t72 = t71;
        asc_storealign((__ubuf__ uint8_t*)(buf45 + t72), v69, (uint16_t)64, (uint16_t)0, mask46);
        int32_t t73 = i62 + 1;
        int32_t t74 = t73 * 32;
        int32_t t75 = t74;
        asc_storealign((__ubuf__ uint8_t*)(buf45 + t75), v70, (uint16_t)64, (uint16_t)0, mask46);
    }
}

__global__ __aicore__ void kernel_project_up_worker(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm0_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm1_arg, KernelArgTensor_int32_t_2d_Device_ss_ds gm2_arg, KernelArgTensor_uint8_t_3d_Device_sss_dds gm7_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm3_arg, KernelArgTensor_uint8_t_3d_Device_sss_dds gm8_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm4_arg, KernelArgTensor_float_2d_Device_ss_ds gm5_arg, KernelArgTensor_float_2d_Device_ss_ds gm6_arg, int64_t arg9, int64_t arg10, int64_t arg11)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ fp8_e4m3fn_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e8m0_t* gm1 = gm1_arg.ptr;
    __gm__ int32_t* gm2 = gm2_arg.ptr;
    __gm__ uint8_t* gm7 = gm7_arg.ptr;
    __gm__ fp8_e8m0_t* gm3 = gm3_arg.ptr;
    __gm__ uint8_t* gm8 = gm8_arg.ptr;
    __gm__ fp8_e8m0_t* gm4 = gm4_arg.ptr;
    __gm__ float* gm5 = gm5_arg.ptr;
    __gm__ float* gm6 = gm6_arg.ptr;
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

    __cc__ float* buf76 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ fp8_e4m3fn_t* buf77 = (__ca__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cb__ fp8_e4m3fn_t* buf78 = (__cb__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cc__ float* buf79 = (__cc__ float*)(uintptr_t)(uint64_t)(16384);
    __ca__ fp8_e4m3fn_t* buf80 = (__ca__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(2048);
    __cb__ fp8_e4m3fn_t* buf81 = (__cb__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(32768);
    __cbuf__ fp8_e8m0_t* buf82 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(71680);
    __cbuf__ fp8_e4m3fn_t* buf83 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(38400);
    __cbuf__ fp8_e8m0_t* buf84 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(71168);
    __cbuf__ fp8_e4m3fn_t* buf85 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(36352);
    __cbuf__ fp8_e8m0_t* buf86 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(35328);
    __cbuf__ fp8_e4m3fn_t* buf87 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(2048);
    __cbuf__ fp8_e8m0_t* buf88 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(34816);
    __cbuf__ fp8_e4m3fn_t* buf89 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ uint8_t* buf90 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ fp8_e4m3fn_t* buf91 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(8192);
    __ubuf__ uint8_t* buf92 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(24576);
    __ubuf__ fp8_e4m3fn_t* buf93 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(32768);
    __cbuf__ fp8_e4m3fn_t* buf94 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(38400);
    __cbuf__ fp8_e4m3fn_t* buf95 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(2048);

    if ASCEND_IS_AIC {
        asc_sync_intra_arrive(PIPE_MTE1, 1);
        asc_sync_intra_arrive(PIPE_MTE1, 17);
        asc_sync_intra_arrive(PIPE_MTE1, 0);
        asc_sync_intra_arrive(PIPE_MTE1, 16);
        int64_t iter96 = 0;
        int64_t iter97 = 0;
        for (int64_t i98 = 0; i98 < 64; i98 = (int64_t)((uint64_t)i98 + (uint64_t)1)) {
            int64_t dim99 = gm2_arg.stride[0];
            int64_t t100 = dim99;
            int64_t t101 = i98;
            int64_t t102 = t101 * t100;
            int32_t ld103 = gm2[t102];
            int32_t t104 = (int32_t)((uint32_t)ld103 + (uint32_t)15);
            int32_t t105 = t104 % 16;
            bool cmp106 = t105 != 0;
            bool cmp107 = t105 < 0;
            bool t108 = cmp106 & cmp107;
            int32_t t109 = t104 / 16;
            int32_t t110 = (int32_t)(uint8_t)t108;
            int32_t t111 = (int32_t)((uint32_t)t109 - (uint32_t)t110);
            int32_t t112 = (int32_t)((uint32_t)t111 * (uint32_t)9);
            int64_t t113 = (int64_t)((uint64_t)arg9 + (uint64_t)arg10);
            int64_t t114 = iter97 % arg10;
            bool cmp115 = t114 != 0;
            bool cmp116 = t114 < 0;
            bool cmp117 = arg10 < 0;
            bool t118 = cmp116 ^ cmp117;
            bool t119 = cmp115 & t118;
            int64_t sel120 = t119 ? arg10 : 0;
            int64_t t121 = (int64_t)((uint64_t)t114 + (uint64_t)sel120);
            int64_t t122 = (int64_t)((uint64_t)t113 - (uint64_t)t121);
            int64_t t123 = t122 % arg10;
            bool cmp124 = t123 != 0;
            bool cmp125 = t123 < 0;
            bool t126 = cmp125 ^ cmp117;
            bool t127 = cmp124 & t126;
            int64_t sel128 = t127 ? arg10 : 0;
            int64_t t129 = (int64_t)((uint64_t)t123 + (uint64_t)sel128);
            int64_t t130 = (int64_t)t112;
            for (int64_t i131 = t129; i131 < t130; i131 = (int64_t)((uint64_t)i131 + (uint64_t)arg10)) {
                int64_t t132 = i131 % 9;
                bool cmp133 = t132 != 0;
                bool cmp134 = t132 < 0;
                bool t135 = cmp133 & cmp134;
                int64_t t136 = i131 / 9;
                int64_t t137 = (int64_t)(uint8_t)t135;
                int64_t t138 = (int64_t)((uint64_t)t136 - (uint64_t)t137);
                int64_t t139 = (int64_t)((uint64_t)t138 * (uint64_t)16);
                int64_t sel140 = t135 ? 9 : 0;
                int64_t t141 = t132 + sel140;
                int64_t t142 = (int64_t)ld103;
                int64_t t143 = (int64_t)((uint64_t)t142 - (uint64_t)t139);
                int64_t t144 = (t143 < 16 ? t143 : 16);
                int64_t t145 = i98 * 160;
                int64_t t146 = (int64_t)((uint64_t)t145 + (uint64_t)t139);
                int64_t t147 = (int64_t)((uint64_t)iter96 + (uint64_t)t139);
                for (int64_t i148 = 0; i148 < 40; i148 = (int64_t)((uint64_t)i148 + (uint64_t)1)) {
                    __cbuf__ fp8_e4m3fn_t* view_root149 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf87) + 0);
                    int64_t t150 = t146 % 16;
                    bool cmp151 = t150 != 0;
                    bool cmp152 = t150 < 0;
                    bool t153 = cmp151 & cmp152;
                    int64_t t154 = t146 / 16;
                    int64_t t155 = (int64_t)(uint8_t)t153;
                    int64_t t156 = (int64_t)((uint64_t)t154 - (uint64_t)t155);
                    int64_t t157 = (t156 < 640 ? t156 : 640);
                    bool cmp158 = t156 <= 640;
                    __cbuf__ fp8_e4m3fn_t* view_root159 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf89) + 0);
                    int64_t t160 = (int64_t)((uint64_t)t156 * (uint64_t)2);
                    int64_t dim161 = gm0_arg.stride[0];
                    int64_t t162 = (int64_t)((uint64_t)t160 * (uint64_t)dim161);
                    int64_t t163 = i148 * 16;
                    int64_t t164 = (int64_t)((uint64_t)t162 + (uint64_t)t163);
                    int64_t t165 = (int64_t)((uint64_t)t164 * (uint64_t)8);
                    int64_t t166 = (int64_t)((uint64_t)t157 * (uint64_t)16);
                    int64_t t167 = (int64_t)((uint64_t)10240 - (uint64_t)t166);
                    int64_t t168 = (t167 < 16 ? t167 : 16);
                    int64_t sel169 = cmp158 ? t168 : 0;
                    int64_t dim170 = gm0_arg.stride[0];
                    asc_lock(PIPE_MTE2, 0);
                    bool cmp171 = (uint64_t)sel169 > (uint64_t)0;
                    if (cmp171) {
                        int64_t t172 = t165;
                        int32_t t173 = (int32_t)sel169;
                        set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(16) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                        asc_copy_gm2l1_nd2nz((__cbuf__ int8_t*)(view_root159 + 0), (__gm__ int8_t*)(gm0 + t172), dim170, 0, t173, 128, 0, false);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    __cbuf__ fp8_e8m0_t* view_root174 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf88) + 0);
                    __cbuf__ fp8_e8m0_t* view_root175 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf86) + 0);
                    int64_t t176 = (int64_t)((uint64_t)640 - (uint64_t)t157);
                    int64_t t177 = (t176 < 1 ? t176 : 1);
                    int64_t sel178 = cmp158 ? t177 : 0;
                    int64_t t179 = (int64_t)((uint64_t)t156 * (uint64_t)2560);
                    int64_t t180 = i148 * 64;
                    int64_t t181 = (int64_t)((uint64_t)t179 + (uint64_t)t180);
                    asc_lock(PIPE_MTE2, 1);
                    bool cmp182 = (uint64_t)sel178 > (uint64_t)0;
                    if (cmp182) {
                        {
                            asc_set_gm2l1_loop_size(1, 1);
                            asc_set_gm2l1_loop1_stride(0, 0);
                            asc_set_gm2l1_loop2_stride(0, 0);
                            uint8_t dma_pad = static_cast<uint8_t>(0);
                            asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                            asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root174 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm1 + t181), sel178, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)2560, 64);
                            asc_set_gm2l1_loop_size(1, 1);
                            asc_set_gm2l1_loop1_stride(0, 0);
                            asc_set_gm2l1_loop2_stride(0, 0);
                            asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                        }
                    }
                    asc_unlock(PIPE_MTE2, 1);
                    int64_t t183 = i98 * 40;
                    int64_t t184 = t183 + i148;
                    int64_t t185 = t141 * 2621440;
                    int64_t t186 = t184 * 64;
                    int64_t t187 = t185 + t186;
                    asc_lock(PIPE_MTE2, 2);
                    {
                        asc_set_gm2l1_loop_size(1, 1);
                        asc_set_gm2l1_loop1_stride(0, 0);
                        asc_set_gm2l1_loop2_stride(0, 0);
                        uint8_t dma_pad = static_cast<uint8_t>(0);
                        asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                        asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root175 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm3 + t187), 16, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)163840, 64);
                        asc_set_gm2l1_loop_size(1, 1);
                        asc_set_gm2l1_loop1_stride(0, 0);
                        asc_set_gm2l1_loop2_stride(0, 0);
                        asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                    }
                    asc_unlock(PIPE_MTE2, 2);
                    __ca__ fp8_e4m3fn_t* view_root188 = reinterpret_cast<__ca__ fp8_e4m3fn_t*>(reinterpret_cast<__ca__ uint8_t*>(buf77) + 0);
                    uint64_t selected_root189 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root188)) / 16;
                    asc_lock(PIPE_MTE1, 3);
                    asc_lock(PIPE_MTE1, 0);
                    asc_lock(PIPE_MTE1, 1);
                    asc_copy_l12l0a(view_root188 + 0, view_root159 + 0, 0, 0, 1, 4, 1, 1);
                    asc_copy_l12l0a_mx(selected_root189, (__cbuf__ fp8_e8m0_t*)(view_root174 + 0), 0, 0, 1, 2, 2, 2);
                    asc_unlock(PIPE_MTE1, 1);
                    asc_unlock(PIPE_MTE1, 0);
                    asc_unlock(PIPE_MTE1, 3);
                    __cb__ fp8_e4m3fn_t* view_root190 = reinterpret_cast<__cb__ fp8_e4m3fn_t*>(reinterpret_cast<__cb__ uint8_t*>(buf78) + 0);
                    uint64_t selected_root191 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root190)) / 16;
                    asc_sync_intra_wait(PIPE_MTE1, 0);
                    asc_sync_intra_wait(PIPE_MTE1, 16);
                    asc_copy_l12l0b_transpose(view_root190 + 0, view_root149 + 0, 0, 0, 8, 8, 8, 16);
                    asc_sync_intra_arrive(PIPE_MTE1, 0);
                    asc_sync_intra_arrive(PIPE_MTE1, 16);
                    asc_lock(PIPE_MTE1, 4);
                    asc_lock(PIPE_MTE1, 2);
                    asc_copy_l12l0b_mx(selected_root191, (__cbuf__ fp8_e8m0_t*)(view_root175 + 0), 0, 0, 16, 2, 2, 2);
                    asc_unlock(PIPE_MTE1, 2);
                    asc_unlock(PIPE_MTE1, 4);
                    bool cmp192 = i148 == 0;
                    __cc__ float* view_root193 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf76) + 0);
                    asc_lock(PIPE_M, 5);
                    asc_lock(PIPE_M, 3);
                    asc_lock(PIPE_M, 4);
                    if (cmp192) {
                        asc_mmad_mx(view_root193 + 0, view_root188, view_root190, 16, 128, 256, 0, true, false, true);
                    } else {
                        asc_mmad_mx(view_root193 + 0, view_root188, view_root190, 16, 128, 256, 0, true, false, false);
                    }
                    asc_unlock(PIPE_M, 4);
                    asc_unlock(PIPE_M, 3);
                    asc_unlock(PIPE_M, 5);
                }
                int64_t t194 = t141 * 256;
                int64_t dim195 = gm5_arg.stride[0];
                int64_t t196 = (int64_t)((uint64_t)t147 * (uint64_t)dim195);
                int64_t t197 = t196;
                int64_t t198 = t194;
                int64_t t199 = t197 + t198;
                int64_t dim200 = gm5_arg.stride[0];
                __cc__ float* view_root201 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf76) + 0);
                bool cmp202 = (uint64_t)t144 > (uint64_t)0;
                if (cmp202) {
                    asc_lock(PIPE_FIX, 5);
                    int32_t t203 = (int32_t)t144;
                    int32_t t204 = (int32_t)dim200;
                    asc_set_l0c2gm_nz2nd(1, 0, 0);
                    asc_copy_l0c2gm(gm5 + t199, view_root201, 256, t203, t204, 16, 0, 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                    asc_unlock(PIPE_FIX, 5);
                }
                for (int64_t i205 = 0; i205 < 40; i205 = (int64_t)((uint64_t)i205 + (uint64_t)1)) {
                    __cbuf__ fp8_e4m3fn_t* view_root206 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf83) + 0);
                    int64_t t207 = t146 % 16;
                    bool cmp208 = t207 != 0;
                    bool cmp209 = t207 < 0;
                    bool t210 = cmp208 & cmp209;
                    int64_t t211 = t146 / 16;
                    int64_t t212 = (int64_t)(uint8_t)t210;
                    int64_t t213 = (int64_t)((uint64_t)t211 - (uint64_t)t212);
                    int64_t t214 = (t213 < 640 ? t213 : 640);
                    bool cmp215 = t213 <= 640;
                    __cbuf__ fp8_e4m3fn_t* view_root216 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf85) + 0);
                    int64_t t217 = (int64_t)((uint64_t)t213 * (uint64_t)2);
                    int64_t dim218 = gm0_arg.stride[0];
                    int64_t t219 = (int64_t)((uint64_t)t217 * (uint64_t)dim218);
                    int64_t t220 = i205 * 16;
                    int64_t t221 = (int64_t)((uint64_t)t219 + (uint64_t)t220);
                    int64_t t222 = (int64_t)((uint64_t)t221 * (uint64_t)8);
                    int64_t t223 = (int64_t)((uint64_t)t214 * (uint64_t)16);
                    int64_t t224 = (int64_t)((uint64_t)10240 - (uint64_t)t223);
                    int64_t t225 = (t224 < 16 ? t224 : 16);
                    int64_t sel226 = cmp215 ? t225 : 0;
                    int64_t dim227 = gm0_arg.stride[0];
                    asc_lock(PIPE_MTE2, 6);
                    bool cmp228 = (uint64_t)sel226 > (uint64_t)0;
                    if (cmp228) {
                        int64_t t229 = t222;
                        int32_t t230 = (int32_t)sel226;
                        set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(16) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                        asc_copy_gm2l1_nd2nz((__cbuf__ int8_t*)(view_root216 + 0), (__gm__ int8_t*)(gm0 + t229), dim227, 0, t230, 128, 0, false);
                    }
                    asc_unlock(PIPE_MTE2, 6);
                    __cbuf__ fp8_e8m0_t* view_root231 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf84) + 0);
                    __cbuf__ fp8_e8m0_t* view_root232 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf82) + 0);
                    int64_t t233 = (int64_t)((uint64_t)640 - (uint64_t)t214);
                    int64_t t234 = (t233 < 1 ? t233 : 1);
                    int64_t sel235 = cmp215 ? t234 : 0;
                    int64_t t236 = (int64_t)((uint64_t)t213 * (uint64_t)2560);
                    int64_t t237 = i205 * 64;
                    int64_t t238 = (int64_t)((uint64_t)t236 + (uint64_t)t237);
                    asc_lock(PIPE_MTE2, 7);
                    bool cmp239 = (uint64_t)sel235 > (uint64_t)0;
                    if (cmp239) {
                        {
                            asc_set_gm2l1_loop_size(1, 1);
                            asc_set_gm2l1_loop1_stride(0, 0);
                            asc_set_gm2l1_loop2_stride(0, 0);
                            uint8_t dma_pad = static_cast<uint8_t>(0);
                            asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                            asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root231 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm1 + t238), sel235, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)2560, 64);
                            asc_set_gm2l1_loop_size(1, 1);
                            asc_set_gm2l1_loop1_stride(0, 0);
                            asc_set_gm2l1_loop2_stride(0, 0);
                            asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                        }
                    }
                    asc_unlock(PIPE_MTE2, 7);
                    int64_t t240 = i98 * 40;
                    int64_t t241 = t240 + i205;
                    int64_t t242 = t141 * 2621440;
                    int64_t t243 = t241 * 64;
                    int64_t t244 = t242 + t243;
                    asc_lock(PIPE_MTE2, 8);
                    {
                        asc_set_gm2l1_loop_size(1, 1);
                        asc_set_gm2l1_loop1_stride(0, 0);
                        asc_set_gm2l1_loop2_stride(0, 0);
                        uint8_t dma_pad = static_cast<uint8_t>(0);
                        asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                        asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root232 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm4 + t244), 16, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)163840, 64);
                        asc_set_gm2l1_loop_size(1, 1);
                        asc_set_gm2l1_loop1_stride(0, 0);
                        asc_set_gm2l1_loop2_stride(0, 0);
                        asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                    }
                    asc_unlock(PIPE_MTE2, 8);
                    __ca__ fp8_e4m3fn_t* view_root245 = reinterpret_cast<__ca__ fp8_e4m3fn_t*>(reinterpret_cast<__ca__ uint8_t*>(buf80) + 0);
                    uint64_t selected_root246 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root245)) / 16;
                    asc_lock(PIPE_MTE1, 9);
                    asc_lock(PIPE_MTE1, 6);
                    asc_lock(PIPE_MTE1, 7);
                    asc_copy_l12l0a(view_root245 + 0, view_root216 + 0, 0, 0, 1, 4, 1, 1);
                    asc_copy_l12l0a_mx(selected_root246, (__cbuf__ fp8_e8m0_t*)(view_root231 + 0), 0, 0, 1, 2, 2, 2);
                    asc_unlock(PIPE_MTE1, 7);
                    asc_unlock(PIPE_MTE1, 6);
                    asc_unlock(PIPE_MTE1, 9);
                    __cb__ fp8_e4m3fn_t* view_root247 = reinterpret_cast<__cb__ fp8_e4m3fn_t*>(reinterpret_cast<__cb__ uint8_t*>(buf81) + 0);
                    uint64_t selected_root248 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root247)) / 16;
                    asc_sync_intra_wait(PIPE_MTE1, 1);
                    asc_sync_intra_wait(PIPE_MTE1, 17);
                    asc_copy_l12l0b_transpose(view_root247 + 0, view_root206 + 0, 0, 0, 8, 8, 8, 16);
                    asc_sync_intra_arrive(PIPE_MTE1, 1);
                    asc_sync_intra_arrive(PIPE_MTE1, 17);
                    asc_lock(PIPE_MTE1, 10);
                    asc_lock(PIPE_MTE1, 8);
                    asc_copy_l12l0b_mx(selected_root248, (__cbuf__ fp8_e8m0_t*)(view_root232 + 0), 0, 0, 16, 2, 2, 2);
                    asc_unlock(PIPE_MTE1, 8);
                    asc_unlock(PIPE_MTE1, 10);
                    bool cmp249 = i205 == 0;
                    __cc__ float* view_root250 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf79) + 0);
                    asc_lock(PIPE_M, 11);
                    asc_lock(PIPE_M, 9);
                    asc_lock(PIPE_M, 10);
                    if (cmp249) {
                        asc_mmad_mx(view_root250 + 0, view_root245, view_root247, 16, 128, 256, 0, true, false, true);
                    } else {
                        asc_mmad_mx(view_root250 + 0, view_root245, view_root247, 16, 128, 256, 0, true, false, false);
                    }
                    asc_unlock(PIPE_M, 10);
                    asc_unlock(PIPE_M, 9);
                    asc_unlock(PIPE_M, 11);
                }
                int64_t dim251 = gm6_arg.stride[0];
                int64_t t252 = (int64_t)((uint64_t)t147 * (uint64_t)dim251);
                int64_t t253 = t252;
                int64_t t254 = t253 + t198;
                int64_t dim255 = gm6_arg.stride[0];
                __cc__ float* view_root256 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf79) + 0);
                if (cmp202) {
                    asc_lock(PIPE_FIX, 11);
                    int32_t t257 = (int32_t)t144;
                    int32_t t258 = (int32_t)dim255;
                    asc_set_l0c2gm_nz2nd(1, 0, 0);
                    asc_copy_l0c2gm(gm6 + t254, view_root256, 256, t257, t258, 16, 0, 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                    asc_unlock(PIPE_FIX, 11);
                }
            }
            int32_t t259 = (int32_t)((uint32_t)t111 * (uint32_t)16);
            int64_t t260 = (int64_t)t259;
            int64_t t261 = (int64_t)((uint64_t)iter96 + (uint64_t)t260);
            int64_t t262 = (int64_t)((uint64_t)iter97 + (uint64_t)t130);
            iter96 = t261;
            iter97 = t262;
        }
    }

    if ASCEND_IS_AIV {
        int64_t iter263 = 0;
        int64_t iter264 = 0;
        for (int64_t i265 = 0; i265 < 64; i265 = (int64_t)((uint64_t)i265 + (uint64_t)1)) {
            int64_t dim266 = gm2_arg.stride[0];
            int64_t t267 = dim266;
            int64_t t268 = i265;
            int64_t t269 = t268 * t267;
            int32_t ld270 = gm2[t269];
            int32_t t271 = (int32_t)((uint32_t)ld270 + (uint32_t)15);
            int32_t t272 = t271 % 16;
            bool cmp273 = t272 != 0;
            bool cmp274 = t272 < 0;
            bool t275 = cmp273 & cmp274;
            int32_t t276 = t271 / 16;
            int32_t t277 = (int32_t)(uint8_t)t275;
            int32_t t278 = (int32_t)((uint32_t)t276 - (uint32_t)t277);
            int32_t t279 = (int32_t)((uint32_t)t278 * (uint32_t)9);
            int64_t t280 = (int64_t)((uint64_t)arg9 + (uint64_t)arg10);
            int64_t t281 = iter264 % arg10;
            bool cmp282 = t281 != 0;
            bool cmp283 = t281 < 0;
            bool cmp284 = arg10 < 0;
            bool t285 = cmp283 ^ cmp284;
            bool t286 = cmp282 & t285;
            int64_t sel287 = t286 ? arg10 : 0;
            int64_t t288 = (int64_t)((uint64_t)t281 + (uint64_t)sel287);
            int64_t t289 = (int64_t)((uint64_t)t280 - (uint64_t)t288);
            int64_t t290 = t289 % arg10;
            bool cmp291 = t290 != 0;
            bool cmp292 = t290 < 0;
            bool t293 = cmp292 ^ cmp284;
            bool t294 = cmp291 & t293;
            int64_t sel295 = t294 ? arg10 : 0;
            int64_t t296 = (int64_t)((uint64_t)t290 + (uint64_t)sel295);
            int64_t t297 = (int64_t)t279;
            for (int64_t i298 = t296; i298 < t297; i298 = (int64_t)((uint64_t)i298 + (uint64_t)arg10)) {
                int64_t t299 = i298 % 9;
                bool cmp300 = t299 != 0;
                bool cmp301 = t299 < 0;
                bool t302 = cmp300 & cmp301;
                int64_t sel303 = t302 ? 9 : 0;
                int64_t t304 = t299 + sel303;
                for (int64_t i305 = 0; i305 < 40; i305 = (int64_t)((uint64_t)i305 + (uint64_t)1)) {
                    int64_t t306 = i265 * 46080;
                    int64_t t307 = t304 * 5120;
                    int64_t t308 = t306 + t307;
                    int64_t t309 = i305 * 128;
                    int64_t t310 = t308 + t309;
                    int64_t t311 = (int64_t)((uint64_t)arg11 * (uint64_t)64);
                    int64_t t312 = (int64_t)((uint64_t)t310 + (uint64_t)t311);
                    int64_t t313 = t312 % 64;
                    bool cmp314 = t313 != 0;
                    bool cmp315 = t313 < 0;
                    bool t316 = cmp314 & cmp315;
                    int64_t t317 = t312 / 64;
                    int64_t t318 = (int64_t)(uint8_t)t316;
                    int64_t t319 = (int64_t)((uint64_t)t317 - (uint64_t)t318);
                    int64_t t320 = (int64_t)((uint64_t)t319 * (uint64_t)8192);
                    asc_lock(PIPE_MTE2, 0);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf90 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm7 + t320), 1, 8192, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    asc_lock(PIPE_V, 0);
                    asc_lock(PIPE_V, 1);
                    kernel_project_up_worker_vf0((__ubuf__ uint8_t*)buf90, (__ubuf__ fp8_e4m3fn_t*)buf91);
                    asc_unlock(PIPE_V, 1);
                    asc_unlock(PIPE_V, 0);
                    __ubuf__ fp8_e4m3fn_t* view_root321 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf91) + 0);
                    __cbuf__ fp8_e4m3fn_t* view_root322 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf95) + 0);
                    int64_t t323 = t311;
                    int64_t t324 = t323 * 32;
                    asc_sync_intra_wait(PIPE_MTE3, 0);
                    asc_lock(PIPE_MTE3, 1);
                    asc_copy_ub2l1(view_root322 + t324, (__ubuf__ int8_t*)view_root321 + 0, 8, 64, 0, 64);
                    asc_unlock(PIPE_MTE3, 1);
                    asc_sync_intra_arrive(PIPE_MTE3, 0);
                }
                for (int64_t i325 = 0; i325 < 40; i325 = (int64_t)((uint64_t)i325 + (uint64_t)1)) {
                    int64_t t326 = i265 * 46080;
                    int64_t t327 = t304 * 5120;
                    int64_t t328 = t326 + t327;
                    int64_t t329 = i325 * 128;
                    int64_t t330 = t328 + t329;
                    int64_t t331 = (int64_t)((uint64_t)arg11 * (uint64_t)64);
                    int64_t t332 = (int64_t)((uint64_t)t330 + (uint64_t)t331);
                    int64_t t333 = t332 % 64;
                    bool cmp334 = t333 != 0;
                    bool cmp335 = t333 < 0;
                    bool t336 = cmp334 & cmp335;
                    int64_t t337 = t332 / 64;
                    int64_t t338 = (int64_t)(uint8_t)t336;
                    int64_t t339 = (int64_t)((uint64_t)t337 - (uint64_t)t338);
                    int64_t t340 = (int64_t)((uint64_t)t339 * (uint64_t)8192);
                    asc_lock(PIPE_MTE2, 2);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf92 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm8 + t340), 1, 8192, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                    }
                    asc_unlock(PIPE_MTE2, 2);
                    asc_lock(PIPE_V, 2);
                    asc_lock(PIPE_V, 3);
                    kernel_project_up_worker_vf1((__ubuf__ uint8_t*)buf92, (__ubuf__ fp8_e4m3fn_t*)buf93);
                    asc_unlock(PIPE_V, 3);
                    asc_unlock(PIPE_V, 2);
                    __ubuf__ fp8_e4m3fn_t* view_root341 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf93) + 0);
                    __cbuf__ fp8_e4m3fn_t* view_root342 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf94) + 0);
                    int64_t t343 = t331;
                    int64_t t344 = t343 * 32;
                    asc_sync_intra_wait(PIPE_MTE3, 1);
                    asc_lock(PIPE_MTE3, 3);
                    asc_copy_ub2l1(view_root342 + t344, (__ubuf__ int8_t*)view_root341 + 0, 8, 64, 0, 64);
                    asc_unlock(PIPE_MTE3, 3);
                    asc_sync_intra_arrive(PIPE_MTE3, 1);
                }
            }
            int32_t t345 = (int32_t)((uint32_t)t278 * (uint32_t)16);
            int64_t t346 = (int64_t)t345;
            int64_t t347 = (int64_t)((uint64_t)iter263 + (uint64_t)t346);
            int64_t t348 = (int64_t)((uint64_t)iter264 + (uint64_t)t297);
            iter263 = t347;
            iter264 = t348;
        }
        asc_sync_intra_wait(PIPE_MTE3, 0);
        asc_sync_intra_wait(PIPE_MTE3, 1);
    }
}
