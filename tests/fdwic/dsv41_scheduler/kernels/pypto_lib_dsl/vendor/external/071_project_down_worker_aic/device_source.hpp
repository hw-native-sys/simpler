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
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__simd_vf__ inline void kernel_project_down_worker_vf0(__ubuf__ uint8_t* buf9, __ubuf__ fp8_e4m3fn_t* buf10)
{
    vector_bool vmask;
    vector_bool mask11 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t v12;
    asc_arange(*(vector_int8_t*)&v12, static_cast<int8_t>(0));
    vector_uint8_t v13;
    asc_duplicate_scalar(v13, 7, mask11);
    vector_uint8_t v14;
    asc_and(v14, v12, v13, mask11);
    vector_uint8_t v15;
    asc_shiftleft_scalar(v15, v14, 2, mask11);
    vector_uint8_t v16;
    asc_add_scalar(v16, v15, 48, mask11);
    vector_uint8_t v17;
    asc_duplicate_scalar(v17, 0, mask11);
    vector_uint8_t v18;
    asc_duplicate_scalar(v18, 48, mask11);
    vector_bool m19;
    asc_eq_scalar(m19, v14, 0, mask11);
    vector_uint8_t v20;
    asc_select(v20, v17, v16, m19);
    vector_bool m21;
    asc_eq_scalar(m21, v14, 1, mask11);
    vector_uint8_t v22;
    asc_select(v22, v18, v20, m21);
    vector_uint8_t v23;
    asc_duplicate_scalar(v23, 8, mask11);
    vector_uint8_t v24;
    asc_and(v24, v12, v23, mask11);
    vector_uint8_t v25;
    asc_shiftleft_scalar(v25, v24, 4, mask11);
    vector_uint8_t v26;
    asc_or(v26, v22, v25, mask11);
    for (int32_t i27 = 0; i27 < 64; i27 = (int32_t)((uint32_t)i27 + (uint32_t)2)) {
        int32_t t28 = i27 * 128;
        int32_t t29 = t28;
        vector_uint8_t v30;
        asc_loadalign(v30, buf9 + t29);
        vector_uint8_t v31;
        asc_shiftright_scalar(v31, v30, 4, mask11);
        vector_uint8_t v32;
        asc_gather(v32, v26, v30);
        vector_uint8_t v33;
        asc_gather(v33, v26, v31);
        vector_uint8_t v34;
        vector_uint8_t v35;
        asc_intlv(v34, v35, v32, v33);
        int32_t t36 = i27 * 32;
        int32_t t37 = t36;
        asc_storealign((__ubuf__ uint8_t*)(buf10 + t37), v34, (uint16_t)64, (uint16_t)0, mask11);
        int32_t t38 = i27 + 1;
        int32_t t39 = t38 * 32;
        int32_t t40 = t39;
        asc_storealign((__ubuf__ uint8_t*)(buf10 + t40), v35, (uint16_t)64, (uint16_t)0, mask11);
    }
}

__global__ __aicore__ void kernel_project_down_worker(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm0_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm1_arg, KernelArgTensor_int32_t_2d_Device_ss_ds gm2_arg, KernelArgTensor_uint8_t_3d_Device_sss_dds gm5_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm3_arg, KernelArgTensor_float_2d_Device_ss_ds gm4_arg, int64_t arg6, int64_t arg7, int64_t arg8)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ fp8_e4m3fn_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e8m0_t* gm1 = gm1_arg.ptr;
    __gm__ int32_t* gm2 = gm2_arg.ptr;
    __gm__ uint8_t* gm5 = gm5_arg.ptr;
    __gm__ fp8_e8m0_t* gm3 = gm3_arg.ptr;
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

    __cc__ float* buf41 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ fp8_e4m3fn_t* buf42 = (__ca__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cb__ fp8_e4m3fn_t* buf43 = (__cb__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ fp8_e8m0_t* buf44 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(35328);
    __cbuf__ fp8_e4m3fn_t* buf45 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(2048);
    __cbuf__ fp8_e8m0_t* buf46 = (__cbuf__ fp8_e8m0_t*)(uintptr_t)(uint64_t)(34816);
    __cbuf__ fp8_e4m3fn_t* buf47 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ uint8_t* buf48 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ fp8_e4m3fn_t* buf49 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(8192);
    __cbuf__ fp8_e4m3fn_t* buf50 = (__cbuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(2048);

    if ASCEND_IS_AIC {
        asc_sync_intra_arrive(PIPE_MTE1, 0);
        asc_sync_intra_arrive(PIPE_MTE1, 16);
        int64_t iter51 = 0;
        int64_t iter52 = 0;
        int64_t iter53 = 0;
        for (int64_t i54 = 0; i54 < 64; i54 = (int64_t)((uint64_t)i54 + (uint64_t)1)) {
            int64_t dim55 = gm2_arg.stride[0];
            int64_t t56 = dim55;
            int64_t t57 = i54;
            int64_t t58 = t57 * t56;
            int32_t ld59 = gm2[t58];
            int32_t t60 = (int32_t)((uint32_t)ld59 + (uint32_t)15);
            int32_t t61 = t60 % 16;
            bool cmp62 = t61 != 0;
            bool cmp63 = t61 < 0;
            bool t64 = cmp62 & cmp63;
            int32_t t65 = t60 / 16;
            int32_t t66 = (int32_t)(uint8_t)t64;
            int32_t t67 = (int32_t)((uint32_t)t65 - (uint32_t)t66);
            int32_t t68 = (int32_t)((uint32_t)t67 * (uint32_t)40);
            int64_t t69 = (int64_t)((uint64_t)arg6 + (uint64_t)arg7);
            int64_t t70 = iter53 % arg7;
            bool cmp71 = t70 != 0;
            bool cmp72 = t70 < 0;
            bool cmp73 = arg7 < 0;
            bool t74 = cmp72 ^ cmp73;
            bool t75 = cmp71 & t74;
            int64_t sel76 = t75 ? arg7 : 0;
            int64_t t77 = (int64_t)((uint64_t)t70 + (uint64_t)sel76);
            int64_t t78 = (int64_t)((uint64_t)t69 - (uint64_t)t77);
            int64_t t79 = t78 % arg7;
            bool cmp80 = t79 != 0;
            bool cmp81 = t79 < 0;
            bool t82 = cmp81 ^ cmp73;
            bool t83 = cmp80 & t82;
            int64_t sel84 = t83 ? arg7 : 0;
            int64_t t85 = (int64_t)((uint64_t)t79 + (uint64_t)sel84);
            int64_t t86 = (int64_t)t68;
            for (int64_t i87 = t85; i87 < t86; i87 = (int64_t)((uint64_t)i87 + (uint64_t)arg7)) {
                int64_t t88 = i87 % 20;
                bool cmp89 = t88 != 0;
                bool cmp90 = t88 < 0;
                bool t91 = cmp89 & cmp90;
                int64_t sel92 = t91 ? 20 : 0;
                int64_t t93 = t88 + sel92;
                int64_t t94 = i87 / 20;
                int64_t t95 = (int64_t)(uint8_t)t91;
                int64_t t96 = (int64_t)((uint64_t)t94 - (uint64_t)t95);
                int64_t t97 = t96 % 2;
                bool cmp98 = t97 != 0;
                bool cmp99 = t97 < 0;
                bool t100 = cmp98 & cmp99;
                int64_t sel101 = t100 ? 2 : 0;
                int64_t t102 = t97 + sel101;
                int64_t t103 = i87 % 40;
                bool cmp104 = t103 != 0;
                bool cmp105 = t103 < 0;
                bool t106 = cmp104 & cmp105;
                int64_t t107 = i87 / 40;
                int64_t t108 = (int64_t)(uint8_t)t106;
                int64_t t109 = (int64_t)((uint64_t)t107 - (uint64_t)t108);
                int64_t t110 = (int64_t)((uint64_t)t109 * (uint64_t)16);
                int64_t t111 = (int64_t)ld59;
                int64_t t112 = (int64_t)((uint64_t)t111 - (uint64_t)t110);
                int64_t t113 = (t112 < 16 ? t112 : 16);
                int64_t t114 = t102 * 9;
                int64_t t115 = t114 % 2;
                bool cmp116 = t115 != 0;
                bool cmp117 = t115 < 0;
                bool t118 = cmp116 & cmp117;
                int64_t t119 = t114 / 2;
                int64_t t120 = (int64_t)(uint8_t)t118;
                int64_t t121 = (int64_t)((uint64_t)t119 - (uint64_t)t120);
                int64_t t122 = (int64_t)((uint64_t)t121 * (uint64_t)256);
                int64_t t123 = t102 + 1;
                int64_t t124 = t123 * 9;
                int64_t t125 = t124 % 2;
                bool cmp126 = t125 != 0;
                bool cmp127 = t125 < 0;
                bool t128 = cmp126 & cmp127;
                int64_t t129 = t124 / 2;
                int64_t t130 = (int64_t)(uint8_t)t128;
                int64_t t131 = (int64_t)((uint64_t)t129 - (uint64_t)t130);
                int64_t t132 = (int64_t)((uint64_t)t131 * (uint64_t)256);
                int64_t t133 = t102 * 976;
                int64_t t134 = (int64_t)((uint64_t)t133 + (uint64_t)iter52);
                int64_t t135 = (int64_t)((uint64_t)t134 + (uint64_t)t110);
                int64_t t136 = (int64_t)((uint64_t)iter51 + (uint64_t)t110);
                int64_t t137 = t122 % 128;
                bool cmp138 = t137 != 0;
                bool cmp139 = t137 < 0;
                bool t140 = cmp138 & cmp139;
                int64_t t141 = t122 / 128;
                int64_t t142 = (int64_t)(uint8_t)t140;
                int64_t t143 = (int64_t)((uint64_t)t141 - (uint64_t)t142);
                int64_t t144 = t132 % 128;
                bool cmp145 = t144 != 0;
                bool cmp146 = t144 < 0;
                bool t147 = cmp145 & cmp146;
                int64_t t148 = t132 / 128;
                int64_t t149 = (int64_t)(uint8_t)t147;
                int64_t t150 = (int64_t)((uint64_t)t148 - (uint64_t)t149);
                for (int64_t i151 = t143; i151 < t150; i151 = (int64_t)((uint64_t)i151 + (uint64_t)1)) {
                    __cbuf__ fp8_e4m3fn_t* view_root152 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf45) + 0);
                    int64_t t153 = t136 % 16;
                    bool cmp154 = t153 != 0;
                    bool cmp155 = t153 < 0;
                    bool t156 = cmp154 & cmp155;
                    int64_t t157 = t136 / 16;
                    int64_t t158 = (int64_t)(uint8_t)t156;
                    int64_t t159 = (int64_t)((uint64_t)t157 - (uint64_t)t158);
                    int64_t t160 = (t159 < 120 ? t159 : 120);
                    bool cmp161 = t159 <= 120;
                    __cbuf__ fp8_e4m3fn_t* view_root162 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf47) + 0);
                    int64_t t163 = (int64_t)((uint64_t)t159 * (uint64_t)2);
                    int64_t dim164 = gm0_arg.stride[0];
                    int64_t t165 = (int64_t)((uint64_t)t163 * (uint64_t)dim164);
                    int64_t t166 = (int64_t)((uint64_t)i151 * (uint64_t)16);
                    int64_t t167 = (int64_t)((uint64_t)t165 + (uint64_t)t166);
                    int64_t t168 = (int64_t)((uint64_t)t167 * (uint64_t)8);
                    int64_t t169 = (int64_t)((uint64_t)t160 * (uint64_t)16);
                    int64_t t170 = (int64_t)((uint64_t)1920 - (uint64_t)t169);
                    int64_t t171 = (t170 < 16 ? t170 : 16);
                    int64_t sel172 = cmp161 ? t171 : 0;
                    int64_t dim173 = gm0_arg.stride[0];
                    asc_lock(PIPE_MTE2, 0);
                    bool cmp174 = (uint64_t)sel172 > (uint64_t)0;
                    if (cmp174) {
                        int64_t t175 = t168;
                        int32_t t176 = (int32_t)sel172;
                        set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(16) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                        asc_copy_gm2l1_nd2nz((__cbuf__ int8_t*)(view_root162 + 0), (__gm__ int8_t*)(gm0 + t175), dim173, 0, t176, 128, 0, false);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    __cbuf__ fp8_e8m0_t* view_root177 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf46) + 0);
                    __cbuf__ fp8_e8m0_t* view_root178 = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf44) + 0);
                    int64_t t179 = (int64_t)((uint64_t)120 - (uint64_t)t160);
                    int64_t t180 = (t179 < 1 ? t179 : 1);
                    int64_t sel181 = cmp161 ? t180 : 0;
                    int64_t t182 = (int64_t)((uint64_t)t159 * (uint64_t)1152);
                    int64_t t183 = (int64_t)((uint64_t)i151 * (uint64_t)64);
                    int64_t t184 = (int64_t)((uint64_t)t182 + (uint64_t)t183);
                    asc_lock(PIPE_MTE2, 1);
                    bool cmp185 = (uint64_t)sel181 > (uint64_t)0;
                    if (cmp185) {
                        {
                            asc_set_gm2l1_loop_size(1, 1);
                            asc_set_gm2l1_loop1_stride(0, 0);
                            asc_set_gm2l1_loop2_stride(0, 0);
                            uint8_t dma_pad = static_cast<uint8_t>(0);
                            asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                            asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root177 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm1 + t184), sel181, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)1152, 64);
                            asc_set_gm2l1_loop_size(1, 1);
                            asc_set_gm2l1_loop1_stride(0, 0);
                            asc_set_gm2l1_loop2_stride(0, 0);
                            asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                        }
                    }
                    asc_unlock(PIPE_MTE2, 1);
                    int64_t t186 = i54 * 18;
                    int64_t t187 = (int64_t)((uint64_t)t186 + (uint64_t)i151);
                    int64_t t188 = t93 * 1179648;
                    int64_t t189 = (int64_t)((uint64_t)t187 * (uint64_t)64);
                    int64_t t190 = (int64_t)((uint64_t)t188 + (uint64_t)t189);
                    asc_lock(PIPE_MTE2, 2);
                    {
                        asc_set_gm2l1_loop_size(1, 1);
                        asc_set_gm2l1_loop1_stride(0, 0);
                        asc_set_gm2l1_loop2_stride(0, 0);
                        uint8_t dma_pad = static_cast<uint8_t>(0);
                        asc_set_gm2l1_pad(static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&dma_pad)));
                        asc_copy_gm2l1_align((__cbuf__ uint8_t*)((__cbuf__ uint8_t*)view_root178 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm3 + t190), 16, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)73728, 64);
                        asc_set_gm2l1_loop_size(1, 1);
                        asc_set_gm2l1_loop1_stride(0, 0);
                        asc_set_gm2l1_loop2_stride(0, 0);
                        asc_set_gm2l1_pad(static_cast<uint32_t>(0));
                    }
                    asc_unlock(PIPE_MTE2, 2);
                    __ca__ fp8_e4m3fn_t* view_root191 = reinterpret_cast<__ca__ fp8_e4m3fn_t*>(reinterpret_cast<__ca__ uint8_t*>(buf42) + 0);
                    uint64_t selected_root192 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root191)) / 16;
                    asc_lock(PIPE_MTE1, 3);
                    asc_lock(PIPE_MTE1, 0);
                    asc_lock(PIPE_MTE1, 1);
                    asc_copy_l12l0a(view_root191 + 0, view_root162 + 0, 0, 0, 1, 4, 1, 1);
                    asc_copy_l12l0a_mx(selected_root192, (__cbuf__ fp8_e8m0_t*)(view_root177 + 0), 0, 0, 1, 2, 2, 2);
                    asc_unlock(PIPE_MTE1, 1);
                    asc_unlock(PIPE_MTE1, 0);
                    asc_unlock(PIPE_MTE1, 3);
                    __cb__ fp8_e4m3fn_t* view_root193 = reinterpret_cast<__cb__ fp8_e4m3fn_t*>(reinterpret_cast<__cb__ uint8_t*>(buf43) + 0);
                    uint64_t selected_root194 = static_cast<uint64_t>(reinterpret_cast<uint64_t>(view_root193)) / 16;
                    asc_sync_intra_wait(PIPE_MTE1, 0);
                    asc_sync_intra_wait(PIPE_MTE1, 16);
                    asc_copy_l12l0b_transpose(view_root193 + 0, view_root152 + 0, 0, 0, 8, 8, 8, 16);
                    asc_sync_intra_arrive(PIPE_MTE1, 0);
                    asc_sync_intra_arrive(PIPE_MTE1, 16);
                    asc_lock(PIPE_MTE1, 4);
                    asc_lock(PIPE_MTE1, 2);
                    asc_copy_l12l0b_mx(selected_root194, (__cbuf__ fp8_e8m0_t*)(view_root178 + 0), 0, 0, 16, 2, 2, 2);
                    asc_unlock(PIPE_MTE1, 2);
                    asc_unlock(PIPE_MTE1, 4);
                    bool cmp195 = i151 == t143;
                    __cc__ float* view_root196 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf41) + 0);
                    asc_lock(PIPE_M, 5);
                    asc_lock(PIPE_M, 3);
                    asc_lock(PIPE_M, 4);
                    if (cmp195) {
                        asc_mmad_mx(view_root196 + 0, view_root191, view_root193, 16, 128, 256, 0, true, false, true);
                    } else {
                        asc_mmad_mx(view_root196 + 0, view_root191, view_root193, 16, 128, 256, 0, true, false, false);
                    }
                    asc_unlock(PIPE_M, 4);
                    asc_unlock(PIPE_M, 3);
                    asc_unlock(PIPE_M, 5);
                }
                int64_t t197 = t93 * 256;
                int64_t dim198 = gm4_arg.stride[0];
                int64_t t199 = (int64_t)((uint64_t)t135 * (uint64_t)dim198);
                int64_t t200 = t199;
                int64_t t201 = t197;
                int64_t t202 = t200 + t201;
                int64_t dim203 = gm4_arg.stride[0];
                __cc__ float* view_root204 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf41) + 0);
                bool cmp205 = (uint64_t)t113 > (uint64_t)0;
                if (cmp205) {
                    asc_lock(PIPE_FIX, 5);
                    int32_t t206 = (int32_t)t113;
                    int32_t t207 = (int32_t)dim203;
                    asc_set_l0c2gm_nz2nd(1, 0, 0);
                    asc_copy_l0c2gm(gm4 + t202, view_root204, 256, t206, t207, 16, 0, 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                    asc_unlock(PIPE_FIX, 5);
                }
            }
            int32_t t208 = (int32_t)((uint32_t)t67 * (uint32_t)16);
            int64_t t209 = (int64_t)t208;
            int64_t t210 = (int64_t)((uint64_t)iter51 + (uint64_t)t209);
            int64_t t211 = (int64_t)ld59;
            int64_t t212 = (int64_t)((uint64_t)iter52 + (uint64_t)t211);
            int64_t t213 = (int64_t)((uint64_t)iter53 + (uint64_t)t86);
            iter51 = t210;
            iter52 = t212;
            iter53 = t213;
        }
    }

    if ASCEND_IS_AIV {
        int64_t iter214 = 0;
        int64_t iter215 = 0;
        int64_t iter216 = 0;
        for (int64_t i217 = 0; i217 < 64; i217 = (int64_t)((uint64_t)i217 + (uint64_t)1)) {
            int64_t dim218 = gm2_arg.stride[0];
            int64_t t219 = dim218;
            int64_t t220 = i217;
            int64_t t221 = t220 * t219;
            int32_t ld222 = gm2[t221];
            int32_t t223 = (int32_t)((uint32_t)ld222 + (uint32_t)15);
            int32_t t224 = t223 % 16;
            bool cmp225 = t224 != 0;
            bool cmp226 = t224 < 0;
            bool t227 = cmp225 & cmp226;
            int32_t t228 = t223 / 16;
            int32_t t229 = (int32_t)(uint8_t)t227;
            int32_t t230 = (int32_t)((uint32_t)t228 - (uint32_t)t229);
            int32_t t231 = (int32_t)((uint32_t)t230 * (uint32_t)40);
            int64_t t232 = (int64_t)((uint64_t)arg6 + (uint64_t)arg7);
            int64_t t233 = iter216 % arg7;
            bool cmp234 = t233 != 0;
            bool cmp235 = t233 < 0;
            bool cmp236 = arg7 < 0;
            bool t237 = cmp235 ^ cmp236;
            bool t238 = cmp234 & t237;
            int64_t sel239 = t238 ? arg7 : 0;
            int64_t t240 = (int64_t)((uint64_t)t233 + (uint64_t)sel239);
            int64_t t241 = (int64_t)((uint64_t)t232 - (uint64_t)t240);
            int64_t t242 = t241 % arg7;
            bool cmp243 = t242 != 0;
            bool cmp244 = t242 < 0;
            bool t245 = cmp244 ^ cmp236;
            bool t246 = cmp243 & t245;
            int64_t sel247 = t246 ? arg7 : 0;
            int64_t t248 = (int64_t)((uint64_t)t242 + (uint64_t)sel247);
            int64_t t249 = (int64_t)t231;
            for (int64_t i250 = t248; i250 < t249; i250 = (int64_t)((uint64_t)i250 + (uint64_t)arg7)) {
                int64_t t251 = i250 % 20;
                bool cmp252 = t251 != 0;
                bool cmp253 = t251 < 0;
                bool t254 = cmp252 & cmp253;
                int64_t sel255 = t254 ? 20 : 0;
                int64_t t256 = t251 + sel255;
                int64_t t257 = i250 / 20;
                int64_t t258 = (int64_t)(uint8_t)t254;
                int64_t t259 = (int64_t)((uint64_t)t257 - (uint64_t)t258);
                int64_t t260 = t259 % 2;
                bool cmp261 = t260 != 0;
                bool cmp262 = t260 < 0;
                bool t263 = cmp261 & cmp262;
                int64_t sel264 = t263 ? 2 : 0;
                int64_t t265 = t260 + sel264;
                int64_t t266 = t265 * 9;
                int64_t t267 = t266 % 2;
                bool cmp268 = t267 != 0;
                bool cmp269 = t267 < 0;
                bool t270 = cmp268 & cmp269;
                int64_t t271 = t266 / 2;
                int64_t t272 = (int64_t)(uint8_t)t270;
                int64_t t273 = (int64_t)((uint64_t)t271 - (uint64_t)t272);
                int64_t t274 = (int64_t)((uint64_t)t273 * (uint64_t)256);
                int64_t t275 = t265 + 1;
                int64_t t276 = t275 * 9;
                int64_t t277 = t276 % 2;
                bool cmp278 = t277 != 0;
                bool cmp279 = t277 < 0;
                bool t280 = cmp278 & cmp279;
                int64_t t281 = t276 / 2;
                int64_t t282 = (int64_t)(uint8_t)t280;
                int64_t t283 = (int64_t)((uint64_t)t281 - (uint64_t)t282);
                int64_t t284 = (int64_t)((uint64_t)t283 * (uint64_t)256);
                int64_t t285 = t274 % 128;
                bool cmp286 = t285 != 0;
                bool cmp287 = t285 < 0;
                bool t288 = cmp286 & cmp287;
                int64_t t289 = t274 / 128;
                int64_t t290 = (int64_t)(uint8_t)t288;
                int64_t t291 = (int64_t)((uint64_t)t289 - (uint64_t)t290);
                int64_t t292 = t284 % 128;
                bool cmp293 = t292 != 0;
                bool cmp294 = t292 < 0;
                bool t295 = cmp293 & cmp294;
                int64_t t296 = t284 / 128;
                int64_t t297 = (int64_t)(uint8_t)t295;
                int64_t t298 = (int64_t)((uint64_t)t296 - (uint64_t)t297);
                for (int64_t i299 = t291; i299 < t298; i299 = (int64_t)((uint64_t)i299 + (uint64_t)1)) {
                    int64_t t300 = i217 * 46080;
                    int64_t t301 = t256 * 2304;
                    int64_t t302 = t300 + t301;
                    int64_t t303 = (int64_t)((uint64_t)i299 * (uint64_t)128);
                    int64_t t304 = (int64_t)((uint64_t)t302 + (uint64_t)t303);
                    int64_t t305 = (int64_t)((uint64_t)arg8 * (uint64_t)64);
                    int64_t t306 = (int64_t)((uint64_t)t304 + (uint64_t)t305);
                    int64_t t307 = t306 % 64;
                    bool cmp308 = t307 != 0;
                    bool cmp309 = t307 < 0;
                    bool t310 = cmp308 & cmp309;
                    int64_t t311 = t306 / 64;
                    int64_t t312 = (int64_t)(uint8_t)t310;
                    int64_t t313 = (int64_t)((uint64_t)t311 - (uint64_t)t312);
                    int64_t t314 = (int64_t)((uint64_t)t313 * (uint64_t)8192);
                    asc_lock(PIPE_MTE2, 0);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf48 + 0), (__gm__ uint8_t*)((__gm__ uint8_t*)gm5 + t314), 1, 8192, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    asc_lock(PIPE_V, 0);
                    asc_lock(PIPE_V, 1);
                    kernel_project_down_worker_vf0((__ubuf__ uint8_t*)buf48, (__ubuf__ fp8_e4m3fn_t*)buf49);
                    asc_unlock(PIPE_V, 1);
                    asc_unlock(PIPE_V, 0);
                    __ubuf__ fp8_e4m3fn_t* view_root315 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf49) + 0);
                    __cbuf__ fp8_e4m3fn_t* view_root316 = reinterpret_cast<__cbuf__ fp8_e4m3fn_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf50) + 0);
                    int64_t t317 = t305;
                    int64_t t318 = t317 * 32;
                    asc_sync_intra_wait(PIPE_MTE3, 0);
                    asc_lock(PIPE_MTE3, 1);
                    asc_copy_ub2l1(view_root316 + t318, (__ubuf__ int8_t*)view_root315 + 0, 8, 64, 0, 64);
                    asc_unlock(PIPE_MTE3, 1);
                    asc_sync_intra_arrive(PIPE_MTE3, 0);
                }
            }
            int32_t t319 = (int32_t)((uint32_t)t230 * (uint32_t)16);
            int64_t t320 = (int64_t)t319;
            int64_t t321 = (int64_t)((uint64_t)iter214 + (uint64_t)t320);
            int64_t t322 = (int64_t)ld222;
            int64_t t323 = (int64_t)((uint64_t)iter215 + (uint64_t)t322);
            int64_t t324 = (int64_t)((uint64_t)iter216 + (uint64_t)t249);
            iter214 = t321;
            iter215 = t323;
            iter216 = t324;
        }
        asc_sync_intra_wait(PIPE_MTE3, 0);
    }
}
