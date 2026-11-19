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

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_bfloat16_t_1d {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_bfloat16_t_2d_Device_ds_ds {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ds_ds(const KernelArgTensor_bfloat16_t_2d& src) {
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

struct KernelArgTensor_float_2d_Device_ds_ds {
    __gm__ float* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ds_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_bfloat16_t_1d_Device_s_s {
    __gm__ bfloat16_t* ptr;
};
static inline KernelArgTensor_bfloat16_t_1d_Device_s_s cannbotdsl_pack_KernelArgTensor_bfloat16_t_1d_Device_s_s(const KernelArgTensor_bfloat16_t_1d& src) {
    return {src.ptr};
}

__simd_vf__ inline void kernel_publish_c1a_vf0(__ubuf__ bfloat16_t* buf0, __ubuf__ bfloat16_t* buf1, __ubuf__ bfloat16_t* buf2)
{
    vector_bool vmask;
    float c3 = 1.0f;
    float c4 = 9.99999997E-7f;
    float c5 = 0.001953125f;
    float c6 = 0.0f;
    vector_bool mask7 = asc_create_mask_b32(PAT_ALL);
    vector_float v8;
    asc_duplicate_scalar(v8, c6, mask7);
    vector_float iter9 = v8;
    for (int32_t i10 = 0; i10 < 512; i10 = (int32_t)((uint32_t)i10 + (uint32_t)64)) {
        int32_t t11 = i10;
        vector_float v12;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v12, (__ubuf__ bfloat16_t*)(buf0 + t11));
        vector_float v13;
        asc_bfloat162float(v13, *(vector_bfloat16_t*)&v12, mask7);
        vector_float v14;
        asc_mul(v14, v13, v13, mask7);
        vector_float v15;
        asc_add(v15, iter9, v14, mask7);
        iter9 = v15;
    }
    vector_float v16;
    asc_reduce_sum(v16, iter9, mask7);
    vector_float v17;
    asc_mul_scalar(v17, v16, c5, mask7);
    vector_float v18;
    asc_add_scalar(v18, v17, c4, mask7);
    vector_float v19;
    asc_sqrt(v19, v18, mask7);
    vector_float v20;
    asc_duplicate_scalar(v20, c3, mask7);
    vector_float v21;
    asc_div(v21, v20, v19, mask7);
    vector_float v22;
    asc_duplicate(v22, v21, mask7);
    for (int32_t i23 = 0; i23 < 512; i23 = (int32_t)((uint32_t)i23 + (uint32_t)64)) {
        int32_t t24 = i23;
        vector_float v25;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v25, (__ubuf__ bfloat16_t*)(buf0 + t24));
        vector_float v26;
        asc_bfloat162float(v26, *(vector_bfloat16_t*)&v25, mask7);
        vector_float v27;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v27, (__ubuf__ bfloat16_t*)(buf1 + t24));
        vector_float v28;
        asc_bfloat162float(v28, *(vector_bfloat16_t*)&v27, mask7);
        vector_float v29;
        asc_mul(v29, v26, v22, mask7);
        vector_float v30;
        asc_mul(v30, v29, v28, mask7);
        vector_bfloat16_t v31;
        asc_float2bfloat16_rn(v31, v30, mask7, ASC_POSITION_EVEN);
        asc_storealign_pack((__ubuf__ uint32_t*)(buf2 + t24), *(vector_uint32_t*)&v31, mask7);
    }
}

__simd_vf__ inline void kernel_publish_c1a_vf1(__ubuf__ bfloat16_t* buf32, __ubuf__ float* buf33, __ubuf__ float* buf34, __ubuf__ float* buf35, __ubuf__ bfloat16_t* buf36, __ubuf__ float* buf37, __ubuf__ uint8_t* buf38)
{
    vector_bool vmask;
    float c39 = 5.0f;
    float c40 = 3.5f;
    float c41 = 2.5f;
    float c42 = 1.75f;
    float c43 = 1.25f;
    float c44 = 0.75f;
    float c45 = 0.25f;
    float c46 = 0.001953125f;
    float c47 = 0.166666672f;
    vector_bool mask48 = asc_create_mask_b32(PAT_ALL);
    uint32_t rem50 = 32;
    vector_bool mask49 = asc_update_mask_b32(rem50);
    uint32_t rem52 = 16;
    vector_bool mask51 = asc_update_mask_b32(rem52);
    for (int32_t i53 = 0; i53 < 512; i53 = (int32_t)((uint32_t)i53 + (uint32_t)64)) {
        int32_t t54 = i53;
        vector_float v55;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v55, (__ubuf__ bfloat16_t*)(buf32 + t54));
        vector_float v56;
        asc_bfloat162float(v56, *(vector_bfloat16_t*)&v55, mask48);
        asc_storealign(buf33 + t54, v56, mask48);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i57 = 0; i57 < 1; i57 = (int32_t)((uint32_t)i57 + (uint32_t)1)) {
        vector_float v58;
        vector_float v59;
        asc_loadalign_deintlv(v58, v59, (__ubuf__ float*)(buf33 + 448));
        vector_float v60;
        asc_loadalign(v60, buf34 + 0);
        vector_float v61;
        asc_loadalign(v61, buf35 + 0);
        vector_float v62;
        asc_mul(v62, v58, v60, mask49);
        vector_float v63;
        asc_mul(v63, v59, v61, mask49);
        vector_float v64;
        asc_mul(v64, v58, v61, mask49);
        vector_float v65;
        asc_mul(v65, v59, v60, mask49);
        vector_float v66;
        asc_sub(v66, v62, v63, mask49);
        vector_float v67;
        asc_add(v67, v64, v65, mask49);
        vector_bfloat16_t v68;
        asc_float2bfloat16_rn(v68, v66, mask49, ASC_POSITION_EVEN);
        vector_bfloat16_t v69;
        asc_float2bfloat16_rn(v69, v67, mask49, ASC_POSITION_EVEN);
        vector_float v70;
        asc_bfloat162float(v70, *(vector_bfloat16_t*)&v68, mask49);
        vector_float v71;
        asc_bfloat162float(v71, *(vector_bfloat16_t*)&v69, mask49);
        vector_uint32_t &r72 = (vector_uint32_t &)v70;
        vector_uint32_t &r73 = (vector_uint32_t &)v71;
        asc_storealign_intlv((__ubuf__ uint32_t*)(buf33 + 448), r72, r73);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i74 = 0; i74 < 32; i74 = (int32_t)((uint32_t)i74 + (uint32_t)1)) {
        int32_t t75 = i74 * 16;
        int32_t t76 = t75;
        vector_float v77;
        asc_loadalign(v77, buf33 + t76);
        vector_float v78;
        asc_abs(v78, v77, mask51);
        vector_float v79;
        asc_reduce_max(v79, v78, mask51);
        vector_float v80;
        asc_mul_scalar(v80, v79, c47, mask51);
        vector_float v81;
        asc_max_scalar(v81, v80, c46, mask51);
        vector_bfloat16_t v82;
        asc_float2bfloat16_rn(v82, v81, mask51, ASC_POSITION_EVEN);
        int32_t t83 = i74;
        asc_storealign_1st(buf36 + t83, v82);
        vector_float v84;
        asc_bfloat162float(v84, *(vector_bfloat16_t*)&v82, mask51);
        vector_float v85;
        asc_duplicate(v85, v84, mask51);
        vector_float v86;
        asc_div(v86, v77, v85, mask51);
        asc_storealign(buf37 + t76, v86, mask51);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i87 = 0; i87 < 512; i87 = (int32_t)((uint32_t)i87 + (uint32_t)64)) {
        int32_t t88 = i87;
        vector_float v89;
        asc_loadalign(v89, buf37 + t88);
        vector_float v90;
        asc_abs(v90, v89, mask48);
        vector_uint32_t v91;
        asc_duplicate_scalar(v91, 0, mask48);
        vector_bool m92;
        asc_gt_scalar(m92, v90, c45, mask48);
        vector_uint32_t v93;
        asc_duplicate_scalar(v93, 1, mask48);
        vector_uint32_t v94;
        asc_select(v94, v93, v91, m92);
        vector_bool m95;
        asc_ge_scalar(m95, v90, c44, mask48);
        vector_uint32_t v96;
        asc_duplicate_scalar(v96, 2, mask48);
        vector_uint32_t v97;
        asc_select(v97, v96, v94, m95);
        vector_bool m98;
        asc_gt_scalar(m98, v90, c43, mask48);
        vector_uint32_t v99;
        asc_duplicate_scalar(v99, 3, mask48);
        vector_uint32_t v100;
        asc_select(v100, v99, v97, m98);
        vector_bool m101;
        asc_ge_scalar(m101, v90, c42, mask48);
        vector_uint32_t v102;
        asc_duplicate_scalar(v102, 4, mask48);
        vector_uint32_t v103;
        asc_select(v103, v102, v100, m101);
        vector_bool m104;
        asc_gt_scalar(m104, v90, c41, mask48);
        vector_uint32_t v105;
        asc_duplicate_scalar(v105, 5, mask48);
        vector_uint32_t v106;
        asc_select(v106, v105, v103, m104);
        vector_bool m107;
        asc_ge_scalar(m107, v90, c40, mask48);
        vector_uint32_t v108;
        asc_duplicate_scalar(v108, 6, mask48);
        vector_uint32_t v109;
        asc_select(v109, v108, v106, m107);
        vector_bool m110;
        asc_gt_scalar(m110, v90, c39, mask48);
        vector_uint32_t v111;
        asc_duplicate_scalar(v111, 7, mask48);
        vector_uint32_t v112;
        asc_select(v112, v111, v109, m110);
        vector_uint32_t &r113 = (vector_uint32_t &)v89;
        vector_uint32_t v114;
        asc_shiftright_scalar(v114, r113, 28, mask48);
        vector_uint32_t v115;
        asc_duplicate_scalar(v115, 8, mask48);
        vector_uint32_t v116;
        asc_and(v116, v114, v115, mask48);
        vector_uint32_t v117;
        asc_or(v117, v112, v116, mask48);
        vector_uint32_t v118;
        vector_uint32_t v119;
        asc_deintlv(v118, v119, v117, v117);
        vector_uint32_t v120;
        asc_shiftleft_scalar(v120, v119, 4, mask49);
        vector_uint32_t v121;
        asc_or(v121, v118, v120, mask49);
        int32_t t122 = i87 / 2;
        int32_t t123 = t122;
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf38 + t123), *(vector_float*)&v121, mask49);
    }
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int64_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int64_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_float_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8 && alignof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_publish_c1a(KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm124_arg, KernelArgTensor_int64_t_1d_Device_d_s gm125_arg, KernelArgTensor_uint8_t_2d_Device_ds_ds gm126_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm127_arg, KernelArgTensor_float_2d_Device_ds_ds gm128_arg, KernelArgTensor_float_2d_Device_ds_ds gm129_arg, KernelArgTensor_bfloat16_t_1d_Device_s_s gm130_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm131_arg, int64_t arg132, int64_t arg133, int32_t arg134)
{
    __gm__ bfloat16_t* gm124 = gm124_arg.ptr;
    __gm__ int64_t* gm125 = gm125_arg.ptr;
    __gm__ uint8_t* gm126 = gm126_arg.ptr;
    __gm__ bfloat16_t* gm127 = gm127_arg.ptr;
    __gm__ float* gm128 = gm128_arg.ptr;
    __gm__ float* gm129 = gm129_arg.ptr;
    __gm__ bfloat16_t* gm130 = gm130_arg.ptr;
    __gm__ bfloat16_t* gm131 = gm131_arg.ptr;
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

    __ubuf__ bfloat16_t* buf135 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ bfloat16_t* buf136 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(1024);
    __ubuf__ bfloat16_t* buf137 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(2048);
    __ubuf__ float* buf138 = (__ubuf__ float*)(uintptr_t)(uint64_t)(3072);
    __ubuf__ float* buf139 = (__ubuf__ float*)(uintptr_t)(uint64_t)(5376);
    __ubuf__ float* buf140 = (__ubuf__ float*)(uintptr_t)(uint64_t)(7424);
    __ubuf__ float* buf141 = (__ubuf__ float*)(uintptr_t)(uint64_t)(7680);
    __ubuf__ uint8_t* buf142 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(7936);
    __ubuf__ bfloat16_t* buf143 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(8192);

    vector_bool vmask;
    {
        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf137 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm130 + 0), 1, 1024, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
    }
    for (int32_t i144 = 0; i144 < arg134; i144 = (int32_t)((uint32_t)i144 + (uint32_t)1)) {
        int64_t t145 = (int64_t)i144;
        int64_t t146 = i144;
        int64_t ld147 = gm125[t146];
        int32_t t148 = i144 % 8;
        bool cmp149 = t148 != 0;
        bool cmp150 = t148 < 0;
        bool t151 = cmp149 & cmp150;
        int32_t t152 = i144 / 8;
        int32_t t153 = (int32_t)(uint8_t)t151;
        int32_t t154 = (int32_t)((uint32_t)t152 - (uint32_t)t153);
        int64_t t155 = (int64_t)t154;
        int64_t t156 = t155 % arg133;
        bool cmp157 = t156 != 0;
        bool cmp158 = t156 < 0;
        bool cmp159 = arg133 < 0;
        bool t160 = cmp158 ^ cmp159;
        bool t161 = cmp157 & t160;
        int64_t sel162 = t161 ? arg133 : 0;
        int64_t t163 = (int64_t)((uint64_t)t156 + (uint64_t)sel162);
        bool cmp164 = ld147 >= 0;
        int64_t ifr165;
        if (cmp164) {
            int64_t t166 = ld147 / 128;
            int64_t t167 = t166 % arg133;
            bool cmp168 = t167 != 0;
            bool cmp169 = t167 < 0;
            bool t170 = cmp169 ^ cmp159;
            bool t171 = cmp168 & t170;
            int64_t sel172 = t171 ? arg133 : 0;
            int64_t t173 = (int64_t)((uint64_t)t167 + (uint64_t)sel172);
            ifr165 = t173;
        } else {
            ifr165 = t163;
        }
        bool cmp174 = ifr165 == arg132;
        if (cmp174) {
            int64_t dim175 = gm124_arg.shape[0];
            int64_t t176 = (t145 < dim175 ? t145 : dim175);
            int64_t t177 = (int64_t)((uint64_t)dim175 - (uint64_t)t176);
            int64_t t178 = (t177 < 1 ? t177 : 1);
            bool cmp179 = t145 <= dim175;
            int64_t sel180 = cmp179 ? t178 : 0;
            int64_t dim181 = gm124_arg.stride[0];
            int64_t t182 = (int64_t)((uint64_t)dim181 * (uint64_t)2);
            int64_t t183 = (int64_t)((uint64_t)t145 * (uint64_t)dim181);
            int64_t t184 = (int64_t)((uint64_t)t183 * (uint64_t)2);
            asc_lock(PIPE_MTE2, 0);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf135 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm124 + t184), sel180, 1024, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t182, 1024);
            }
            asc_unlock(PIPE_MTE2, 0);
            asc_lock(PIPE_V, 0);
            asc_lock(PIPE_V, 1);
            kernel_publish_c1a_vf0((__ubuf__ bfloat16_t*)buf135, (__ubuf__ bfloat16_t*)buf137, (__ubuf__ bfloat16_t*)buf136);
            asc_unlock(PIPE_V, 1);
            asc_unlock(PIPE_V, 0);
            int64_t dim185 = gm131_arg.stride[0];
            int64_t t186 = (int64_t)((uint64_t)t145 * (uint64_t)dim185);
            int64_t t187 = (int64_t)((uint64_t)t186 * (uint64_t)16);
            int64_t t188 = (int64_t)((uint64_t)t187 / (uint64_t)8);
            int64_t t189 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm131) + (uint64_t)t188);
            __gm__ bfloat16_t* root190 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t189);
            __ubuf__ bfloat16_t* view_root191 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf136) + 0);
            int64_t dim192 = gm131_arg.stride[0];
            int64_t t193 = dim192;
            int64_t t194 = t193 * 2;
            int64_t t195 = t194;
            asc_lock(PIPE_MTE3, 1);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root190 + 0, (__ubuf__ bfloat16_t*)view_root191 + 0, 1, 1024, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 1);
            bool ifr196;
            if (cmp164) {
                int64_t dim197 = gm126_arg.shape[0];
                bool cmp198 = ld147 < dim197;
                ifr196 = cmp198;
            } else {
                ifr196 = false;
            }
            if (ifr196) {
                int64_t dim199 = gm128_arg.shape[0];
                int64_t t200 = (t145 < dim199 ? t145 : dim199);
                int64_t t201 = (int64_t)((uint64_t)dim199 - (uint64_t)t200);
                int64_t t202 = (t201 < 1 ? t201 : 1);
                bool cmp203 = t145 <= dim199;
                int64_t sel204 = cmp203 ? t202 : 0;
                int64_t dim205 = gm128_arg.stride[0];
                int64_t t206 = (int64_t)((uint64_t)dim205 * (uint64_t)4);
                int64_t t207 = (int64_t)((uint64_t)t145 * (uint64_t)dim205);
                int64_t t208 = (int64_t)((uint64_t)t207 * (uint64_t)4);
                asc_lock(PIPE_MTE2, 3);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf140 + 0), (__gm__ float*)((__gm__ uint8_t*)gm128 + t208), sel204, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t206, 256);
                }
                asc_unlock(PIPE_MTE2, 3);
                int64_t dim209 = gm129_arg.shape[0];
                int64_t t210 = (t145 < dim209 ? t145 : dim209);
                int64_t t211 = (int64_t)((uint64_t)dim209 - (uint64_t)t210);
                int64_t t212 = (t211 < 1 ? t211 : 1);
                bool cmp213 = t145 <= dim209;
                int64_t sel214 = cmp213 ? t212 : 0;
                int64_t dim215 = gm129_arg.stride[0];
                int64_t t216 = (int64_t)((uint64_t)dim215 * (uint64_t)4);
                int64_t t217 = (int64_t)((uint64_t)t145 * (uint64_t)dim215);
                int64_t t218 = (int64_t)((uint64_t)t217 * (uint64_t)4);
                asc_lock(PIPE_MTE2, 4);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf141 + 0), (__gm__ float*)((__gm__ uint8_t*)gm129 + t218), sel214, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t216, 256);
                }
                asc_unlock(PIPE_MTE2, 4);
                asc_lock(PIPE_V, 3);
                asc_lock(PIPE_V, 4);
                asc_lock(PIPE_V, 6);
                asc_lock(PIPE_V, 5);
                kernel_publish_c1a_vf1((__ubuf__ bfloat16_t*)buf136, (__ubuf__ float*)buf138, (__ubuf__ float*)buf140, (__ubuf__ float*)buf141, (__ubuf__ bfloat16_t*)buf143, (__ubuf__ float*)buf139, (__ubuf__ uint8_t*)buf142);
                asc_unlock(PIPE_V, 5);
                asc_unlock(PIPE_V, 6);
                asc_unlock(PIPE_V, 4);
                asc_unlock(PIPE_V, 3);
                int64_t dim219 = gm126_arg.stride[0];
                int64_t t220 = (int64_t)((uint64_t)ld147 * (uint64_t)dim219);
                int64_t t221 = (int64_t)((uint64_t)t220 * (uint64_t)8);
                int64_t t222 = (int64_t)((uint64_t)t221 / (uint64_t)8);
                int64_t t223 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm126) + (uint64_t)t222);
                __gm__ uint8_t* root224 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t223);
                __ubuf__ uint8_t* view_root225 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf142) + 0);
                int64_t dim226 = gm126_arg.stride[0];
                asc_lock(PIPE_MTE3, 5);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root224 + 0, (__ubuf__ uint8_t*)view_root225 + 0, 1, 256, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 5);
                int64_t dim227 = gm127_arg.stride[0];
                int64_t t228 = (int64_t)((uint64_t)ld147 * (uint64_t)dim227);
                int64_t t229 = (int64_t)((uint64_t)t228 * (uint64_t)16);
                int64_t t230 = (int64_t)((uint64_t)t229 / (uint64_t)8);
                int64_t t231 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm127) + (uint64_t)t230);
                __gm__ bfloat16_t* root232 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t231);
                __ubuf__ bfloat16_t* view_root233 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf143) + 0);
                int64_t dim234 = gm127_arg.stride[0];
                int64_t t235 = dim234;
                int64_t t236 = t235 * 2;
                int64_t t237 = t236;
                asc_lock(PIPE_MTE3, 6);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root232 + 0, (__ubuf__ bfloat16_t*)view_root233 + 0, 1, 64, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 6);
            }
        }
    }
}
