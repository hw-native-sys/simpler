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

struct KernelArgTensor_float_2d_Device_dd_ds {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_dd_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_dd_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.shape[0], src.shape[1]}, {src.stride[0]}};
}

struct KernelArgTensor_int32_t_1d_Device_d_s {
    __gm__ int32_t* ptr;
    int64_t shape[1];
};
static inline KernelArgTensor_int32_t_1d_Device_d_s cannbotdsl_pack_KernelArgTensor_int32_t_1d_Device_d_s(const KernelArgTensor_int32_t_1d& src) {
    return {src.ptr, {src.shape[0]}};
}

struct KernelArgTensor_int32_t_2d_Device_dd_ds {
    __gm__ int32_t* ptr;
    int64_t shape[2];
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_dd_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_dd_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.shape[0], src.shape[1]}, {src.stride[0]}};
}

__simd_vf__ inline void kernel_generate_pool_vf0(int32_t arg0, __ubuf__ int32_t* buf1)
{
    vector_bool vmask;
    vector_bool mask2 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i3 = 0; i3 < 32; i3 = (int32_t)((uint32_t)i3 + (uint32_t)1)) {
        int32_t t4 = i3 * 64;
        vector_int32_t v5;
        asc_arange(v5, t4);
        vector_bool m6;
        asc_lt_scalar(m6, v5, arg0, mask2);
        vector_int32_t v7;
        asc_duplicate_scalar(v7, -1);
        vector_int32_t v8;
        asc_select(v8, v5, v7, m6);
        int32_t t9 = t4;
        asc_storealign(buf1 + t9, v8, mask2);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf1(__ubuf__ uint32_t* buf10)
{
    vector_bool vmask;
    vector_bool mask11 = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t v12;
    asc_duplicate_scalar(v12, 0, mask11);
    for (int32_t i13 = 0; i13 < 4; i13 = (int32_t)((uint32_t)i13 + (uint32_t)1)) {
        int32_t t14 = i13 * 64;
        int32_t t15 = t14;
        asc_storealign(buf10 + t15, v12, mask11);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf2(int32_t arg16, __ubuf__ float* buf17, int32_t arg18, int32_t arg19, __ubuf__ uint32_t* buf20, int32_t arg21)
{
    vector_bool vmask;
    float c22 = 0.0f;
    float c23 = 1.00000002E+30f;
    float c24 = -1.00000002E+30f;
    for (int32_t i25 = 0; i25 < arg21; i25 = (int32_t)((uint32_t)i25 + (uint32_t)1)) {
        int32_t t26 = (int32_t)((uint32_t)i25 * (uint32_t)64);
        int32_t t27 = (int32_t)((uint32_t)arg16 - (uint32_t)t26);
        uint32_t rem29 = t27;
        vector_bool mask28 = asc_update_mask_b32(rem29);
        vector_uint32_t v30;
        asc_arange(*(vector_int32_t*)&v30, static_cast<int32_t>(t26));
        vector_uint32_t v31;
        asc_shiftleft_scalar(v31, v30, 3, mask28);
        vector_float v32;
        asc_gather(v32, (__ubuf__ float*)(buf17 + 0), v31, mask28);
        vector_float iter33 = v32;
        for (int32_t i34 = 1; i34 < 8; i34 = (int32_t)((uint32_t)i34 + (uint32_t)1)) {
            vector_uint32_t v35;
            asc_add_scalar(v35, v31, i34, mask28);
            vector_float v36;
            asc_gather(v36, (__ubuf__ float*)(buf17 + 0), v35, mask28);
            vector_float v37;
            asc_max(v37, iter33, v36, mask28);
            iter33 = v37;
        }
        vector_float v38;
        asc_max_scalar(v38, iter33, c24, mask28);
        vector_uint32_t v39;
        asc_add_scalar(v39, v30, arg18, mask28);
        int32_t t40 = (int32_t)((uint32_t)arg19 - (uint32_t)1);
        vector_bool m41;
        asc_eq_scalar(m41, v39, t40, mask28);
        vector_float v42;
        asc_duplicate_scalar(v42, c23, mask28);
        vector_float v43;
        asc_select(v43, v42, v38, m41);
        vector_float v44;
        asc_add_scalar(v44, v43, c22, mask28);
        vector_uint32_t &r45 = (vector_uint32_t &)v44;
        vector_uint32_t v46;
        asc_duplicate_scalar(v46, -2147483648);
        vector_uint32_t v47;
        asc_xor(v47, r45, v46, mask28);
        vector_uint32_t v48;
        asc_duplicate_scalar(v48, -1);
        vector_uint32_t v49;
        asc_xor(v49, r45, v48, mask28);
        vector_int32_t &r50 = (vector_int32_t &)r45;
        vector_bool m51;
        asc_lt_scalar(m51, r50, 0, mask28);
        vector_uint32_t v52;
        asc_select(v52, v49, v47, m51);
        int32_t t53 = t26;
        asc_storealign(buf20 + t53, v52, mask28);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf3(int32_t arg54, __ubuf__ uint32_t* buf55, int32_t arg56, __ubuf__ uint16_t* buf57, __ubuf__ uint32_t* buf58)
{
    vector_bool vmask;
    uint32_t rem60 = 128;
    vector_bool mask59 = asc_update_mask_b16(rem60);
    vector_bool mask61 = asc_create_mask_b32(PAT_ALL);
    vector_uint16_t v62;
    asc_duplicate_scalar(v62, 0, mask59);
    vector_uint16_t iter63 = v62;
    vector_uint16_t iter64 = v62;
    for (int32_t i65 = 0; i65 < arg56; i65 = (int32_t)((uint32_t)i65 + (uint32_t)1)) {
        int32_t t66 = (int32_t)((uint32_t)i65 * (uint32_t)64);
        int32_t t67 = (int32_t)((uint32_t)arg54 - (uint32_t)t66);
        uint32_t rem69 = t67;
        vector_bool mask68 = asc_update_mask_b32(rem69);
        uint32_t rem71 = t67;
        vector_bool mask70 = asc_update_mask_b8(rem71);
        vector_uint8_t v72;
        asc_arange(*(vector_int8_t*)&v72, static_cast<int8_t>(0));
        vector_bool m73;
        asc_lt_scalar(m73, v72, 64, mask70);
        int32_t t74 = t66;
        vector_uint32_t v75;
        asc_loadalign(v75, buf55 + t74);
        vector_uint32_t v76;
        asc_duplicate_scalar(v76, 0);
        vector_uint32_t v77;
        asc_and(v77, v75, v76, mask68);
        vector_bool m78;
        asc_eq_scalar(m78, v77, 0, mask68);
        vector_uint32_t v79;
        asc_duplicate_scalar(v79, 1);
        vector_uint32_t v80;
        asc_select(v80, v79, v76, m78);
        vector_uint16_t v81;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v81, *(vector_uint32_t*)&v80);
        vector_uint8_t v82;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v82, *(vector_uint16_t*)&v81);
        vector_bool m83;
        asc_eq_scalar(m83, v82, 1, m73);
        vector_uint32_t v84;
        asc_shiftright_scalar(v84, v75, 24, mask68);
        vector_uint32_t v85;
        asc_duplicate_scalar(v85, 255);
        vector_uint32_t v86;
        asc_and(v86, v84, v85, mask68);
        vector_uint16_t v87;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v87, *(vector_uint32_t*)&v86);
        vector_uint8_t v88;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v88, *(vector_uint16_t*)&v87);
        vector_uint16_t v89 = iter63;
        asc_cumulative_histogram_bin0(v89, v88, m83);
        vector_uint16_t v90 = iter64;
        asc_cumulative_histogram_bin1(v90, v88, m83);
        iter63 = v89;
        iter64 = v90;
    }
    for (int32_t i91 = 0; i91 < 1; i91 = (int32_t)((uint32_t)i91 + (uint32_t)1)) {
        asc_storealign(buf57 + 0, iter63, mask59);
        asc_storealign(buf57 + 128, iter64, mask59);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i92 = 0; i92 < 4; i92 = (int32_t)((uint32_t)i92 + (uint32_t)1)) {
        int32_t t93 = i92 * 64;
        int32_t t94 = t93;
        vector_uint16_t v95;
        asc_loadalign(v95, buf57 + t94);
        vector_uint32_t v96;
        asc_unpack_lower(v96, v95);
        vector_uint32_t v97;
        asc_loadalign(v97, buf58 + t94);
        vector_uint32_t v98;
        asc_add(v98, v97, v96, mask61);
        asc_storealign(buf58 + t94, v98, mask61);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf4(__ubuf__ uint32_t* buf99)
{
    vector_bool vmask;
    vector_bool mask100 = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t v101;
    asc_duplicate_scalar(v101, 0, mask100);
    for (int32_t i102 = 0; i102 < 4; i102 = (int32_t)((uint32_t)i102 + (uint32_t)1)) {
        int32_t t103 = i102 * 64;
        int32_t t104 = t103;
        asc_storealign(buf99 + t104, v101, mask100);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf5(int32_t arg105, __ubuf__ float* buf106, int32_t arg107, int32_t arg108, __ubuf__ uint32_t* buf109, int32_t arg110)
{
    vector_bool vmask;
    float c111 = 0.0f;
    float c112 = 1.00000002E+30f;
    float c113 = -1.00000002E+30f;
    for (int32_t i114 = 0; i114 < arg110; i114 = (int32_t)((uint32_t)i114 + (uint32_t)1)) {
        int32_t t115 = (int32_t)((uint32_t)i114 * (uint32_t)64);
        int32_t t116 = (int32_t)((uint32_t)arg105 - (uint32_t)t115);
        uint32_t rem118 = t116;
        vector_bool mask117 = asc_update_mask_b32(rem118);
        vector_uint32_t v119;
        asc_arange(*(vector_int32_t*)&v119, static_cast<int32_t>(t115));
        vector_uint32_t v120;
        asc_shiftleft_scalar(v120, v119, 3, mask117);
        vector_float v121;
        asc_gather(v121, (__ubuf__ float*)(buf106 + 0), v120, mask117);
        vector_float iter122 = v121;
        for (int32_t i123 = 1; i123 < 8; i123 = (int32_t)((uint32_t)i123 + (uint32_t)1)) {
            vector_uint32_t v124;
            asc_add_scalar(v124, v120, i123, mask117);
            vector_float v125;
            asc_gather(v125, (__ubuf__ float*)(buf106 + 0), v124, mask117);
            vector_float v126;
            asc_max(v126, iter122, v125, mask117);
            iter122 = v126;
        }
        vector_float v127;
        asc_max_scalar(v127, iter122, c113, mask117);
        vector_uint32_t v128;
        asc_add_scalar(v128, v119, arg107, mask117);
        int32_t t129 = (int32_t)((uint32_t)arg108 - (uint32_t)1);
        vector_bool m130;
        asc_eq_scalar(m130, v128, t129, mask117);
        vector_float v131;
        asc_duplicate_scalar(v131, c112, mask117);
        vector_float v132;
        asc_select(v132, v131, v127, m130);
        vector_float v133;
        asc_add_scalar(v133, v132, c111, mask117);
        vector_uint32_t &r134 = (vector_uint32_t &)v133;
        vector_uint32_t v135;
        asc_duplicate_scalar(v135, -2147483648);
        vector_uint32_t v136;
        asc_xor(v136, r134, v135, mask117);
        vector_uint32_t v137;
        asc_duplicate_scalar(v137, -1);
        vector_uint32_t v138;
        asc_xor(v138, r134, v137, mask117);
        vector_int32_t &r139 = (vector_int32_t &)r134;
        vector_bool m140;
        asc_lt_scalar(m140, r139, 0, mask117);
        vector_uint32_t v141;
        asc_select(v141, v138, v136, m140);
        int32_t t142 = t115;
        asc_storealign(buf109 + t142, v141, mask117);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf6(int32_t arg143, __ubuf__ uint32_t* buf144, int32_t arg145, int32_t arg146, __ubuf__ uint16_t* buf147, __ubuf__ uint32_t* buf148)
{
    vector_bool vmask;
    uint32_t rem150 = 128;
    vector_bool mask149 = asc_update_mask_b16(rem150);
    vector_bool mask151 = asc_create_mask_b32(PAT_ALL);
    vector_uint16_t v152;
    asc_duplicate_scalar(v152, 0, mask149);
    vector_uint16_t iter153 = v152;
    vector_uint16_t iter154 = v152;
    for (int32_t i155 = 0; i155 < arg146; i155 = (int32_t)((uint32_t)i155 + (uint32_t)1)) {
        int32_t t156 = (int32_t)((uint32_t)i155 * (uint32_t)64);
        int32_t t157 = (int32_t)((uint32_t)arg143 - (uint32_t)t156);
        uint32_t rem159 = t157;
        vector_bool mask158 = asc_update_mask_b32(rem159);
        uint32_t rem161 = t157;
        vector_bool mask160 = asc_update_mask_b8(rem161);
        vector_uint8_t v162;
        asc_arange(*(vector_int8_t*)&v162, static_cast<int8_t>(0));
        vector_bool m163;
        asc_lt_scalar(m163, v162, 64, mask160);
        int32_t t164 = t156;
        vector_uint32_t v165;
        asc_loadalign(v165, buf144 + t164);
        vector_uint32_t v166;
        asc_duplicate_scalar(v166, -16777216);
        vector_uint32_t v167;
        asc_and(v167, v165, v166, mask158);
        vector_bool m168;
        asc_eq_scalar(m168, v167, arg145, mask158);
        vector_uint32_t v169;
        asc_duplicate_scalar(v169, 1);
        vector_uint32_t v170;
        asc_duplicate_scalar(v170, 0);
        vector_uint32_t v171;
        asc_select(v171, v169, v170, m168);
        vector_uint16_t v172;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v172, *(vector_uint32_t*)&v171);
        vector_uint8_t v173;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v173, *(vector_uint16_t*)&v172);
        vector_bool m174;
        asc_eq_scalar(m174, v173, 1, m163);
        vector_uint32_t v175;
        asc_shiftright_scalar(v175, v165, 16, mask158);
        vector_uint32_t v176;
        asc_duplicate_scalar(v176, 255);
        vector_uint32_t v177;
        asc_and(v177, v175, v176, mask158);
        vector_uint16_t v178;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v178, *(vector_uint32_t*)&v177);
        vector_uint8_t v179;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v179, *(vector_uint16_t*)&v178);
        vector_uint16_t v180 = iter153;
        asc_cumulative_histogram_bin0(v180, v179, m174);
        vector_uint16_t v181 = iter154;
        asc_cumulative_histogram_bin1(v181, v179, m174);
        iter153 = v180;
        iter154 = v181;
    }
    for (int32_t i182 = 0; i182 < 1; i182 = (int32_t)((uint32_t)i182 + (uint32_t)1)) {
        asc_storealign(buf147 + 0, iter153, mask149);
        asc_storealign(buf147 + 128, iter154, mask149);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i183 = 0; i183 < 4; i183 = (int32_t)((uint32_t)i183 + (uint32_t)1)) {
        int32_t t184 = i183 * 64;
        int32_t t185 = t184;
        vector_uint16_t v186;
        asc_loadalign(v186, buf147 + t185);
        vector_uint32_t v187;
        asc_unpack_lower(v187, v186);
        vector_uint32_t v188;
        asc_loadalign(v188, buf148 + t185);
        vector_uint32_t v189;
        asc_add(v189, v188, v187, mask151);
        asc_storealign(buf148 + t185, v189, mask151);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf7(__ubuf__ uint32_t* buf190)
{
    vector_bool vmask;
    vector_bool mask191 = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t v192;
    asc_duplicate_scalar(v192, 0, mask191);
    for (int32_t i193 = 0; i193 < 4; i193 = (int32_t)((uint32_t)i193 + (uint32_t)1)) {
        int32_t t194 = i193 * 64;
        int32_t t195 = t194;
        asc_storealign(buf190 + t195, v192, mask191);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf8(int32_t arg196, __ubuf__ float* buf197, int32_t arg198, int32_t arg199, __ubuf__ uint32_t* buf200, int32_t arg201)
{
    vector_bool vmask;
    float c202 = 0.0f;
    float c203 = 1.00000002E+30f;
    float c204 = -1.00000002E+30f;
    for (int32_t i205 = 0; i205 < arg201; i205 = (int32_t)((uint32_t)i205 + (uint32_t)1)) {
        int32_t t206 = (int32_t)((uint32_t)i205 * (uint32_t)64);
        int32_t t207 = (int32_t)((uint32_t)arg196 - (uint32_t)t206);
        uint32_t rem209 = t207;
        vector_bool mask208 = asc_update_mask_b32(rem209);
        vector_uint32_t v210;
        asc_arange(*(vector_int32_t*)&v210, static_cast<int32_t>(t206));
        vector_uint32_t v211;
        asc_shiftleft_scalar(v211, v210, 3, mask208);
        vector_float v212;
        asc_gather(v212, (__ubuf__ float*)(buf197 + 0), v211, mask208);
        vector_float iter213 = v212;
        for (int32_t i214 = 1; i214 < 8; i214 = (int32_t)((uint32_t)i214 + (uint32_t)1)) {
            vector_uint32_t v215;
            asc_add_scalar(v215, v211, i214, mask208);
            vector_float v216;
            asc_gather(v216, (__ubuf__ float*)(buf197 + 0), v215, mask208);
            vector_float v217;
            asc_max(v217, iter213, v216, mask208);
            iter213 = v217;
        }
        vector_float v218;
        asc_max_scalar(v218, iter213, c204, mask208);
        vector_uint32_t v219;
        asc_add_scalar(v219, v210, arg198, mask208);
        int32_t t220 = (int32_t)((uint32_t)arg199 - (uint32_t)1);
        vector_bool m221;
        asc_eq_scalar(m221, v219, t220, mask208);
        vector_float v222;
        asc_duplicate_scalar(v222, c203, mask208);
        vector_float v223;
        asc_select(v223, v222, v218, m221);
        vector_float v224;
        asc_add_scalar(v224, v223, c202, mask208);
        vector_uint32_t &r225 = (vector_uint32_t &)v224;
        vector_uint32_t v226;
        asc_duplicate_scalar(v226, -2147483648);
        vector_uint32_t v227;
        asc_xor(v227, r225, v226, mask208);
        vector_uint32_t v228;
        asc_duplicate_scalar(v228, -1);
        vector_uint32_t v229;
        asc_xor(v229, r225, v228, mask208);
        vector_int32_t &r230 = (vector_int32_t &)r225;
        vector_bool m231;
        asc_lt_scalar(m231, r230, 0, mask208);
        vector_uint32_t v232;
        asc_select(v232, v229, v227, m231);
        int32_t t233 = t206;
        asc_storealign(buf200 + t233, v232, mask208);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf9(int32_t arg234, __ubuf__ uint32_t* buf235, int32_t arg236, int32_t arg237, __ubuf__ uint16_t* buf238, __ubuf__ uint32_t* buf239)
{
    vector_bool vmask;
    uint32_t rem241 = 128;
    vector_bool mask240 = asc_update_mask_b16(rem241);
    vector_bool mask242 = asc_create_mask_b32(PAT_ALL);
    vector_uint16_t v243;
    asc_duplicate_scalar(v243, 0, mask240);
    vector_uint16_t iter244 = v243;
    vector_uint16_t iter245 = v243;
    for (int32_t i246 = 0; i246 < arg237; i246 = (int32_t)((uint32_t)i246 + (uint32_t)1)) {
        int32_t t247 = (int32_t)((uint32_t)i246 * (uint32_t)64);
        int32_t t248 = (int32_t)((uint32_t)arg234 - (uint32_t)t247);
        uint32_t rem250 = t248;
        vector_bool mask249 = asc_update_mask_b32(rem250);
        uint32_t rem252 = t248;
        vector_bool mask251 = asc_update_mask_b8(rem252);
        vector_uint8_t v253;
        asc_arange(*(vector_int8_t*)&v253, static_cast<int8_t>(0));
        vector_bool m254;
        asc_lt_scalar(m254, v253, 64, mask251);
        int32_t t255 = t247;
        vector_uint32_t v256;
        asc_loadalign(v256, buf235 + t255);
        vector_uint32_t v257;
        asc_duplicate_scalar(v257, -65536);
        vector_uint32_t v258;
        asc_and(v258, v256, v257, mask249);
        vector_bool m259;
        asc_eq_scalar(m259, v258, arg236, mask249);
        vector_uint32_t v260;
        asc_duplicate_scalar(v260, 1);
        vector_uint32_t v261;
        asc_duplicate_scalar(v261, 0);
        vector_uint32_t v262;
        asc_select(v262, v260, v261, m259);
        vector_uint16_t v263;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v263, *(vector_uint32_t*)&v262);
        vector_uint8_t v264;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v264, *(vector_uint16_t*)&v263);
        vector_bool m265;
        asc_eq_scalar(m265, v264, 1, m254);
        vector_uint32_t v266;
        asc_shiftright_scalar(v266, v256, 8, mask249);
        vector_uint32_t v267;
        asc_duplicate_scalar(v267, 255);
        vector_uint32_t v268;
        asc_and(v268, v266, v267, mask249);
        vector_uint16_t v269;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v269, *(vector_uint32_t*)&v268);
        vector_uint8_t v270;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v270, *(vector_uint16_t*)&v269);
        vector_uint16_t v271 = iter244;
        asc_cumulative_histogram_bin0(v271, v270, m265);
        vector_uint16_t v272 = iter245;
        asc_cumulative_histogram_bin1(v272, v270, m265);
        iter244 = v271;
        iter245 = v272;
    }
    for (int32_t i273 = 0; i273 < 1; i273 = (int32_t)((uint32_t)i273 + (uint32_t)1)) {
        asc_storealign(buf238 + 0, iter244, mask240);
        asc_storealign(buf238 + 128, iter245, mask240);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i274 = 0; i274 < 4; i274 = (int32_t)((uint32_t)i274 + (uint32_t)1)) {
        int32_t t275 = i274 * 64;
        int32_t t276 = t275;
        vector_uint16_t v277;
        asc_loadalign(v277, buf238 + t276);
        vector_uint32_t v278;
        asc_unpack_lower(v278, v277);
        vector_uint32_t v279;
        asc_loadalign(v279, buf239 + t276);
        vector_uint32_t v280;
        asc_add(v280, v279, v278, mask242);
        asc_storealign(buf239 + t276, v280, mask242);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf10(__ubuf__ uint32_t* buf281)
{
    vector_bool vmask;
    vector_bool mask282 = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t v283;
    asc_duplicate_scalar(v283, 0, mask282);
    for (int32_t i284 = 0; i284 < 4; i284 = (int32_t)((uint32_t)i284 + (uint32_t)1)) {
        int32_t t285 = i284 * 64;
        int32_t t286 = t285;
        asc_storealign(buf281 + t286, v283, mask282);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf11(int32_t arg287, __ubuf__ float* buf288, int32_t arg289, int32_t arg290, __ubuf__ uint32_t* buf291, int32_t arg292)
{
    vector_bool vmask;
    float c293 = 0.0f;
    float c294 = 1.00000002E+30f;
    float c295 = -1.00000002E+30f;
    for (int32_t i296 = 0; i296 < arg292; i296 = (int32_t)((uint32_t)i296 + (uint32_t)1)) {
        int32_t t297 = (int32_t)((uint32_t)i296 * (uint32_t)64);
        int32_t t298 = (int32_t)((uint32_t)arg287 - (uint32_t)t297);
        uint32_t rem300 = t298;
        vector_bool mask299 = asc_update_mask_b32(rem300);
        vector_uint32_t v301;
        asc_arange(*(vector_int32_t*)&v301, static_cast<int32_t>(t297));
        vector_uint32_t v302;
        asc_shiftleft_scalar(v302, v301, 3, mask299);
        vector_float v303;
        asc_gather(v303, (__ubuf__ float*)(buf288 + 0), v302, mask299);
        vector_float iter304 = v303;
        for (int32_t i305 = 1; i305 < 8; i305 = (int32_t)((uint32_t)i305 + (uint32_t)1)) {
            vector_uint32_t v306;
            asc_add_scalar(v306, v302, i305, mask299);
            vector_float v307;
            asc_gather(v307, (__ubuf__ float*)(buf288 + 0), v306, mask299);
            vector_float v308;
            asc_max(v308, iter304, v307, mask299);
            iter304 = v308;
        }
        vector_float v309;
        asc_max_scalar(v309, iter304, c295, mask299);
        vector_uint32_t v310;
        asc_add_scalar(v310, v301, arg289, mask299);
        int32_t t311 = (int32_t)((uint32_t)arg290 - (uint32_t)1);
        vector_bool m312;
        asc_eq_scalar(m312, v310, t311, mask299);
        vector_float v313;
        asc_duplicate_scalar(v313, c294, mask299);
        vector_float v314;
        asc_select(v314, v313, v309, m312);
        vector_float v315;
        asc_add_scalar(v315, v314, c293, mask299);
        vector_uint32_t &r316 = (vector_uint32_t &)v315;
        vector_uint32_t v317;
        asc_duplicate_scalar(v317, -2147483648);
        vector_uint32_t v318;
        asc_xor(v318, r316, v317, mask299);
        vector_uint32_t v319;
        asc_duplicate_scalar(v319, -1);
        vector_uint32_t v320;
        asc_xor(v320, r316, v319, mask299);
        vector_int32_t &r321 = (vector_int32_t &)r316;
        vector_bool m322;
        asc_lt_scalar(m322, r321, 0, mask299);
        vector_uint32_t v323;
        asc_select(v323, v320, v318, m322);
        int32_t t324 = t297;
        asc_storealign(buf291 + t324, v323, mask299);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf12(int32_t arg325, __ubuf__ uint32_t* buf326, int32_t arg327, int32_t arg328, __ubuf__ uint16_t* buf329, __ubuf__ uint32_t* buf330)
{
    vector_bool vmask;
    uint32_t rem332 = 128;
    vector_bool mask331 = asc_update_mask_b16(rem332);
    vector_bool mask333 = asc_create_mask_b32(PAT_ALL);
    vector_uint16_t v334;
    asc_duplicate_scalar(v334, 0, mask331);
    vector_uint16_t iter335 = v334;
    vector_uint16_t iter336 = v334;
    for (int32_t i337 = 0; i337 < arg328; i337 = (int32_t)((uint32_t)i337 + (uint32_t)1)) {
        int32_t t338 = (int32_t)((uint32_t)i337 * (uint32_t)64);
        int32_t t339 = (int32_t)((uint32_t)arg325 - (uint32_t)t338);
        uint32_t rem341 = t339;
        vector_bool mask340 = asc_update_mask_b32(rem341);
        uint32_t rem343 = t339;
        vector_bool mask342 = asc_update_mask_b8(rem343);
        vector_uint8_t v344;
        asc_arange(*(vector_int8_t*)&v344, static_cast<int8_t>(0));
        vector_bool m345;
        asc_lt_scalar(m345, v344, 64, mask342);
        int32_t t346 = t338;
        vector_uint32_t v347;
        asc_loadalign(v347, buf326 + t346);
        vector_uint32_t v348;
        asc_duplicate_scalar(v348, -256);
        vector_uint32_t v349;
        asc_and(v349, v347, v348, mask340);
        vector_bool m350;
        asc_eq_scalar(m350, v349, arg327, mask340);
        vector_uint32_t v351;
        asc_duplicate_scalar(v351, 1);
        vector_uint32_t v352;
        asc_duplicate_scalar(v352, 0);
        vector_uint32_t v353;
        asc_select(v353, v351, v352, m350);
        vector_uint16_t v354;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v354, *(vector_uint32_t*)&v353);
        vector_uint8_t v355;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v355, *(vector_uint16_t*)&v354);
        vector_bool m356;
        asc_eq_scalar(m356, v355, 1, m345);
        vector_uint32_t v357;
        asc_shiftright_scalar(v357, v347, 0, mask340);
        vector_uint32_t v358;
        asc_duplicate_scalar(v358, 255);
        vector_uint32_t v359;
        asc_and(v359, v357, v358, mask340);
        vector_uint16_t v360;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint16_t*)&v360, *(vector_uint32_t*)&v359);
        vector_uint8_t v361;
        cannbotdsl_92::cannbotdsl_pack_to_low(*(vector_uint8_t*)&v361, *(vector_uint16_t*)&v360);
        vector_uint16_t v362 = iter335;
        asc_cumulative_histogram_bin0(v362, v361, m356);
        vector_uint16_t v363 = iter336;
        asc_cumulative_histogram_bin1(v363, v361, m356);
        iter335 = v362;
        iter336 = v363;
    }
    for (int32_t i364 = 0; i364 < 1; i364 = (int32_t)((uint32_t)i364 + (uint32_t)1)) {
        asc_storealign(buf329 + 0, iter335, mask331);
        asc_storealign(buf329 + 128, iter336, mask331);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i365 = 0; i365 < 4; i365 = (int32_t)((uint32_t)i365 + (uint32_t)1)) {
        int32_t t366 = i365 * 64;
        int32_t t367 = t366;
        vector_uint16_t v368;
        asc_loadalign(v368, buf329 + t367);
        vector_uint32_t v369;
        asc_unpack_lower(v369, v368);
        vector_uint32_t v370;
        asc_loadalign(v370, buf330 + t367);
        vector_uint32_t v371;
        asc_add(v371, v370, v369, mask333);
        asc_storealign(buf330 + t367, v371, mask333);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf13(int32_t arg372, __ubuf__ float* buf373, int32_t arg374, __ubuf__ uint32_t* buf375, int32_t arg376)
{
    vector_bool vmask;
    float c377 = 0.0f;
    float c378 = 1.00000002E+30f;
    float c379 = -1.00000002E+30f;
    for (int32_t i380 = 0; i380 < arg376; i380 = (int32_t)((uint32_t)i380 + (uint32_t)1)) {
        int32_t t381 = (int32_t)((uint32_t)i380 * (uint32_t)64);
        int32_t t382 = (int32_t)((uint32_t)arg372 - (uint32_t)t381);
        uint32_t rem384 = t382;
        vector_bool mask383 = asc_update_mask_b32(rem384);
        vector_uint32_t v385;
        asc_arange(*(vector_int32_t*)&v385, static_cast<int32_t>(t381));
        vector_uint32_t v386;
        asc_shiftleft_scalar(v386, v385, 3, mask383);
        vector_float v387;
        asc_gather(v387, (__ubuf__ float*)(buf373 + 0), v386, mask383);
        vector_float iter388 = v387;
        for (int32_t i389 = 1; i389 < 8; i389 = (int32_t)((uint32_t)i389 + (uint32_t)1)) {
            vector_uint32_t v390;
            asc_add_scalar(v390, v386, i389, mask383);
            vector_float v391;
            asc_gather(v391, (__ubuf__ float*)(buf373 + 0), v390, mask383);
            vector_float v392;
            asc_max(v392, iter388, v391, mask383);
            iter388 = v392;
        }
        vector_float v393;
        asc_max_scalar(v393, iter388, c379, mask383);
        vector_uint32_t v394;
        asc_add_scalar(v394, v385, 0, mask383);
        int32_t t395 = (int32_t)((uint32_t)arg374 - (uint32_t)1);
        vector_bool m396;
        asc_eq_scalar(m396, v394, t395, mask383);
        vector_float v397;
        asc_duplicate_scalar(v397, c378, mask383);
        vector_float v398;
        asc_select(v398, v397, v393, m396);
        vector_float v399;
        asc_add_scalar(v399, v398, c377, mask383);
        vector_uint32_t &r400 = (vector_uint32_t &)v399;
        vector_uint32_t v401;
        asc_duplicate_scalar(v401, -2147483648);
        vector_uint32_t v402;
        asc_xor(v402, r400, v401, mask383);
        vector_uint32_t v403;
        asc_duplicate_scalar(v403, -1);
        vector_uint32_t v404;
        asc_xor(v404, r400, v403, mask383);
        vector_int32_t &r405 = (vector_int32_t &)r400;
        vector_bool m406;
        asc_lt_scalar(m406, r405, 0, mask383);
        vector_uint32_t v407;
        asc_select(v407, v404, v402, m406);
        int32_t t408 = t381;
        asc_storealign(buf375 + t408, v407, mask383);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf14(int32_t arg409, __ubuf__ int32_t* buf410, int32_t arg411, __ubuf__ uint32_t* buf412, int32_t arg413, int32_t arg414, __ubuf__ int32_t* buf415)
{
    vector_bool vmask;
    vector_store_unalign _sua_buf410;
    vector_bool mask416 = asc_create_mask_b32(PAT_ALL);
    vector_int32_t v417;
    asc_duplicate_scalar(v417, arg409, mask416);
    cannbotdsl_92::cannbotdsl_squeeze_store_init();
    vector_int32_t iter418 = v417;
    for (int32_t i419 = 0; i419 < arg414; i419 = (int32_t)((uint32_t)i419 + (uint32_t)1)) {
        int32_t t420 = (int32_t)((uint32_t)i419 * (uint32_t)64);
        int32_t t421 = (int32_t)((uint32_t)arg411 - (uint32_t)t420);
        uint32_t rem423 = t421;
        vector_bool mask422 = asc_update_mask_b32(rem423);
        int32_t t424 = t420;
        vector_uint32_t v425;
        asc_loadalign(v425, buf412 + t424);
        vector_bool m426;
        asc_eq_scalar(m426, v425, arg413, mask422);
        vector_int32_t v427;
        asc_unsqueeze(v427, m426);
        vector_bool m428;
        asc_lt(m428, v427, iter418, mask422);
        vector_bool m429;
        asc_and(m429, m426, m428, mask422);
        vector_bool m430;
        asc_gt_scalar(m430, v425, arg413, mask422);
        vector_bool m431;
        asc_or(m431, m430, m429, mask422);
        vector_int32_t v432;
        asc_arange(v432, t420);
        vector_int32_t v433;
        cannbotdsl_92::cannbotdsl_squeeze_store(buf410 + 0, v433, _sua_buf410, v432, m431);
        vector_int32_t v434;
        asc_duplicate_scalar(v434, 1);
        vector_int32_t v435;
        asc_duplicate_scalar(v435, 0);
        vector_int32_t v436;
        asc_select(v436, v434, v435, m426);
        vector_int32_t v437;
        asc_reduce_sum(v437, v436, mask422);
        vector_int32_t v438;
        asc_duplicate(v438, v437, mask416);
        vector_int32_t v439;
        asc_sub(v439, iter418, v438, mask416);
        iter418 = v439;
    }
    for (int32_t i440 = 0; i440 < 1; i440 = (int32_t)((uint32_t)i440 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_finalize(buf410 + 0, _sua_buf410);
        asc_storealign_1st(buf415 + 0, iter418);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_generate_pool_vf15(int32_t arg441, __ubuf__ float* buf442, int32_t arg443, int32_t arg444, __ubuf__ uint32_t* buf445, int32_t arg446)
{
    vector_bool vmask;
    float c447 = 0.0f;
    float c448 = 1.00000002E+30f;
    float c449 = -1.00000002E+30f;
    for (int32_t i450 = 0; i450 < arg446; i450 = (int32_t)((uint32_t)i450 + (uint32_t)1)) {
        int32_t t451 = (int32_t)((uint32_t)i450 * (uint32_t)64);
        int32_t t452 = (int32_t)((uint32_t)arg441 - (uint32_t)t451);
        uint32_t rem454 = t452;
        vector_bool mask453 = asc_update_mask_b32(rem454);
        vector_uint32_t v455;
        asc_arange(*(vector_int32_t*)&v455, static_cast<int32_t>(t451));
        vector_uint32_t v456;
        asc_shiftleft_scalar(v456, v455, 3, mask453);
        vector_float v457;
        asc_gather(v457, (__ubuf__ float*)(buf442 + 0), v456, mask453);
        vector_float iter458 = v457;
        for (int32_t i459 = 1; i459 < 8; i459 = (int32_t)((uint32_t)i459 + (uint32_t)1)) {
            vector_uint32_t v460;
            asc_add_scalar(v460, v456, i459, mask453);
            vector_float v461;
            asc_gather(v461, (__ubuf__ float*)(buf442 + 0), v460, mask453);
            vector_float v462;
            asc_max(v462, iter458, v461, mask453);
            iter458 = v462;
        }
        vector_float v463;
        asc_max_scalar(v463, iter458, c449, mask453);
        vector_uint32_t v464;
        asc_add_scalar(v464, v455, arg443, mask453);
        int32_t t465 = (int32_t)((uint32_t)arg444 - (uint32_t)1);
        vector_bool m466;
        asc_eq_scalar(m466, v464, t465, mask453);
        vector_float v467;
        asc_duplicate_scalar(v467, c448, mask453);
        vector_float v468;
        asc_select(v468, v467, v463, m466);
        vector_float v469;
        asc_add_scalar(v469, v468, c447, mask453);
        vector_uint32_t &r470 = (vector_uint32_t &)v469;
        vector_uint32_t v471;
        asc_duplicate_scalar(v471, -2147483648);
        vector_uint32_t v472;
        asc_xor(v472, r470, v471, mask453);
        vector_uint32_t v473;
        asc_duplicate_scalar(v473, -1);
        vector_uint32_t v474;
        asc_xor(v474, r470, v473, mask453);
        vector_int32_t &r475 = (vector_int32_t &)r470;
        vector_bool m476;
        asc_lt_scalar(m476, r475, 0, mask453);
        vector_uint32_t v477;
        asc_select(v477, v474, v472, m476);
        int32_t t478 = t451;
        asc_storealign(buf445 + t478, v477, mask453);
    }
}

__simd_vf__ inline void kernel_generate_pool_vf16(int32_t arg479, __ubuf__ int32_t* buf480, int32_t arg481, __ubuf__ uint32_t* buf482, int32_t arg483, int32_t arg484, int32_t arg485, __ubuf__ int32_t* buf486)
{
    vector_bool vmask;
    vector_store_unalign _sua_buf480;
    vector_bool mask487 = asc_create_mask_b32(PAT_ALL);
    vector_int32_t v488;
    asc_duplicate_scalar(v488, arg479, mask487);
    vector_int32_t iter489 = v488;
    for (int32_t i490 = 0; i490 < arg485; i490 = (int32_t)((uint32_t)i490 + (uint32_t)1)) {
        int32_t t491 = (int32_t)((uint32_t)i490 * (uint32_t)64);
        int32_t t492 = (int32_t)((uint32_t)arg481 - (uint32_t)t491);
        uint32_t rem494 = t492;
        vector_bool mask493 = asc_update_mask_b32(rem494);
        int32_t t495 = t491;
        vector_uint32_t v496;
        asc_loadalign(v496, buf482 + t495);
        vector_bool m497;
        asc_eq_scalar(m497, v496, arg483, mask493);
        vector_int32_t v498;
        asc_unsqueeze(v498, m497);
        vector_bool m499;
        asc_lt(m499, v498, iter489, mask493);
        vector_bool m500;
        asc_and(m500, m497, m499, mask493);
        vector_bool m501;
        asc_gt_scalar(m501, v496, arg483, mask493);
        vector_bool m502;
        asc_or(m502, m501, m500, mask493);
        int32_t t503 = (int32_t)((uint32_t)arg484 + (uint32_t)t491);
        vector_int32_t v504;
        asc_arange(v504, t503);
        vector_int32_t v505;
        cannbotdsl_92::cannbotdsl_squeeze_store(buf480 + 0, v505, _sua_buf480, v504, m502);
        vector_int32_t v506;
        asc_duplicate_scalar(v506, 1);
        vector_int32_t v507;
        asc_duplicate_scalar(v507, 0);
        vector_int32_t v508;
        asc_select(v508, v506, v507, m497);
        vector_int32_t v509;
        asc_reduce_sum(v509, v508, mask493);
        vector_int32_t v510;
        asc_duplicate(v510, v509, mask487);
        vector_int32_t v511;
        asc_sub(v511, iter489, v510, mask487);
        iter489 = v511;
    }
    for (int32_t i512 = 0; i512 < 1; i512 = (int32_t)((uint32_t)i512 + (uint32_t)1)) {
        cannbotdsl_92::cannbotdsl_squeeze_store_finalize(buf480 + 0, _sua_buf480);
        asc_storealign_1st(buf486 + 0, iter489);
    }
    asc_mem_bar(VST_VLD);
}

static_assert(sizeof(KernelArgTensor_float_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_float_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_generate_pool(KernelArgTensor_float_2d_Device_dd_ds gm513_arg, KernelArgTensor_int32_t_1d_Device_d_s gm514_arg, KernelArgTensor_int32_t_2d_Device_dd_ds gm515_arg, KernelArgTensor_int32_t_1d_Device_d_s gm516_arg, int64_t arg517, int64_t arg518, int32_t arg519)
{
    __gm__ float* gm513 = gm513_arg.ptr;
    __gm__ int32_t* gm514 = gm514_arg.ptr;
    __gm__ int32_t* gm515 = gm515_arg.ptr;
    __gm__ int32_t* gm516 = gm516_arg.ptr;
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

    __ubuf__ float* buf520 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ uint32_t* buf521 = (__ubuf__ uint32_t*)(uintptr_t)(uint64_t)(32768);
    __ubuf__ uint32_t* buf522 = (__ubuf__ uint32_t*)(uintptr_t)(uint64_t)(36864);
    __ubuf__ uint16_t* buf523 = (__ubuf__ uint16_t*)(uintptr_t)(uint64_t)(37888);
    __ubuf__ int32_t* buf524 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(38656);
    __ubuf__ int32_t* buf525 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(46848);

    vector_bool vmask;
    int64_t dim526 = gm515_arg.shape[0];
    for (int64_t i527 = arg517; i527 < dim526; i527 = (int64_t)((uint64_t)i527 + (uint64_t)arg518)) {
        int64_t dim528 = gm513_arg.shape[1];
        int64_t t529 = (int64_t)arg519;
        bool cmp530 = i527 < t529;
        int32_t ifr531;
        if (cmp530) {
            int64_t t532 = i527;
            int32_t ld533 = gm514[t532];
            bool cmp534 = ld533 < 0;
            int32_t sel535 = cmp534 ? 0 : ld533;
            int64_t t536 = (int64_t)sel535;
            bool cmp537 = t536 > dim528;
            int32_t t538 = (int32_t)dim528;
            int32_t sel539 = cmp537 ? t538 : sel535;
            ifr531 = sel539;
        } else {
            ifr531 = 0;
        }
        int32_t t540 = (int32_t)((uint32_t)ifr531 + (uint32_t)7);
        int32_t t541 = t540 % 8;
        bool cmp542 = t541 != 0;
        bool cmp543 = t541 < 0;
        bool t544 = cmp542 & cmp543;
        int32_t t545 = t540 / 8;
        int32_t t546 = (int32_t)(uint8_t)t544;
        int32_t t547 = (int32_t)((uint32_t)t545 - (uint32_t)t546);
        int64_t dim548 = gm515_arg.shape[1];
        int64_t t549 = (int64_t)t547;
        bool cmp550 = t549 > dim548;
        int32_t t551 = (int32_t)dim548;
        int32_t sel552 = cmp550 ? t551 : t547;
        bool cmp553 = sel552 > 2048;
        int32_t sel554 = cmp553 ? 2048 : sel552;
        asc_lock(PIPE_V, 2);
        kernel_generate_pool_vf0(sel554, (__ubuf__ int32_t*)buf524);
        asc_unlock(PIPE_V, 2);
        bool cmp555 = t547 > sel554;
        if (cmp555) {
            int32_t t556 = (int32_t)((uint32_t)t547 - (uint32_t)sel554);
            int32_t t557 = (int32_t)((uint32_t)t556 + (uint32_t)1);
            kernel_generate_pool_vf1((__ubuf__ uint32_t*)buf522);
            int32_t t558 = (int32_t)((uint32_t)t547 + (uint32_t)1023);
            int32_t t559 = t558 % 1024;
            bool cmp560 = t559 != 0;
            bool cmp561 = t559 < 0;
            bool t562 = cmp560 & cmp561;
            int32_t t563 = t558 / 1024;
            int32_t t564 = (int32_t)(uint8_t)t562;
            int32_t t565 = (int32_t)((uint32_t)t563 - (uint32_t)t564);
            for (int32_t i566 = 0; i566 < t565; i566 = (int32_t)((uint32_t)i566 + (uint32_t)1)) {
                int64_t t567 = (int64_t)i566;
                int64_t dim568 = gm513_arg.shape[0];
                int64_t t569 = (i527 < dim568 ? i527 : dim568);
                int64_t t570 = (int64_t)((uint64_t)dim568 - (uint64_t)t569);
                int64_t t571 = (t570 < 1 ? t570 : 1);
                bool cmp572 = i527 <= dim568;
                int64_t sel573 = cmp572 ? t571 : 0;
                int64_t dim574 = gm513_arg.stride[0];
                int64_t t575 = (int64_t)((uint64_t)dim574 * (uint64_t)4);
                int64_t t576 = dim528 / 8192;
                int64_t t577 = (t567 < t576 ? t567 : t576);
                int64_t t578 = (int64_t)((uint64_t)t577 * (uint64_t)8192);
                int64_t t579 = (int64_t)((uint64_t)dim528 - (uint64_t)t578);
                int64_t t580 = (t579 < 8192 ? t579 : 8192);
                bool cmp581 = t567 <= t576;
                int64_t sel582 = cmp581 ? t580 : 0;
                int64_t t583 = (int64_t)((uint64_t)i527 * (uint64_t)dim574);
                int64_t t584 = (int64_t)((uint64_t)t583 * (uint64_t)4);
                int64_t t585 = (int64_t)((uint64_t)t567 * (uint64_t)32768);
                int64_t t586 = (int64_t)((uint64_t)t584 + (uint64_t)t585);
                int64_t t587 = (int64_t)((uint64_t)sel582 * (uint64_t)4);
                bool cmp588 = t587 == 0;
                int64_t sel589 = cmp588 ? 0 : sel573;
                int64_t sel590 = cmp588 ? 4 : t587;
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf520 + 0), (__gm__ float*)((__gm__ uint8_t*)gm513 + t586), sel589, sel590, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t575, 32768);
                }
                asc_unlock(PIPE_MTE2, 0);
                int32_t t591 = (int32_t)((uint32_t)i566 * (uint32_t)1024);
                int32_t t592 = (int32_t)((uint32_t)t547 - (uint32_t)t591);
                bool cmp593 = t592 > 1024;
                int32_t sel594 = cmp593 ? 1024 : t592;
                int32_t t595 = (int32_t)((uint32_t)sel594 + (uint32_t)63);
                int32_t t596 = t595 % 64;
                bool cmp597 = t596 != 0;
                bool cmp598 = t596 < 0;
                bool t599 = cmp597 & cmp598;
                int32_t t600 = t595 / 64;
                int32_t t601 = (int32_t)(uint8_t)t599;
                int32_t t602 = (int32_t)((uint32_t)t600 - (uint32_t)t601);
                asc_lock(PIPE_V, 0);
                kernel_generate_pool_vf2(sel594, (__ubuf__ float*)buf520, t591, t547, (__ubuf__ uint32_t*)buf521, t602);
                asc_unlock(PIPE_V, 0);
                kernel_generate_pool_vf3(sel594, (__ubuf__ uint32_t*)buf521, t602, (__ubuf__ uint16_t*)buf523, (__ubuf__ uint32_t*)buf522);
            }
            asc_sync();
            int32_t iter603 = 255;
            int32_t iter604 = 0;
            for (int64_t i605 = 0; i605 < 8; i605 = (int64_t)((uint64_t)i605 + (uint64_t)1)) {
                int32_t t606 = (int32_t)((uint32_t)iter604 + (uint32_t)iter603);
                int32_t t607 = t606 % 2;
                bool cmp608 = t607 != 0;
                bool cmp609 = t607 < 0;
                bool t610 = cmp608 & cmp609;
                int32_t t611 = t606 / 2;
                int32_t t612 = (int32_t)(uint8_t)t610;
                int32_t t613 = (int32_t)((uint32_t)t611 - (uint32_t)t612);
                asc_lock(PIPE_S, 1);
                int64_t t614 = t613;
                uint32_t ld615 = buf522[t614];
                asc_unlock(PIPE_S, 1);
                int32_t scalar616 = (int32_t)ld615;
                bool cmp617 = scalar616 >= t557;
                int32_t sel618 = cmp617 ? t613 : iter603;
                int32_t t619 = (int32_t)((uint32_t)t613 + (uint32_t)1);
                int32_t sel620 = cmp617 ? iter604 : t619;
                iter603 = sel618;
                iter604 = sel620;
            }
            bool cmp621 = iter604 > 0;
            int32_t ifr622;
            if (cmp621) {
                int32_t t623 = (int32_t)((uint32_t)iter604 - (uint32_t)1);
                asc_lock(PIPE_S, 1);
                int64_t t624 = t623;
                uint32_t ld625 = buf522[t624];
                asc_unlock(PIPE_S, 1);
                int32_t scalar626 = (int32_t)ld625;
                ifr622 = scalar626;
            } else {
                ifr622 = 0;
            }
            int32_t t627 = (int32_t)((uint32_t)t557 - (uint32_t)ifr622);
            int32_t t628 = (int32_t)((uint32_t)iter604 * (uint32_t)16777216);
            asc_lock(PIPE_V, 1);
            kernel_generate_pool_vf4((__ubuf__ uint32_t*)buf522);
            asc_unlock(PIPE_V, 1);
            for (int32_t i629 = 0; i629 < t565; i629 = (int32_t)((uint32_t)i629 + (uint32_t)1)) {
                int64_t t630 = (int64_t)i629;
                int64_t dim631 = gm513_arg.shape[0];
                int64_t t632 = (i527 < dim631 ? i527 : dim631);
                int64_t t633 = (int64_t)((uint64_t)dim631 - (uint64_t)t632);
                int64_t t634 = (t633 < 1 ? t633 : 1);
                bool cmp635 = i527 <= dim631;
                int64_t sel636 = cmp635 ? t634 : 0;
                int64_t dim637 = gm513_arg.stride[0];
                int64_t t638 = (int64_t)((uint64_t)dim637 * (uint64_t)4);
                int64_t t639 = dim528 / 8192;
                int64_t t640 = (t630 < t639 ? t630 : t639);
                int64_t t641 = (int64_t)((uint64_t)t640 * (uint64_t)8192);
                int64_t t642 = (int64_t)((uint64_t)dim528 - (uint64_t)t641);
                int64_t t643 = (t642 < 8192 ? t642 : 8192);
                bool cmp644 = t630 <= t639;
                int64_t sel645 = cmp644 ? t643 : 0;
                int64_t t646 = (int64_t)((uint64_t)i527 * (uint64_t)dim637);
                int64_t t647 = (int64_t)((uint64_t)t646 * (uint64_t)4);
                int64_t t648 = (int64_t)((uint64_t)t630 * (uint64_t)32768);
                int64_t t649 = (int64_t)((uint64_t)t647 + (uint64_t)t648);
                int64_t t650 = (int64_t)((uint64_t)sel645 * (uint64_t)4);
                bool cmp651 = t650 == 0;
                int64_t sel652 = cmp651 ? 0 : sel636;
                int64_t sel653 = cmp651 ? 4 : t650;
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf520 + 0), (__gm__ float*)((__gm__ uint8_t*)gm513 + t649), sel652, sel653, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t638, 32768);
                }
                asc_unlock(PIPE_MTE2, 0);
                int32_t t654 = (int32_t)((uint32_t)i629 * (uint32_t)1024);
                int32_t t655 = (int32_t)((uint32_t)t547 - (uint32_t)t654);
                bool cmp656 = t655 > 1024;
                int32_t sel657 = cmp656 ? 1024 : t655;
                int32_t t658 = (int32_t)((uint32_t)sel657 + (uint32_t)63);
                int32_t t659 = t658 % 64;
                bool cmp660 = t659 != 0;
                bool cmp661 = t659 < 0;
                bool t662 = cmp660 & cmp661;
                int32_t t663 = t658 / 64;
                int32_t t664 = (int32_t)(uint8_t)t662;
                int32_t t665 = (int32_t)((uint32_t)t663 - (uint32_t)t664);
                asc_lock(PIPE_V, 0);
                kernel_generate_pool_vf5(sel657, (__ubuf__ float*)buf520, t654, t547, (__ubuf__ uint32_t*)buf521, t665);
                asc_unlock(PIPE_V, 0);
                kernel_generate_pool_vf6(sel657, (__ubuf__ uint32_t*)buf521, t628, t665, (__ubuf__ uint16_t*)buf523, (__ubuf__ uint32_t*)buf522);
            }
            asc_sync();
            int32_t iter666 = 255;
            int32_t iter667 = 0;
            for (int64_t i668 = 0; i668 < 8; i668 = (int64_t)((uint64_t)i668 + (uint64_t)1)) {
                int32_t t669 = (int32_t)((uint32_t)iter667 + (uint32_t)iter666);
                int32_t t670 = t669 % 2;
                bool cmp671 = t670 != 0;
                bool cmp672 = t670 < 0;
                bool t673 = cmp671 & cmp672;
                int32_t t674 = t669 / 2;
                int32_t t675 = (int32_t)(uint8_t)t673;
                int32_t t676 = (int32_t)((uint32_t)t674 - (uint32_t)t675);
                asc_lock(PIPE_S, 1);
                int64_t t677 = t676;
                uint32_t ld678 = buf522[t677];
                asc_unlock(PIPE_S, 1);
                int32_t scalar679 = (int32_t)ld678;
                bool cmp680 = scalar679 >= t627;
                int32_t sel681 = cmp680 ? t676 : iter666;
                int32_t t682 = (int32_t)((uint32_t)t676 + (uint32_t)1);
                int32_t sel683 = cmp680 ? iter667 : t682;
                iter666 = sel681;
                iter667 = sel683;
            }
            bool cmp684 = iter667 > 0;
            int32_t ifr685;
            if (cmp684) {
                int32_t t686 = (int32_t)((uint32_t)iter667 - (uint32_t)1);
                asc_lock(PIPE_S, 1);
                int64_t t687 = t686;
                uint32_t ld688 = buf522[t687];
                asc_unlock(PIPE_S, 1);
                int32_t scalar689 = (int32_t)ld688;
                ifr685 = scalar689;
            } else {
                ifr685 = 0;
            }
            int32_t t690 = (int32_t)((uint32_t)t627 - (uint32_t)ifr685);
            int32_t t691 = (int32_t)((uint32_t)iter667 * (uint32_t)65536);
            int32_t t692 = (int32_t)((uint32_t)t628 + (uint32_t)t691);
            asc_lock(PIPE_V, 1);
            kernel_generate_pool_vf7((__ubuf__ uint32_t*)buf522);
            asc_unlock(PIPE_V, 1);
            for (int32_t i693 = 0; i693 < t565; i693 = (int32_t)((uint32_t)i693 + (uint32_t)1)) {
                int64_t t694 = (int64_t)i693;
                int64_t dim695 = gm513_arg.shape[0];
                int64_t t696 = (i527 < dim695 ? i527 : dim695);
                int64_t t697 = (int64_t)((uint64_t)dim695 - (uint64_t)t696);
                int64_t t698 = (t697 < 1 ? t697 : 1);
                bool cmp699 = i527 <= dim695;
                int64_t sel700 = cmp699 ? t698 : 0;
                int64_t dim701 = gm513_arg.stride[0];
                int64_t t702 = (int64_t)((uint64_t)dim701 * (uint64_t)4);
                int64_t t703 = dim528 / 8192;
                int64_t t704 = (t694 < t703 ? t694 : t703);
                int64_t t705 = (int64_t)((uint64_t)t704 * (uint64_t)8192);
                int64_t t706 = (int64_t)((uint64_t)dim528 - (uint64_t)t705);
                int64_t t707 = (t706 < 8192 ? t706 : 8192);
                bool cmp708 = t694 <= t703;
                int64_t sel709 = cmp708 ? t707 : 0;
                int64_t t710 = (int64_t)((uint64_t)i527 * (uint64_t)dim701);
                int64_t t711 = (int64_t)((uint64_t)t710 * (uint64_t)4);
                int64_t t712 = (int64_t)((uint64_t)t694 * (uint64_t)32768);
                int64_t t713 = (int64_t)((uint64_t)t711 + (uint64_t)t712);
                int64_t t714 = (int64_t)((uint64_t)sel709 * (uint64_t)4);
                bool cmp715 = t714 == 0;
                int64_t sel716 = cmp715 ? 0 : sel700;
                int64_t sel717 = cmp715 ? 4 : t714;
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf520 + 0), (__gm__ float*)((__gm__ uint8_t*)gm513 + t713), sel716, sel717, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t702, 32768);
                }
                asc_unlock(PIPE_MTE2, 0);
                int32_t t718 = (int32_t)((uint32_t)i693 * (uint32_t)1024);
                int32_t t719 = (int32_t)((uint32_t)t547 - (uint32_t)t718);
                bool cmp720 = t719 > 1024;
                int32_t sel721 = cmp720 ? 1024 : t719;
                int32_t t722 = (int32_t)((uint32_t)sel721 + (uint32_t)63);
                int32_t t723 = t722 % 64;
                bool cmp724 = t723 != 0;
                bool cmp725 = t723 < 0;
                bool t726 = cmp724 & cmp725;
                int32_t t727 = t722 / 64;
                int32_t t728 = (int32_t)(uint8_t)t726;
                int32_t t729 = (int32_t)((uint32_t)t727 - (uint32_t)t728);
                asc_lock(PIPE_V, 0);
                kernel_generate_pool_vf8(sel721, (__ubuf__ float*)buf520, t718, t547, (__ubuf__ uint32_t*)buf521, t729);
                asc_unlock(PIPE_V, 0);
                kernel_generate_pool_vf9(sel721, (__ubuf__ uint32_t*)buf521, t692, t729, (__ubuf__ uint16_t*)buf523, (__ubuf__ uint32_t*)buf522);
            }
            asc_sync();
            int32_t iter730 = 255;
            int32_t iter731 = 0;
            for (int64_t i732 = 0; i732 < 8; i732 = (int64_t)((uint64_t)i732 + (uint64_t)1)) {
                int32_t t733 = (int32_t)((uint32_t)iter731 + (uint32_t)iter730);
                int32_t t734 = t733 % 2;
                bool cmp735 = t734 != 0;
                bool cmp736 = t734 < 0;
                bool t737 = cmp735 & cmp736;
                int32_t t738 = t733 / 2;
                int32_t t739 = (int32_t)(uint8_t)t737;
                int32_t t740 = (int32_t)((uint32_t)t738 - (uint32_t)t739);
                asc_lock(PIPE_S, 1);
                int64_t t741 = t740;
                uint32_t ld742 = buf522[t741];
                asc_unlock(PIPE_S, 1);
                int32_t scalar743 = (int32_t)ld742;
                bool cmp744 = scalar743 >= t690;
                int32_t sel745 = cmp744 ? t740 : iter730;
                int32_t t746 = (int32_t)((uint32_t)t740 + (uint32_t)1);
                int32_t sel747 = cmp744 ? iter731 : t746;
                iter730 = sel745;
                iter731 = sel747;
            }
            bool cmp748 = iter731 > 0;
            int32_t ifr749;
            if (cmp748) {
                int32_t t750 = (int32_t)((uint32_t)iter731 - (uint32_t)1);
                asc_lock(PIPE_S, 1);
                int64_t t751 = t750;
                uint32_t ld752 = buf522[t751];
                asc_unlock(PIPE_S, 1);
                int32_t scalar753 = (int32_t)ld752;
                ifr749 = scalar753;
            } else {
                ifr749 = 0;
            }
            int32_t t754 = (int32_t)((uint32_t)t690 - (uint32_t)ifr749);
            int32_t t755 = (int32_t)((uint32_t)iter731 * (uint32_t)256);
            int32_t t756 = (int32_t)((uint32_t)t692 + (uint32_t)t755);
            asc_lock(PIPE_V, 1);
            kernel_generate_pool_vf10((__ubuf__ uint32_t*)buf522);
            asc_unlock(PIPE_V, 1);
            for (int32_t i757 = 0; i757 < t565; i757 = (int32_t)((uint32_t)i757 + (uint32_t)1)) {
                int64_t t758 = (int64_t)i757;
                int64_t dim759 = gm513_arg.shape[0];
                int64_t t760 = (i527 < dim759 ? i527 : dim759);
                int64_t t761 = (int64_t)((uint64_t)dim759 - (uint64_t)t760);
                int64_t t762 = (t761 < 1 ? t761 : 1);
                bool cmp763 = i527 <= dim759;
                int64_t sel764 = cmp763 ? t762 : 0;
                int64_t dim765 = gm513_arg.stride[0];
                int64_t t766 = (int64_t)((uint64_t)dim765 * (uint64_t)4);
                int64_t t767 = dim528 / 8192;
                int64_t t768 = (t758 < t767 ? t758 : t767);
                int64_t t769 = (int64_t)((uint64_t)t768 * (uint64_t)8192);
                int64_t t770 = (int64_t)((uint64_t)dim528 - (uint64_t)t769);
                int64_t t771 = (t770 < 8192 ? t770 : 8192);
                bool cmp772 = t758 <= t767;
                int64_t sel773 = cmp772 ? t771 : 0;
                int64_t t774 = (int64_t)((uint64_t)i527 * (uint64_t)dim765);
                int64_t t775 = (int64_t)((uint64_t)t774 * (uint64_t)4);
                int64_t t776 = (int64_t)((uint64_t)t758 * (uint64_t)32768);
                int64_t t777 = (int64_t)((uint64_t)t775 + (uint64_t)t776);
                int64_t t778 = (int64_t)((uint64_t)sel773 * (uint64_t)4);
                bool cmp779 = t778 == 0;
                int64_t sel780 = cmp779 ? 0 : sel764;
                int64_t sel781 = cmp779 ? 4 : t778;
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf520 + 0), (__gm__ float*)((__gm__ uint8_t*)gm513 + t777), sel780, sel781, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t766, 32768);
                }
                asc_unlock(PIPE_MTE2, 0);
                int32_t t782 = (int32_t)((uint32_t)i757 * (uint32_t)1024);
                int32_t t783 = (int32_t)((uint32_t)t547 - (uint32_t)t782);
                bool cmp784 = t783 > 1024;
                int32_t sel785 = cmp784 ? 1024 : t783;
                int32_t t786 = (int32_t)((uint32_t)sel785 + (uint32_t)63);
                int32_t t787 = t786 % 64;
                bool cmp788 = t787 != 0;
                bool cmp789 = t787 < 0;
                bool t790 = cmp788 & cmp789;
                int32_t t791 = t786 / 64;
                int32_t t792 = (int32_t)(uint8_t)t790;
                int32_t t793 = (int32_t)((uint32_t)t791 - (uint32_t)t792);
                asc_lock(PIPE_V, 0);
                kernel_generate_pool_vf11(sel785, (__ubuf__ float*)buf520, t782, t547, (__ubuf__ uint32_t*)buf521, t793);
                asc_unlock(PIPE_V, 0);
                kernel_generate_pool_vf12(sel785, (__ubuf__ uint32_t*)buf521, t756, t793, (__ubuf__ uint16_t*)buf523, (__ubuf__ uint32_t*)buf522);
            }
            asc_sync();
            int32_t iter794 = 255;
            int32_t iter795 = 0;
            for (int64_t i796 = 0; i796 < 8; i796 = (int64_t)((uint64_t)i796 + (uint64_t)1)) {
                int32_t t797 = (int32_t)((uint32_t)iter795 + (uint32_t)iter794);
                int32_t t798 = t797 % 2;
                bool cmp799 = t798 != 0;
                bool cmp800 = t798 < 0;
                bool t801 = cmp799 & cmp800;
                int32_t t802 = t797 / 2;
                int32_t t803 = (int32_t)(uint8_t)t801;
                int32_t t804 = (int32_t)((uint32_t)t802 - (uint32_t)t803);
                int64_t t805 = t804;
                uint32_t ld806 = buf522[t805];
                int32_t scalar807 = (int32_t)ld806;
                bool cmp808 = scalar807 >= t754;
                int32_t sel809 = cmp808 ? t804 : iter794;
                int32_t t810 = (int32_t)((uint32_t)t804 + (uint32_t)1);
                int32_t sel811 = cmp808 ? iter795 : t810;
                iter794 = sel809;
                iter795 = sel811;
            }
            bool cmp812 = iter795 > 0;
            int32_t ifr813;
            if (cmp812) {
                int32_t t814 = (int32_t)((uint32_t)iter795 - (uint32_t)1);
                int64_t t815 = t814;
                uint32_t ld816 = buf522[t815];
                int32_t scalar817 = (int32_t)ld816;
                ifr813 = scalar817;
            } else {
                ifr813 = 0;
            }
            int64_t t818 = iter795;
            uint32_t ld819 = buf522[t818];
            int32_t scalar820 = (int32_t)ld819;
            int32_t t821 = (int32_t)((uint32_t)scalar820 - (uint32_t)ifr813);
            int32_t t822 = (int32_t)((uint32_t)t754 - (uint32_t)ifr813);
            int32_t t823 = (int32_t)((uint32_t)t756 + (uint32_t)iter795);
            int32_t t824 = (int32_t)((uint32_t)t821 - (uint32_t)t822);
            int32_t t825 = (int32_t)((uint32_t)t824 + (uint32_t)1);
            int64_t dim826 = gm513_arg.shape[0];
            int64_t t827 = (i527 < dim826 ? i527 : dim826);
            int64_t t828 = (int64_t)((uint64_t)dim826 - (uint64_t)t827);
            int64_t t829 = (t828 < 1 ? t828 : 1);
            bool cmp830 = i527 <= dim826;
            int64_t sel831 = cmp830 ? t829 : 0;
            int64_t dim832 = gm513_arg.stride[0];
            int64_t t833 = (int64_t)((uint64_t)dim832 * (uint64_t)4);
            int64_t t834 = dim528 / 8192;
            int64_t t835 = (t834 < 0 ? t834 : 0);
            int64_t t836 = (int64_t)((uint64_t)t835 * (uint64_t)8192);
            int64_t t837 = (int64_t)((uint64_t)dim528 - (uint64_t)t836);
            int64_t t838 = (t837 < 8192 ? t837 : 8192);
            bool cmp839 = t834 >= 0;
            int64_t sel840 = cmp839 ? t838 : 0;
            int64_t t841 = (int64_t)((uint64_t)i527 * (uint64_t)dim832);
            int64_t t842 = (int64_t)((uint64_t)t841 * (uint64_t)4);
            int64_t t843 = (int64_t)((uint64_t)sel840 * (uint64_t)4);
            bool cmp844 = t843 == 0;
            int64_t sel845 = cmp844 ? 0 : sel831;
            int64_t sel846 = cmp844 ? 4 : t843;
            asc_lock(PIPE_MTE2, 0);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf520 + 0), (__gm__ float*)((__gm__ uint8_t*)gm513 + t842), sel845, sel846, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t833, 32768);
            }
            asc_unlock(PIPE_MTE2, 0);
            bool cmp847 = t547 > 1024;
            int32_t sel848 = cmp847 ? 1024 : t547;
            int32_t t849 = (int32_t)((uint32_t)sel848 + (uint32_t)63);
            int32_t t850 = t849 % 64;
            bool cmp851 = t850 != 0;
            bool cmp852 = t850 < 0;
            bool t853 = cmp851 & cmp852;
            int32_t t854 = t849 / 64;
            int32_t t855 = (int32_t)(uint8_t)t853;
            int32_t t856 = (int32_t)((uint32_t)t854 - (uint32_t)t855);
            asc_lock(PIPE_V, 0);
            kernel_generate_pool_vf13(sel848, (__ubuf__ float*)buf520, t547, (__ubuf__ uint32_t*)buf521, t856);
            asc_unlock(PIPE_V, 0);
            kernel_generate_pool_vf14(t825, (__ubuf__ int32_t*)buf524, sel848, (__ubuf__ uint32_t*)buf521, t823, t856, (__ubuf__ int32_t*)buf525);
            asc_sync();
            asc_lock(PIPE_S, 3);
            int32_t ld857 = buf525[0];
            asc_unlock(PIPE_S, 3);
            int32_t iter858 = ld857;
            for (int32_t i859 = 1; i859 < t565; i859 = (int32_t)((uint32_t)i859 + (uint32_t)1)) {
                int64_t t860 = (int64_t)i859;
                int64_t t861 = (t860 < t834 ? t860 : t834);
                int64_t t862 = (int64_t)((uint64_t)t861 * (uint64_t)8192);
                int64_t t863 = (int64_t)((uint64_t)dim528 - (uint64_t)t862);
                int64_t t864 = (t863 < 8192 ? t863 : 8192);
                bool cmp865 = t860 <= t834;
                int64_t sel866 = cmp865 ? t864 : 0;
                int64_t t867 = (int64_t)((uint64_t)t860 * (uint64_t)32768);
                int64_t t868 = (int64_t)((uint64_t)t842 + (uint64_t)t867);
                int64_t t869 = (int64_t)((uint64_t)sel866 * (uint64_t)4);
                bool cmp870 = t869 == 0;
                int64_t sel871 = cmp870 ? 0 : sel831;
                int64_t sel872 = cmp870 ? 4 : t869;
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf520 + 0), (__gm__ float*)((__gm__ uint8_t*)gm513 + t868), sel871, sel872, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t833, 32768);
                }
                asc_unlock(PIPE_MTE2, 0);
                int32_t t873 = (int32_t)((uint32_t)i859 * (uint32_t)1024);
                int32_t t874 = (int32_t)((uint32_t)t547 - (uint32_t)t873);
                bool cmp875 = t874 > 1024;
                int32_t sel876 = cmp875 ? 1024 : t874;
                int32_t t877 = (int32_t)((uint32_t)sel876 + (uint32_t)63);
                int32_t t878 = t877 % 64;
                bool cmp879 = t878 != 0;
                bool cmp880 = t878 < 0;
                bool t881 = cmp879 & cmp880;
                int32_t t882 = t877 / 64;
                int32_t t883 = (int32_t)(uint8_t)t881;
                int32_t t884 = (int32_t)((uint32_t)t882 - (uint32_t)t883);
                asc_lock(PIPE_V, 0);
                kernel_generate_pool_vf15(sel876, (__ubuf__ float*)buf520, t873, t547, (__ubuf__ uint32_t*)buf521, t884);
                asc_unlock(PIPE_V, 0);
                asc_lock(PIPE_V, 3);
                kernel_generate_pool_vf16(iter858, (__ubuf__ int32_t*)buf524, sel876, (__ubuf__ uint32_t*)buf521, t823, t873, t884, (__ubuf__ int32_t*)buf525);
                asc_unlock(PIPE_V, 3);
                asc_sync();
                asc_lock(PIPE_S, 3);
                int32_t ld885 = buf525[0];
                asc_unlock(PIPE_S, 3);
                iter858 = ld885;
            }
        }
        int64_t dim886 = gm515_arg.stride[0];
        int64_t t887 = (int64_t)((uint64_t)i527 * (uint64_t)dim886);
        int64_t t888 = (int64_t)((uint64_t)t887 * (uint64_t)32);
        int64_t t889 = (int64_t)((uint64_t)t888 / (uint64_t)8);
        int64_t t890 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm515) + (uint64_t)t889);
        __gm__ int32_t* root891 = (__gm__ int32_t*)(uintptr_t)(uint64_t)(t890);
        __ubuf__ int32_t* view_root892 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf524) + 0);
        int64_t t893 = dim548;
        int64_t dim894 = gm515_arg.stride[0];
        int64_t t895 = dim894;
        int64_t t896 = t893 * 4;
        int64_t t897 = t895 * 4;
        bool cmp898 = (uint64_t)dim548 > (uint64_t)0;
        if (cmp898) {
            int64_t t899 = t896;
            int64_t t900 = t897;
            asc_lock(PIPE_MTE3, 2);
            int32_t t901 = (int32_t)t899;
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int32_t*)root891 + 0, (__ubuf__ int32_t*)view_root892 + 0, 1, t901, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 2);
        }
    }
    bool cmp902 = arg517 == 0;
    if (cmp902) {
        for (int64_t i903 = 0; i903 < dim526; i903 = (int64_t)((uint64_t)i903 + (uint64_t)1)) {
            int64_t dim904 = gm513_arg.shape[1];
            int64_t t905 = (int64_t)arg519;
            bool cmp906 = i903 < t905;
            int32_t ifr907;
            if (cmp906) {
                int64_t t908 = i903;
                int32_t ld909 = gm514[t908];
                bool cmp910 = ld909 < 0;
                int32_t sel911 = cmp910 ? 0 : ld909;
                int64_t t912 = (int64_t)sel911;
                bool cmp913 = t912 > dim904;
                int32_t t914 = (int32_t)dim904;
                int32_t sel915 = cmp913 ? t914 : sel911;
                ifr907 = sel915;
            } else {
                ifr907 = 0;
            }
            int32_t t916 = (int32_t)((uint32_t)ifr907 + (uint32_t)7);
            int32_t t917 = t916 % 8;
            bool cmp918 = t917 != 0;
            bool cmp919 = t917 < 0;
            bool t920 = cmp918 & cmp919;
            int32_t t921 = t916 / 8;
            int32_t t922 = (int32_t)(uint8_t)t920;
            int32_t t923 = (int32_t)((uint32_t)t921 - (uint32_t)t922);
            int64_t dim924 = gm515_arg.shape[1];
            int64_t t925 = (int64_t)t923;
            bool cmp926 = t925 > dim924;
            int32_t t927 = (int32_t)dim924;
            int32_t sel928 = cmp926 ? t927 : t923;
            bool cmp929 = sel928 > 2048;
            int32_t sel930 = cmp929 ? 2048 : sel928;
            int64_t t931 = i903;
            gm516[t931] = sel930;
        }
    }
}
