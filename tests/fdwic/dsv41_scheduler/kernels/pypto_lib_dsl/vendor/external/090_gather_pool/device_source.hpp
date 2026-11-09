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

struct KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds {
    __gm__ fp8_e8m0_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds cannbotdsl_pack_KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds(const KernelArgTensor_fp8_e8m0_t_2d& src) {
    return {src.ptr, {src.shape[1]}, {src.stride[0]}};
}

struct KernelArgTensor_int32_t_2d_Device_sd_ds {
    __gm__ int32_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_int32_t_2d_Device_sd_ds cannbotdsl_pack_KernelArgTensor_int32_t_2d_Device_sd_ds(const KernelArgTensor_int32_t_2d& src) {
    return {src.ptr, {src.shape[1]}, {src.stride[0]}};
}

__simd_vf__ inline void kernel_gather_pool_vf0(__ubuf__ uint8_t* buf0, __ubuf__ uint8_t* buf1, __ubuf__ int32_t* buf2, __ubuf__ int32_t* buf3)
{
    vector_bool vmask;
    vector_bool mask4 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t v5;
    asc_duplicate_scalar(v5, 0, mask4);
    vector_uint8_t v6;
    asc_duplicate_scalar(v6, 127, mask4);
    for (int32_t i7 = 0; i7 < 16384; i7 = (int32_t)((uint32_t)i7 + (uint32_t)256)) {
        int32_t t8 = i7;
        asc_storealign(buf0 + t8, v5, mask4);
    }
    for (int32_t i9 = 0; i9 < 1024; i9 = (int32_t)((uint32_t)i9 + (uint32_t)256)) {
        int32_t t10 = i9;
        asc_storealign(buf1 + t10, v6, mask4);
    }
    vector_bool mask11 = asc_create_mask_b32(PAT_ALL);
    vector_int32_t v12;
    asc_duplicate_scalar(v12, 0, mask11);
    vector_int32_t v13;
    asc_duplicate_scalar(v13, -1, mask11);
    for (int32_t i14 = 0; i14 < 256; i14 = (int32_t)((uint32_t)i14 + (uint32_t)64)) {
        int32_t t15 = i14;
        asc_storealign(buf2 + t15, v12, mask11);
        asc_storealign(buf3 + t15, v13, mask11);
    }
}

__simd_vf__ inline void kernel_gather_pool_vf1(int32_t arg16, int32_t arg17, __ubuf__ int32_t* buf18, int32_t arg19, __ubuf__ int32_t* buf20)
{
    vector_bool vmask;
    for (int32_t i21 = 0; i21 < 1; i21 = (int32_t)((uint32_t)i21 + (uint32_t)1)) {
        uint32_t rem23 = 8;
        vector_bool mask22 = asc_update_mask_b32(rem23);
        vector_int32_t v24;
        asc_arange(v24, 0);
        vector_bool m25;
        asc_lt_scalar(m25, v24, arg16, mask22);
        vector_int32_t v26;
        asc_duplicate_scalar(v26, 1, mask22);
        vector_int32_t v27;
        asc_duplicate_scalar(v27, 0, mask22);
        vector_int32_t v28;
        asc_select(v28, v26, v27, m25);
        vector_int32_t v29;
        asc_add_scalar(v29, v24, arg17, mask22);
        vector_int32_t v30;
        asc_duplicate_scalar(v30, -1, mask22);
        vector_int32_t v31;
        asc_select(v31, v29, v30, m25);
        asc_storealign(buf18 + arg19, v28, mask22);
        asc_storealign(buf20 + arg19, v31, mask22);
    }
}

static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 32 && alignof(KernelArgTensor_int32_t_2d_Device_dd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_1d_Device_d_s) == 16 && alignof(KernelArgTensor_int32_t_1d_Device_d_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds) == 24 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_sd_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_sd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_sd_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_sd_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_gather_pool(KernelArgTensor_uint8_t_2d_Device_ds_ds gm32_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ds_ds gm33_arg, KernelArgTensor_int32_t_2d_Device_dd_ds gm34_arg, KernelArgTensor_int32_t_1d_Device_d_s gm35_arg, KernelArgTensor_int32_t_1d_Device_d_s gm36_arg, KernelArgTensor_int32_t_2d_Device_dd_ds gm37_arg, KernelArgTensor_int32_t_1d_Device_d_s gm38_arg, KernelArgTensor_uint8_t_2d_Device_ds_ds gm39_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_sd_ds gm40_arg, KernelArgTensor_int32_t_2d_Device_sd_ds gm41_arg, KernelArgTensor_int32_t_2d_Device_sd_ds gm42_arg, int64_t arg43, int64_t arg44, int32_t arg45)
{
    __gm__ uint8_t* gm32 = gm32_arg.ptr;
    __gm__ fp8_e8m0_t* gm33 = gm33_arg.ptr;
    __gm__ int32_t* gm34 = gm34_arg.ptr;
    __gm__ int32_t* gm35 = gm35_arg.ptr;
    __gm__ int32_t* gm36 = gm36_arg.ptr;
    __gm__ int32_t* gm37 = gm37_arg.ptr;
    __gm__ int32_t* gm38 = gm38_arg.ptr;
    __gm__ uint8_t* gm39 = gm39_arg.ptr;
    __gm__ fp8_e8m0_t* gm40 = gm40_arg.ptr;
    __gm__ int32_t* gm41 = gm41_arg.ptr;
    __gm__ int32_t* gm42 = gm42_arg.ptr;
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

    __ubuf__ uint8_t* buf46 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ uint8_t* buf47 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(16384);
    __ubuf__ int32_t* buf48 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(17408);
    __ubuf__ int32_t* buf49 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(18432);

    vector_bool vmask;
    int64_t dim50 = gm37_arg.shape[1];
    int64_t t51 = (int64_t)((uint64_t)dim50 + (uint64_t)31);
    int64_t t52 = t51 % 32;
    bool cmp53 = t52 != 0;
    bool cmp54 = t52 < 0;
    bool t55 = cmp53 & cmp54;
    int64_t t56 = t51 / 32;
    int64_t t57 = (int64_t)(uint8_t)t55;
    int64_t t58 = (int64_t)((uint64_t)t56 - (uint64_t)t57);
    int64_t t59 = (int64_t)arg45;
    int64_t t60 = (int64_t)((uint64_t)t59 * (uint64_t)t58);
    for (int64_t i61 = arg43; i61 < t60; i61 = (int64_t)((uint64_t)i61 + (uint64_t)arg44)) {
        int64_t t62 = i61 % t58;
        bool cmp63 = t62 != 0;
        bool cmp64 = t62 < 0;
        bool cmp65 = t58 < 0;
        bool t66 = cmp64 ^ cmp65;
        bool t67 = cmp63 & t66;
        int64_t t68 = i61 / t58;
        int64_t t69 = (int64_t)(uint8_t)t67;
        int64_t t70 = (int64_t)((uint64_t)t68 - (uint64_t)t69);
        int64_t sel71 = t67 ? t58 : 0;
        int64_t t72 = (int64_t)((uint64_t)t62 + (uint64_t)sel71);
        int64_t t73 = t70;
        int32_t ld74 = gm38[t73];
        bool cmp75 = ld74 < 0;
        int32_t sel76 = cmp75 ? 0 : ld74;
        int64_t t77 = (int64_t)sel76;
        bool cmp78 = t77 > dim50;
        int32_t t79 = (int32_t)dim50;
        int32_t sel80 = cmp78 ? t79 : sel76;
        int32_t ld81 = gm35[t73];
        int32_t ld82 = gm36[t73];
        bool cmp83 = ld82 < 0;
        int32_t sel84 = cmp83 ? 0 : ld82;
        asc_lock(PIPE_V, 0);
        asc_lock(PIPE_V, 3);
        kernel_gather_pool_vf0((__ubuf__ uint8_t*)buf46, (__ubuf__ uint8_t*)buf47, (__ubuf__ int32_t*)buf48, (__ubuf__ int32_t*)buf49);
        asc_unlock(PIPE_V, 3);
        asc_unlock(PIPE_V, 0);
        for (int64_t i85 = 0; i85 < 32; i85 = (int64_t)((uint64_t)i85 + (uint64_t)1)) {
            int64_t t86 = (int64_t)((uint64_t)t72 * (uint64_t)32);
            int64_t t87 = (int64_t)((uint64_t)t86 + (uint64_t)i85);
            int64_t t88 = (int64_t)sel80;
            bool cmp89 = t87 < t88;
            if (cmp89) {
                int64_t dim90 = gm37_arg.stride[0];
                int64_t t91 = dim90;
                int64_t t92 = t87;
                int64_t t93 = t73 * t91;
                int64_t t94 = t93 + t92;
                int32_t ld95 = gm37[t94];
                bool cmp96 = ld95 >= 0;
                bool ifr97;
                if (cmp96) {
                    int32_t t98 = (int32_t)((uint32_t)sel84 + (uint32_t)7);
                    int32_t t99 = t98 % 8;
                    bool cmp100 = t99 != 0;
                    bool cmp101 = t99 < 0;
                    bool t102 = cmp100 & cmp101;
                    int32_t t103 = t98 / 8;
                    int32_t t104 = (int32_t)(uint8_t)t102;
                    int32_t t105 = (int32_t)((uint32_t)t103 - (uint32_t)t104);
                    bool cmp106 = ld95 < t105;
                    ifr97 = cmp106;
                } else {
                    ifr97 = false;
                }
                if (ifr97) {
                    int32_t t107 = ld95 / 16;
                    bool cmp108 = ld81 >= 0;
                    bool ifr109;
                    if (cmp108) {
                        int64_t dim110 = gm34_arg.shape[0];
                        int64_t t111 = (int64_t)ld81;
                        bool cmp112 = t111 < dim110;
                        bool ifr113;
                        if (cmp112) {
                            int64_t dim114 = gm34_arg.shape[1];
                            int64_t t115 = (int64_t)t107;
                            bool cmp116 = t115 < dim114;
                            ifr113 = cmp116;
                        } else {
                            ifr113 = false;
                        }
                        ifr109 = ifr113;
                    } else {
                        ifr109 = false;
                    }
                    if (ifr109) {
                        int64_t dim117 = gm34_arg.stride[0];
                        int64_t t118 = dim117;
                        int64_t t119 = ld81;
                        int64_t t120 = t107;
                        int64_t t121 = t119 * t118;
                        int64_t t122 = t121 + t120;
                        int32_t ld123 = gm34[t122];
                        bool cmp124 = ld123 >= 0;
                        bool ifr125;
                        if (cmp124) {
                            int64_t dim126 = gm32_arg.shape[0];
                            int64_t t127 = dim126 % 128;
                            bool cmp128 = t127 != 0;
                            bool cmp129 = t127 < 0;
                            bool t130 = cmp128 & cmp129;
                            int64_t t131 = dim126 / 128;
                            int64_t t132 = (int64_t)(uint8_t)t130;
                            int64_t t133 = (int64_t)((uint64_t)t131 - (uint64_t)t132);
                            int64_t t134 = (int64_t)ld123;
                            bool cmp135 = t134 < t133;
                            ifr125 = cmp135;
                        } else {
                            ifr125 = false;
                        }
                        if (ifr125) {
                            int32_t t136 = (int32_t)((uint32_t)ld123 * (uint32_t)16);
                            int32_t t137 = ld95 % 16;
                            int32_t t138 = (int32_t)((uint32_t)t136 + (uint32_t)t137);
                            int64_t t139 = (int64_t)t138;
                            int64_t dim140 = gm32_arg.shape[0];
                            int64_t t141 = dim140 / 8;
                            int64_t t142 = (t139 < t141 ? t139 : t141);
                            int64_t t143 = (int64_t)((uint64_t)t142 * (uint64_t)8);
                            int64_t t144 = (int64_t)((uint64_t)dim140 - (uint64_t)t143);
                            int64_t t145 = (t144 < 8 ? t144 : 8);
                            bool cmp146 = t139 <= t141;
                            int64_t sel147 = cmp146 ? t145 : 0;
                            int64_t dim148 = gm32_arg.stride[0];
                            int64_t t149 = (int64_t)((uint64_t)t139 * (uint64_t)dim148);
                            int64_t t150 = (int64_t)((uint64_t)t149 * (uint64_t)8);
                            int64_t t151 = i85 * 512;
                            asc_lock(PIPE_MTE2, 0);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf46 + t151), (__gm__ uint8_t*)((__gm__ uint8_t*)gm32 + t150), sel147, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)dim148, 64);
                            }
                            asc_unlock(PIPE_MTE2, 0);
                            int64_t dim152 = gm33_arg.shape[0];
                            int64_t t153 = (t139 < dim152 ? t139 : dim152);
                            int64_t t154 = (int64_t)((uint64_t)dim152 - (uint64_t)t153);
                            int64_t t155 = (t154 < 1 ? t154 : 1);
                            bool cmp156 = t139 <= dim152;
                            int64_t sel157 = cmp156 ? t155 : 0;
                            int64_t dim158 = gm33_arg.stride[0];
                            int64_t t159 = (int64_t)((uint64_t)t139 * (uint64_t)dim158);
                            int64_t t160 = i85 * 32;
                            asc_lock(PIPE_MTE2, 1);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf47 + t160), (__gm__ uint8_t*)((__gm__ uint8_t*)gm33 + t159), sel157, 32, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)dim158, 1024);
                            }
                            asc_unlock(PIPE_MTE2, 1);
                            int32_t t161 = (int32_t)((uint32_t)ld95 * (uint32_t)8);
                            int32_t t162 = (int32_t)((uint32_t)sel84 - (uint32_t)t161);
                            bool cmp163 = t162 > 8;
                            int32_t sel164 = cmp163 ? 8 : t162;
                            int32_t t165 = (int32_t)((uint32_t)t138 * (uint32_t)8);
                            int32_t t166 = (int32_t)i85;
                            int32_t t167 = t166 * 8;
                            int64_t t168 = t167;
                            asc_lock(PIPE_V, 2);
                            kernel_gather_pool_vf1(sel164, t165, (__ubuf__ int32_t*)buf48, (int32_t)t168, (__ubuf__ int32_t*)buf49);
                            asc_unlock(PIPE_V, 2);
                        }
                    }
                }
            }
        }
        int64_t t169 = (int64_t)((uint64_t)i61 * (uint64_t)256);
        int64_t dim170 = gm39_arg.stride[0];
        int64_t t171 = (int64_t)((uint64_t)t169 * (uint64_t)dim170);
        int64_t t172 = (int64_t)((uint64_t)t171 * (uint64_t)8);
        int64_t t173 = (int64_t)((uint64_t)t172 / (uint64_t)8);
        int64_t t174 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm39) + (uint64_t)t173);
        __gm__ uint8_t* root175 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t174);
        __ubuf__ uint8_t* view_root176 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf46) + 0);
        int64_t dim177 = gm39_arg.stride[0];
        asc_lock(PIPE_MTE3, 0);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root175 + 0, (__ubuf__ uint8_t*)view_root176 + 0, 256, 64, static_cast<uint8_t>(0), (uint64_t)dim177, 64);
        asc_unlock(PIPE_MTE3, 0);
        int64_t dim178 = gm40_arg.stride[0];
        int64_t t179 = (int64_t)((uint64_t)dim178 * (uint64_t)8);
        int64_t t180 = t179 / 8;
        int64_t t181 = (int64_t)((uint64_t)i61 * (uint64_t)8192);
        int64_t t182 = (int64_t)((uint64_t)t181 / (uint64_t)8);
        int64_t t183 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm40) + (uint64_t)t182);
        __gm__ uint8_t* root184 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t183);
        __ubuf__ uint8_t* view_root185 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf47) + 0);
        asc_lock(PIPE_MTE3, 1);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root184 + 0, (__ubuf__ uint8_t*)view_root185 + 0, 1, 1024, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 1);
        int64_t t186 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm41) + (uint64_t)t182);
        __gm__ int32_t* root187 = (__gm__ int32_t*)(uintptr_t)(uint64_t)(t186);
        __ubuf__ int32_t* view_root188 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf48) + 0);
        int64_t dim189 = gm41_arg.stride[0];
        int64_t t190 = dim189;
        int64_t t191 = t190 * 4;
        int64_t t192 = t191;
        asc_lock(PIPE_MTE3, 2);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int32_t*)root187 + 0, (__ubuf__ int32_t*)view_root188 + 0, 1, 1024, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 2);
        int64_t t193 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm42) + (uint64_t)t182);
        __gm__ int32_t* root194 = (__gm__ int32_t*)(uintptr_t)(uint64_t)(t193);
        __ubuf__ int32_t* view_root195 = reinterpret_cast<__ubuf__ int32_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf49) + 0);
        int64_t dim196 = gm42_arg.stride[0];
        int64_t t197 = dim196;
        int64_t t198 = t197 * 4;
        int64_t t199 = t198;
        asc_lock(PIPE_MTE3, 3);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ int32_t*)root194 + 0, (__ubuf__ int32_t*)view_root195 + 0, 1, 1024, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 3);
    }
}
