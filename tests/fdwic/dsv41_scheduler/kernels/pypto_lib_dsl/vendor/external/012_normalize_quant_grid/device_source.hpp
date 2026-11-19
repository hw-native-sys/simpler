// CANNBOTDSL_HOST_ABI_VERSION=9
#include "c_api/asc_simd.h"
#include "basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h"
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

struct KernelArgTensor_bfloat16_t_2d_Device_ss_ds {
    __gm__ bfloat16_t* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ss_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

struct KernelArgTensor_bfloat16_t_1d_Device_s_s {
    __gm__ bfloat16_t* ptr;
};
static inline KernelArgTensor_bfloat16_t_1d_Device_s_s cannbotdsl_pack_KernelArgTensor_bfloat16_t_1d_Device_s_s(const KernelArgTensor_bfloat16_t_1d& src) {
    return {src.ptr};
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

__simd_vf__ inline void kernel_normalize_quant_grid_vf0(__ubuf__ float* buf0)
{
    vector_bool vmask;
    float c1 = 0.0f;
    vector_bool mask2 = asc_create_mask_b32(PAT_ALL);
    vector_float v3;
    asc_duplicate_scalar(v3, c1, mask2);
    for (int32_t i4 = 0; i4 < 1344; i4 = (int32_t)((uint32_t)i4 + (uint32_t)64)) {
        int32_t t5 = i4;
        asc_storealign(buf0 + t5, v3, mask2);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_normalize_quant_grid_vf1(__ubuf__ bfloat16_t* buf6, __ubuf__ bfloat16_t* buf7, __ubuf__ bfloat16_t* buf8, __ubuf__ float* buf9, __ubuf__ fp8_e4m3fn_t* buf10, int32_t arg11, __ubuf__ uint8_t* buf12)
{
    vector_bool vmask;
    float c13 = 448.0f;
    float c14 = -448.0f;
    float c15 = 1.0f;
    float c16 = 9.99999997E-7f;
    float c17 = 7.81250012E-4f;
    float c18 = 0.0f;
    vector_bool mask19 = asc_create_mask_b32(PAT_ALL);
    vector_float v20;
    asc_duplicate_scalar(v20, c18, mask19);
    vector_float iter21 = v20;
    for (int32_t i22 = 0; i22 < 1280; i22 = (int32_t)((uint32_t)i22 + (uint32_t)64)) {
        int32_t t23 = i22;
        vector_float v24;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v24, (__ubuf__ bfloat16_t*)(buf6 + t23));
        vector_float v25;
        asc_bfloat162float(v25, *(vector_bfloat16_t*)&v24, mask19);
        vector_float v26;
        asc_mul(v26, v25, v25, mask19);
        vector_float v27;
        asc_add(v27, iter21, v26, mask19);
        iter21 = v27;
    }
    vector_float v28;
    asc_reduce_sum(v28, iter21, mask19);
    vector_float v29;
    asc_mul_scalar(v29, v28, c17, mask19);
    vector_float v30;
    asc_add_scalar(v30, v29, c16, mask19);
    vector_float v31;
    asc_sqrt(v31, v30, mask19);
    vector_float v32;
    asc_duplicate_scalar(v32, c15, mask19);
    vector_float v33;
    {
        static constexpr AscendC::Reg::DivSpecificMode div_mode_34 = {
            AscendC::Reg::MaskMergeMode::ZEROING, true, AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE};
        AscendC::Reg::Div<float, &div_mode_34, AscendC::Reg::RegTensor<float>>(*(AscendC::Reg::RegTensor<float>*)&v33, *(AscendC::Reg::RegTensor<float>*)&v32, *(AscendC::Reg::RegTensor<float>*)&v31, mask19);
    }
    vector_float v35;
    asc_duplicate(v35, v33, mask19);
    for (int32_t i36 = 0; i36 < 1280; i36 = (int32_t)((uint32_t)i36 + (uint32_t)64)) {
        int32_t t37 = i36;
        vector_float v38;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v38, (__ubuf__ bfloat16_t*)(buf6 + t37));
        vector_float v39;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v39, (__ubuf__ bfloat16_t*)(buf7 + t37));
        vector_float v40;
        asc_bfloat162float(v40, *(vector_bfloat16_t*)&v38, mask19);
        vector_float v41;
        asc_bfloat162float(v41, *(vector_bfloat16_t*)&v39, mask19);
        vector_float v42;
        asc_mul(v42, v40, v35, mask19);
        vector_float v43;
        asc_mul(v43, v42, v41, mask19);
        vector_bfloat16_t v44;
        asc_float2bfloat16_rn(v44, v43, mask19, ASC_POSITION_EVEN);
        asc_storealign_pack((__ubuf__ uint32_t*)(buf8 + t37), *(vector_uint32_t*)&v44, mask19);
        vector_float v45;
        asc_bfloat162float(v45, *(vector_bfloat16_t*)&v44, mask19);
        asc_storealign(buf9 + t37, v45, mask19);
    }
    asc_mem_bar(VST_VLD);
    uint32_t rem47 = 32;
    vector_bool mask46 = asc_update_mask_b32(rem47);
    for (int32_t i48 = 0; i48 < 40; i48 = (int32_t)((uint32_t)i48 + (uint32_t)1)) {
        int32_t t49 = i48 * 32;
        int32_t t50 = t49;
        vector_float v51;
        asc_loadalign(v51, buf9 + t50);
        vector_float v52;
        asc_abs(v52, v51, mask46);
        vector_float v53;
        asc_reduce_max(v53, v52, mask46);
        vector_int32_t &r54 = (vector_int32_t &)v53;
        vector_int32_t v55;
        asc_shiftright_scalar(v55, r54, 23, mask46);
        vector_int32_t v56;
        asc_add_scalar(v56, v55, -8, mask46);
        vector_int32_t v57;
        asc_max_scalar(v57, v56, 0, mask46);
        vector_int32_t v58;
        asc_duplicate(v58, v57, mask46);
        vector_int32_t v59;
        asc_duplicate_scalar(v59, 254, mask46);
        vector_int32_t v60;
        asc_sub(v60, v59, v58, mask46);
        vector_int32_t v61;
        asc_min_scalar(v61, v58, 1, mask46);
        vector_int32_t v62;
        asc_mul(v62, v60, v61, mask46);
        vector_int32_t v63;
        asc_shiftleft_scalar(v63, v62, 23, mask46);
        vector_float &r64 = (vector_float &)v63;
        vector_float v65;
        asc_mul(v65, v51, r64, mask46);
        vector_float v66;
        asc_max_scalar(v66, v65, c14, mask46);
        vector_float v67;
        asc_min_scalar(v67, v66, c13, mask46);
        vector_fp8_e4m3fn_t v68;
        asc_float2e4m3_rn(v68, v67, mask46);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf10 + t50), *(vector_float*)&v68, mask46);
        vector_uint8_t v69;
        asc_int322uint8(v69, v58, mask46);
        int32_t t70 = i48 / 2;
        int32_t t71 = t70 * 32;
        int32_t t72 = (int32_t)((uint32_t)arg11 * (uint32_t)2);
        int32_t t73 = (int32_t)((uint32_t)t71 + (uint32_t)t72);
        int32_t t74 = i48 % 2;
        int32_t t75 = (int32_t)((uint32_t)t73 + (uint32_t)t74);
        int32_t t76 = t75;
        asc_storealign_1st(buf12 + t76, v69);
    }
}

__simd_vf__ inline void kernel_normalize_quant_grid_vf2(__ubuf__ float* buf77, __ubuf__ fp8_e4m3fn_t* buf78, int32_t arg79, __ubuf__ uint8_t* buf80)
{
    vector_bool vmask;
    float c81 = 448.0f;
    float c82 = -448.0f;
    uint32_t rem84 = 32;
    vector_bool mask83 = asc_update_mask_b32(rem84);
    for (int32_t i85 = 0; i85 < 40; i85 = (int32_t)((uint32_t)i85 + (uint32_t)1)) {
        int32_t t86 = i85 * 32;
        int32_t t87 = t86;
        vector_float v88;
        asc_loadalign(v88, buf77 + t87);
        vector_float v89;
        asc_abs(v89, v88, mask83);
        vector_float v90;
        asc_reduce_max(v90, v89, mask83);
        vector_int32_t &r91 = (vector_int32_t &)v90;
        vector_int32_t v92;
        asc_shiftright_scalar(v92, r91, 23, mask83);
        vector_int32_t v93;
        asc_add_scalar(v93, v92, -8, mask83);
        vector_int32_t v94;
        asc_max_scalar(v94, v93, 0, mask83);
        vector_int32_t v95;
        asc_duplicate(v95, v94, mask83);
        vector_int32_t v96;
        asc_duplicate_scalar(v96, 254, mask83);
        vector_int32_t v97;
        asc_sub(v97, v96, v95, mask83);
        vector_int32_t v98;
        asc_min_scalar(v98, v95, 1, mask83);
        vector_int32_t v99;
        asc_mul(v99, v97, v98, mask83);
        vector_int32_t v100;
        asc_shiftleft_scalar(v100, v99, 23, mask83);
        vector_float &r101 = (vector_float &)v100;
        vector_float v102;
        asc_mul(v102, v88, r101, mask83);
        vector_float v103;
        asc_max_scalar(v103, v102, c82, mask83);
        vector_float v104;
        asc_min_scalar(v104, v103, c81, mask83);
        vector_fp8_e4m3fn_t v105;
        asc_float2e4m3_rn(v105, v104, mask83);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf78 + t87), *(vector_float*)&v105, mask83);
        vector_uint8_t v106;
        asc_int322uint8(v106, v95, mask83);
        int32_t t107 = i85 / 2;
        int32_t t108 = t107 * 32;
        int32_t t109 = (int32_t)((uint32_t)arg79 * (uint32_t)2);
        int32_t t110 = (int32_t)((uint32_t)t108 + (uint32_t)t109);
        int32_t t111 = i85 % 2;
        int32_t t112 = (int32_t)((uint32_t)t110 + (uint32_t)t111);
        int32_t t113 = t112;
        asc_storealign_1st(buf80 + t113, v106);
    }
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8 && alignof(KernelArgTensor_bfloat16_t_1d_Device_s_s) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_normalize_quant_grid(KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm114_arg, KernelArgTensor_bfloat16_t_1d_Device_s_s gm115_arg, KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm116_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm117_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm118_arg, int64_t arg119, int64_t arg120, int32_t arg121)
{
    __gm__ bfloat16_t* gm114 = gm114_arg.ptr;
    __gm__ bfloat16_t* gm115 = gm115_arg.ptr;
    __gm__ bfloat16_t* gm116 = gm116_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm117 = gm117_arg.ptr;
    __gm__ fp8_e8m0_t* gm118 = gm118_arg.ptr;
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

    __ubuf__ bfloat16_t* buf122 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ bfloat16_t* buf123 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(2560);
    __ubuf__ bfloat16_t* buf124 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(5120);
    __ubuf__ float* buf125 = (__ubuf__ float*)(uintptr_t)(uint64_t)(7680);
    __ubuf__ fp8_e4m3fn_t* buf126 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(13056);
    __ubuf__ uint8_t* buf127 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(14336);

    vector_bool vmask;
    for (int64_t i128 = arg119; i128 < 5; i128 = (int64_t)((uint64_t)i128 + (uint64_t)arg120)) {
        int64_t t129 = (int64_t)((uint64_t)i128 * (uint64_t)16);
        int64_t t130 = (int64_t)arg121;
        int64_t t131 = (int64_t)((uint64_t)t130 - (uint64_t)t129);
        int64_t t132 = (t131 < 16 ? t131 : 16);
        int64_t t133 = (t132 > 0 ? t132 : 0);
        int64_t t134 = (t129 < t130 ? t129 : t130);
        asc_lock(PIPE_MTE2, 1);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf123 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm115 + 0), 1, 2560, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 1);
        for (int64_t i135 = 0; i135 < 16; i135 = (int64_t)((uint64_t)i135 + (uint64_t)1)) {
            kernel_normalize_quant_grid_vf0((__ubuf__ float*)buf125);
            bool cmp136 = i135 < t133;
            if (cmp136) {
                int64_t t137 = (i135 < t133 ? i135 : t133);
                int64_t t138 = (int64_t)((uint64_t)t133 - (uint64_t)t137);
                int64_t t139 = (t138 < 1 ? t138 : 1);
                bool cmp140 = i135 <= t133;
                int64_t sel141 = cmp140 ? t139 : 0;
                int64_t dim142 = gm114_arg.stride[0];
                int64_t t143 = (int64_t)((uint64_t)dim142 * (uint64_t)2);
                int64_t t144 = (int64_t)((uint64_t)i135 + (uint64_t)t134);
                int64_t t145 = (int64_t)((uint64_t)t144 * (uint64_t)dim142);
                int64_t t146 = (int64_t)((uint64_t)t145 * (uint64_t)2);
                asc_lock(PIPE_MTE2, 0);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf122 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm114 + t146), sel141, 2560, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t143, 2560);
                }
                asc_unlock(PIPE_MTE2, 0);
                int32_t t147 = (int32_t)i135;
                asc_lock(PIPE_V, 0);
                asc_lock(PIPE_V, 1);
                asc_lock(PIPE_V, 2);
                asc_lock(PIPE_V, 3);
                asc_lock(PIPE_V, 4);
                kernel_normalize_quant_grid_vf1((__ubuf__ bfloat16_t*)buf122, (__ubuf__ bfloat16_t*)buf123, (__ubuf__ bfloat16_t*)buf124, (__ubuf__ float*)buf125, (__ubuf__ fp8_e4m3fn_t*)buf126, t147, (__ubuf__ uint8_t*)buf127);
                asc_unlock(PIPE_V, 4);
                asc_unlock(PIPE_V, 3);
                asc_unlock(PIPE_V, 2);
                asc_unlock(PIPE_V, 1);
                asc_unlock(PIPE_V, 0);
                int64_t dim148 = gm116_arg.stride[0];
                int64_t t149 = (int64_t)((uint64_t)t144 * (uint64_t)dim148);
                int64_t t150 = (int64_t)((uint64_t)t149 * (uint64_t)16);
                int64_t t151 = (int64_t)((uint64_t)t150 / (uint64_t)8);
                int64_t t152 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm116) + (uint64_t)t151);
                __gm__ bfloat16_t* root153 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t152);
                __ubuf__ bfloat16_t* view_root154 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf124) + 0);
                int64_t dim155 = gm116_arg.stride[0];
                int64_t t156 = dim155;
                int64_t t157 = t156 * 2;
                int64_t t158 = t157;
                asc_lock(PIPE_MTE3, 2);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root153 + 0, (__ubuf__ bfloat16_t*)view_root154 + 0, 1, 2560, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 2);
            } else {
                int32_t t159 = (int32_t)i135;
                asc_lock(PIPE_V, 3);
                asc_lock(PIPE_V, 4);
                kernel_normalize_quant_grid_vf2((__ubuf__ float*)buf125, (__ubuf__ fp8_e4m3fn_t*)buf126, t159, (__ubuf__ uint8_t*)buf127);
                asc_unlock(PIPE_V, 4);
                asc_unlock(PIPE_V, 3);
            }
            int64_t t160 = (int64_t)((uint64_t)i135 + (uint64_t)t129);
            int64_t dim161 = gm117_arg.stride[0];
            int64_t t162 = (int64_t)((uint64_t)t160 * (uint64_t)dim161);
            int64_t t163 = (int64_t)((uint64_t)t162 * (uint64_t)8);
            int64_t t164 = (int64_t)((uint64_t)t163 / (uint64_t)8);
            int64_t t165 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm117) + (uint64_t)t164);
            __gm__ fp8_e4m3fn_t* root166 = (__gm__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(t165);
            __ubuf__ fp8_e4m3fn_t* view_root167 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf126) + 0);
            int64_t dim168 = gm117_arg.stride[0];
            asc_lock(PIPE_MTE3, 3);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ fp8_e4m3fn_t*)root166 + 0, (__ubuf__ fp8_e4m3fn_t*)view_root167 + 0, 1, 1280, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 3);
        }
        int64_t dim169 = gm118_arg.stride[0];
        int64_t t170 = (int64_t)((uint64_t)dim169 * (uint64_t)8);
        int64_t t171 = t170 / 8;
        int64_t t172 = (int64_t)((uint64_t)i128 * (uint64_t)5120);
        int64_t t173 = (int64_t)((uint64_t)t172 / (uint64_t)8);
        int64_t t174 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm118) + (uint64_t)t173);
        __gm__ uint8_t* root175 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t174);
        __ubuf__ uint8_t* view_root176 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf127) + 0);
        asc_lock(PIPE_MTE3, 4);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root175 + 0, (__ubuf__ uint8_t*)view_root176 + 0, 1, 640, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 4);
    }
}
