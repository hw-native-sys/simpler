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

struct KernelArgTensor_fp8_e4m3fn_t_2d {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_int32_t_2d {
    __gm__ int32_t* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_float_2d {
    __gm__ float* ptr;
    int64_t shape[2];
    int64_t stride[2];
};

struct KernelArgTensor_bfloat16_t_2d_Device_ds_ds {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_bfloat16_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_bfloat16_t_2d_Device_ds_ds(const KernelArgTensor_bfloat16_t_2d& src) {
    return {src.ptr, {src.shape[0]}, {src.stride[0]}};
}

struct KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds(const KernelArgTensor_fp8_e4m3fn_t_2d& src) {
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

struct KernelArgTensor_float_2d_Device_ss_ds {
    __gm__ float* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ss_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.stride[0]}};
}

static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__simd_vf__ inline void kernel_swa_queries_vf0(int64_t arg10, __ubuf__ int32_t* buf11, int64_t arg12, __ubuf__ fp8_e4m3fn_t* buf13, __ubuf__ bfloat16_t* buf14, __ubuf__ bfloat16_t* buf15)
{
    vector_bool vmask;
    float c16 = 0.0f;
    uint32_t rem18 = 32;
    vector_bool mask17 = asc_update_mask_b32(rem18);
    uint32_t rem20 = 128;
    vector_bool mask19 = asc_update_mask_b8(rem20);
    vector_float v21;
    asc_duplicate_scalar(v21, c16, mask17);
    for (int32_t i22 = 0; i22 < 16; i22 = (int32_t)((uint32_t)i22 + (uint32_t)1)) {
        int32_t t23 = (int32_t)arg10;
        int32_t t24 = (int32_t)((uint32_t)t23 + (uint32_t)i22);
        int32_t t25 = t24;
        vector_int32_t v26;
asc_loadalign_brc(v26, buf11 + t25);
        vector_bool m27;
        asc_ge_scalar(m27, v26, 0, mask17);
        int32_t t28 = (int32_t)arg12;
        vector_bool m29;
        asc_lt_scalar(m29, v26, t28, mask17);
        vector_bool m30;
        asc_and(m30, m27, m29, mask17);
        for (int32_t i31 = 0; i31 < 16; i31 = (int32_t)((uint32_t)i31 + (uint32_t)1)) {
            int32_t t32 = i22 * 512;
            int32_t t33 = i31 * 32;
            int32_t t34 = t32 + t33;
            int32_t t35 = t34;
            vector_uint32_t v36;
            cannbotdsl_92::cannbotdsl_loadalign_unpack4(*(vector_uint8_t*)&v36, (__ubuf__ uint8_t*)(buf13 + t35));
            vector_fp8_e4m3fn_t &r37 = (vector_fp8_e4m3fn_t &)v36;
            vector_float v38;
            asc_e4m32float(v38, *(vector_fp8_e4m3fn_t*)&r37, mask19);
            int32_t t39 = i22 * 16;
            int32_t t40 = t39 + i31;
            int32_t t41 = t40;
            vector_bfloat16_t v42;
asc_loadalign_brc(v42, buf14 + t41);
            vector_float v43;
            asc_bfloat162float(v43, *(vector_bfloat16_t*)&v42, mask17);
            vector_float v44;
            asc_mul(v44, v38, v43, mask17);
            vector_float v45;
            asc_select(v45, v44, v21, m30);
            vector_bfloat16_t v46;
            asc_float2bfloat16_rn(v46, v45, mask17, ASC_POSITION_EVEN);
            asc_storealign_pack((__ubuf__ uint32_t*)(buf15 + t35), *(vector_uint32_t*)&v46, mask17);
        }
    }
}

__simd_vf__ inline void kernel_swa_queries_vf1(__ubuf__ bfloat16_t* buf47, __ubuf__ bfloat16_t* buf48)
{
    vector_bool vmask;
    vector_bool mask49 = asc_create_mask_b16(PAT_ALL);
    for (int32_t i50 = 0; i50 < 16; i50 = (int32_t)((uint32_t)i50 + (uint32_t)1)) {
        for (int32_t i51 = 0; i51 < 512; i51 = (int32_t)((uint32_t)i51 + (uint32_t)128)) {
            int32_t t52 = i50 * 512;
            int32_t t53 = t52 + i51;
            int32_t t54 = t53;
            vector_bfloat16_t v55;
            asc_loadalign(v55, buf47 + t54);
            int32_t t56 = i50 * 16;
            int32_t t57 = i51 / 16;
            int32_t t58 = t57 * 272;
            int32_t t59 = t56 + t58;
            int32_t t60 = t59;
            asc_storealign(buf48 + t60, v55, (uint16_t)17, (uint16_t)0, mask49);
        }
    }
}

__simd_vf__ inline void kernel_swa_queries_vf2(__ubuf__ int32_t* buf61, int32_t arg62, __ubuf__ float* buf63, __ubuf__ float* buf64, __ubuf__ float* buf65, __ubuf__ bfloat16_t* buf66)
{
    vector_bool vmask;
    float c67 = 0.0441941731f;
    float c68 = -1.00000002E+30f;
    float c69 = 0.0f;
    vector_bool mask70 = asc_create_mask_b32(PAT_ALL);
    vector_bool mask71 = asc_create_mask_b16(PAT_ALL);
    vector_float v72;
    asc_duplicate_scalar(v72, c69, mask70);
    vector_float v73;
    asc_duplicate_scalar(v73, c68, mask70);
    vector_int32_t v74;
    asc_loadalign(v74, buf61 + 0);
    vector_int32_t v75;
    asc_loadalign(v75, buf61 + 64);
    vector_bool m76;
    asc_ge_scalar(m76, v74, 0, mask70);
    vector_bool m77;
    asc_lt_scalar(m77, v74, arg62, mask70);
    vector_bool m78;
    asc_and(m78, m76, m77, mask70);
    vector_bool m79;
    asc_ge_scalar(m79, v75, 0, mask70);
    vector_bool m80;
    asc_lt_scalar(m80, v75, arg62, mask70);
    vector_bool m81;
    asc_and(m81, m79, m80, mask70);
    for (int32_t i82 = 0; i82 < 32; i82 = (int32_t)((uint32_t)i82 + (uint32_t)1)) {
        int32_t t83 = i82 * 128;
        int32_t t84 = t83;
        vector_float v85;
        asc_loadalign(v85, buf63 + t84);
        vector_float v86;
        asc_mul_scalar(v86, v85, c67, mask70);
        int32_t t87 = t83 + 64;
        int32_t t88 = t87;
        vector_float v89;
        asc_loadalign(v89, buf63 + t88);
        vector_float v90;
        asc_mul_scalar(v90, v89, c67, mask70);
        vector_float v91;
        asc_select(v91, v86, v73, m78);
        vector_float v92;
        asc_select(v92, v90, v73, m81);
        vector_float v93;
        asc_max(v93, v91, v92, mask70);
        vector_float v94;
        asc_reduce_max(v94, v93, mask70);
        vector_float v95;
        asc_duplicate(v95, v94, mask70);
        vector_float v96;
        asc_sub(v96, v91, v95, mask70);
        vector_float v97;
        asc_exp(v97, v96, mask70);
        vector_float v98;
        asc_sub(v98, v92, v95, mask70);
        vector_float v99;
        asc_exp(v99, v98, mask70);
        vector_float v100;
        asc_select(v100, v97, v72, m78);
        vector_float v101;
        asc_select(v101, v99, v72, m81);
        vector_float v102;
        asc_reduce_sum(v102, v100, mask70);
        vector_float v103;
        asc_reduce_sum(v103, v101, mask70);
        vector_float v104;
        asc_add(v104, v102, v103, mask70);
        vector_float v105;
        asc_duplicate(v105, v104, mask70);
        int32_t t106 = i82;
        vector_float v107;
asc_loadalign_brc(v107, buf64 + t106);
        vector_float v108;
        asc_max(v108, v95, v107, mask70);
        vector_float v109;
        asc_sub(v109, v95, v108, mask70);
        vector_float v110;
        asc_exp(v110, v109, mask70);
        vector_float v111;
        asc_sub(v111, v107, v108, mask70);
        vector_float v112;
        asc_exp(v112, v111, mask70);
        vector_float v113;
        asc_mul(v113, v105, v110, mask70);
        vector_float v114;
        asc_add(v114, v113, v112, mask70);
        vector_float v115;
        {
            static constexpr AscendC::Reg::DivSpecificMode div_mode_116 = {
                AscendC::Reg::MaskMergeMode::ZEROING, true, AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE};
            AscendC::Reg::Div<float, &div_mode_116, AscendC::Reg::RegTensor<float>>(*(AscendC::Reg::RegTensor<float>*)&v115, *(AscendC::Reg::RegTensor<float>*)&v110, *(AscendC::Reg::RegTensor<float>*)&v114, mask70);
        }
        asc_storealign_1st(buf65 + t106, v115);
        vector_bfloat16_t v117;
        asc_float2bfloat16_rn(v117, v100, mask70, ASC_POSITION_EVEN);
        vector_bfloat16_t v118;
        asc_float2bfloat16_rn(v118, v101, mask70, ASC_POSITION_EVEN);
        vector_bfloat16_t v119;
        vector_bfloat16_t v120;
        asc_deintlv(v119, v120, v117, v118);
        int32_t t121 = i82 * 16;
        int32_t t122 = t121;
        asc_storealign(buf66 + t122, v119, (uint16_t)33, (uint16_t)0, mask71);
    }
}

__simd_vf__ inline void kernel_swa_queries_vf3(__ubuf__ float* buf123, __ubuf__ float* buf124, __ubuf__ bfloat16_t* buf125)
{
    vector_bool vmask;
    vector_bool mask126 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i127 = 0; i127 < 32; i127 = (int32_t)((uint32_t)i127 + (uint32_t)1)) {
        int32_t t128 = i127;
        vector_float v129;
asc_loadalign_brc(v129, buf123 + t128);
        for (int32_t i130 = 0; i130 < 128; i130 = (int32_t)((uint32_t)i130 + (uint32_t)64)) {
            int32_t t131 = i127 * 128;
            int32_t t132 = t131 + i130;
            int32_t t133 = t132;
            vector_float v134;
            asc_loadalign(v134, buf124 + t133);
            vector_float v135;
            asc_mul(v135, v134, v129, mask126);
            vector_bfloat16_t v136;
            asc_float2bfloat16_rn(v136, v135, mask126, ASC_POSITION_EVEN);
            asc_storealign_pack((__ubuf__ uint32_t*)(buf125 + t133), *(vector_uint32_t*)&v136, mask126);
        }
    }
}

__global__ __aicore__ void kernel_swa_queries(KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm0_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds gm1_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm2_arg, KernelArgTensor_int32_t_2d_Device_ds_ds gm3_arg, KernelArgTensor_float_2d_Device_ss_ds gm4_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm5_arg, int64_t arg6, int64_t arg7, int64_t arg8, int32_t arg9)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ bfloat16_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm1 = gm1_arg.ptr;
    __gm__ bfloat16_t* gm2 = gm2_arg.ptr;
    __gm__ int32_t* gm3 = gm3_arg.ptr;
    __gm__ float* gm4 = gm4_arg.ptr;
    __gm__ bfloat16_t* gm5 = gm5_arg.ptr;
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

    __ubuf__ float* buf137 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf138 = (__ubuf__ float*)(uintptr_t)(uint64_t)(16384);
    __cc__ float* buf139 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ bfloat16_t* buf140 = (__ca__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cb__ bfloat16_t* buf141 = (__cb__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ bfloat16_t* buf142 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(65536);
    __cbuf__ bfloat16_t* buf143 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(196608);
    __cbuf__ bfloat16_t* buf144 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf145 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf146 = (__ubuf__ float*)(uintptr_t)(uint64_t)(16384);
    __ubuf__ int32_t* buf147 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(32768);
    __ubuf__ fp8_e4m3fn_t* buf148 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(33280);
    __ubuf__ bfloat16_t* buf149 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(41984);
    __ubuf__ bfloat16_t* buf150 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(42496);
    __ubuf__ bfloat16_t* buf151 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(58880);
    __ubuf__ bfloat16_t* buf152 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(76288);
    __ubuf__ float* buf153 = (__ubuf__ float*)(uintptr_t)(uint64_t)(84736);
    __ubuf__ float* buf154 = (__ubuf__ float*)(uintptr_t)(uint64_t)(84864);
    __ubuf__ bfloat16_t* buf155 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(84992);
    __cbuf__ bfloat16_t* buf156 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(65536);
    __cbuf__ bfloat16_t* buf157 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(196608);

    if ASCEND_IS_AIC {
        asc_sync_intra_arrive(PIPE_MTE1, 1);
        asc_sync_intra_arrive(PIPE_MTE1, 17);
        asc_sync_intra_arrive(PIPE_MTE1, 0);
        asc_sync_intra_arrive(PIPE_MTE1, 16);
        int64_t t158 = (int64_t)arg9;
        for (int64_t i159 = arg6; i159 < t158; i159 = (int64_t)((uint64_t)i159 + (uint64_t)arg7)) {
            __cbuf__ bfloat16_t* view_root160 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf144) + 0);
            int64_t t161 = (int64_t)((uint64_t)i159 * (uint64_t)4);
            int64_t dim162 = gm0_arg.stride[0];
            int64_t t163 = (int64_t)((uint64_t)t161 * (uint64_t)dim162);
            int64_t t164 = (int64_t)((uint64_t)t163 * (uint64_t)16);
            int64_t dim165 = gm0_arg.shape[0];
            int64_t t166 = dim165 / 64;
            int64_t t167 = (i159 < t166 ? i159 : t166);
            int64_t t168 = (int64_t)((uint64_t)t167 * (uint64_t)64);
            int64_t t169 = (int64_t)((uint64_t)dim165 - (uint64_t)t168);
            int64_t t170 = (t169 < 64 ? t169 : 64);
            bool cmp171 = i159 <= t166;
            int64_t sel172 = cmp171 ? t170 : 0;
            int64_t dim173 = gm0_arg.stride[0];
            int64_t t174 = dim173;
            int64_t t175 = t174 * 2;
            int64_t t176 = t175;
            asc_lock(PIPE_MTE2, 0);
            bool cmp177 = (uint64_t)sel172 > (uint64_t)0;
            if (cmp177) {
                int64_t t178 = t164;
                int32_t t179 = (int32_t)sel172;
                set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(64) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                asc_copy_gm2l1_nd2nz(view_root160 + 0, gm0 + t178, t176, 0, t179, 512, 0, false);
            }
            asc_unlock(PIPE_MTE2, 0);
            asc_sync_intra_wait(PIPE_MTE1, 0);
            asc_sync_intra_wait(PIPE_MTE1, 16);
            for (int64_t i180 = 0; i180 < 4; i180 = (int64_t)((uint64_t)i180 + (uint64_t)1)) {
                __ca__ bfloat16_t* view_root181 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf140) + 0);
                int64_t t182 = i180 * 8;
                asc_lock(PIPE_MTE1, 1);
                asc_lock(PIPE_MTE1, 0);
                int32_t t183 = (int32_t)t182;
                asc_copy_l12l0a(view_root181 + 0, view_root160 + 0, 0, t183, 4, 8, 4, 4);
                asc_unlock(PIPE_MTE1, 0);
                asc_unlock(PIPE_MTE1, 1);
                __cb__ bfloat16_t* view_root184 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf141) + 0);
                __cbuf__ bfloat16_t* view_root185 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf142) + 0);
                asc_lock(PIPE_MTE1, 2);
                asc_copy_l12l0b(view_root184 + 0, view_root185 + 0, 0, t183, 8, 8, 8, 8);
                asc_unlock(PIPE_MTE1, 2);
                bool cmp186 = i180 == 0;
                __cc__ float* view_root187 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf139) + 0);
                asc_lock(PIPE_M, 3);
                asc_lock(PIPE_M, 1);
                asc_lock(PIPE_M, 2);
                if (cmp186) {
                    asc_mmad(view_root187 + 0, view_root181, view_root184, 64, 128, 128, 0, true, false, true);
                } else {
                    asc_mmad(view_root187 + 0, view_root181, view_root184, 64, 128, 128, 0, true, false, false);
                }
                asc_unlock(PIPE_M, 2);
                asc_unlock(PIPE_M, 1);
                asc_unlock(PIPE_M, 3);
            }
            __ubuf__ float* view_root188 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf137) + 0);
            __cc__ float* view_root189 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf139) + 0);
            asc_sync_intra_wait(PIPE_FIX, 2);
            asc_sync_intra_wait(PIPE_FIX, 18);
            asc_lock(PIPE_FIX, 3);
            asc_set_l0c2gm_nz2nd(1, 0, 0);
            asc_copy_l0c2ub((__ubuf__ float*)view_root188 + 0, view_root189, 128, 64, 128, 64, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
            asc_unlock(PIPE_FIX, 3);
            asc_sync_intra_arrive(PIPE_FIX, 2);
            asc_sync_intra_arrive(PIPE_FIX, 18);
            __cbuf__ bfloat16_t* view_root190 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf143) + 0);
            asc_sync_intra_wait(PIPE_MTE1, 1);
            asc_sync_intra_wait(PIPE_MTE1, 17);
            for (int64_t i191 = 0; i191 < 4; i191 = (int64_t)((uint64_t)i191 + (uint64_t)1)) {
                __ca__ bfloat16_t* view_root192 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf140) + 0);
                asc_lock(PIPE_MTE1, 1);
                asc_copy_l12l0a(view_root192 + 0, view_root190 + 0, 0, 0, 4, 8, 4, 4);
                asc_unlock(PIPE_MTE1, 1);
                __cb__ bfloat16_t* view_root193 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf141) + 0);
                __cbuf__ bfloat16_t* view_root194 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf142) + 0);
                int64_t t195 = i191 * 8;
                asc_lock(PIPE_MTE1, 2);
                int32_t t196 = (int32_t)t195;
                asc_copy_l12l0b_transpose(view_root193 + 0, view_root194 + 0, 0, t196, 8, 8, 8, 8);
                asc_unlock(PIPE_MTE1, 2);
                asc_lock(PIPE_M, 3);
                asc_lock(PIPE_M, 1);
                asc_lock(PIPE_M, 2);
                asc_mmad(view_root189 + 0, view_root192, view_root193, 64, 128, 128, 0, true, false, true);
                asc_unlock(PIPE_M, 2);
                asc_unlock(PIPE_M, 1);
                asc_unlock(PIPE_M, 3);
                __ubuf__ float* view_root197 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf138) + 0);
                asc_sync_intra_wait(PIPE_FIX, 3);
                asc_sync_intra_wait(PIPE_FIX, 19);
                asc_lock(PIPE_FIX, 3);
                asc_set_l0c2gm_nz2nd(1, 0, 0);
                asc_copy_l0c2ub((__ubuf__ float*)view_root197 + 0, view_root189, 128, 64, 128, 64, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                asc_unlock(PIPE_FIX, 3);
                asc_sync_intra_arrive(PIPE_FIX, 3);
                asc_sync_intra_arrive(PIPE_FIX, 19);
            }
            asc_sync_intra_arrive(PIPE_MTE1, 0);
            asc_sync_intra_arrive(PIPE_MTE1, 16);
            asc_sync_intra_arrive(PIPE_MTE1, 1);
            asc_sync_intra_arrive(PIPE_MTE1, 17);
        }
        asc_sync_intra_wait(PIPE_FIX, 2);
        asc_sync_intra_wait(PIPE_FIX, 18);
        asc_sync_intra_wait(PIPE_FIX, 3);
        asc_sync_intra_wait(PIPE_FIX, 19);
    }

    if ASCEND_IS_AIV {
        asc_sync_intra_arrive(PIPE_V, 3);
        asc_sync_intra_arrive(PIPE_V, 2);
        int64_t t198 = (int64_t)arg9;
        for (int64_t i199 = arg6; i199 < t198; i199 = (int64_t)((uint64_t)i199 + (uint64_t)arg7)) {
            int64_t dim200 = gm1_arg.shape[0];
            int64_t dim201 = gm3_arg.stride[0];
            int64_t t202 = (int64_t)((uint64_t)i199 * (uint64_t)dim201);
            int64_t t203 = (int64_t)((uint64_t)t202 * (uint64_t)4);
            asc_lock(PIPE_MTE2, 0);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf147 + 0), (__gm__ int32_t*)((__gm__ uint8_t*)gm3 + t203), 1, 512, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
            }
            asc_unlock(PIPE_MTE2, 0);
            int64_t t204 = (int64_t)((uint64_t)arg8 * (uint64_t)128);
            asc_lock(PIPE_MTE2, 5);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf154 + 0), (__gm__ float*)((__gm__ uint8_t*)gm4 + t204), 1, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
            }
            asc_unlock(PIPE_MTE2, 5);
            asc_sync_intra_wait(PIPE_MTE3, 0);
            for (int64_t i205 = 0; i205 < 4; i205 = (int64_t)((uint64_t)i205 + (uint64_t)1)) {
                for (int64_t i206 = 0; i206 < 16; i206 = (int64_t)((uint64_t)i206 + (uint64_t)1)) {
                    int64_t t207 = (int64_t)((uint64_t)arg8 * (uint64_t)64);
                    int64_t t208 = i205 * 16;
                    int64_t t209 = (int64_t)((uint64_t)t207 + (uint64_t)t208);
                    int64_t t210 = (int64_t)((uint64_t)t209 + (uint64_t)i206);
                    int64_t t211 = t202;
                    int64_t t212 = t210;
                    int64_t t213 = t211 + t212;
                    int32_t ld214 = gm3[t213];
                    int64_t t215 = (int64_t)((uint64_t)dim200 - (uint64_t)1);
                    int64_t t216 = (int64_t)ld214;
                    int64_t t217 = (t216 < t215 ? t216 : t215);
                    int64_t t218 = (t217 > 0 ? t217 : 0);
                    int64_t t219 = (t218 < dim200 ? t218 : dim200);
                    int64_t t220 = (int64_t)((uint64_t)dim200 - (uint64_t)t219);
                    int64_t t221 = (t220 < 1 ? t220 : 1);
                    bool cmp222 = t218 <= dim200;
                    int64_t sel223 = cmp222 ? t221 : 0;
                    int64_t dim224 = gm1_arg.stride[0];
                    int64_t t225 = (int64_t)((uint64_t)t218 * (uint64_t)dim224);
                    int64_t t226 = i206 * 512;
                    asc_lock(PIPE_MTE2, 1);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ fp8_e4m3fn_t*)((__ubuf__ uint8_t*)buf148 + t226), (__gm__ fp8_e4m3fn_t*)((__gm__ uint8_t*)gm1 + t225), sel223, 512, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)dim224, 512);
                    }
                    asc_unlock(PIPE_MTE2, 1);
                    int64_t dim227 = gm2_arg.shape[0];
                    int64_t t228 = (t218 < dim227 ? t218 : dim227);
                    int64_t t229 = (int64_t)((uint64_t)dim227 - (uint64_t)t228);
                    int64_t t230 = (t229 < 1 ? t229 : 1);
                    bool cmp231 = t218 <= dim227;
                    int64_t sel232 = cmp231 ? t230 : 0;
                    int64_t dim233 = gm2_arg.stride[0];
                    int64_t t234 = (int64_t)((uint64_t)dim233 * (uint64_t)2);
                    int64_t t235 = (int64_t)((uint64_t)t218 * (uint64_t)dim233);
                    int64_t t236 = (int64_t)((uint64_t)t235 * (uint64_t)2);
                    int64_t t237 = i206 * 32;
                    asc_lock(PIPE_MTE2, 2);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf149 + t237), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm2 + t236), sel232, 32, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t234, 32);
                    }
                    asc_unlock(PIPE_MTE2, 2);
                }
                int64_t t238 = (int64_t)((uint64_t)arg8 * (uint64_t)64);
                int64_t t239 = i205 * 16;
                int64_t t240 = (int64_t)((uint64_t)t238 + (uint64_t)t239);
                asc_lock(PIPE_V, 0);
                asc_lock(PIPE_V, 1);
                asc_lock(PIPE_V, 2);
                kernel_swa_queries_vf0(t240, (__ubuf__ int32_t*)buf147, dim200, (__ubuf__ fp8_e4m3fn_t*)buf148, (__ubuf__ bfloat16_t*)buf149, (__ubuf__ bfloat16_t*)buf150);
                asc_unlock(PIPE_V, 2);
                asc_unlock(PIPE_V, 1);
                asc_unlock(PIPE_V, 0);
                asc_lock(PIPE_V, 3);
                kernel_swa_queries_vf1((__ubuf__ bfloat16_t*)buf150, (__ubuf__ bfloat16_t*)buf151);
                asc_unlock(PIPE_V, 3);
                int64_t t241 = (int64_t)((uint64_t)arg8 * (uint64_t)4);
                int64_t t242 = (int64_t)((uint64_t)t241 + (uint64_t)i205);
                __ubuf__ bfloat16_t* view_root243 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf151) + 0);
                __cbuf__ bfloat16_t* view_root244 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf156) + 0);
                int64_t t245 = (int64_t)((uint64_t)t242 * (uint64_t)256);
                asc_lock(PIPE_MTE3, 3);
                int64_t t246 = t245;
                asc_copy_ub2l1(view_root244 + t246, (__ubuf__ bfloat16_t*)view_root243 + 0, 32, 16, 1, 112);
                asc_unlock(PIPE_MTE3, 3);
            }
            asc_sync_intra_arrive(PIPE_MTE3, 0);
            int32_t t247 = (int32_t)dim200;
            asc_sync_intra_wait(PIPE_V, 2);
            asc_lock(PIPE_V, 0);
            asc_lock(PIPE_V, 5);
            asc_lock(PIPE_V, 4);
            kernel_swa_queries_vf2((__ubuf__ int32_t*)buf147, t247, (__ubuf__ float*)buf145, (__ubuf__ float*)buf154, (__ubuf__ float*)buf153, (__ubuf__ bfloat16_t*)buf152);
            asc_unlock(PIPE_V, 4);
            asc_unlock(PIPE_V, 5);
            asc_unlock(PIPE_V, 0);
            asc_sync_intra_arrive(PIPE_V, 2);
            __ubuf__ bfloat16_t* view_root248 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf152) + 0);
            __cbuf__ bfloat16_t* view_root249 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf157) + 0);
            int64_t t250 = (int64_t)((uint64_t)arg8 * (uint64_t)32);
            int64_t t251 = t250;
            int64_t t252 = t251 * 16;
            asc_sync_intra_wait(PIPE_MTE3, 1);
            asc_lock(PIPE_MTE3, 4);
            asc_copy_ub2l1(view_root249 + t252, (__ubuf__ bfloat16_t*)view_root248 + 0, 8, 32, 1, 32);
            asc_unlock(PIPE_MTE3, 4);
            asc_sync_intra_arrive(PIPE_MTE3, 1);
            for (int64_t i253 = 0; i253 < 4; i253 = (int64_t)((uint64_t)i253 + (uint64_t)1)) {
                asc_sync_intra_wait(PIPE_V, 3);
                asc_lock(PIPE_V, 6);
                kernel_swa_queries_vf3((__ubuf__ float*)buf153, (__ubuf__ float*)buf146, (__ubuf__ bfloat16_t*)buf155);
                asc_unlock(PIPE_V, 6);
                asc_sync_intra_arrive(PIPE_V, 3);
                int64_t t254 = (int64_t)((uint64_t)i199 * (uint64_t)64);
                int64_t dim255 = gm5_arg.stride[0];
                int64_t t256 = (int64_t)((uint64_t)t254 * (uint64_t)dim255);
                int64_t t257 = i253 * 128;
                int64_t t258 = (int64_t)((uint64_t)t256 + (uint64_t)t257);
                int64_t t259 = (int64_t)((uint64_t)t258 * (uint64_t)16);
                int64_t t260 = (int64_t)((uint64_t)t259 / (uint64_t)8);
                int64_t t261 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm5) + (uint64_t)t260);
                __gm__ bfloat16_t* root262 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t261);
                __ubuf__ bfloat16_t* view_root263 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf155) + 0);
                int64_t dim264 = gm5_arg.stride[0];
                int64_t t265 = dim264;
                int64_t dim266 = gm5_arg.shape[0];
                int64_t t267 = dim266 / 64;
                int64_t t268 = (i199 < t267 ? i199 : t267);
                int64_t t269 = (int64_t)((uint64_t)t268 * (uint64_t)64);
                int64_t t270 = (int64_t)((uint64_t)dim266 - (uint64_t)t269);
                int64_t t271 = (t270 < 64 ? t270 : 64);
                bool cmp272 = i199 <= t267;
                int64_t sel273 = cmp272 ? t271 : 0;
                int64_t t274 = sel273 / 2;
                int64_t t275 = sel273 % 2;
                int64_t t276 = t275 + 1;
                int64_t t277 = t276 / 2;
                int64_t t278 = (int64_t)((uint64_t)t274 + (uint64_t)t277);
                int64_t t279 = (int64_t)((uint64_t)arg8 * (uint64_t)t278);
                int64_t t280 = t279;
                int64_t t281 = t280 * t265;
                int64_t t282 = t265 * 2;
                int64_t t283 = t282;
                asc_lock(PIPE_MTE3, 6);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root262 + t281, (__ubuf__ bfloat16_t*)view_root263 + 0, 32, 256, static_cast<uint8_t>(0), (uint64_t)t283, 256);
                asc_unlock(PIPE_MTE3, 6);
            }
        }
        asc_sync_intra_wait(PIPE_MTE3, 0);
        asc_sync_intra_wait(PIPE_MTE3, 1);
    }
}
