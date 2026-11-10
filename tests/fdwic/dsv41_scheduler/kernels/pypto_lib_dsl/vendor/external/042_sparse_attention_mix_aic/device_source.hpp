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

struct KernelArgTensor_uint8_t_2d {
    __gm__ uint8_t* ptr;
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

struct KernelArgTensor_uint8_t_2d_Device_ds_ds {
    __gm__ uint8_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};
static inline KernelArgTensor_uint8_t_2d_Device_ds_ds cannbotdsl_pack_KernelArgTensor_uint8_t_2d_Device_ds_ds(const KernelArgTensor_uint8_t_2d& src) {
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
static_assert(sizeof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_uint8_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_int32_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 24 && alignof(KernelArgTensor_bfloat16_t_2d_Device_ds_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int32_t) == 4 && alignof(int32_t) == 4, "device argument ABI mismatch");
__simd_vf__ inline void kernel_sparse_attention_mix_vf0(__ubuf__ float* buf13, __ubuf__ float* buf14)
{
    vector_bool vmask;
    float c15 = -1.00000002E+30f;
    float c16 = 0.0f;
    uint32_t rem18 = 32;
    vector_bool mask17 = asc_update_mask_b32(rem18);
    vector_bool mask19 = asc_create_mask_b32(PAT_ALL);
    vector_float v20;
    asc_duplicate_scalar(v20, c16, mask19);
    vector_float v21;
    asc_duplicate_scalar(v21, c15, mask17);
    asc_storealign(buf13 + 0, v21, mask17);
    asc_storealign(buf13 + 32, v20, mask17);
    for (int32_t i22 = 0; i22 < 256; i22 = (int32_t)((uint32_t)i22 + (uint32_t)1)) {
        int32_t t23 = i22 * 64;
        int32_t t24 = t23;
        asc_storealign(buf14 + t24, v20, mask19);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_sparse_attention_mix_vf1(int64_t arg25, __ubuf__ int32_t* buf26, int64_t arg27, __ubuf__ fp8_e4m3fn_t* buf28, __ubuf__ bfloat16_t* buf29, __ubuf__ bfloat16_t* buf30)
{
    vector_bool vmask;
    float c31 = 0.0f;
    uint32_t rem33 = 32;
    vector_bool mask32 = asc_update_mask_b32(rem33);
    uint32_t rem35 = 128;
    vector_bool mask34 = asc_update_mask_b8(rem35);
    vector_float v36;
    asc_duplicate_scalar(v36, c31, mask32);
    for (int32_t i37 = 0; i37 < 16; i37 = (int32_t)((uint32_t)i37 + (uint32_t)1)) {
        int32_t t38 = (int32_t)arg25;
        int32_t t39 = (int32_t)((uint32_t)t38 + (uint32_t)i37);
        int32_t t40 = t39;
        vector_int32_t v41;
asc_loadalign_brc(v41, buf26 + t40);
        vector_bool m42;
        asc_ge_scalar(m42, v41, 0, mask32);
        int32_t t43 = (int32_t)arg27;
        vector_bool m44;
        asc_lt_scalar(m44, v41, t43, mask32);
        vector_bool m45;
        asc_and(m45, m42, m44, mask32);
        for (int32_t i46 = 0; i46 < 16; i46 = (int32_t)((uint32_t)i46 + (uint32_t)1)) {
            int32_t t47 = i37 * 512;
            int32_t t48 = i46 * 32;
            int32_t t49 = t47 + t48;
            int32_t t50 = t49;
            vector_uint32_t v51;
            cannbotdsl_92::cannbotdsl_loadalign_unpack4(*(vector_uint8_t*)&v51, (__ubuf__ uint8_t*)(buf28 + t50));
            vector_fp8_e4m3fn_t &r52 = (vector_fp8_e4m3fn_t &)v51;
            vector_float v53;
            asc_e4m32float(v53, *(vector_fp8_e4m3fn_t*)&r52, mask34);
            int32_t t54 = i37 * 16;
            int32_t t55 = t54 + i46;
            int32_t t56 = t55;
            vector_bfloat16_t v57;
asc_loadalign_brc(v57, buf29 + t56);
            vector_float v58;
            asc_bfloat162float(v58, *(vector_bfloat16_t*)&v57, mask32);
            vector_float v59;
            asc_mul(v59, v53, v58, mask32);
            vector_float v60;
            asc_select(v60, v59, v36, m45);
            vector_bfloat16_t v61;
            asc_float2bfloat16_rn(v61, v60, mask32, ASC_POSITION_EVEN);
            asc_storealign_pack((__ubuf__ uint32_t*)(buf30 + t50), *(vector_uint32_t*)&v61, mask32);
        }
    }
}

__simd_vf__ inline void kernel_sparse_attention_mix_vf2(__ubuf__ bfloat16_t* buf62, __ubuf__ bfloat16_t* buf63)
{
    vector_bool vmask;
    vector_bool mask64 = asc_create_mask_b16(PAT_ALL);
    for (int32_t i65 = 0; i65 < 16; i65 = (int32_t)((uint32_t)i65 + (uint32_t)1)) {
        for (int32_t i66 = 0; i66 < 512; i66 = (int32_t)((uint32_t)i66 + (uint32_t)128)) {
            int32_t t67 = i65 * 512;
            int32_t t68 = t67 + i66;
            int32_t t69 = t68;
            vector_bfloat16_t v70;
            asc_loadalign(v70, buf62 + t69);
            int32_t t71 = i65 * 16;
            int32_t t72 = i66 / 16;
            int32_t t73 = t72 * 272;
            int32_t t74 = t71 + t73;
            int32_t t75 = t74;
            asc_storealign(buf63 + t75, v70, (uint16_t)17, (uint16_t)0, mask64);
        }
    }
}

__simd_vf__ inline void kernel_sparse_attention_mix_vf3(int64_t arg76, __ubuf__ int32_t* buf77, int64_t arg78, __ubuf__ uint8_t* buf79, __ubuf__ bfloat16_t* buf80, __ubuf__ bfloat16_t* buf81)
{
    vector_bool vmask;
    bfloat16_t c82 = __builtin_bit_cast(bfloat16_t, (uint16_t)0);
    vector_bool mask83 = asc_create_mask_b8(PAT_ALL);
    vector_bool mask84 = asc_create_mask_b16(PAT_ALL);
    vector_bool mask85 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t v86;
    asc_duplicate_scalar(v86, c82, mask84);
    for (int32_t i87 = 0; i87 < 16; i87 = (int32_t)((uint32_t)i87 + (uint32_t)1)) {
        int32_t t88 = (int32_t)arg76;
        int32_t t89 = (int32_t)((uint32_t)t88 + (uint32_t)i87);
        int32_t t90 = t89;
        vector_int32_t v91;
asc_loadalign_brc(v91, buf77 + t90);
        vector_bool m92;
        asc_ge_scalar(m92, v91, 0, mask85);
        int32_t t93 = (int32_t)arg78;
        vector_bool m94;
        asc_lt_scalar(m94, v91, t93, mask85);
        vector_bool m95;
        asc_and(m95, m92, m94, mask85);
        vector_uint32_t v96;
        asc_duplicate_scalar(v96, 1);
        vector_uint32_t v97;
        asc_duplicate_scalar(v97, 0);
        vector_uint32_t v98;
        asc_select(v98, v96, v97, m95);
        vector_uint16_t &r99 = (vector_uint16_t &)v98;
        vector_uint16_t v100;
        asc_duplicate(v100, r99, mask84);
        vector_bool m101;
        asc_eq_scalar(m101, v100, 1, mask84);
        for (int32_t i102 = 0; i102 < 512; i102 = (int32_t)((uint32_t)i102 + (uint32_t)128)) {
            int32_t t103 = i87 * 256;
            int32_t t104 = i102 / 2;
            int32_t t105 = t103 + t104;
            int32_t t106 = t105;
            vector_uint32_t v107;
            cannbotdsl_92::cannbotdsl_loadalign_unpack4(*(vector_uint8_t*)&v107, (__ubuf__ uint8_t*)(buf79 + t106));
            vector_fp4x2_e2m1_t &r108 = (vector_fp4x2_e2m1_t &)v107;
            vector_bfloat16_t v109;
            asc_e2m1x22bfloat16(v109, *(vector_fp4x2_e2m1_t*)&r108, mask83, ASC_DISPERSE_FIRST_QUARTER);
            int32_t t110 = i87 * 32;
            int32_t t111 = i102 / 16;
            int32_t t112 = t110 + t111;
            int32_t t113 = t112;
            vector_uint16_t v114;
            vlds(v114, (__ubuf__ uint16_t*)(buf80 + t113), 0, E2B_B16);
            vector_bfloat16_t &r115 = (vector_bfloat16_t &)v114;
            vector_bfloat16_t v116;
            asc_mul(v116, v109, r115, mask84);
            vector_bfloat16_t v117;
            asc_select(v117, v116, v86, m101);
            int32_t t118 = i87 * 16;
            int32_t t119 = t111 * 272;
            int32_t t120 = t118 + t119;
            int32_t t121 = t120;
            asc_storealign(buf81 + t121, v117, (uint16_t)17, (uint16_t)0, mask84);
        }
    }
}

__simd_vf__ inline void kernel_sparse_attention_mix_vf4(__ubuf__ int32_t* buf122, int32_t arg123, __ubuf__ float* buf124, __ubuf__ float* buf125, __ubuf__ bfloat16_t* buf126)
{
    vector_bool vmask;
    float c127 = 0.0441941731f;
    float c128 = -1.00000002E+30f;
    float c129 = 0.0f;
    vector_bool mask130 = asc_create_mask_b32(PAT_ALL);
    vector_bool mask131 = asc_create_mask_b16(PAT_ALL);
    vector_float v132;
    asc_duplicate_scalar(v132, c129, mask130);
    vector_float v133;
    asc_duplicate_scalar(v133, c128, mask130);
    vector_int32_t v134;
    asc_loadalign(v134, buf122 + 0);
    vector_int32_t v135;
    asc_loadalign(v135, buf122 + 64);
    vector_bool m136;
    asc_ge_scalar(m136, v134, 0, mask130);
    vector_bool m137;
    asc_lt_scalar(m137, v134, arg123, mask130);
    vector_bool m138;
    asc_and(m138, m136, m137, mask130);
    vector_bool m139;
    asc_ge_scalar(m139, v135, 0, mask130);
    vector_bool m140;
    asc_lt_scalar(m140, v135, arg123, mask130);
    vector_bool m141;
    asc_and(m141, m139, m140, mask130);
    for (int32_t i142 = 0; i142 < 32; i142 = (int32_t)((uint32_t)i142 + (uint32_t)1)) {
        int32_t t143 = i142 * 128;
        int32_t t144 = t143;
        vector_float v145;
        asc_loadalign(v145, buf124 + t144);
        vector_float v146;
        asc_mul_scalar(v146, v145, c127, mask130);
        int32_t t147 = t143 + 64;
        int32_t t148 = t147;
        vector_float v149;
        asc_loadalign(v149, buf124 + t148);
        vector_float v150;
        asc_mul_scalar(v150, v149, c127, mask130);
        vector_float v151;
        asc_select(v151, v146, v133, m138);
        vector_float v152;
        asc_select(v152, v150, v133, m141);
        vector_float v153;
        asc_max(v153, v151, v152, mask130);
        vector_float v154;
        asc_reduce_max(v154, v153, mask130);
        vector_float v155;
        asc_duplicate(v155, v154, mask130);
        int32_t t156 = i142;
        vector_float v157;
asc_loadalign_brc(v157, buf125 + t156);
        vector_float v158;
        asc_max(v158, v155, v157, mask130);
        vector_float v159;
        asc_sub(v159, v157, v158, mask130);
        vector_float v160;
        asc_exp(v160, v159, mask130);
        vector_float v161;
        asc_sub(v161, v151, v158, mask130);
        vector_float v162;
        asc_exp(v162, v161, mask130);
        vector_float v163;
        asc_sub(v163, v152, v158, mask130);
        vector_float v164;
        asc_exp(v164, v163, mask130);
        vector_float v165;
        asc_select(v165, v162, v132, m138);
        vector_float v166;
        asc_select(v166, v164, v132, m141);
        vector_float v167;
        asc_reduce_sum(v167, v165, mask130);
        vector_float v168;
        asc_reduce_sum(v168, v166, mask130);
        vector_float v169;
        asc_add(v169, v167, v168, mask130);
        vector_float v170;
        asc_duplicate(v170, v169, mask130);
        int32_t t171 = i142 + 32;
        int32_t t172 = t171;
        vector_float v173;
asc_loadalign_brc(v173, buf125 + t172);
        vector_float v174;
        asc_mul(v174, v173, v160, mask130);
        vector_float v175;
        asc_add(v175, v174, v170, mask130);
        asc_storealign_1st(buf125 + t156, v158);
        asc_storealign_1st(buf125 + t172, v175);
        int32_t t176 = i142 + 64;
        int32_t t177 = t176;
        asc_storealign_1st(buf125 + t177, v160);
        vector_bfloat16_t v178;
        asc_float2bfloat16_rn(v178, v165, mask130, ASC_POSITION_EVEN);
        vector_bfloat16_t v179;
        asc_float2bfloat16_rn(v179, v166, mask130, ASC_POSITION_EVEN);
        vector_bfloat16_t v180;
        vector_bfloat16_t v181;
        asc_deintlv(v180, v181, v178, v179);
        int32_t t182 = i142 * 16;
        int32_t t183 = t182;
        asc_storealign(buf126 + t183, v180, (uint16_t)33, (uint16_t)0, mask131);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_sparse_attention_mix_vf5(__ubuf__ float* buf184, int64_t arg185, __ubuf__ float* buf186, __ubuf__ float* buf187)
{
    vector_bool vmask;
    vector_bool mask188 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i189 = 0; i189 < 32; i189 = (int32_t)((uint32_t)i189 + (uint32_t)1)) {
        int32_t t190 = i189 + 64;
        int32_t t191 = t190;
        vector_float v192;
asc_loadalign_brc(v192, buf184 + t191);
        for (int32_t i193 = 0; i193 < 128; i193 = (int32_t)((uint32_t)i193 + (uint32_t)64)) {
            int32_t t194 = i189 * 512;
            int32_t t195 = (int32_t)arg185;
            int32_t t196 = (int32_t)((uint32_t)t195 * (uint32_t)128);
            int32_t t197 = (int32_t)((uint32_t)t194 + (uint32_t)t196);
            int32_t t198 = (int32_t)((uint32_t)t197 + (uint32_t)i193);
            int32_t t199 = t198;
            vector_float v200;
            asc_loadalign(v200, buf186 + t199);
            int32_t t201 = i189 * 128;
            int32_t t202 = t201 + i193;
            int32_t t203 = t202;
            vector_float v204;
            asc_loadalign(v204, buf187 + t203);
            vector_float v205;
            asc_mul(v205, v200, v192, mask188);
            vector_float v206;
            asc_add(v206, v205, v204, mask188);
            asc_storealign(buf186 + t199, v206, mask188);
        }
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_sparse_attention_mix_vf6(__ubuf__ float* buf207, __ubuf__ float* buf208, int64_t arg209, __ubuf__ float* buf210, __ubuf__ bfloat16_t* buf211)
{
    vector_bool vmask;
    vector_bool mask212 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i213 = 0; i213 < 32; i213 = (int32_t)((uint32_t)i213 + (uint32_t)1)) {
        int32_t t214 = i213;
        vector_float v215;
asc_loadalign_brc(v215, buf207 + t214);
        int32_t t216 = i213 + 32;
        int32_t t217 = t216;
        vector_float v218;
asc_loadalign_brc(v218, buf207 + t217);
        vector_float v219;
asc_loadalign_brc(v219, buf208 + t214);
        vector_float v220;
        asc_max(v220, v215, v219, mask212);
        vector_float v221;
        asc_sub(v221, v215, v220, mask212);
        vector_float v222;
        asc_exp(v222, v221, mask212);
        vector_float v223;
        asc_sub(v223, v219, v220, mask212);
        vector_float v224;
        asc_exp(v224, v223, mask212);
        vector_float v225;
        asc_mul(v225, v218, v222, mask212);
        vector_float v226;
        asc_add(v226, v225, v224, mask212);
        vector_float v227;
        {
            static constexpr AscendC::Reg::DivSpecificMode div_mode_228 = {
                AscendC::Reg::MaskMergeMode::ZEROING, true, AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE};
            AscendC::Reg::Div<float, &div_mode_228, AscendC::Reg::RegTensor<float>>(*(AscendC::Reg::RegTensor<float>*)&v227, *(AscendC::Reg::RegTensor<float>*)&v222, *(AscendC::Reg::RegTensor<float>*)&v226, mask212);
        }
        for (int32_t i229 = 0; i229 < 128; i229 = (int32_t)((uint32_t)i229 + (uint32_t)64)) {
            int32_t t230 = i213 * 512;
            int32_t t231 = (int32_t)arg209;
            int32_t t232 = (int32_t)((uint32_t)t231 * (uint32_t)128);
            int32_t t233 = (int32_t)((uint32_t)t230 + (uint32_t)t232);
            int32_t t234 = (int32_t)((uint32_t)t233 + (uint32_t)i229);
            int32_t t235 = t234;
            vector_float v236;
            asc_loadalign(v236, buf210 + t235);
            vector_float v237;
            asc_mul(v237, v236, v227, mask212);
            vector_bfloat16_t v238;
            asc_float2bfloat16_rn(v238, v237, mask212, ASC_POSITION_EVEN);
            int32_t t239 = i213 * 128;
            int32_t t240 = t239 + i229;
            int32_t t241 = t240;
            asc_storealign_pack((__ubuf__ uint32_t*)(buf211 + t241), *(vector_uint32_t*)&v238, mask212);
        }
    }
}

__global__ __aicore__ void kernel_sparse_attention_mix(KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm0_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds gm1_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm2_arg, KernelArgTensor_uint8_t_2d_Device_ds_ds gm3_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm4_arg, KernelArgTensor_int32_t_2d_Device_ds_ds gm5_arg, KernelArgTensor_int32_t_2d_Device_ds_ds gm6_arg, KernelArgTensor_float_2d_Device_ss_ds gm7_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm8_arg, int64_t arg9, int64_t arg10, int64_t arg11, int32_t arg12)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ bfloat16_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm1 = gm1_arg.ptr;
    __gm__ bfloat16_t* gm2 = gm2_arg.ptr;
    __gm__ uint8_t* gm3 = gm3_arg.ptr;
    __gm__ bfloat16_t* gm4 = gm4_arg.ptr;
    __gm__ int32_t* gm5 = gm5_arg.ptr;
    __gm__ int32_t* gm6 = gm6_arg.ptr;
    __gm__ float* gm7 = gm7_arg.ptr;
    __gm__ bfloat16_t* gm8 = gm8_arg.ptr;
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

    __ubuf__ float* buf242 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf243 = (__ubuf__ float*)(uintptr_t)(uint64_t)(16384);
    __cc__ float* buf244 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ bfloat16_t* buf245 = (__ca__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cb__ bfloat16_t* buf246 = (__cb__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ bfloat16_t* buf247 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(65536);
    __cbuf__ bfloat16_t* buf248 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(196608);
    __cbuf__ bfloat16_t* buf249 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf250 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf251 = (__ubuf__ float*)(uintptr_t)(uint64_t)(16384);
    __ubuf__ int32_t* buf252 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(32768);
    __ubuf__ fp8_e4m3fn_t* buf253 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(33280);
    __ubuf__ uint8_t* buf254 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(41984);
    __ubuf__ bfloat16_t* buf255 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(46336);
    __ubuf__ bfloat16_t* buf256 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(46848);
    __ubuf__ bfloat16_t* buf257 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(47872);
    __ubuf__ bfloat16_t* buf258 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(64256);
    __ubuf__ bfloat16_t* buf259 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(81664);
    __ubuf__ float* buf260 = (__ubuf__ float*)(uintptr_t)(uint64_t)(90112);
    __ubuf__ float* buf261 = (__ubuf__ float*)(uintptr_t)(uint64_t)(90496);
    __ubuf__ float* buf262 = (__ubuf__ float*)(uintptr_t)(uint64_t)(156032);
    __ubuf__ bfloat16_t* buf263 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(156160);
    __cbuf__ bfloat16_t* buf264 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(65536);
    __cbuf__ bfloat16_t* buf265 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(196608);

    if ASCEND_IS_AIC {
        asc_sync_intra_arrive(PIPE_MTE1, 1);
        asc_sync_intra_arrive(PIPE_MTE1, 17);
        asc_sync_intra_arrive(PIPE_MTE1, 0);
        asc_sync_intra_arrive(PIPE_MTE1, 16);
        int64_t t266 = (int64_t)arg12;
        for (int64_t i267 = arg9; i267 < t266; i267 = (int64_t)((uint64_t)i267 + (uint64_t)arg10)) {
            __cbuf__ bfloat16_t* view_root268 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf249) + 0);
            int64_t t269 = (int64_t)((uint64_t)i267 * (uint64_t)4);
            int64_t dim270 = gm0_arg.stride[0];
            int64_t t271 = (int64_t)((uint64_t)t269 * (uint64_t)dim270);
            int64_t t272 = (int64_t)((uint64_t)t271 * (uint64_t)16);
            int64_t dim273 = gm0_arg.shape[0];
            int64_t t274 = dim273 / 64;
            int64_t t275 = (i267 < t274 ? i267 : t274);
            int64_t t276 = (int64_t)((uint64_t)t275 * (uint64_t)64);
            int64_t t277 = (int64_t)((uint64_t)dim273 - (uint64_t)t276);
            int64_t t278 = (t277 < 64 ? t277 : 64);
            bool cmp279 = i267 <= t274;
            int64_t sel280 = cmp279 ? t278 : 0;
            int64_t dim281 = gm0_arg.stride[0];
            int64_t t282 = dim281;
            int64_t t283 = t282 * 2;
            int64_t t284 = t283;
            asc_lock(PIPE_MTE2, 0);
            bool cmp285 = (uint64_t)sel280 > (uint64_t)0;
            if (cmp285) {
                int64_t t286 = t272;
                int32_t t287 = (int32_t)sel280;
                set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(64) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
                asc_copy_gm2l1_nd2nz(view_root268 + 0, gm0 + t286, t284, 0, t287, 512, 0, false);
            }
            asc_unlock(PIPE_MTE2, 0);
            for (int64_t i288 = 0; i288 < 5; i288 = (int64_t)((uint64_t)i288 + (uint64_t)1)) {
                asc_sync_intra_wait(PIPE_MTE1, 0);
                asc_sync_intra_wait(PIPE_MTE1, 16);
                for (int64_t i289 = 0; i289 < 4; i289 = (int64_t)((uint64_t)i289 + (uint64_t)1)) {
                    __ca__ bfloat16_t* view_root290 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf245) + 0);
                    int64_t t291 = i289 * 8;
                    asc_lock(PIPE_MTE1, 1);
                    asc_lock(PIPE_MTE1, 0);
                    int32_t t292 = (int32_t)t291;
                    asc_copy_l12l0a(view_root290 + 0, view_root268 + 0, 0, t292, 4, 8, 4, 4);
                    asc_unlock(PIPE_MTE1, 0);
                    asc_unlock(PIPE_MTE1, 1);
                    __cb__ bfloat16_t* view_root293 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf246) + 0);
                    __cbuf__ bfloat16_t* view_root294 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf247) + 0);
                    asc_lock(PIPE_MTE1, 2);
                    asc_copy_l12l0b(view_root293 + 0, view_root294 + 0, 0, t292, 8, 8, 8, 8);
                    asc_unlock(PIPE_MTE1, 2);
                    bool cmp295 = i289 == 0;
                    __cc__ float* view_root296 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf244) + 0);
                    asc_lock(PIPE_M, 3);
                    asc_lock(PIPE_M, 1);
                    asc_lock(PIPE_M, 2);
                    if (cmp295) {
                        asc_mmad(view_root296 + 0, view_root290, view_root293, 64, 128, 128, 0, true, false, true);
                    } else {
                        asc_mmad(view_root296 + 0, view_root290, view_root293, 64, 128, 128, 0, true, false, false);
                    }
                    asc_unlock(PIPE_M, 2);
                    asc_unlock(PIPE_M, 1);
                    asc_unlock(PIPE_M, 3);
                }
                __ubuf__ float* view_root297 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf242) + 0);
                __cc__ float* view_root298 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf244) + 0);
                asc_sync_intra_wait(PIPE_FIX, 2);
                asc_sync_intra_wait(PIPE_FIX, 18);
                asc_lock(PIPE_FIX, 3);
                asc_set_l0c2gm_nz2nd(1, 0, 0);
                asc_copy_l0c2ub((__ubuf__ float*)view_root297 + 0, view_root298, 128, 64, 128, 64, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                asc_unlock(PIPE_FIX, 3);
                asc_sync_intra_arrive(PIPE_FIX, 2);
                asc_sync_intra_arrive(PIPE_FIX, 18);
                __cbuf__ bfloat16_t* view_root299 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf248) + 0);
                asc_sync_intra_wait(PIPE_MTE1, 1);
                asc_sync_intra_wait(PIPE_MTE1, 17);
                for (int64_t i300 = 0; i300 < 4; i300 = (int64_t)((uint64_t)i300 + (uint64_t)1)) {
                    __ca__ bfloat16_t* view_root301 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf245) + 0);
                    asc_lock(PIPE_MTE1, 1);
                    asc_copy_l12l0a(view_root301 + 0, view_root299 + 0, 0, 0, 4, 8, 4, 4);
                    asc_unlock(PIPE_MTE1, 1);
                    __cb__ bfloat16_t* view_root302 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf246) + 0);
                    __cbuf__ bfloat16_t* view_root303 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf247) + 0);
                    int64_t t304 = i300 * 8;
                    asc_lock(PIPE_MTE1, 2);
                    int32_t t305 = (int32_t)t304;
                    asc_copy_l12l0b_transpose(view_root302 + 0, view_root303 + 0, 0, t305, 8, 8, 8, 8);
                    asc_unlock(PIPE_MTE1, 2);
                    asc_lock(PIPE_M, 3);
                    asc_lock(PIPE_M, 1);
                    asc_lock(PIPE_M, 2);
                    asc_mmad(view_root298 + 0, view_root301, view_root302, 64, 128, 128, 0, true, false, true);
                    asc_unlock(PIPE_M, 2);
                    asc_unlock(PIPE_M, 1);
                    asc_unlock(PIPE_M, 3);
                    __ubuf__ float* view_root306 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf243) + 0);
                    asc_sync_intra_wait(PIPE_FIX, 3);
                    asc_sync_intra_wait(PIPE_FIX, 19);
                    asc_lock(PIPE_FIX, 3);
                    asc_set_l0c2gm_nz2nd(1, 0, 0);
                    asc_copy_l0c2ub((__ubuf__ float*)view_root306 + 0, view_root298, 128, 64, 128, 64, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
                    asc_unlock(PIPE_FIX, 3);
                    asc_sync_intra_arrive(PIPE_FIX, 3);
                    asc_sync_intra_arrive(PIPE_FIX, 19);
                }
                asc_sync_intra_arrive(PIPE_MTE1, 0);
                asc_sync_intra_arrive(PIPE_MTE1, 16);
                asc_sync_intra_arrive(PIPE_MTE1, 1);
                asc_sync_intra_arrive(PIPE_MTE1, 17);
            }
        }
        asc_sync_intra_wait(PIPE_FIX, 2);
        asc_sync_intra_wait(PIPE_FIX, 18);
        asc_sync_intra_wait(PIPE_FIX, 3);
        asc_sync_intra_wait(PIPE_FIX, 19);
    }

    if ASCEND_IS_AIV {
        asc_sync_intra_arrive(PIPE_V, 3);
        asc_sync_intra_arrive(PIPE_V, 2);
        int64_t t307 = (int64_t)arg12;
        for (int64_t i308 = arg9; i308 < t307; i308 = (int64_t)((uint64_t)i308 + (uint64_t)arg10)) {
            int64_t t309 = (int64_t)((uint64_t)arg11 * (uint64_t)128);
            asc_lock(PIPE_MTE2, 7);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf262 + 0), (__gm__ float*)((__gm__ uint8_t*)gm7 + t309), 1, 128, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
            }
            asc_unlock(PIPE_MTE2, 7);
            kernel_sparse_attention_mix_vf0((__ubuf__ float*)buf260, (__ubuf__ float*)buf261);
            for (int64_t i310 = 0; i310 < 5; i310 = (int64_t)((uint64_t)i310 + (uint64_t)1)) {
                int64_t dim311 = gm1_arg.shape[0];
                bool cmp312 = i310 == 0;
                int64_t ifr313;
                if (cmp312) {
                    int64_t dim314 = gm5_arg.stride[0];
                    int64_t t315 = (int64_t)((uint64_t)i308 * (uint64_t)dim314);
                    int64_t t316 = (int64_t)((uint64_t)t315 * (uint64_t)4);
                    asc_lock(PIPE_MTE2, 0);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf252 + 0), (__gm__ int32_t*)((__gm__ uint8_t*)gm5 + t316), 1, 512, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    asc_sync_intra_wait(PIPE_MTE3, 0);
                    for (int64_t i317 = 0; i317 < 4; i317 = (int64_t)((uint64_t)i317 + (uint64_t)1)) {
                        int64_t t318 = (int64_t)((uint64_t)arg11 * (uint64_t)64);
                        int64_t t319 = i317 * 16;
                        int64_t t320 = (int64_t)((uint64_t)t318 + (uint64_t)t319);
                        for (int64_t i321 = 0; i321 < 16; i321 = (int64_t)((uint64_t)i321 + (uint64_t)1)) {
                            int64_t t322 = (int64_t)((uint64_t)t320 + (uint64_t)i321);
                            int64_t t323 = t315;
                            int64_t t324 = t322;
                            int64_t t325 = t323 + t324;
                            int32_t ld326 = gm5[t325];
                            int64_t t327 = (int64_t)ld326;
                            int64_t t328 = (int64_t)((uint64_t)dim311 - (uint64_t)1);
                            int64_t t329 = (t327 < t328 ? t327 : t328);
                            int64_t t330 = (t329 > 0 ? t329 : 0);
                            int64_t t331 = (t330 < dim311 ? t330 : dim311);
                            int64_t t332 = (int64_t)((uint64_t)dim311 - (uint64_t)t331);
                            int64_t t333 = (t332 < 1 ? t332 : 1);
                            bool cmp334 = t330 <= dim311;
                            int64_t sel335 = cmp334 ? t333 : 0;
                            int64_t dim336 = gm1_arg.stride[0];
                            int64_t t337 = (int64_t)((uint64_t)t330 * (uint64_t)dim336);
                            int64_t t338 = i321 * 512;
                            asc_lock(PIPE_MTE2, 1);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ fp8_e4m3fn_t*)((__ubuf__ uint8_t*)buf253 + t338), (__gm__ fp8_e4m3fn_t*)((__gm__ uint8_t*)gm1 + t337), sel335, 512, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)dim336, 512);
                            }
                            asc_unlock(PIPE_MTE2, 1);
                            int64_t dim339 = gm2_arg.shape[0];
                            int64_t t340 = (t330 < dim339 ? t330 : dim339);
                            int64_t t341 = (int64_t)((uint64_t)dim339 - (uint64_t)t340);
                            int64_t t342 = (t341 < 1 ? t341 : 1);
                            bool cmp343 = t330 <= dim339;
                            int64_t sel344 = cmp343 ? t342 : 0;
                            int64_t dim345 = gm2_arg.stride[0];
                            int64_t t346 = (int64_t)((uint64_t)dim345 * (uint64_t)2);
                            int64_t t347 = (int64_t)((uint64_t)t330 * (uint64_t)dim345);
                            int64_t t348 = (int64_t)((uint64_t)t347 * (uint64_t)2);
                            int64_t t349 = i321 * 32;
                            asc_lock(PIPE_MTE2, 3);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf255 + t349), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm2 + t348), sel344, 32, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t346, 32);
                            }
                            asc_unlock(PIPE_MTE2, 3);
                        }
                        asc_lock(PIPE_V, 0);
                        asc_lock(PIPE_V, 1);
                        asc_lock(PIPE_V, 3);
                        kernel_sparse_attention_mix_vf1(t320, (__ubuf__ int32_t*)buf252, dim311, (__ubuf__ fp8_e4m3fn_t*)buf253, (__ubuf__ bfloat16_t*)buf255, (__ubuf__ bfloat16_t*)buf257);
                        asc_unlock(PIPE_V, 3);
                        asc_unlock(PIPE_V, 1);
                        asc_unlock(PIPE_V, 0);
                        asc_lock(PIPE_V, 5);
                        kernel_sparse_attention_mix_vf2((__ubuf__ bfloat16_t*)buf257, (__ubuf__ bfloat16_t*)buf258);
                        asc_unlock(PIPE_V, 5);
                        int64_t t350 = (int64_t)((uint64_t)arg11 * (uint64_t)4);
                        int64_t t351 = (int64_t)((uint64_t)t350 + (uint64_t)i317);
                        __ubuf__ bfloat16_t* view_root352 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf258) + 0);
                        __cbuf__ bfloat16_t* view_root353 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf264) + 0);
                        int64_t t354 = (int64_t)((uint64_t)t351 * (uint64_t)256);
                        asc_lock(PIPE_MTE3, 5);
                        int64_t t355 = t354;
                        asc_copy_ub2l1(view_root353 + t355, (__ubuf__ bfloat16_t*)view_root352 + 0, 32, 16, 1, 112);
                        asc_unlock(PIPE_MTE3, 5);
                    }
                    asc_sync_intra_arrive(PIPE_MTE3, 0);
                    ifr313 = dim311;
                } else {
                    int64_t t356 = i310 - 1;
                    int64_t t357 = t356 * 128;
                    int64_t dim358 = gm6_arg.stride[0];
                    int64_t t359 = (int64_t)((uint64_t)i308 * (uint64_t)dim358);
                    int64_t t360 = (int64_t)((uint64_t)t359 * (uint64_t)4);
                    int64_t t361 = t356 * 512;
                    int64_t t362 = (int64_t)((uint64_t)t360 + (uint64_t)t361);
                    asc_lock(PIPE_MTE2, 0);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf252 + 0), (__gm__ int32_t*)((__gm__ uint8_t*)gm6 + t362), 1, 512, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)0, 0);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    asc_sync_intra_wait(PIPE_MTE3, 0);
                    for (int64_t i363 = 0; i363 < 4; i363 = (int64_t)((uint64_t)i363 + (uint64_t)1)) {
                        int64_t t364 = (int64_t)((uint64_t)arg11 * (uint64_t)64);
                        int64_t t365 = i363 * 16;
                        int64_t t366 = (int64_t)((uint64_t)t364 + (uint64_t)t365);
                        for (int64_t i367 = 0; i367 < 16; i367 = (int64_t)((uint64_t)i367 + (uint64_t)1)) {
                            int64_t t368 = (int64_t)((uint64_t)t366 + (uint64_t)i367);
                            int64_t t369 = t359;
                            int64_t t370 = t357;
                            int64_t t371 = t369 + t370;
                            int64_t t372 = t368;
                            int64_t t373 = t371 + t372;
                            int32_t ld374 = gm6[t373];
                            int64_t t375 = (int64_t)ld374;
                            int64_t dim376 = gm3_arg.shape[0];
                            int64_t t377 = (int64_t)((uint64_t)dim376 - (uint64_t)1);
                            int64_t t378 = (t375 < t377 ? t375 : t377);
                            int64_t t379 = (t378 > 0 ? t378 : 0);
                            int64_t t380 = (t379 < dim376 ? t379 : dim376);
                            int64_t t381 = (int64_t)((uint64_t)dim376 - (uint64_t)t380);
                            int64_t t382 = (t381 < 1 ? t381 : 1);
                            bool cmp383 = t379 <= dim376;
                            int64_t sel384 = cmp383 ? t382 : 0;
                            int64_t dim385 = gm3_arg.stride[0];
                            int64_t t386 = (int64_t)((uint64_t)t379 * (uint64_t)dim385);
                            int64_t t387 = i367 * 256;
                            asc_lock(PIPE_MTE2, 2);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ uint8_t*)((__ubuf__ uint8_t*)buf254 + t387), (__gm__ uint8_t*)((__gm__ uint8_t*)gm3 + t386), sel384, 256, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)dim385, 256);
                            }
                            asc_unlock(PIPE_MTE2, 2);
                            int64_t dim388 = gm4_arg.shape[0];
                            int64_t t389 = (t379 < dim388 ? t379 : dim388);
                            int64_t t390 = (int64_t)((uint64_t)dim388 - (uint64_t)t389);
                            int64_t t391 = (t390 < 1 ? t390 : 1);
                            bool cmp392 = t379 <= dim388;
                            int64_t sel393 = cmp392 ? t391 : 0;
                            int64_t dim394 = gm4_arg.stride[0];
                            int64_t t395 = (int64_t)((uint64_t)dim394 * (uint64_t)2);
                            int64_t t396 = (int64_t)((uint64_t)t379 * (uint64_t)dim394);
                            int64_t t397 = (int64_t)((uint64_t)t396 * (uint64_t)2);
                            int64_t t398 = i367 * 64;
                            asc_lock(PIPE_MTE2, 4);
                            {
                                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf256 + t398), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm4 + t397), sel393, 64, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t395, 64);
                            }
                            asc_unlock(PIPE_MTE2, 4);
                        }
                        int64_t dim399 = gm3_arg.shape[0];
                        asc_lock(PIPE_V, 0);
                        asc_lock(PIPE_V, 2);
                        asc_lock(PIPE_V, 4);
                        asc_lock(PIPE_V, 5);
                        kernel_sparse_attention_mix_vf3(t366, (__ubuf__ int32_t*)buf252, dim399, (__ubuf__ uint8_t*)buf254, (__ubuf__ bfloat16_t*)buf256, (__ubuf__ bfloat16_t*)buf258);
                        asc_unlock(PIPE_V, 5);
                        asc_unlock(PIPE_V, 4);
                        asc_unlock(PIPE_V, 2);
                        asc_unlock(PIPE_V, 0);
                        int64_t t400 = (int64_t)((uint64_t)arg11 * (uint64_t)4);
                        int64_t t401 = (int64_t)((uint64_t)t400 + (uint64_t)i363);
                        __ubuf__ bfloat16_t* view_root402 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf258) + 0);
                        __cbuf__ bfloat16_t* view_root403 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf264) + 0);
                        int64_t t404 = (int64_t)((uint64_t)t401 * (uint64_t)256);
                        asc_lock(PIPE_MTE3, 5);
                        int64_t t405 = t404;
                        asc_copy_ub2l1(view_root403 + t405, (__ubuf__ bfloat16_t*)view_root402 + 0, 32, 16, 1, 112);
                        asc_unlock(PIPE_MTE3, 5);
                    }
                    asc_sync_intra_arrive(PIPE_MTE3, 0);
                    int64_t dim406 = gm3_arg.shape[0];
                    ifr313 = dim406;
                }
                int32_t t407 = (int32_t)ifr313;
                asc_sync_intra_wait(PIPE_V, 2);
                asc_lock(PIPE_V, 0);
                asc_lock(PIPE_V, 6);
                kernel_sparse_attention_mix_vf4((__ubuf__ int32_t*)buf252, t407, (__ubuf__ float*)buf250, (__ubuf__ float*)buf260, (__ubuf__ bfloat16_t*)buf259);
                asc_unlock(PIPE_V, 6);
                asc_unlock(PIPE_V, 0);
                asc_sync_intra_arrive(PIPE_V, 2);
                __ubuf__ bfloat16_t* view_root408 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf259) + 0);
                __cbuf__ bfloat16_t* view_root409 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf265) + 0);
                int64_t t410 = (int64_t)((uint64_t)arg11 * (uint64_t)32);
                int64_t t411 = t410;
                int64_t t412 = t411 * 16;
                asc_sync_intra_wait(PIPE_MTE3, 1);
                asc_lock(PIPE_MTE3, 6);
                asc_copy_ub2l1(view_root409 + t412, (__ubuf__ bfloat16_t*)view_root408 + 0, 8, 32, 1, 32);
                asc_unlock(PIPE_MTE3, 6);
                asc_sync_intra_arrive(PIPE_MTE3, 1);
                for (int64_t i413 = 0; i413 < 4; i413 = (int64_t)((uint64_t)i413 + (uint64_t)1)) {
                    asc_sync_intra_wait(PIPE_V, 3);
                    kernel_sparse_attention_mix_vf5((__ubuf__ float*)buf260, i413, (__ubuf__ float*)buf261, (__ubuf__ float*)buf251);
                    asc_sync_intra_arrive(PIPE_V, 3);
                }
            }
            for (int64_t i414 = 0; i414 < 4; i414 = (int64_t)((uint64_t)i414 + (uint64_t)1)) {
                asc_lock(PIPE_V, 7);
                asc_lock(PIPE_V, 8);
                kernel_sparse_attention_mix_vf6((__ubuf__ float*)buf260, (__ubuf__ float*)buf262, i414, (__ubuf__ float*)buf261, (__ubuf__ bfloat16_t*)buf263);
                asc_unlock(PIPE_V, 8);
                asc_unlock(PIPE_V, 7);
                int64_t t415 = (int64_t)((uint64_t)i308 * (uint64_t)64);
                int64_t dim416 = gm8_arg.stride[0];
                int64_t t417 = (int64_t)((uint64_t)t415 * (uint64_t)dim416);
                int64_t t418 = i414 * 128;
                int64_t t419 = (int64_t)((uint64_t)t417 + (uint64_t)t418);
                int64_t t420 = (int64_t)((uint64_t)t419 * (uint64_t)16);
                int64_t t421 = (int64_t)((uint64_t)t420 / (uint64_t)8);
                int64_t t422 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm8) + (uint64_t)t421);
                __gm__ bfloat16_t* root423 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t422);
                __ubuf__ bfloat16_t* view_root424 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf263) + 0);
                int64_t dim425 = gm8_arg.stride[0];
                int64_t t426 = dim425;
                int64_t dim427 = gm8_arg.shape[0];
                int64_t t428 = dim427 / 64;
                int64_t t429 = (i308 < t428 ? i308 : t428);
                int64_t t430 = (int64_t)((uint64_t)t429 * (uint64_t)64);
                int64_t t431 = (int64_t)((uint64_t)dim427 - (uint64_t)t430);
                int64_t t432 = (t431 < 64 ? t431 : 64);
                bool cmp433 = i308 <= t428;
                int64_t sel434 = cmp433 ? t432 : 0;
                int64_t t435 = sel434 / 2;
                int64_t t436 = sel434 % 2;
                int64_t t437 = t436 + 1;
                int64_t t438 = t437 / 2;
                int64_t t439 = (int64_t)((uint64_t)t435 + (uint64_t)t438);
                int64_t t440 = (int64_t)((uint64_t)arg11 * (uint64_t)t439);
                int64_t t441 = t440;
                int64_t t442 = t441 * t426;
                int64_t t443 = t426 * 2;
                int64_t t444 = t443;
                asc_lock(PIPE_MTE3, 8);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root423 + t442, (__ubuf__ bfloat16_t*)view_root424 + 0, 32, 256, static_cast<uint8_t>(0), (uint64_t)t444, 256);
                asc_unlock(PIPE_MTE3, 8);
            }
        }
        asc_sync_intra_wait(PIPE_MTE3, 0);
        asc_sync_intra_wait(PIPE_MTE3, 1);
    }
}
