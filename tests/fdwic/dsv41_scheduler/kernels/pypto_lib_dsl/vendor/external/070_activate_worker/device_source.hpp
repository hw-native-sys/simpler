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

struct KernelArgTensor_float_2d_Device_ss_ds {
    __gm__ float* ptr;
    int64_t stride[1];
};
static inline KernelArgTensor_float_2d_Device_ss_ds cannbotdsl_pack_KernelArgTensor_float_2d_Device_ss_ds(const KernelArgTensor_float_2d& src) {
    return {src.ptr, {src.stride[0]}};
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

__simd_vf__ inline void kernel_activate_worker_vf0(__ubuf__ float* buf0)
{
    vector_bool vmask;
    float c1 = 0.0f;
    vector_bool mask2 = asc_create_mask_b32(PAT_ALL);
    vector_float v3;
    asc_duplicate_scalar(v3, c1, mask2);
    for (int32_t i4 = 0; i4 < 2368; i4 = (int32_t)((uint32_t)i4 + (uint32_t)64)) {
        int32_t t5 = i4;
        asc_storealign(buf0 + t5, v3, mask2);
    }
}

__simd_vf__ inline void kernel_activate_worker_vf1(__ubuf__ float* buf6, __ubuf__ float* buf7, float arg8, __ubuf__ float* buf9)
{
    vector_bool vmask;
    float c10 = -1.0f;
    float c11 = -10.0f;
    float c12 = 10.0f;
    float c13 = 1.0f;
    vector_bool mask14 = asc_create_mask_b32(PAT_ALL);
    vector_float v15;
    asc_duplicate_scalar(v15, c13, mask14);
    for (int32_t i16 = 0; i16 < 2304; i16 = (int32_t)((uint32_t)i16 + (uint32_t)64)) {
        int32_t t17 = i16;
        vector_float v18;
        asc_loadalign(v18, buf6 + t17);
        vector_float v19;
        asc_loadalign(v19, buf7 + t17);
        vector_float v20;
        asc_min_scalar(v20, v18, c12, mask14);
        vector_float v21;
        asc_max_scalar(v21, v19, c11, mask14);
        vector_float v22;
        asc_min_scalar(v22, v21, c12, mask14);
        vector_float v23;
        asc_mul_scalar(v23, v20, c10, mask14);
        vector_float v24;
        asc_exp(v24, v23, mask14);
        vector_float v25;
        asc_add_scalar(v25, v24, c13, mask14);
        vector_float v26;
        {
            static constexpr AscendC::Reg::DivSpecificMode div_mode_27 = {
                AscendC::Reg::MaskMergeMode::ZEROING, true, AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE};
            AscendC::Reg::Div<float, &div_mode_27, AscendC::Reg::RegTensor<float>>(*(AscendC::Reg::RegTensor<float>*)&v26, *(AscendC::Reg::RegTensor<float>*)&v15, *(AscendC::Reg::RegTensor<float>*)&v25, mask14);
        }
        vector_float v28;
        asc_mul(v28, v20, v26, mask14);
        vector_float v29;
        asc_mul(v29, v28, v22, mask14);
        vector_float v30;
        asc_mul_scalar(v30, v29, arg8, mask14);
        asc_storealign(buf9 + t17, v30, mask14);
    }
}

__simd_vf__ inline void kernel_activate_worker_vf2(__ubuf__ float* buf31, __ubuf__ fp8_e4m3fn_t* buf32, int32_t arg33, __ubuf__ uint8_t* buf34)
{
    vector_bool vmask;
    float c35 = -448.0f;
    float c36 = 448.0f;
    float c37 = 0.00223214296f;
    float c38 = 9.99999974E-5f;
    asc_mem_bar(VST_VLD);
    uint32_t rem40 = 32;
    vector_bool mask39 = asc_update_mask_b32(rem40);
    for (int32_t i41 = 0; i41 < 72; i41 = (int32_t)((uint32_t)i41 + (uint32_t)1)) {
        int32_t t42 = i41 * 32;
        int32_t t43 = t42;
        vector_float v44;
        asc_loadalign(v44, buf31 + t43);
        vector_float v45;
        asc_abs(v45, v44, mask39);
        vector_float v46;
        asc_reduce_max(v46, v45, mask39);
        vector_float v47;
        asc_max_scalar(v47, v46, c38, mask39);
        vector_float v48;
        asc_mul_scalar(v48, v47, c37, mask39);
        vector_int32_t &r49 = (vector_int32_t &)v48;
        vector_int32_t v50;
        asc_add_scalar(v50, r49, 8388607, mask39);
        vector_int32_t v51;
        asc_shiftright_scalar(v51, v50, 23, mask39);
        vector_int32_t v52;
        asc_duplicate(v52, v51, mask39);
        vector_int32_t v53;
        asc_duplicate_scalar(v53, 254, mask39);
        vector_int32_t v54;
        asc_sub(v54, v53, v52, mask39);
        vector_int32_t v55;
        asc_shiftleft_scalar(v55, v54, 23, mask39);
        vector_float &r56 = (vector_float &)v55;
        vector_float v57;
        asc_mul(v57, v44, r56, mask39);
        vector_float v58;
        asc_min_scalar(v58, v57, c36, mask39);
        vector_float v59;
        asc_max_scalar(v59, v58, c35, mask39);
        vector_fp8_e4m3fn_t v60;
        asc_float2e4m3_rn(v60, v59, mask39);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf32 + t43), *(vector_float*)&v60, mask39);
        vector_uint8_t v61;
        asc_int322uint8(v61, v52, mask39);
        int32_t t62 = i41 / 2;
        int32_t t63 = t62 * 32;
        int32_t t64 = (int32_t)((uint32_t)arg33 * (uint32_t)2);
        int32_t t65 = (int32_t)((uint32_t)t63 + (uint32_t)t64);
        int32_t t66 = i41 % 2;
        int32_t t67 = (int32_t)((uint32_t)t65 + (uint32_t)t66);
        int32_t t68 = t67;
        asc_storealign_1st(buf34 + t68, v61);
    }
}

static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_float_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_float_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_int32_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 16 && alignof(KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
static_assert(sizeof(int64_t) == 8 && alignof(int64_t) == 8, "device argument ABI mismatch");
__vector__ __global__ __aicore__ void kernel_activate_worker(KernelArgTensor_float_2d_Device_ss_ds gm69_arg, KernelArgTensor_float_2d_Device_ss_ds gm70_arg, KernelArgTensor_float_2d_Device_ss_ds gm71_arg, KernelArgTensor_int32_t_2d_Device_ss_ds gm72_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm73_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm74_arg, int64_t arg75, int64_t arg76)
{
    __gm__ float* gm69 = gm69_arg.ptr;
    __gm__ float* gm70 = gm70_arg.ptr;
    __gm__ float* gm71 = gm71_arg.ptr;
    __gm__ int32_t* gm72 = gm72_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm73 = gm73_arg.ptr;
    __gm__ fp8_e8m0_t* gm74 = gm74_arg.ptr;
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

    __ubuf__ float* buf77 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf78 = (__ubuf__ float*)(uintptr_t)(uint64_t)(9216);
    __ubuf__ float* buf79 = (__ubuf__ float*)(uintptr_t)(uint64_t)(18432);
    __ubuf__ fp8_e4m3fn_t* buf80 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(27904);
    __ubuf__ uint8_t* buf81 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(30208);

    vector_bool vmask;
    int64_t iter82 = 0;
    int64_t iter83 = 0;
    for (int64_t i84 = 0; i84 < 64; i84 = (int64_t)((uint64_t)i84 + (uint64_t)1)) {
        int64_t dim85 = gm72_arg.stride[0];
        int64_t t86 = dim85;
        int64_t t87 = i84;
        int64_t t88 = t87 * t86;
        int32_t ld89 = gm72[t88];
        int32_t t90 = (int32_t)((uint32_t)ld89 + (uint32_t)15);
        int32_t t91 = t90 % 16;
        bool cmp92 = t91 != 0;
        bool cmp93 = t91 < 0;
        bool t94 = cmp92 & cmp93;
        int32_t t95 = t90 / 16;
        int32_t t96 = (int32_t)(uint8_t)t94;
        int32_t t97 = (int32_t)((uint32_t)t95 - (uint32_t)t96);
        int64_t t98 = (int64_t)((uint64_t)arg75 + (uint64_t)arg76);
        int64_t t99 = iter83 % arg76;
        bool cmp100 = t99 != 0;
        bool cmp101 = t99 < 0;
        bool cmp102 = arg76 < 0;
        bool t103 = cmp101 ^ cmp102;
        bool t104 = cmp100 & t103;
        int64_t sel105 = t104 ? arg76 : 0;
        int64_t t106 = (int64_t)((uint64_t)t99 + (uint64_t)sel105);
        int64_t t107 = (int64_t)((uint64_t)t98 - (uint64_t)t106);
        int64_t t108 = t107 % arg76;
        bool cmp109 = t108 != 0;
        bool cmp110 = t108 < 0;
        bool t111 = cmp110 ^ cmp102;
        bool t112 = cmp109 & t111;
        int64_t sel113 = t112 ? arg76 : 0;
        int64_t t114 = (int64_t)((uint64_t)t108 + (uint64_t)sel113);
        int64_t t115 = (int64_t)t97;
        for (int64_t i116 = t114; i116 < t115; i116 = (int64_t)((uint64_t)i116 + (uint64_t)arg76)) {
            int64_t t117 = (int64_t)((uint64_t)i116 * (uint64_t)16);
            int64_t t118 = (int64_t)((uint64_t)iter82 + (uint64_t)t117);
            for (int64_t i119 = 0; i119 < 16; i119 = (int64_t)((uint64_t)i119 + (uint64_t)1)) {
                kernel_activate_worker_vf0((__ubuf__ float*)buf79);
                int64_t t120 = (int64_t)((uint64_t)t117 + (uint64_t)i119);
                int64_t t121 = (int64_t)ld89;
                bool cmp122 = t120 < t121;
                if (cmp122) {
                    int64_t t123 = (int64_t)((uint64_t)t118 + (uint64_t)i119);
                    int64_t t124 = (t123 < 1920 ? t123 : 1920);
                    int64_t t125 = (int64_t)((uint64_t)1920 - (uint64_t)t124);
                    int64_t t126 = (t125 < 1 ? t125 : 1);
                    bool cmp127 = t123 <= 1920;
                    int64_t sel128 = cmp127 ? t126 : 0;
                    int64_t dim129 = gm69_arg.stride[0];
                    int64_t t130 = (int64_t)((uint64_t)dim129 * (uint64_t)4);
                    int64_t t131 = (int64_t)((uint64_t)t123 * (uint64_t)dim129);
                    int64_t t132 = (int64_t)((uint64_t)t131 * (uint64_t)4);
                    asc_lock(PIPE_MTE2, 0);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf77 + 0), (__gm__ float*)((__gm__ uint8_t*)gm69 + t132), sel128, 9216, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t130, 9216);
                    }
                    asc_unlock(PIPE_MTE2, 0);
                    int64_t dim133 = gm70_arg.stride[0];
                    int64_t t134 = (int64_t)((uint64_t)dim133 * (uint64_t)4);
                    int64_t t135 = (int64_t)((uint64_t)t123 * (uint64_t)dim133);
                    int64_t t136 = (int64_t)((uint64_t)t135 * (uint64_t)4);
                    asc_lock(PIPE_MTE2, 1);
                    {
                        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf78 + 0), (__gm__ float*)((__gm__ uint8_t*)gm70 + t136), sel128, 9216, 0, 0, false, static_cast<uint8_t>(0), (uint64_t)t134, 9216);
                    }
                    asc_unlock(PIPE_MTE2, 1);
                    int64_t t137 = i84 * 160;
                    int64_t t138 = (int64_t)((uint64_t)t137 + (uint64_t)t117);
                    int64_t t139 = (int64_t)((uint64_t)t138 + (uint64_t)i119);
                    int64_t dim140 = gm71_arg.stride[0];
                    int64_t t141 = dim140;
                    int64_t t142 = t139;
                    int64_t t143 = t142 * t141;
                    float ld144 = gm71[t143];
                    asc_lock(PIPE_V, 0);
                    asc_lock(PIPE_V, 1);
                    kernel_activate_worker_vf1((__ubuf__ float*)buf77, (__ubuf__ float*)buf78, ld144, (__ubuf__ float*)buf79);
                    asc_unlock(PIPE_V, 1);
                    asc_unlock(PIPE_V, 0);
                }
                int32_t t145 = (int32_t)i119;
                asc_lock(PIPE_V, 2);
                asc_lock(PIPE_V, 3);
                kernel_activate_worker_vf2((__ubuf__ float*)buf79, (__ubuf__ fp8_e4m3fn_t*)buf80, t145, (__ubuf__ uint8_t*)buf81);
                asc_unlock(PIPE_V, 3);
                asc_unlock(PIPE_V, 2);
                int64_t t146 = (int64_t)((uint64_t)t118 + (uint64_t)i119);
                int64_t dim147 = gm73_arg.stride[0];
                int64_t t148 = (int64_t)((uint64_t)t146 * (uint64_t)dim147);
                int64_t t149 = (int64_t)((uint64_t)t148 * (uint64_t)8);
                int64_t t150 = (int64_t)((uint64_t)t149 / (uint64_t)8);
                int64_t t151 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm73) + (uint64_t)t150);
                __gm__ fp8_e4m3fn_t* root152 = (__gm__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(t151);
                __ubuf__ fp8_e4m3fn_t* view_root153 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf80) + 0);
                int64_t dim154 = gm73_arg.stride[0];
                asc_lock(PIPE_MTE3, 2);
                cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ fp8_e4m3fn_t*)root152 + 0, (__ubuf__ fp8_e4m3fn_t*)view_root153 + 0, 1, 2304, static_cast<uint8_t>(0), (uint64_t)0, 0);
                asc_unlock(PIPE_MTE3, 2);
            }
            int64_t dim155 = gm74_arg.stride[0];
            int64_t t156 = (int64_t)((uint64_t)dim155 * (uint64_t)8);
            int64_t t157 = t156 / 8;
            int64_t t158 = (int64_t)((uint64_t)t118 * (uint64_t)576);
            int64_t t159 = (int64_t)((uint64_t)t158 / (uint64_t)8);
            int64_t t160 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm74) + (uint64_t)t159);
            __gm__ uint8_t* root161 = (__gm__ uint8_t*)(uintptr_t)(uint64_t)(t160);
            __ubuf__ uint8_t* view_root162 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf81) + 0);
            asc_lock(PIPE_MTE3, 3);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)root161 + 0, (__ubuf__ uint8_t*)view_root162 + 0, 1, 1152, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 3);
        }
        int32_t t163 = (int32_t)((uint32_t)t97 * (uint32_t)16);
        int64_t t164 = (int64_t)t163;
        int64_t t165 = (int64_t)((uint64_t)iter82 + (uint64_t)t164);
        int64_t t166 = (int64_t)((uint64_t)iter83 + (uint64_t)t115);
        iter82 = t165;
        iter83 = t166;
    }
}
