#include "kernel_operator.h"
#include "runtime/rt.h"
#include "kernel/pa_entry.cce"

extern "C" void get_ffts_info(uint64_t *addr, uint32_t *len) {
    rtGetC2cCtrlAddr(addr, len);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q_gm,
    uint8_t *k_gm,
    uint8_t *v_gm,
    uint8_t *block_tables_gm,
    uint8_t *mask_gm,
    uint8_t *deq_scale1_gm,
    uint8_t *offset1_gm,
    uint8_t *deq_scale2_gm,
    uint8_t *offset2_gm,
    uint8_t *razorOffset_gm,
    uint8_t *scale_gm,
    uint8_t *logN_gm,
    uint8_t *eye_gm,
    uint8_t *o_gm,
    uint8_t *s_gm,
    uint8_t *p_gm,
    uint8_t *o_tmp_gm,
    uint8_t *go_gm,
    uint8_t *o_core_tmp_gm,
    uint8_t *l_gm,
    uint8_t *gm_k16,
    uint8_t *gm_v16,
    uint8_t *tiling_para_gm)
{
    void *ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&ffts_addr), &ffts_len);

    paged_attention_mask<<<block_dim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr,
        (__gm__ uint8_t *)q_gm,
        (__gm__ uint8_t *)k_gm,
        (__gm__ uint8_t *)v_gm,
        (__gm__ uint8_t *)block_tables_gm,
        (__gm__ uint8_t *)mask_gm,
        (__gm__ uint8_t *)deq_scale1_gm,
        (__gm__ uint8_t *)offset1_gm,
        (__gm__ uint8_t *)deq_scale2_gm,
        (__gm__ uint8_t *)offset2_gm,
        (__gm__ uint8_t *)razorOffset_gm,
        (__gm__ uint8_t *)scale_gm,
        (__gm__ uint8_t *)logN_gm,
        (__gm__ uint8_t *)eye_gm,
        (__gm__ uint8_t *)o_gm,
        (__gm__ uint8_t *)s_gm,
        (__gm__ uint8_t *)p_gm,
        (__gm__ uint8_t *)o_tmp_gm,
        (__gm__ uint8_t *)go_gm,
        (__gm__ uint8_t *)o_core_tmp_gm,
        (__gm__ uint8_t *)l_gm,
        (__gm__ uint8_t *)gm_k16,
        (__gm__ uint8_t *)gm_v16,
        (__gm__ uint8_t *)tiling_para_gm);
}
