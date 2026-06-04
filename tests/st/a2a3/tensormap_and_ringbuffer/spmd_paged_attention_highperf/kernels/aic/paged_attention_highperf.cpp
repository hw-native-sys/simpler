/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#include <cstdint>

#ifdef __CPU_SIM
#include <pto/pto-inst.hpp>
#endif

#include "tensor.h"

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

static float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000) >> 13));
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
#ifdef __DAV_VEC__
    (void)args;
    return;
#else
    auto *query_t = reinterpret_cast<Tensor *>(args[0]);
    auto *key_t = reinterpret_cast<Tensor *>(args[1]);
    auto *value_t = reinterpret_cast<Tensor *>(args[2]);
    auto *block_table_t = reinterpret_cast<Tensor *>(args[3]);
    auto *out_t = reinterpret_cast<Tensor *>(args[4]);

    auto *query = reinterpret_cast<uint16_t *>(query_t->buffer.addr) + query_t->start_offset;
    auto *key = reinterpret_cast<uint16_t *>(key_t->buffer.addr) + key_t->start_offset;
    auto *value = reinterpret_cast<uint16_t *>(value_t->buffer.addr) + value_t->start_offset;
    auto *block_table = reinterpret_cast<int32_t *>(block_table_t->buffer.addr) + block_table_t->start_offset;
    auto *out = reinterpret_cast<uint16_t *>(out_t->buffer.addr) + out_t->start_offset;

    const int batch = static_cast<int>(query_t->shapes[0]);
    const int num_heads = static_cast<int>(query_t->shapes[1]);
    const int head_dim = static_cast<int>(query_t->shapes[2]);
    const int block_size = static_cast<int>(key_t->shapes[1]);
    const int num_kv_heads = static_cast<int>(key_t->shapes[2]);
    const int blocks_per_batch = static_cast<int>(key_t->shapes[0]) / batch;
    const int max_blocks_per_query = static_cast<int>(block_table_t->shapes[1]);
    const int heads_per_kv = num_heads / num_kv_heads;
    const int seq_len = blocks_per_batch * block_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            const int kv_head = h / heads_per_kv;
            float max_score = -INFINITY;
            for (int token = 0; token < seq_len; ++token) {
                const int block_col = std::min(token / block_size, max_blocks_per_query - 1);
                const int block_id = block_table[b * max_blocks_per_query + block_col];
                const int block_token = token % block_size;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    const int q_idx = (b * num_heads + h) * head_dim + d;
                    const int k_idx = ((block_id * block_size + block_token) * num_kv_heads + kv_head) * head_dim + d;
                    score += half_to_float(query[q_idx]) * half_to_float(key[k_idx]);
                }
                max_score = std::max(max_score, score * scale);
            }

            float denom = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float accum = 0.0f;
                for (int token = 0; token < seq_len; ++token) {
                    const int block_col = std::min(token / block_size, max_blocks_per_query - 1);
                    const int block_id = block_table[b * max_blocks_per_query + block_col];
                    const int block_token = token % block_size;
                    float score = 0.0f;
                    for (int kd = 0; kd < head_dim; ++kd) {
                        const int q_idx = (b * num_heads + h) * head_dim + kd;
                        const int k_idx =
                            ((block_id * block_size + block_token) * num_kv_heads + kv_head) * head_dim + kd;
                        score += half_to_float(query[q_idx]) * half_to_float(key[k_idx]);
                    }
                    const float weight = std::exp(score * scale - max_score);
                    if (d == 0) {
                        denom += weight;
                    }
                    const int v_idx = ((block_id * block_size + block_token) * num_kv_heads + kv_head) * head_dim + d;
                    accum += weight * half_to_float(value[v_idx]);
                }
                const int out_idx = (b * num_heads + h) * head_dim + d;
                out[out_idx] = float_to_half(accum / denom);
            }
        }
    }
#endif
}

#else

#include "intrinsic.h"

#define PTO_PA_NO_GLOBAL_ENTRY
#include "../kernel/pa_entry.cce"
#undef PTO_PA_NO_GLOBAL_ENTRY

static __aicore__ __attribute__((always_inline)) __gm__ uint8_t *tensor_data(__gm__ int64_t *args, int idx) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[idx]);
    return reinterpret_cast<__gm__ uint8_t *>(tensor->buffer.addr);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ uint8_t *q_gm = tensor_data(args, 0);
    __gm__ uint8_t *k_gm = tensor_data(args, 1);
    __gm__ uint8_t *v_gm = tensor_data(args, 2);
    __gm__ uint8_t *block_tables_gm = tensor_data(args, 3);
    __gm__ uint8_t *o_gm = tensor_data(args, 4);
    __gm__ uint8_t *s_gm = tensor_data(args, 5);
    __gm__ uint8_t *p_gm = tensor_data(args, 6);
    __gm__ uint8_t *o_tmp_gm = tensor_data(args, 7);
    __gm__ uint8_t *go_gm = tensor_data(args, 8);
    __gm__ uint8_t *o_core_tmp_gm = tensor_data(args, 9);
    __gm__ uint8_t *l_gm = tensor_data(args, 10);
    __gm__ uint8_t *gm_k16 = tensor_data(args, 11);
    __gm__ uint8_t *gm_v16 = tensor_data(args, 12);
    __gm__ uint8_t *tiling_para_gm = tensor_data(args, 13);
    __gm__ uint8_t *null_gm = tensor_data(args, 14);
    const uint32_t pto_block_idx = static_cast<uint32_t>(get_block_idx(args));
    const uint32_t pto_block_num = static_cast<uint32_t>(get_block_num(args));
#ifdef __DAV_C220_VEC__
    const uint32_t pto_sub_block_id = static_cast<uint32_t>(get_sub_block_id(args));
#else
    const uint32_t pto_sub_block_id = 0;
#endif

    paged_attention_mask_body(
        nullptr, pto_block_idx, pto_block_num, pto_sub_block_id, q_gm, k_gm, v_gm, block_tables_gm, null_gm, null_gm,
        null_gm, null_gm, null_gm, null_gm, null_gm, null_gm, null_gm, o_gm, s_gm, p_gm, o_tmp_gm, go_gm, o_core_tmp_gm,
        l_gm, gm_k16, gm_v16, tiling_para_gm
    );
}

#endif
