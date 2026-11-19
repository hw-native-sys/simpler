// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Engram projection, activation quantization and residual gating.
#include "mx_quant.hpp"
#include "rmsnorm.hpp"
#include "mxfp8_projection.hpp"
#if defined(__CPU_SIM)
#include <cmath>
#endif

// External calls place tensors before scalars. Split vector modes occupy args[5].
namespace engram_abi {
struct Quantize {
    static constexpr int payload = 1, activation_scale = 2, rows = 3, padded = 4;
};
struct Projection {
    static constexpr int payload = 0, weight = 1, activation_scale = 2, weight_scale = 3, kv = 4;
    static constexpr int rows = 5, row_tile = 6;
};
struct Gate {
    static constexpr int kv = 0, norm = 1, hidden = 2, output = 3, rows = 4;
};
} // namespace engram_abi

namespace engram_quant {
// Engram activation quantization with M16-packed E8M0 scales.
using namespace mx_quant;
namespace {
#if defined(__DAV_VEC__)
constexpr int width = 6144, columns = 256;
using Input = Tile<TileType::Vec, bfloat16_t, 16, columns, BLayout::RowMajor, DYNAMIC, columns>;
using Shape = TileShape2D<bfloat16_t, DYNAMIC, columns>;
using GlobalStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
__aicore__ inline int valid_rows(int active, int work) {
    int rows = active - work / (width / columns) * 16;
    return rows < 0 ? 0 : (rows > 16 ? 16 : rows);
}
template <typename Abi> __aicore__ inline void load_work(__gm__ int64_t* args, int work, int slot) {
    int rows = valid_rows(args[Abi::rows], work);
    if (rows > 0) {
        Input input(rows);
        TASSIGN(input, slot * 16384);
        int t0 = work / (width / columns) * 16, k0 = work % (width / columns) * columns;
        GlobalTensor<bfloat16_t, Shape, GlobalStride> source(
            data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + k0, Shape(rows, columns),
            GlobalStride(1, 1, 1, stride(args, 0), 1));
        TLOAD(input, source);
    }
}
#endif
} // namespace
template <typename Abi> __aicore__ inline void run(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    int first = get_block_idx(args),
        workers = get_block_num(args);
    int total = args[Abi::padded] / 16 * (width / columns);
    if (first >= total)
        return;
    load_work<Abi>(args, first, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    // Each 16-KiB slot contains input, payload, and packed scales.
    // V-to-MTE2 credits protect input reuse; MTE3-to-V credits protect output reuse.
    int iteration = 0;
    for (int work = first; work < total; work += workers, ++iteration) {
        int slot = iteration % 2;
        event_t current = slot == 0 ? EVENT_ID0 : EVENT_ID1;
        event_t next = slot == 0 ? EVENT_ID1 : EVENT_ID0;
        if (work + workers < total) {
            if (iteration > 0)
                wait_flag(PIPE_V, PIPE_MTE2, next);
            load_work<Abi>(args, work + workers, 1 - slot);
            set_flag(PIPE_MTE2, PIPE_V, next);
        }
        int t0 = work / (width / columns) * 16, k0 = work % (width / columns) * columns;
        int rows = valid_rows(args[Abi::rows], work);
        Input input(rows);
        Vec<uint8_t, 16, columns> output;
        Vec<uint8_t, 1, 16 * columns / 32> scales;
        TASSIGN(input, slot * 16384);
        TASSIGN(output, slot * 16384 + 8192);
        TASSIGN(scales, slot * 16384 + 12288);
        wait_flag(PIPE_MTE2, PIPE_V, current);
        wait_flag(PIPE_MTE3, PIPE_V, current);
        quantize<columns, Input, true>(output.data(), scales.data(), input.data(), rows);
        set_flag(PIPE_V, PIPE_MTE2, current);
        set_flag(PIPE_V, PIPE_MTE3, current);
        wait_flag(PIPE_V, PIPE_MTE3, current);
        GlobalTensor<uint8_t, TileShape2D<uint8_t, 16, columns>, GlobalStride> target(
            data<uint8_t>(args, Abi::payload) + t0 * width + k0, TileShape2D<uint8_t, 16, columns>(),
            GlobalStride(1, 1, 1, width, 1));
        GlobalTensor<uint8_t, TileShape2D<uint8_t, 1, 128>, BaseShape2D<uint8_t, 1, 128>> scale_target(
            data<uint8_t>(args, Abi::activation_scale) + t0 * (width / 32) + k0 / columns * 128);
        TSTORE(target, output);
        TSTORE(scale_target, scales);
        set_flag(PIPE_MTE3, PIPE_V, current);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    if (iteration > 1)
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
#endif
}
} // namespace engram_quant

namespace engram_projection {
using namespace mxfp8_projection;
template <typename Abi, int MT> __aicore__ inline void project_rows(__gm__ int64_t* args) {
    constexpr int NT = MT == 80 ? 512 : 256, K = 6144, N = 25600;
    constexpr int InnerK = MT == 80 ? 64 : 128;
    constexpr int COLUMNS = N / NT;
    const int tasks = (args[Abi::rows] + MT - 1) / MT * COLUMNS;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        const int row = task / COLUMNS * MT;
        const int n = task % COLUMNS * NT;
        const int rows = args[Abi::rows] - row < MT ? args[Abi::rows] - row : MT;
        project<K, N, NT, MT, false, InnerK>(data<F>(args, Abi::payload) + row * K,
                                                data<F>(args, Abi::weight),
                                                data<S>(args, Abi::activation_scale) + row * (K / 32),
                                                data<S>(args, Abi::weight_scale),
                                                data<bfloat16_t>(args, Abi::kv) + row * N, rows, N, n);
    }
}
template <typename Abi> __aicore__ inline void run(__gm__ int64_t* args) {
    if (args[Abi::row_tile] == 256)
        project_rows<Abi, 256>(args);
    else if (args[Abi::row_tile] == 128)
        project_rows<Abi, 128>(args);
    else
        project_rows<Abi, 80>(args);
}
} // namespace engram_projection

namespace engram_gate {
// Per-row Engram normalization, signed-sqrt gate and rounded residual update.
using namespace pto;
using mx_quant::data;
namespace {
constexpr int D = 5120;
template <typename T> using Row = Tile<TileType::Vec, T, 1, D>;
template <typename T> using GM = GlobalTensor<T, TileShape2D<T, 1, D>, BaseShape2D<T, 1, D>>;
#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void gate(Row<float>::TileDType __out__ output, Row<float>::TileDType __in__ hidden,
                                   Row<bfloat16_t>::TileDType __in__ key,
                                   Row<bfloat16_t>::TileDType __in__ value,
                                   Row<float>::TileDType __in__ weight) {
    auto out = (__ubuf__ float*)__cce_get_tile_ptr(output);
    auto x = (__ubuf__ float*)__cce_get_tile_ptr(hidden);
    auto k = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(key);
    auto v = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(value);
    auto w = (__ubuf__ float*)__cce_get_tile_ptr(weight);
#if defined(__CPU_SIM)
    float sx = 0, sk = 0, dot = 0;
    for (int i = 0; i < D; ++i) {
        float ki = static_cast<float>(k[i]);
        sx += x[i] * x[i];
        sk += ki * ki;
        dot += (x[i] * w[i]) * ki;
    }
    dot *= (1.0f / std::sqrt(sx / D + 1e-6f)) * (1.0f / std::sqrt(sk / D + 1e-6f)) * 0.013975424859373685f;
    float signed_root = std::copysign(std::sqrt(std::max(std::abs(dot), 1e-6f)), dot);
    float g = 1.0f / (1.0f + std::exp(-signed_root));
    for (int i = 0; i < D; ++i)
        out[i] = static_cast<float>(static_cast<bfloat16_t>(x[i] + g * static_cast<float>(v[i])));
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        RegTensor<float> sx, sk, dot;
        vdup(sx, 0.0f, full, MODE_ZEROING);
        vdup(sk, 0.0f, full, MODE_ZEROING);
        vdup(dot, 0.0f, full, MODE_ZEROING);
        for (uint16_t chunk = 0; chunk < D / 64; ++chunk) {
            RegTensor<float> xv, kv, wv, square, product;
            RegTensor<bfloat16_t> kb;
            vlds(xv, x, chunk * 64, NORM);
            vlds(kb, k, chunk * 64, UNPK_B16);
            vcvt(kv, kb, full, PART_EVEN);
            vlds(wv, w, chunk * 64, NORM);
            vmul(square, xv, xv, full, MODE_ZEROING);
            vadd(sx, sx, square, full, MODE_ZEROING);
            vmul(square, kv, kv, full, MODE_ZEROING);
            vadd(sk, sk, square, full, MODE_ZEROING);
            vmul(product, xv, wv, full, MODE_ZEROING);
            vmul(product, product, kv, full, MODE_ZEROING);
            vadd(dot, dot, product, full, MODE_ZEROING);
        }
        vcadd(sx, sx, full, MODE_ZEROING);
        vcadd(sk, sk, full, MODE_ZEROING);
        vcadd(dot, dot, full, MODE_ZEROING);
        vdup(sx, sx, full, POS_LOWEST, MODE_ZEROING);
        vdup(sk, sk, full, POS_LOWEST, MODE_ZEROING);
        vdup(dot, dot, full, POS_LOWEST, MODE_ZEROING);
        vmuls(sx, sx, 1.0f / D, full, MODE_ZEROING);
        vmuls(sk, sk, 1.0f / D, full, MODE_ZEROING);
        RegTensor<float> ix, ik, root, negative, one, g;
        recipe_rms::inverse_root(ix, sx, full);
        recipe_rms::inverse_root(ik, sk, full);
        vmul(dot, dot, ix, full, MODE_ZEROING);
        vmul(dot, dot, ik, full, MODE_ZEROING);
        vmuls(dot, dot, 0.013975424859373685f, full, MODE_ZEROING);
        vabs(root, dot, full);
        vmaxs(root, root, 1.0e-6f, full, MODE_ZEROING);
        vsqrt(root, root, full, MODE_ZEROING);
        MaskReg nonnegative;
        vcmps_ge(nonnegative, dot, 0.0f, full);
        vmuls(negative, root, -1.0f, full, MODE_ZEROING);
        vsel(root, root, negative, nonnegative);
        vmuls(root, root, -1.0f, full, MODE_ZEROING);
        vexp(root, root, full, MODE_ZEROING);
        vadds(root, root, 1.0f, full, MODE_ZEROING);
        vdup(one, 1.0f, full, MODE_ZEROING);
        vdiv(g, one, root, full, MODE_ZEROING);
        for (uint16_t chunk = 0; chunk < D / 64; ++chunk) {
            RegTensor<float> xv, vv, y;
            RegTensor<bfloat16_t> vb, rounded;
            vlds(xv, x, chunk * 64, NORM);
            vlds(vb, v, chunk * 64, UNPK_B16);
            vcvt(vv, vb, full, PART_EVEN);
            vmul(y, vv, g, full, MODE_ZEROING);
            vadd(y, xv, y, full, MODE_ZEROING);
            vcvt(rounded, y, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(y, rounded, full, PART_EVEN);
            vsts(y, out, chunk * 64, NORM_B32, full);
        }
    }
#endif
}
#endif
} // namespace
template <typename Abi> __aicore__ inline void run(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    const int first = get_block_idx(args),
              workers = get_block_num(args);
    if (first >= args[Abi::rows] * 4)
        return;
    Row<float> hidden, weight, output;
    Row<bfloat16_t> key, value;
    TASSIGN(hidden, 0);
    TASSIGN(key, 20480);
    TASSIGN(value, 30720);
    TASSIGN(weight, 40960);
    TASSIGN(output, 61440);
    // Retain one copy's weight when the worker stride preserves that copy.
    GM<float> wg(data<float>(args, Abi::norm) + (first % 4) * D);
    TLOAD(weight, wg);
    for (int work = first; work < args[Abi::rows] * 4; work += workers) {
        int row = work / 4, copy = work % 4;
        if (workers % 4 != 0 && work != first) {
            GM<float> next_weight(data<float>(args, Abi::norm) + copy * D);
            TLOAD(weight, next_weight);
        }
        GM<float> xg(data<float>(args, Abi::hidden) + work * D);
        GM<bfloat16_t> kg(data<bfloat16_t>(args, Abi::kv) + row * 5 * D + copy * D);
        GM<bfloat16_t> vg(data<bfloat16_t>(args, Abi::kv) + row * 5 * D + 4 * D);
        TLOAD(hidden, xg);
        TLOAD(key, kg);
        TLOAD(value, vg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        gate(output.data(), hidden.data(), key.data(), value.data(), weight.data());
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GM<float> yg(data<float>(args, Abi::output) + work * D);
        TSTORE(yg, output);
        pipe_barrier(PIPE_ALL);
    }
#endif
}
} // namespace engram_gate

namespace engram_gate_pairs {
#if defined(__DAV_VEC__)
using namespace pto;
using mx_quant::data;
constexpr int D = 5120, InputSlot = 71680;
using H = Tile<TileType::Vec, float, 1, 2 * D>;
using K = Tile<TileType::Vec, bfloat16_t, 1, 2 * D>;
using V = Tile<TileType::Vec, bfloat16_t, 1, D>;
using HG = GlobalTensor<float, TileShape2D<float, 1, 2 * D>, BaseShape2D<float, 1, 4 * D>>;
using KG = GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, 1, 2 * D>, BaseShape2D<bfloat16_t, 1, 5 * D>>;
template <typename Abi> __aicore__ inline void load_work(__gm__ int64_t* args, int work, int slot) {
    H h;
    K k;
    V v;
    TASSIGN(h, slot * InputSlot);
    TASSIGN(k, slot * InputSlot + 40960);
    TASSIGN(v, slot * InputSlot + 61440);
    HG hg(data<float>(args, Abi::hidden) + work * 2 * D);
    KG kg(data<bfloat16_t>(args, Abi::kv) + (work / 2) * 5 * D + (work % 2) * 2 * D);
    engram_gate::GM<bfloat16_t> vg(data<bfloat16_t>(args, Abi::kv) + (work / 2) * 5 * D + 4 * D);
    TLOAD(h, hg);
    TLOAD(k, kg);
    TLOAD(v, vg);
}
#endif
template <typename Abi> __aicore__ inline void run(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    int first = get_block_idx(args);
    int workers = get_block_num(args);
    int total = args[Abi::rows] * 2;
    if (first >= total)
        return;
    H weights;
    TASSIGN(weights, 143360);
    HG wg(data<float>(args, Abi::norm) + (first % 2) * 2 * D);
    TLOAD(weights, wg);
    load_work<Abi>(args, first, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    int iteration = 0;
    for (int work = first; work < total; work += workers, ++iteration) {
        int slot = iteration % 2;
        event_t current = slot == 0 ? EVENT_ID0 : EVENT_ID1;
        event_t next = slot == 0 ? EVENT_ID1 : EVENT_ID0;
        if (work != first && workers % 2 != 0) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
            HG next_weight(data<float>(args, Abi::norm) + (work % 2) * 2 * D);
            TLOAD(weights, next_weight);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        }
        if (work + workers < total) {
            if (iteration > 0)
                wait_flag(PIPE_V, PIPE_MTE2, next);
            load_work<Abi>(args, work + workers, 1 - slot);
            set_flag(PIPE_MTE2, PIPE_V, next);
        }
        wait_flag(PIPE_MTE2, PIPE_V, current);
        for (int copy = 0; copy < 2; ++copy) {
            event_t output_event = copy == 0 ? EVENT_ID2 : EVENT_ID3;
            engram_gate::Row<float> h, w, out;
            engram_gate::Row<bfloat16_t> k, v;
            TASSIGN(h, slot * InputSlot + copy * D * 4);
            TASSIGN(k, slot * InputSlot + 40960 + copy * D * 2);
            TASSIGN(v, slot * InputSlot + 61440);
            TASSIGN(w, 143360 + copy * D * 4);
            TASSIGN(out, 184320 + copy * D * 4);
            wait_flag(PIPE_MTE3, PIPE_V, output_event);
            engram_gate::gate(out.data(), h.data(), k.data(), v.data(), w.data());
            set_flag(PIPE_V, PIPE_MTE3, output_event);
            wait_flag(PIPE_V, PIPE_MTE3, output_event);
            engram_gate::GM<float> og(data<float>(args, Abi::output) + (work * 2 + copy) * D);
            TSTORE(og, out);
            set_flag(PIPE_MTE3, PIPE_V, output_event);
        }
        set_flag(PIPE_V, PIPE_MTE2, current);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    if (iteration > 1)
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    pipe_barrier(PIPE_ALL);
#endif
}
} // namespace engram_gate_pairs

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_CUBE__)
    engram_projection::run<engram_abi::Projection>(args);
#elif defined(__DAV_VEC__)
    if (args[5] == 1) {
        engram_quant::run<engram_abi::Quantize>(args);
    } else if (args[engram_abi::Gate::rows] >= 256) {
        engram_gate_pairs::run<engram_abi::Gate>(args);
    } else {
        engram_gate::run<engram_abi::Gate>(args);
    }
#endif
}
