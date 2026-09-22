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

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "host_build_graph/task_id.h"
#include "host_build_graph/types.h"

struct GraphScopeResult {
    bool execute_block{true};
    bool recording{false};
    TaskId task_id{TaskId::invalid()};
    // The recording this call opened, non-null exactly when `recording` is set.
    // The recording thread hands it back to graph_prepare so binding needs no
    // lookup: several Definitions record at once, and finding one by key would
    // mean traversing the in-flight map under its mutex on a path whose whole
    // purpose is to not contend with the submitting thread.
    void *recording_handle{nullptr};
    // The formal parameters the recorded body must read: the entry's own deep copy, not
    // the caller's arguments, which are only lent for the duration of the submit call.
    // Set exactly when `recording` is, and valid until graph_commit drains the entry --
    // which is after every job that could read it has finished.
    //
    // Const because the boundary is written once, by the submitting thread in graph_begin,
    // and only read after that -- by the recorder, and by later same-key submissions
    // comparing against it. Nothing may write it once it is handed out here.
    //
    // The body must read these and not the caller's arguments even on the synchronous
    // fallback path: recording resolves a task slot's origin against this object's slot
    // array, so an origin from any other object falls outside it and the parameter is
    // recorded as static.
    const GraphTaskArgs *params{nullptr};
};

using GraphSubmitResult = GraphScopeResult;

constexpr uint64_t graph_hash_byte(uint64_t h, uint8_t b) { return (h ^ static_cast<uint64_t>(b)) * 1099511628211ULL; }

// FNV-1a, mixing eight bytes per multiply instead of one. The only thing hashed
// with it is a Graph's `full_key` — a callable hash, a graph id, a config-value
// count and the config values themselves, a few bytes per call.
//
// Streaming: callers chain several calls (see rt_graph_make_key below) and the word
// split makes the result depend on where they split, so two different chainings of
// the same bytes disagree. That is harmless here because a key is only ever produced
// by this one call sequence and never compared against a differently-split digest.
// The value is recomputed on both sides of every run and is never persisted, so it
// carries no cross-version contract.
inline uint64_t graph_hash_bytes(uint64_t h, const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= bytes; i += sizeof(uint64_t)) {
        uint64_t word = 0;
        __builtin_memcpy(&word, p + i, sizeof(word));
        h = (h ^ word) * 1099511628211ULL;
    }
    for (; i < bytes; ++i) {
        h = graph_hash_byte(h, p[i]);
    }
    return h;
}

constexpr uint64_t graph_const_hash_impl(const char *s, uint64_t h) {
    return (*s == '\0') ? h : graph_const_hash_impl(s + 1, graph_hash_byte(h, static_cast<uint8_t>(*s)));
}

constexpr uint64_t GRAPH_KEY(const char *s) { return graph_const_hash_impl(s, 1469598103934665603ULL); }

inline bool rt_graph_args_cacheable(const GraphTaskArgs &args) {
    if (args.has_error() || args.tensor_count() <= 0 || args.tensor_count() > GRAPH_MAX_TENSOR_ARGS) {
        return false;
    }
    return true;
}

inline uint64_t rt_graph_make_key(uint64_t graph_id) { return graph_id; }

template <typename T>
inline uint64_t graph_hash_config_value(uint64_t hash, T value) {
    using Value = std::remove_cv_t<std::remove_reference_t<T>>;
    static_assert(
        std::is_integral_v<Value> || std::is_same_v<Value, float> || std::is_same_v<Value, double>,
        "Graph construction parameters must be integral, float, or double values"
    );
    constexpr uint8_t category = std::is_same_v<Value, bool> ? 1 :
                                 std::is_integral_v<Value>   ? (std::is_signed_v<Value> ? 2 : 3) :
                                                               4;
    constexpr uint8_t width = sizeof(Value);
    hash = graph_hash_byte(hash, category);
    hash = graph_hash_byte(hash, width);
    return graph_hash_bytes(hash, &value, sizeof(value));
}

template <typename... Config>
inline uint64_t rt_graph_make_key(uint64_t graph_id, Config... config) {
    uint64_t hash = graph_hash_bytes(1469598103934665603ULL, &graph_id, sizeof(graph_id));
    const uint32_t count = sizeof...(Config);
    hash = graph_hash_bytes(hash, &count, sizeof(count));
    ((hash = graph_hash_config_value(hash, config)), ...);
    return hash;
}
