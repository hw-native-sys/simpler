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

#include <stdint.h>

#include <memory>
#include <mutex>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>

#include "host_build_graph/graph_cache.h"

namespace simpler::hbg {

// Mix one scalar field in. Category and width join the bytes so that two fields holding the
// same bit pattern at different types -- int32_t{1} and the float whose pattern that is --
// do not hash alike.
//
// Exposed for a Config that builds its hash in steps; the usual spelling is config_hash.
template <typename T>
inline uint64_t config_hash_field(uint64_t hash, T value) {
    using Value = std::remove_cv_t<std::remove_reference_t<T>>;
    static_assert(
        std::is_integral_v<Value> || std::is_same_v<Value, float> || std::is_same_v<Value, double>,
        "config_hash takes integral, float, or double fields; hash anything else with "
        "graph_hash_bytes on top of config_hash_field"
    );
    constexpr uint8_t category = std::is_same_v<Value, bool> ? 1 :
                                 std::is_integral_v<Value>   ? (std::is_signed_v<Value> ? 2 : 3) :
                                                               4;
    constexpr uint8_t width = sizeof(Value);
    hash = graph_hash_byte(hash, category);
    hash = graph_hash_byte(hash, width);
    return graph_hash_bytes(hash, &value, sizeof(value));
}

// A Config's hash from its fields: `return config_hash(layer, route_by_hash);`
//
// Every field a Config compares in operator== belongs here, and in the same order. Unequal Configs may share a hash:
// equality separates them in the registry. Mixing every field keeps those collision buckets small.
//
// The first field hashes from zero. FNV's offset basis would buy a little avalanche quality
// and cost a constant a Config author has no use for; every field already mixes in its
// category and width ahead of its bytes, and this value is recomputed on both sides of every
// run rather than persisted. A single zero-valued field still does not collide with
// DefaultConfig's zero, because those two bytes precede it.
template <typename... Fields>
inline uint64_t config_hash(const Fields &...fields) {
    static_assert(sizeof...(Fields) >= 1, "config_hash: a Config with no fields hashes as DefaultConfig");
    uint64_t hash = 0;
    ((hash = config_hash_field(hash, fields)), ...);
    return hash;
}

/**
 * A Graph's construction parameters: the values its body reads as fixed data, as opposed to
 * the boundary scalars it forwards per invocation.
 *
 * A Config decides which Definition a submission reaches. Its hash selects a registry
 * bucket; the Graph identity, Config type, and operator== select an entry within it.
 * Configs equal under operator== must return the same hash.
 * Each distinct identity receives a unique key for the host cache, so a hash collision
 * records another Definition rather than replaying the wrong one or rejecting the run.
 *
 * A Config derives from ConfigBase and declares two ordinary member functions:
 *
 *     struct ExtendConfig : simpler::hbg::ConfigBase {
 *         int layer;
 *         bool route_by_hash;
 *
 *         uint64_t hash() const { return config_hash(layer, route_by_hash); }
 *         bool operator==(const ExtendConfig &o) const {
 *             return layer == o.layer && route_by_hash == o.route_by_hash;
 *         }
 *     };
 *
 * Neither is virtual, and ConfigBase declares neither: both are called where the Config's
 * own type is known, which is the submitting template in orchestration_api.h. Omitting
 * either is a compile error at the submission, through the traits below.
 *
 * A Config must be copy-constructible and self-contained -- the retained copy outlives the
 * submit call that lent the caller's object, and the recording job that reads it runs on
 * another thread. A pointer or string_view into caller storage dangles.
 */
class ConfigBase {
public:
    // The only member, and it carries two jobs: the retained Config is held by
    // unique_ptr<ConfigBase> so deletion has to be virtual, and a polymorphic base is what
    // lets the comparison recover the derived type with dynamic_cast.
    virtual ~ConfigBase() = default;
};

// The Config of a Graph that takes no construction parameters. Submitting it and submitting
// nothing are the same submission: no values to hash, and no values to disagree on.
struct DefaultConfig : ConfigBase {
    uint64_t hash() const { return 0; }
    bool operator==(const DefaultConfig &) const { return true; }
};

// Whether T declares `uint64_t hash() const`.
template <typename T, typename = void>
inline constexpr bool has_config_hash_v = false;
template <typename T>
inline constexpr bool has_config_hash_v<T, std::void_t<decltype(static_cast<uint64_t (T::*)() const>(&T::hash))>> =
    true;

// Whether T declares `bool operator==(const T &) const` itself.
//
// The cast to the exact member-function type is what makes this a question about T rather
// than about what T happens to compare with: a bare
// `declval<const T &>() == declval<const T &>()` probe is satisfied by any operator reachable
// through a conversion, so a Config that declares none could still answer yes.
template <typename T, typename = void>
inline constexpr bool has_config_equality_v = false;
template <typename T>
inline constexpr bool
    has_config_equality_v<T, std::void_t<decltype(static_cast<bool (T::*)(const T &) const>(&T::operator==))>> = true;

// What a Config must provide. Neither hash() nor operator== is virtual, and ConfigBase
// declares neither: both are called where ConfigT is known, which is the submitting
// template. So this is where a Config that omits one is rejected, and the message is the one
// below rather than a template expansion.
template <typename ConfigT>
inline void assert_config() {
    static_assert(std::is_base_of_v<ConfigBase, ConfigT>, "a Graph Config must derive from simpler::hbg::ConfigBase");
    static_assert(has_config_hash_v<ConfigT>, "a Graph Config must provide `uint64_t hash() const`");
    static_assert(
        has_config_equality_v<ConfigT>, "a Graph Config must provide `bool operator==(const ItsOwnType &) const`"
    );
}

// One exact Graph identity retained for the bind. Config stays in the orchestration
// library; only graph_key crosses the RuntimeOps boundary.
struct RetainedGraphConfig {
    uint64_t graph_id;
    uint64_t graph_key;
    std::unique_ptr<ConfigBase> config;
};

// Digest buckets may contain different Graph ids, Config types, or Config values.
// Entries are only inserted during a bind and cleared together after recorder jobs
// finish, matching the lifetime of the host Definition cache.
inline std::unordered_multimap<uint64_t, RetainedGraphConfig> &config_registry() {
    static std::unordered_multimap<uint64_t, RetainedGraphConfig> registry;
    return registry;
}

// Nested submissions can reach the registry from a recorder thread before graph_begin
// rejects the nesting. Lookup, insertion, and key allocation share one critical section.
inline std::mutex &config_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline void clear_config_registry() {
    const std::lock_guard<std::mutex> lock(config_registry_mutex());
    config_registry().clear();
}

// Return the same key for an equal identity, or retain a copy with a fresh key.
// Keys are insertion ordinals, independent of digest values. With no per-entry erase,
// size() + 1 is unique until the bind ends; clearing the registry resets that namespace.
// The host mixes this key with the callable's address. For a fixed callable its final
// XOR and odd multiplication are bijective, so distinct ordinals stay distinct there.
// RTTI and operator== remain local to the orchestration library that owns the Config.
template <typename ConfigT>
[[nodiscard]] inline uint64_t retain_config(uint64_t graph_id, const ConfigT &config) {
    assert_config<ConfigT>();
    const uint64_t digest = rt_graph_make_key(graph_id, config.hash());
    const std::lock_guard<std::mutex> lock(config_registry_mutex());
    auto &registry = config_registry();
    const auto range = registry.equal_range(digest);
    for (auto it = range.first; it != range.second; ++it) {
        const auto &entry = it->second;
        if (entry.graph_id != graph_id || typeid(*entry.config) != typeid(ConfigT)) continue;
        if (dynamic_cast<const ConfigT &>(*entry.config) == config) return entry.graph_key;
    }
    const uint64_t graph_key = registry.size() + 1;
    registry.emplace(digest, RetainedGraphConfig{graph_id, graph_key, std::make_unique<ConfigT>(config)});
    return graph_key;
}

}  // namespace simpler::hbg
