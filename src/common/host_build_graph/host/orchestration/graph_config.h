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
#include <unordered_map>

#include "host_build_graph/host/graph_cache.h"

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
// Every field a Config compares in operator== belongs here, and in the same order. A field
// that reaches the comparison but not the hash makes two distinct Configs collide on one
// key, which is detected but only as a failure.
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
 * A Config decides which Definition a submission reaches. Its hash joins the Graph key, so
 * two different Configs of one Graph record two Definitions; and the Config a key was first
 * submitted with is retained and compared against every later submission on that key, which
 * is what turns a hash collision into a detectable one rather than a silently wrong replay.
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

// The Config each Graph key was submitted with during the current bind.
//
// Held by base pointer so one table covers every Config type -- which is what lets a
// collision between two *different* Config types be caught at all: a per-type table would
// put them in separate maps where neither sees the other.
//
// Cleared at the end of each bind, through the .so's framework_finish_orch. That pairs it
// with the host's Definition cache, which is a local of run_host_orchestration and dies with
// it: an entry here means that key has a Definition, and both go away together. A table that
// outlived the cache would report a collision between two Configs that, the cache having
// been emptied in between, could never have been confused for one another.
//
// Two threads reach this table. The orchestration entry submits a Graph on its own thread,
// and a recorded body runs on a recorder worker that outlives the submit call which handed
// it over -- so a Graph submitted from inside a recorded body arrives here from that worker
// while the entry thread is still submitting. That nesting is refused, but the refusal is
// reached through graph_begin, which is downstream of this table: the entry below is already
// written by then. Every access therefore holds config_registry_mutex().
inline std::unordered_map<uint64_t, std::unique_ptr<ConfigBase>> &config_registry() {
    static std::unordered_map<uint64_t, std::unique_ptr<ConfigBase>> registry;
    return registry;
}

// Guards config_registry(). Held across a whole retain -- the lookup, the insert and the
// comparison are one decision, and two threads that each found the key absent would both
// insert, leaving whichever lost the race recorded as the key's owner.
inline std::mutex &config_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

// Release every key, as framework_finish_orch does at the end of a bind.
inline void clear_config_registry() {
    const std::lock_guard<std::mutex> lock(config_registry_mutex());
    config_registry().clear();
}

// Retain this key's Config, or check it against the one already retained. False means two
// Configs disagree on one key.
//
// Two Configs disagreeing on one key means the key no longer identifies the Graph it was
// built from, so the Definition cache would hand back the wrong body. The caller reports it:
// a Config's hash() is written by whoever declared the Config, so a collision is something a
// caller can produce, and this runtime's rule for those is a fatal naming the reason rather
// than an abort. The report lives at the submission because that is where the runtime handle
// the fatal goes through is in scope.
//
// dynamic_cast rather than a type tag: the retained object, this comparison, and every
// Config type reachable here all live in the one orchestration .so, so there is no cross-.so
// RTTI identity to worry about.
template <typename ConfigT>
[[nodiscard]] inline bool retain_config(uint64_t graph_key, const ConfigT &config) {
    const std::lock_guard<std::mutex> lock(config_registry_mutex());
    auto &registry = config_registry();
    auto it = registry.find(graph_key);
    if (it == registry.end()) {
        registry.emplace(graph_key, std::make_unique<ConfigT>(config));
        return true;
    }
    const auto *retained = dynamic_cast<const ConfigT *>(it->second.get());
    return retained != nullptr && *retained == config;
}

}  // namespace simpler::hbg
