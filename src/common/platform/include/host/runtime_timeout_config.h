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

#ifndef SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_RUNTIME_TIMEOUT_CONFIG_H
#define SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_RUNTIME_TIMEOUT_CONFIG_H

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

constexpr const char *PTO2_OP_EXECUTE_TIMEOUT_US_ENV = "PTO2_OP_EXECUTE_TIMEOUT_US";
constexpr const char *PTO2_STREAM_SYNC_TIMEOUT_MS_ENV = "PTO2_STREAM_SYNC_TIMEOUT_MS";
constexpr const char *PTO2_SCHEDULER_TIMEOUT_MS_ENV = "PTO2_SCHEDULER_TIMEOUT_MS";

// Covers the host stream-sync window before the AICPU scheduler no-progress
// timer is armed: cold kernel registration, orchestration SO dlopen, runtime
// init, and AICore handshake. The host cannot know the later orchestration
// wall-clock maximum from env parsing alone; callers must size stream-sync for
// that graph-specific producer window.
constexpr int32_t RUNTIME_TIMEOUT_SCHEDULER_ARMING_GUARD_MS = 1500;

struct RuntimeTimeoutConfig {
    uint64_t op_execute_timeout_us;
    int32_t stream_sync_timeout_ms;
    int32_t scheduler_timeout_ms;
};

struct HostRuntimeTimeoutConfig {
    uint64_t op_execute_timeout_us;
    int32_t stream_sync_timeout_ms;
    // AICPU scheduler no-progress watchdog override (ms). 0 means "no env
    // override" — the device falls back to its compile-time default
    // (SCHEDULER_TIMEOUT_CYCLES). Latched once per device into InitArgs.
    int32_t scheduler_timeout_ms{0};
};

struct RuntimeTimeoutParseStatus {
    bool op_execute_env_set{false};
    bool op_execute_valid{true};
    bool stream_sync_env_set{false};
    bool stream_sync_valid{true};
    bool scheduler_env_set{false};
    bool scheduler_valid{true};
};

enum class RuntimeTimeoutOrderStatus {
    OK,
    SCHEDULER_NOT_BELOW_OP_EXECUTE,
    OP_EXECUTE_NOT_BELOW_STREAM_SYNC,
    STREAM_SYNC_NOT_COVERING_SCHEDULER_GUARD,
};

inline const char *runtime_timeout_order_status_name(RuntimeTimeoutOrderStatus status) {
    switch (status) {
    case RuntimeTimeoutOrderStatus::OK:
        return "OK";
    case RuntimeTimeoutOrderStatus::SCHEDULER_NOT_BELOW_OP_EXECUTE:
        return "scheduler timeout must be below op-execute timeout";
    case RuntimeTimeoutOrderStatus::OP_EXECUTE_NOT_BELOW_STREAM_SYNC:
        return "op-execute timeout must be below stream-sync timeout";
    case RuntimeTimeoutOrderStatus::STREAM_SYNC_NOT_COVERING_SCHEDULER_GUARD:
        return "stream-sync timeout must cover scheduler timeout plus scheduler-arming guard";
    }
    return "unknown timeout ordering error";
}

inline std::string trim_runtime_timeout_token(const std::string &input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

inline bool parse_runtime_timeout_uint(const char *raw, uint64_t min_value, uint64_t max_value, uint64_t *out_value) {
    if (raw == nullptr || out_value == nullptr) {
        return false;
    }
    std::string token = trim_runtime_timeout_token(raw);
    if (token.empty() || token[0] == '-') {
        return false;
    }

    char *endptr = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(token.c_str(), &endptr, 10);
    if (errno == ERANGE || endptr == token.c_str() || *endptr != '\0') {
        return false;
    }

    uint64_t value = static_cast<uint64_t>(parsed);
    if (value < min_value || value > max_value) {
        return false;
    }

    *out_value = value;
    return true;
}

inline bool apply_runtime_timeout_override(
    const char * /*name*/, const char *raw, uint64_t min_value, uint64_t max_value, uint64_t *out_value
) {
    uint64_t parsed = 0;
    if (!parse_runtime_timeout_uint(raw, min_value, max_value, &parsed)) {
        return false;
    }
    *out_value = parsed;
    return true;
}

inline bool apply_runtime_timeout_override(
    const char *name, const char *raw, uint64_t min_value, uint64_t max_value, int32_t *out_value
) {
    uint64_t parsed = 0;
    if (!apply_runtime_timeout_override(name, raw, min_value, max_value, &parsed)) {
        return false;
    }
    if (parsed > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    *out_value = static_cast<int32_t>(parsed);
    return true;
}

inline RuntimeTimeoutConfig
resolve_runtime_timeout_config(const RuntimeTimeoutConfig &defaults, RuntimeTimeoutParseStatus *status = nullptr) {
    RuntimeTimeoutConfig cfg = defaults;
    if (status != nullptr) {
        *status = RuntimeTimeoutParseStatus{};
    }
    const char *op_env = std::getenv(PTO2_OP_EXECUTE_TIMEOUT_US_ENV);
    if (op_env != nullptr) {
        if (status != nullptr) status->op_execute_env_set = true;
        bool ok = apply_runtime_timeout_override(
            PTO2_OP_EXECUTE_TIMEOUT_US_ENV, op_env, 1, std::numeric_limits<uint64_t>::max(), &cfg.op_execute_timeout_us
        );
        if (status != nullptr) status->op_execute_valid = ok;
    }
    const char *sync_env = std::getenv(PTO2_STREAM_SYNC_TIMEOUT_MS_ENV);
    if (sync_env != nullptr) {
        if (status != nullptr) status->stream_sync_env_set = true;
        bool ok = apply_runtime_timeout_override(
            PTO2_STREAM_SYNC_TIMEOUT_MS_ENV, sync_env, 1, static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
            &cfg.stream_sync_timeout_ms
        );
        if (status != nullptr) status->stream_sync_valid = ok;
    }
    const char *sched_env = std::getenv(PTO2_SCHEDULER_TIMEOUT_MS_ENV);
    if (sched_env != nullptr) {
        if (status != nullptr) status->scheduler_env_set = true;
        bool ok = apply_runtime_timeout_override(
            PTO2_SCHEDULER_TIMEOUT_MS_ENV, sched_env, 1, static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
            &cfg.scheduler_timeout_ms
        );
        if (status != nullptr) status->scheduler_valid = ok;
    }
    return cfg;
}

inline RuntimeTimeoutOrderStatus validate_runtime_timeout_order(const RuntimeTimeoutConfig &cfg) {
    uint64_t scheduler_timeout_us = static_cast<uint64_t>(cfg.scheduler_timeout_ms) * 1000;
    uint64_t stream_sync_timeout_us = static_cast<uint64_t>(cfg.stream_sync_timeout_ms) * 1000;
    uint64_t scheduler_guarded_stream_budget_ms =
        static_cast<uint64_t>(cfg.scheduler_timeout_ms) + RUNTIME_TIMEOUT_SCHEDULER_ARMING_GUARD_MS;
    if (scheduler_timeout_us >= cfg.op_execute_timeout_us) {
        return RuntimeTimeoutOrderStatus::SCHEDULER_NOT_BELOW_OP_EXECUTE;
    }
    if (cfg.op_execute_timeout_us >= stream_sync_timeout_us) {
        return RuntimeTimeoutOrderStatus::OP_EXECUTE_NOT_BELOW_STREAM_SYNC;
    }
    if (static_cast<uint64_t>(cfg.stream_sync_timeout_ms) <= scheduler_guarded_stream_budget_ms) {
        return RuntimeTimeoutOrderStatus::STREAM_SYNC_NOT_COVERING_SCHEDULER_GUARD;
    }
    return RuntimeTimeoutOrderStatus::OK;
}

inline bool runtime_timeout_platform_is_sim(const char *platform) {
    return platform != nullptr && std::strstr(platform, "sim") != nullptr;
}

inline RuntimeTimeoutOrderStatus
validate_runtime_timeout_order_for_platform(const RuntimeTimeoutConfig &cfg, const char *platform) {
    if (runtime_timeout_platform_is_sim(platform)) {
        return RuntimeTimeoutOrderStatus::OK;
    }
    return validate_runtime_timeout_order(cfg);
}

#endif  // SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_RUNTIME_TIMEOUT_CONFIG_H
