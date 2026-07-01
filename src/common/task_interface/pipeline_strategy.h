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
#ifndef SRC_COMMON_TASK_INTERFACE_PIPELINE_STRATEGY_H_
#define SRC_COMMON_TASK_INTERFACE_PIPELINE_STRATEGY_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>

enum class PipelineStrategy : int32_t {
    BASELINE = 0,
    S2_O2_SPLIT_CTRL_STRATEGY2 = 2,
};

constexpr int32_t PIPELINE_LAYOUT_MAX_THREADS = 6;
constexpr int32_t PIPELINE_STRATEGY_UNSET_BASELINE = -1;

struct PipelineLayout {
    PipelineStrategy strategy{PipelineStrategy::BASELINE};
    int32_t scheduler_threads{0};
    int32_t orchestrator_threads{1};
    int32_t scheduler_index_by_thread[PIPELINE_LAYOUT_MAX_THREADS]{-1, -1, -1, -1, -1, -1};
    int32_t orchestrator_stage_by_thread[PIPELINE_LAYOUT_MAX_THREADS]{-1, -1, -1, -1, -1, -1};
    const char *name{"baseline"};
    const char *cluster_layout{"baseline"};
};

static inline PipelineLayout resolve_pipeline_layout(int32_t raw_strategy) {
    switch (raw_strategy) {
    case 2:
        return {
            PipelineStrategy::S2_O2_SPLIT_CTRL_STRATEGY2,
            2,
            2,
            {0, 1, -1, -1, -1, -1},
            {-1, -1, 0, 1, -1, -1},
            "2S2O_split_ctrl_strategy2",
            "cluster0=S0,S1;cluster1=O0,O1",
        };
    default:
        return {
            PipelineStrategy::BASELINE, 0,          1,          {-1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1},   "baseline", "baseline",
        };
    }
}

static inline int32_t resolve_pipeline_strategy_with_env(int32_t config_strategy) {
    const char *env = std::getenv("SIMPLER_PIPELINE_STRATEGY");
    if (env == nullptr || env[0] == '\0') {
        return config_strategy;
    }
    char *end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end == env) {
        return config_strategy;
    }
    return static_cast<int32_t>(parsed);
}

static inline bool resolve_pipeline_defer_submit_disabled_with_env() {
    const char *env = std::getenv("SIMPLER_PIPELINE_DEFER_SUBMIT");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0 ||
           std::strcmp(env, "off") == 0 || std::strcmp(env, "OFF") == 0 || std::strcmp(env, "no") == 0 ||
           std::strcmp(env, "NO") == 0;
}

#endif  // SRC_COMMON_TASK_INTERFACE_PIPELINE_STRATEGY_H_
