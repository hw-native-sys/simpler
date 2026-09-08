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

#include <string>

#include "call_config.h"
#include "common/args_dump.h"
#include "common/chip_swimlane_profiling.h"
#include "host/pmu_collector.h"

/**
 * One run's diagnostics configuration, resolved from its own CallConfig.
 *
 * The runner also holds these values as members, bound by apply_call_config().
 * Those members are correct for anything that reads them while their run holds
 * the execution claim — arming at launch, teardown at reap. They are not correct
 * for `prepare_execution`, which can run for a successor while a predecessor is
 * still in flight: apply_call_config() is skipped for that overlap, so the
 * members still describe the predecessor.
 *
 * A prepare-phase reader therefore resolves the values here, from the CallConfig
 * it was handed, rather than reading the members. The derivations are the same
 * ones the setters apply, kept in one place so the two cannot disagree.
 */
struct DfxRunConfig {
    ChipSwimlaneLevel chip_swimlane_level{ChipSwimlaneLevel::DISABLED};
    DumpArgsLevel dump_args_level{DumpArgsLevel::OFF};
    PmuEventType pmu_event_type{};
    bool pmu_enabled{false};
    bool dep_gen_enabled{false};
    bool scope_stats_enabled{false};
    bool capture_clock_anchors{false};
    std::string output_prefix;

    bool chip_swimlane_enabled() const { return chip_swimlane_level != ChipSwimlaneLevel::DISABLED; }
    bool dump_args_enabled() const { return dump_args_level != DumpArgsLevel::OFF; }

    static DfxRunConfig from(const CallConfig &config) {
        DfxRunConfig resolved;
        resolved.chip_swimlane_level = static_cast<ChipSwimlaneLevel>(config.enable_chip_swimlane);
        resolved.dump_args_level = static_cast<DumpArgsLevel>(config.enable_dump_args);
        resolved.pmu_enabled = config.enable_pmu > 0;
        resolved.pmu_event_type = resolve_pmu_event_type(config.enable_pmu);
        resolved.dep_gen_enabled = config.enable_dep_gen != 0;
        resolved.scope_stats_enabled = config.enable_scope_stats != 0;
        resolved.capture_clock_anchors = config.capture_clock_anchors != 0;
        resolved.output_prefix = config.output_prefix;
        return resolved;
    }
};
