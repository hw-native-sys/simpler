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
#include "common/core_type.h"
#include "host/pmu_collector.h"

/**
 * The diagnostics metadata one run owns, handed to a retaining collector at
 * admission rather than read at close.
 *
 * The collector holds exactly one resident copy of each of these, written by
 * whichever run launched last. While launches are exclusive that is also this
 * run's, but a successor that submits while its predecessor still executes
 * replaces them before the predecessor closes -- so the predecessor's artifact
 * would carry its successor's core types. Passing them in is what makes each
 * value belong to the run whose bucket keeps it.
 *
 * `core_types` points at the caller's storage and is read only during the
 * admission call; the bucket takes its own copy.
 */
struct RunLocalDfxMetadata {
    const CoreType *core_types{nullptr};
    int core_type_count{0};
    bool host_orchestrated{false};
};

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
    std::string output_prefix;

    bool chip_swimlane_enabled() const { return chip_swimlane_level != ChipSwimlaneLevel::DISABLED; }
    bool dump_args_enabled() const { return dump_args_level != DumpArgsLevel::OFF; }

    /** Mirrors CallConfig::diagnostics_any(). */
    bool diagnostics_any() const {
        return chip_swimlane_enabled() || dump_args_enabled() || pmu_enabled || dep_gen_enabled || scope_stats_enabled;
    }

    static DfxRunConfig from(const CallConfig &config) {
        DfxRunConfig resolved;
        resolved.chip_swimlane_level = static_cast<ChipSwimlaneLevel>(config.enable_chip_swimlane);
        resolved.dump_args_level = static_cast<DumpArgsLevel>(config.enable_dump_args);
        resolved.pmu_enabled = config.enable_pmu > 0;
        resolved.pmu_event_type = resolve_pmu_event_type(config.enable_pmu);
        resolved.dep_gen_enabled = config.enable_dep_gen != 0;
        resolved.scope_stats_enabled = config.enable_scope_stats != 0;
        resolved.output_prefix = config.output_prefix;
        return resolved;
    }
};
