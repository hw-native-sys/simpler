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

#include <memory>
#include <string>
#include <vector>

#include "common/chip_swimlane_profiling.h"
#include "host/clock_correlation.h"
#include "host/dfx_run_config.h"
#include "host/host_phase_records.h"

/**
 * One run's host-orchestration phase state, held per pipeline slot.
 *
 * A host-orchestrating runtime records these during its bind, which is
 * preparation — and a prepared successor prepares while its predecessor is
 * still executing. So unlike the collector pools, which `arm_collectors_for_run`
 * builds under the execution claim, this cannot be deferred to launch: the
 * records describe the bind, and the `HostOrchestrationBegin` anchor means the
 * instant host orchestration began. What it gets instead is one of these per
 * concurrently-live run, so a successor's bind writes its own.
 *
 * The collector on the far side is still resident and single, so everything
 * destined for it is captured here and published in `publish_to_collector()`,
 * from the launch arming, under the claim.
 */
struct HostPhaseRunState {
    /** Per-event records for this run's bind. Non-copyable and non-movable. */
    simpler::dfx::HostPhaseRecordStore records;

    /**
     * This run's configuration, stamped by `begin_host_phase_run()` before its
     * bind. The runner's own members describe whichever run last held the
     * execution claim, which on an overlapping prepare is the predecessor.
     */
    ChipSwimlaneLevel chip_swimlane_level{ChipSwimlaneLevel::DISABLED};
    std::string output_prefix;

    std::unique_ptr<simpler::dfx::ClockCorrelationProvider> clock_correlation;
    /** Sampled during bind; replayed into the collector at launch. */
    std::vector<simpler::dfx::ClockAnchorSample> orchestration_begin_anchors;
    std::string clock_provider_name;
    std::string clock_provider_unit;
    /**
     * Whether this run's orchestrator phase records come from the host rather
     * than the AICPU, which decides whether the collector sizes a device orch
     * phase pool for it. Only a host-orchestrating bind sets it.
     */
    bool host_orchestrated{false};

    bool wants_records(bool producer_wants_records) const {
        return (producer_wants_records && !output_prefix.empty()) || host_orchestrated;
    }

    /** Forget the previous run in this slot, before a new one binds into it. */
    void begin(const DfxRunConfig &dfx) {
        chip_swimlane_level = dfx.chip_swimlane_level;
        output_prefix = dfx.output_prefix;
        host_orchestrated = false;
        orchestration_begin_anchors.clear();
        clock_provider_name.clear();
        clock_provider_unit.clear();
    }
};
