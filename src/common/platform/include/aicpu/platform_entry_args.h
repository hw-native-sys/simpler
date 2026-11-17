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
/**
 * The launch package's entry-argument header, as one AICPU thread received it.
 *
 * The platform entry holds the only pointer to this launch's arguments and the
 * runtime entry takes a `Runtime *` alone, so a runtime that reads entry values
 * from the launch package needs them forwarded — as the register tables and
 * profiling bases already are. Globals inside the AICPU SO rather than
 * `thread_local`, per docs/dynamic-linking.md.
 *
 * **One view slot per gate survivor, and only that survivor touches it.**
 * Concurrent threads must not overwrite each other's published header. HBG
 * also uses the current thread's argument-copy base for its Graph Definition.
 * TMR's relocated input is read-only RTS-owned storage and may be shared by
 * the launch's threads; per-thread view slots do not make the input private.
 * Distinct threads write distinct view objects, and each thread reads the view
 * it published. No atomic is needed for these slots.
 *
 * `exec_idx` is `platform_aicpu_affinity_thread_idx()`, the gate's deterministic
 * survivor position, which is also the index `AicpuExecutor::run` assigns roles
 * by. Publish only after the gate has kept this thread: a dropped thread returns
 * from the entry and its arguments go with it.
 *
 * An index the gate never issued — a variant with no filter gate, where the
 * index reads -1 — carries no published view and reads as the descriptor route,
 * which is what those variants use.
 *
 * RTS relocates the TMR entry pointer into its device argument allocation.
 * The runtime checks source, counts and pointer before consuming that input.
 * HBG retains its separately validated base-plus-offset Graph Definition view.
 *
 * Implementation: common/platform/shared/aicpu/platform_entry_args.cpp
 */

#pragma once

#include <cstdint>
#include <cstddef>

#include "common/launch_entry_args.h"

/**
 * One thread's view of its launch package: the entry region, and the Graph
 * Definition section when this run carries one.
 *
 * `entry_data` is RTS's relocated host-input pointer; `args_base` and the graph
 * fields describe HBG's Graph Definition section. Each thread publishes its
 * own view after the affinity gate.
 */
struct PlatformEntryArgs {
    const void *args_base{nullptr};
    const void *entry_data{nullptr};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    uint32_t source{static_cast<uint32_t>(EntryArgsSource::Descriptor)};
    uint32_t graph_offset{0};
    uint32_t graph_bytes{0};
    uint32_t graph_source{static_cast<uint32_t>(GraphSectionSource::None)};
    uint32_t cache_maintenance{static_cast<uint32_t>(EntryCacheMaintenance::Full)};
};
static_assert(offsetof(PlatformEntryArgs, cache_maintenance) == 40);
static_assert(sizeof(PlatformEntryArgs) == 48);

/**
 * Publish this thread's view, after the affinity gate has kept it.
 *
 * @param exec_idx  This thread's gate survivor index; an out-of-range index
 *                  publishes nothing rather than writing a slot it does not own
 */
void set_platform_entry_args(int32_t exec_idx, const PlatformEntryArgs &view);

/**
 * This thread's own published view.
 *
 * @param exec_idx  The same index the caller published under. An index outside
 *                  the gate's range, or one that published nothing, reads as the
 *                  descriptor route
 */
PlatformEntryArgs get_platform_entry_args(int32_t exec_idx);
