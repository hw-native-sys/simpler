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
 * **One slot per gate survivor, and only that survivor touches it.** Every
 * launched thread runs the kernel entry with its *own* copy of the launch
 * arguments, so a single shared slot would be two things at once: a write-write
 * race between threads storing into it, and a pointer whose owner is whichever
 * thread happened to win — possibly one the affinity gate dropped, whose copy
 * is gone by the time a reader dereferences it. Publishing per survivor index
 * removes both: distinct threads write distinct objects, and the thread that
 * reads a slot is the thread that wrote it, still inside the call that owns the
 * block. No atomic is needed because no object here is shared, and nothing is
 * serialized on the dispatch path.
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
 * Nothing here does pointer arithmetic. The base and the offset stay separate
 * until the runtime has checked the offset, the counts and the source against
 * the descriptor's own, because a payload address formed from unchecked values
 * is the thing that must not exist.
 *
 * Implementation: common/platform/shared/aicpu/platform_entry_args.cpp
 */

#pragma once

#include <cstdint>

#include "common/launch_entry_args.h"

/** One thread's view of the launch package's entry region. */
struct PlatformEntryArgs {
    const void *args_base{nullptr};
    uint32_t offset{0};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    uint32_t source{static_cast<uint32_t>(EntryArgsSource::Descriptor)};
};

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
