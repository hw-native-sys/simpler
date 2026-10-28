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

#include <cstddef>
#include <cstdint>

#include "scheduler_types.h"

#if defined(__CCE_AICORE__)
#define SCHEDULER_SSBUF __ssbuf__
#else
#define SCHEDULER_SSBUF
#endif

// A5 exposes one 3 KiB SSBUF per 1AIC+2AIV cluster. HBG owns the final
// contiguous KiB and deliberately leaves the low 2 KiB available to user
// kernels. All addresses below are relative to the hardware SSBUF base.
constexpr uint32_t SCHEDULER_SSBUF_HARDWARE_SIZE = PLATFORM_SSBUF_SIZE;
constexpr uint32_t SCHEDULER_SSBUF_REGION_OFFSET = 2 * 1024;
constexpr uint32_t SCHEDULER_SSBUF_REGION_SIZE = 1024;
constexpr uint32_t SCHEDULER_SSBUF_INIT_GENERATION = 1;

struct alignas(64) SchedulerSsbufHeader {
    volatile uint32_t init_generation;
    volatile uint32_t region_size;
    uint8_t reserved[56];
};

struct alignas(32) SchedulerSsbufDispatchControl {
    // Low 32 bits: ready generation; high 32 bits: signed timing slot.
    volatile uint64_t publication;
    int64_t task_ref;
    uint8_t reserved[16];
};

struct alignas(32) SchedulerSsbufCompletionInbox {
    // Low/high halves are slot 0/1 generations, owned by one Executor.
    volatile uint64_t publication;
    uint8_t reserved[24];
};

struct alignas(64) SchedulerSsbufLane {
    // Completions of this lane's Mix subtask, which the coordinator polls so it
    // can close the dispatch's tracker. The lane retires its own slot; what
    // crosses SSBUF is the fact, not the retirement.
    SchedulerSsbufCompletionInbox mix_completion;
    // Written by the coordinator alone, into whichever slot the index does not
    // name.
    SchedulerSsbufDispatchControl dispatch[SCHEDULER_LANE_SLOT_COUNT];
    // Written by the lane alone. Reading it is how the coordinator tells a
    // dispatch it left here from one the lane has already taken up.
    volatile uint64_t current;
    uint8_t reserved[24];
};

struct alignas(64) SchedulerSsbufRegion {
    // +0x000: initialization and layout contract
    SchedulerSsbufHeader header;
    // +0x040: one 128-byte lane per cluster worker
    SchedulerSsbufLane lanes[PLATFORM_CORES_PER_BLOCKDIM];
    // The region's size is fixed by the hardware window HBG owns, so the room
    // the lanes do not take is declared rather than left implicit.
    uint8_t reserved
        [SCHEDULER_SSBUF_REGION_SIZE - sizeof(SchedulerSsbufHeader) -
         PLATFORM_CORES_PER_BLOCKDIM * sizeof(SchedulerSsbufLane)];
};

static_assert(SCHEDULER_SSBUF_REGION_OFFSET + SCHEDULER_SSBUF_REGION_SIZE == SCHEDULER_SSBUF_HARDWARE_SIZE);
static_assert(sizeof(SchedulerSsbufHeader) == 64);
static_assert(sizeof(SchedulerSsbufDispatchControl) == 32);
static_assert(sizeof(SchedulerSsbufCompletionInbox) == 32);
static_assert(sizeof(SchedulerSsbufLane) == 128);
static_assert(sizeof(SchedulerSsbufRegion) == SCHEDULER_SSBUF_REGION_SIZE);
static_assert(SCHEDULER_PENDING_SLOT_COUNT == 1);
static_assert(offsetof(SchedulerSsbufDispatchControl, publication) % 8 == 0);
static_assert(offsetof(SchedulerSsbufCompletionInbox, publication) % 8 == 0);
static_assert(offsetof(SchedulerSsbufLane, mix_completion) == 0x00);
static_assert(offsetof(SchedulerSsbufLane, dispatch) == 0x20);
static_assert(offsetof(SchedulerSsbufLane, current) == 0x60);
static_assert(offsetof(SchedulerSsbufRegion, lanes[0]) == 0x040);
static_assert(offsetof(SchedulerSsbufRegion, lanes[1]) == 0x0c0);
static_assert(offsetof(SchedulerSsbufRegion, lanes[2]) == 0x140);
// Over half the region is unclaimed. A cross-core dispatch channel is what that
// room is for; sizing one means recomputing this, not assuming the space is
// there.
static_assert(offsetof(SchedulerSsbufRegion, reserved) == 0x1c0);
static_assert(sizeof(SchedulerSsbufRegion::reserved) == SCHEDULER_SSBUF_REGION_SIZE - 0x1c0);

inline __aicore__ SCHEDULER_SSBUF SchedulerSsbufRegion *scheduler_ssbuf_region(uint64_t hardware_base) {
    return reinterpret_cast<SCHEDULER_SSBUF SchedulerSsbufRegion *>(hardware_base + SCHEDULER_SSBUF_REGION_OFFSET);
}

inline __aicore__ uint64_t scheduler_ssbuf_pack_ready(uint32_t generation, int32_t timing_slot) {
    return generation | (static_cast<uint64_t>(static_cast<uint32_t>(timing_slot)) << 32);
}

inline __aicore__ uint64_t scheduler_ssbuf_update_completion(uint64_t publication, uint32_t slot, uint32_t generation) {
    const uint32_t shift = slot * 32;
    return (publication & ~(UINT64_C(0xffffffff) << shift)) | (static_cast<uint64_t>(generation) << shift);
}

// The consumer supplies an acquire barrier on a hit before consuming payload.
inline __aicore__ uint64_t scheduler_ssbuf_load_relaxed(const SCHEDULER_SSBUF volatile uint64_t *token) {
#if defined(__CCE_AICORE__)
    return *token;
#else
    return __atomic_load_n(token, __ATOMIC_RELAXED);
#endif
}

// The caller completes payload writes with a release barrier before publishing.
inline __aicore__ void scheduler_ssbuf_store_relaxed(SCHEDULER_SSBUF volatile uint64_t *token, uint64_t value) {
#if defined(__CCE_AICORE__)
    *token = value;
#else
    __atomic_store_n(token, value, __ATOMIC_RELAXED);
#endif
}

// These are SPSC publication tokens, not atomic read-modify-write state.
// Volatile SSBUF accesses use DSB for payload-before-token release and
// token-before-payload acquire ordering.
inline __aicore__ uint32_t scheduler_ssbuf_load_acquire(const SCHEDULER_SSBUF volatile uint32_t *token) {
#if defined(__CCE_AICORE__)
    const uint32_t value = *token;
    dsb((mem_dsb_t)0);
    return value;
#else
    return __atomic_load_n(token, __ATOMIC_ACQUIRE);
#endif
}

inline __aicore__ void scheduler_ssbuf_store_release(SCHEDULER_SSBUF volatile uint32_t *token, uint32_t value) {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
    *token = value;
#else
    __atomic_store_n(token, value, __ATOMIC_RELEASE);
#endif
}

inline __aicore__ void scheduler_ssbuf_initialize(SCHEDULER_SSBUF SchedulerSsbufRegion *region) {
    // SSBUF may contain user data from an earlier launch. Only generation
    // tokens gate consumers; publish the header token after every lane token
    // and all immutable metadata have been initialized. Consumers enter only
    // after AICPU observes bootstrap_complete and releases the launch gate.
    // Reserved bytes are unspecified and must not be read; new fields require
    // explicit initialization when the layout is extended.
    scheduler_ssbuf_store_release(&region->header.init_generation, 0);
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        for (uint32_t slot = 0; slot < SCHEDULER_LANE_SLOT_COUNT; ++slot)
            scheduler_ssbuf_store_relaxed(&region->lanes[lane].dispatch[slot].publication, 0);
        scheduler_ssbuf_store_relaxed(&region->lanes[lane].current, 0);
        scheduler_ssbuf_store_relaxed(&region->lanes[lane].mix_completion.publication, 0);
    }
    region->header.region_size = SCHEDULER_SSBUF_REGION_SIZE;
    scheduler_ssbuf_store_release(&region->header.init_generation, SCHEDULER_SSBUF_INIT_GENERATION);
}

inline __aicore__ bool scheduler_ssbuf_is_initialized(const SCHEDULER_SSBUF SchedulerSsbufRegion *region) {
    return scheduler_ssbuf_load_acquire(&region->header.init_generation) == SCHEDULER_SSBUF_INIT_GENERATION &&
           region->header.region_size == SCHEDULER_SSBUF_REGION_SIZE;
}
