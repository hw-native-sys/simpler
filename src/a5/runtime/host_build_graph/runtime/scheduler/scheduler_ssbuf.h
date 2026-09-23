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
    volatile uint32_t scheduler_lane;
    uint8_t reserved[52];
};

struct alignas(32) SchedulerSsbufDispatchControl {
    // Low 32 bits: ready generation; high 32 bits: signed timing slot.
    volatile uint64_t publication;
    int64_t task_id;
    uint8_t reserved[16];
};

struct alignas(32) SchedulerSsbufCompletionInbox {
    // Low/high halves are slot 0/1 generations, owned by one Executor.
    volatile uint64_t publication;
    uint8_t reserved[24];
};

// Keep shared slots cache-line aligned without over-aligning core-local copies.
struct alignas(64) SchedulerSsbufTaskTrace {
    SchedulerExecutorTaskTrace payload;
};

struct alignas(64) SchedulerSsbufLane {
    SchedulerSsbufDispatchControl dispatch[SCHEDULER_PENDING_SLOT_COUNT];
    SchedulerSsbufTaskTrace traces[SCHEDULER_PENDING_SLOT_COUNT];
    SchedulerSsbufCompletionInbox completion;
    uint8_t reserved[32];
};

struct alignas(64) SchedulerSsbufRegion {
    // +0x000: initialization and layout contract
    SchedulerSsbufHeader header;
    // +0x040, +0x140, +0x240: one 256-byte lane per cluster worker
    SchedulerSsbufLane lanes[PLATFORM_CORES_PER_BLOCKDIM];
    // +0x340: reserved through the end of the 1 KiB HBG region
    uint8_t reserved[192];
};

static_assert(SCHEDULER_SSBUF_REGION_OFFSET + SCHEDULER_SSBUF_REGION_SIZE == SCHEDULER_SSBUF_HARDWARE_SIZE);
static_assert(sizeof(SchedulerSsbufHeader) == 64);
static_assert(sizeof(SchedulerSsbufDispatchControl) == 32);
static_assert(sizeof(SchedulerSsbufCompletionInbox) == 32);
static_assert(sizeof(SchedulerSsbufTaskTrace) == 64);
static_assert(sizeof(SchedulerSsbufLane) == 256);
static_assert(sizeof(SchedulerSsbufRegion) == SCHEDULER_SSBUF_REGION_SIZE);
static_assert(SCHEDULER_PENDING_SLOT_COUNT == 2);
static_assert(offsetof(SchedulerSsbufDispatchControl, publication) % 8 == 0);
static_assert(offsetof(SchedulerSsbufCompletionInbox, publication) % 8 == 0);
static_assert(offsetof(SchedulerSsbufLane, dispatch[0]) == 0x00);
static_assert(offsetof(SchedulerSsbufLane, dispatch[1]) == 0x20);
static_assert(offsetof(SchedulerSsbufLane, traces[0]) == 0x40);
static_assert(offsetof(SchedulerSsbufLane, traces[1]) == 0x80);
static_assert(offsetof(SchedulerSsbufLane, completion) == 0xc0);
static_assert(offsetof(SchedulerSsbufRegion, lanes[0]) == 0x040);
static_assert(offsetof(SchedulerSsbufRegion, lanes[1]) == 0x140);
static_assert(offsetof(SchedulerSsbufRegion, lanes[2]) == 0x240);
static_assert(offsetof(SchedulerSsbufRegion, reserved) == 0x340);

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

inline __aicore__ void
scheduler_ssbuf_initialize(SCHEDULER_SSBUF SchedulerSsbufRegion *region, uint32_t scheduler_lane) {
    // SSBUF may contain user data from an earlier launch. Only generation
    // tokens gate consumers; publish the header token after every lane token
    // and all immutable metadata have been initialized. Consumers enter only
    // after AICPU observes bootstrap_complete and releases the launch gate.
    // Reserved bytes are unspecified and must not be read; new fields require
    // explicit initialization when the layout is extended.
    scheduler_ssbuf_store_release(&region->header.init_generation, 0);
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
            scheduler_ssbuf_store_relaxed(&region->lanes[lane].dispatch[slot].publication, 0);
        }
        scheduler_ssbuf_store_relaxed(&region->lanes[lane].completion.publication, 0);
    }
    region->header.region_size = SCHEDULER_SSBUF_REGION_SIZE;
    region->header.scheduler_lane = scheduler_lane;
    scheduler_ssbuf_store_release(&region->header.init_generation, SCHEDULER_SSBUF_INIT_GENERATION);
}

inline __aicore__ bool
scheduler_ssbuf_is_initialized(const SCHEDULER_SSBUF SchedulerSsbufRegion *region, uint32_t scheduler_lane) {
    return scheduler_ssbuf_load_acquire(&region->header.init_generation) == SCHEDULER_SSBUF_INIT_GENERATION &&
           region->header.region_size == SCHEDULER_SSBUF_REGION_SIZE && region->header.scheduler_lane == scheduler_lane;
}
