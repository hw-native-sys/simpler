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
 * The percounter scheduler's GM wire structs, shared by the host that builds
 * them, the AICPU that publishes the topology into them, and the AICore lanes
 * that read them.
 *
 * Host-only planning (the layout planner and the table builder) lives in
 * host/region_plan.h; this header is compiled by the AICore
 * target too, so it includes nothing host-only.
 *
 * Every offset here is from the region base, never a pointer, so the whole
 * region survives one memcpy to the device.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// scheduler_protocol_types.h, which scheduler_layout.h pulls in, names LaneMask
// without including it; host_build_graph's own translation units happen to have
// it in scope by then, and this runtime's do not.
#include "common/lane_mask.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/ssbuf.h"
#include "scheduler/scheduler_types.h"

// Continues the SCHEDULER_RUNTIME_MODE_* numbering scheduler_layout.h starts,
// which reaches 4. The value is published into SchedulerBootstrapInputs and read
// back by every tier of this runtime, so it has to agree with those four even
// though the host_build_graph runtime never produces it.
constexpr uint32_t SCHEDULER_RUNTIME_MODE_PERCOUNTER = 5;

namespace simpler::percounter {

inline constexpr uint32_t PERCOUNTER_CORE_TYPE_COUNT = 2;  // 0 = AIC, 1 = AIV

// Ordinary candidates per lane; the MIX and cohort slots are additional.
inline constexpr uint32_t PERCOUNTER_PENDING_CAPACITY = 1;
static_assert(PERCOUNTER_PENDING_CAPACITY > 0);
inline constexpr uint32_t PERCOUNTER_MIX_PAYLOAD_SLOT = PERCOUNTER_PENDING_CAPACITY;
inline constexpr uint32_t PERCOUNTER_SYNC_PAYLOAD_SLOT = PERCOUNTER_MIX_PAYLOAD_SLOT + 1;
inline constexpr uint32_t PERCOUNTER_PAYLOAD_SLOT_COUNT = PERCOUNTER_SYNC_PAYLOAD_SLOT + 1;

/**
 * One cluster's MIX rendezvous and preparation records, held in SSBUF.
 *
 * SSBUF offers no atomics -- `scheduler/ssbuf.h` has an acquire load and a
 * release store and nothing else -- so every field here has exactly one writer.
 * `task_id`, `armed_generation`, `active_mask`, `block_idx`, `block_num` and
 * `cohort_block` and the preparation record are written only by AIV0, which is why the
 * claimer is a fixed lane role rather than an election; entry `i` of the two
 * arrays is written only by cluster lane `i`.
 *
 * `armed_generation` is monotone with odd meaning armed, so one word carries
 * both the generation and the free/busy state, and a value never returns to
 * mean something else. The arrays hold the generation a lane joined at rather
 * than a flag, so an arrival left over from an older generation reads back as
 * absent. Both properties need the line cleared before any lane reads it:
 * SSBUF survives a launch, and a stale odd value would send a lane after a
 * stale task id.
 *
 * Every field is `volatile`: each is read by lanes that did not write it, mostly
 * inside a spin, and a plain member would let the compiler hoist the load.
 */
struct alignas(64) PercounterMixSignal {
    volatile int64_t task_id;
    volatile uint32_t armed_generation;  // odd = armed, even = free
    // Which cluster lanes this rendezvous gathers. It is staged before the
    // generation turns odd and cleared on every path that frees or abandons
    // the line, so a mask under an even generation is an arming in flight for
    // exactly those lanes -- which is what holds one of them back from a
    // cohort block the holds were read too early to see.
    volatile uint32_t active_mask;
    volatile uint32_t arrived[PLATFORM_CORES_PER_BLOCKDIM];   // entry i written by cluster lane i
    volatile uint32_t finished[PLATFORM_CORES_PER_BLOCKDIM];  // entry i written by cluster lane i
    // A MIX task with `block_num > 1` is one rendezvous per block, so the block
    // is part of the payload and moves under the same release as the task id.
    volatile int32_t block_idx;
    volatile int32_t block_num;
    // Zero when this arming is a draw from the MIX list, non-zero when it is a
    // cohort's own block. The rendezvous is identical either way; what differs
    // is where its end is reported, to the cohort record rather than to the MIX
    // done ticket. It carries the cohort's sequence plus one rather than a one
    // so that the value names the cohort in a dump, but nothing compares it:
    // the queue keeps one cohort outstanding, so a cohort rendezvous on this
    // line is always the one whose block the reader holds.
    volatile uint32_t cohort_block;
    // Entry `i` carries the sequence plus one of the cohort whose block cluster
    // lane `i` holds and has not finished, and zero while it holds none; lane
    // `i` is its only writer. It is published before that block can be served,
    // and for a block one lane runs alone it is cleared only after that kernel
    // returns; a MIX cohort's claimer clears it where it arms the line instead.
    //
    // Only a draw from the MIX list reads it, and only for whether it is zero:
    // the queue keeps one cohort outstanding, so which cohort a lane owes is
    // never in question. It is what an arming can see of a lane's other
    // commitment, not a lock -- the reads are not atomic with the fetch_add
    // that takes a block, so the lane's own service order resolves what they
    // could not exclude.
    volatile uint32_t cohort_hold[PLATFORM_CORES_PER_BLOCKDIM];

    // Preparation does not reserve any participant. A named lane may prepare
    // this payload while the task's producers still run, then serve other work.
    // Only armed_generation admits the rendezvous. The claimer writes this
    // separate record under an odd-writing/even-published sequence.
    volatile int64_t prepare_task_id;
    volatile int32_t prepare_block_idx;
    volatile int32_t prepare_block_num;
    volatile uint32_t prepare_mask;
    volatile uint32_t prepare_generation;
};
static_assert(sizeof(PercounterMixSignal) == 128, "MIX rendezvous and preparation own two cache lines");
static_assert(alignof(PercounterMixSignal) == 64, "two clusters must not share a line");
// Every member is volatile, so the type is not trivially copyable: the line is
// written field by field under a seqlock, never memcpy'd as a whole.
static_assert(std::is_standard_layout_v<PercounterMixSignal>);

/**
 * Percounter's view of the SSBUF bytes HBG owns.
 *
 * The same bytes resident lays out as SchedulerSsbufRegion, as a separate type:
 * the two schedulers never run in the same launch, and the resident layout is a
 * per-lane dispatch and completion mailbox percounter has no use for.
 */
struct alignas(64) PercounterSsbufRegion {
    PercounterMixSignal mix;
    uint8_t reserved[PERCOUNTER_SSBUF_REGION_SIZE - sizeof(PercounterMixSignal)];
};
static_assert(sizeof(PercounterSsbufRegion) == PERCOUNTER_SSBUF_REGION_SIZE);

inline __aicore__ PERCOUNTER_SSBUF PercounterSsbufRegion *percounter_ssbuf_region(uint64_t hardware_base) {
    return reinterpret_cast<PERCOUNTER_SSBUF PercounterSsbufRegion *>(hardware_base + PERCOUNTER_SSBUF_REGION_OFFSET);
}

/**
 * Put the cluster's line in the state the protocol starts from.
 *
 * SSBUF survives a launch, so nothing the host does covers this. Only the
 * cluster's AIV0 calls it, and only before any lane of the cluster has read the
 * line, which the per-cluster ready word in GM establishes.
 */
inline __aicore__ void percounter_mix_signal_clear(PERCOUNTER_SSBUF PercounterMixSignal *signal) {
    signal->task_id = -1;
    signal->active_mask = 0;
    signal->block_idx = 0;
    signal->block_num = 1;
    signal->cohort_block = 0;
    signal->prepare_task_id = -1;
    signal->prepare_block_idx = 0;
    signal->prepare_block_num = 1;
    signal->prepare_mask = 0;
    percounter_ssbuf_store_release(&signal->prepare_generation, 0);
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        signal->arrived[lane] = 0;
        signal->finished[lane] = 0;
        signal->cohort_hold[lane] = 0;
    }
    // Last, with release ordering: a lane that sees an even generation must see
    // the cleared payload behind it.
    percounter_ssbuf_store_release(&signal->armed_generation, 0);
}

/** Whether the cluster's slot currently holds a task. */
inline __aicore__ bool percounter_mix_armed(uint32_t generation) { return (generation & 1U) != 0; }

/**
 * Tickets per core type: one per group of PERCOUNTER_TICKET_GROUP_LANES lanes.
 *
 * A ticket serves one draw at a time, about 0.35 us each on a5. Lanes whose
 * kernels all take the same time finish together and draw together, so one
 * ticket per type queued 32 AIC draws: 5.6 us a draw on paged_attention_unroll
 * Case1 and 4.5 us on manual_scope Case1, more than the rest of a task's
 * scheduling. A group of lanes shares a ticket over its own blocks of the list
 * (see percounter_group_position), so a burst queues one group.
 *
 * Fixed shards alone would not be work-conserving: a lane whose group's share
 * is spent would idle while another group still had work. So a lane draws from
 * its own group's ticket first and from the others' once its own is spent, and
 * leaves only when every group's is.
 */
inline constexpr uint32_t PERCOUNTER_TICKET_CAPACITY = 16;
inline constexpr uint32_t PERCOUNTER_TICKET_GROUP_LANES = 8;

/**
 * One counter per 64 B cache line.
 *
 * The atomic unit serialises per line rather than per address, so N counters
 * sharing a line cost N times the latency: 6371 cycles for 16 per line against
 * 446 for one, measured on a5. Packing would stay correct -- no update is lost --
 * only slow.
 */
struct alignas(64) PercounterCounter {
    // 64-bit to match percounter_gm_fetch_add, the runtime's only atomic add.
    uint64_t value;
    // The rest describes a Graph body task, written with the preset by the
    // AICPU before the task's entry is published, and never changed after: the
    // one line a lane reads to run the task. Unused for a task-table task.
    uint64_t storage_address;  // the invocation's ChipTaskStorage array
    uint32_t body_index;       // this task's index in it, and in its body
    uint32_t flags;            // PERCOUNTER_BODY_*
    uint32_t shell_task_id;
    uint32_t fanin_count;
    // Region offset of this task's producers, as uint64 body indices. Unused for
    // a root, which waits on its shell's fanin instead.
    uint64_t fanin_offset;
    uint32_t invocation;
    uint8_t reserved[20];
};
static_assert(sizeof(PercounterCounter) == 64, "a counter owns exactly one cache line");
static_assert(std::is_trivially_copyable_v<PercounterCounter> && std::is_standard_layout_v<PercounterCounter>);
static_assert(alignof(PercounterCounter) == 64, "counters must not straddle lines");

/**
 * A counter is published when it reads **positive as a signed value**.
 *
 * The host presets it to `1 - total_subtasks` and every subtask adds one, so the
 * value walks `1 - total .. 0` and lands on exactly 1. Nothing has to detect the
 * last arriver, so a task of N subtasks costs N atomics, not N + 1.
 *
 * Signed `> 0` rather than `== 1`: an equality test would turn a counter that
 * overshot into a consumer that waits forever. With `> 0`, a defect that ran a
 * subtask twice shows up as a wrong result rather than as an op timeout.
 *
 * A task completed inline at bind is preset to 1 directly, the value it would
 * have reached.
 */
inline constexpr uint64_t PERCOUNTER_COUNTER_PUBLISHED = 1;

inline __aicore__ bool percounter_counter_published(uint64_t value) { return static_cast<int64_t>(value) > 0; }

inline constexpr uint64_t percounter_counter_preset(int64_t total_subtasks) {
    return static_cast<uint64_t>(static_cast<int64_t>(PERCOUNTER_COUNTER_PUBLISHED) - total_subtasks);
}

/** A ticket, line-isolated for the same reason the counters are. */
struct alignas(64) PercounterTicket {
    uint64_t next;
    uint8_t reserved[56];
};
static_assert(sizeof(PercounterTicket) == 64, "a ticket owns exactly one cache line");
static_assert(std::is_trivially_copyable_v<PercounterTicket> && std::is_standard_layout_v<PercounterTicket>);

// The MIX list takes two tickets where each core-type list takes one: a draw
// point at index 0 and a completion count at index 1. The exit rule needs the
// second. Consuming the draw and arming the slot are two steps, so between them
// the lane that took the last MIX task shows its cluster "the list is claimed
// out" while its own slot still reads free, and its peers would leave right
// before it needed them. A count of *completed* MIX tasks rises only after a
// rendezvous is over.
inline constexpr uint32_t PERCOUNTER_MIX_TICKET_COUNT = 2;
inline constexpr uint32_t PERCOUNTER_MIX_TICKET_DRAW = 0;
inline constexpr uint32_t PERCOUNTER_MIX_TICKET_DONE = 1;

/**
 * The sync-start cohorts waiting to be gathered, as a queue.
 *
 * **An order list hands out work that may run one after another; a cohort's
 * blocks must run at the same time.** So a cohort's blocks are in no list: the
 * task holds a single *arming* entry (PERCOUNTER_ORDER_BLOCK_IDX_ARM), and its
 * blocks are handed out from here.
 *
 * **Only the open cohort may be claimed from, and it leaves the slot the moment
 * it is fully claimed, not when it finishes.**
 *
 * The first rule is what makes a queue safe without revocation. The state
 * nothing can resolve is two cohorts each holding part of the pool -- 32
 * clusters, two cohorts of 24 holding 20 each, neither able to fill. With one
 * claimable at a time there is never more than one partially claimed cohort,
 * and the open one fills as long as no cohort is wider than the pool. Every
 * sync-start task is held to that by `scheduler_resident_worker_demand_checked`
 * -- a task-table one in upstream's own walk, a Graph body one in
 * `build_graph_template`.
 *
 * Both are measured against the bind's worker count, and the pool a cohort
 * actually meets is the one the handshake reports, which `cluster_count`
 * carries. A run whose reported topology is smaller than the bind assumed --
 * every core of some cluster silent, since a partial cluster fails the
 * handshake outright -- can therefore still hold a cohort it cannot fill. The
 * reported count is never compared against the admitted width; upstream's
 * resident path is bounded the same way and has the same gap.
 *
 * The second rule frees the slot at the gather rather than at the end of
 * execution, so a lane holding a block cannot read the record to learn that
 * its cohort filled; `released` answers that, and a holder of sequence `s`
 * runs once `released > s`.
 *
 * It does **not** mean the next cohort collects claims while this one runs.
 * `percounter_sync_promote` opens one only once `completed` has reached
 * `claimed_blocks`, so at most one cohort is outstanding at a time. That is
 * what the freedom from deadlock rests on, and its cost is that a cohort's
 * gather cannot overlap its predecessor's execution.
 */
struct alignas(64) PercounterSyncQueue {
    // The open cohort. A graph with no cohort currently open pays one load on
    // this line and nothing else.
    uint64_t open_generation;  // even = free, odd = open; CAS to take the slot
    int64_t open_task_id;      // -1 = slot taken, payload not published yet
    uint32_t open_block_num;
    uint32_t open_active_mask;
    uint64_t open_seq;  // this cohort's position in the promotion order
    uint64_t open_armed_cycles;
    uint8_t reserved0[24];
    // Claims are not counted here but per cohort, in the second half of the
    // descriptor ring (see `percounter_sync_claims`): a claim that lands after
    // its cohort was replaced must not consume a block of the replacement.
    alignas(64) uint8_t reserved1[64];
    // Promotion order: a cohort of sequence `s` has been fully claimed once this
    // exceeds `s`. Monotone and never reused.
    alignas(64) uint64_t released;
    uint8_t reserved2[56];
    // Hands out sequence numbers; monotone, and the ring is sized to the
    // graph's cohort count, so it never wraps. The next sequence to promote is
    // `open_generation / 2`, not a cursor of its own: every cohort adds exactly
    // two to the generation, and reading both from the word the promotion CAS
    // checks is what keeps a stale cursor from promoting one cohort twice.
    alignas(64) uint64_t tail;
    uint8_t reserved3[56];
    // Cohort blocks that have actually *run*, which the exit rule waits on.
    // `released` says only that every block has an owner. A MIX cohort block is
    // not even run by the lane that claimed it -- the claimer arms its cluster
    // and is free at once -- so without this count the cluster's other lanes
    // could leave before discovering the rendezvous they are needed for.
    alignas(64) uint64_t completed;
    // Blocks owed by every cohort released so far, which `completed` reaches
    // exactly when none of them is outstanding. Shares this line with it: the
    // promotion reads the pair, and a cohort that is still outstanding is one
    // that can still be waiting for a lane.
    uint64_t claimed_blocks;
    uint8_t reserved4[48];
};
static_assert(sizeof(PercounterSyncQueue) == 320, "the sync queue owns five isolated cache lines");

/**
 * The descriptor ring holds `2 * cohort_count` words: the descriptors at
 * `[0, cohort_count)` and each cohort's claim count at `cohort_count + seq`.
 *
 * **A claim count belongs to one cohort for the whole run.** A shared count
 * reset at each promotion let a lane that read "one block left" add after the
 * last block was taken, the cohort released and the next one promoted -- its
 * add then consumed a block index of the next cohort that nobody would run, and
 * that cohort never filled. With 2-block cohorts and idle lanes that window is
 * routine. A late add to a count of its own lands past the block count and is
 * discarded.
 */
inline __aicore__ __gm__ uint64_t *percounter_sync_claims(__gm__ uint64_t *ring, uint64_t cohort_count, uint64_t seq) {
    return ring + cohort_count + seq;
}
static_assert(std::is_trivially_copyable_v<PercounterSyncQueue> && std::is_standard_layout_v<PercounterSyncQueue>);

/** Whether a cohort is currently open for claims. Same parity rule as MIX. */
inline __aicore__ bool percounter_sync_armed(uint64_t generation) { return (generation & 1U) != 0; }

/**
 * One queued cohort, packed into the word the ring stores.
 *
 *   bits  0..31  task id
 *   bits 32..47  block count
 *   bits 48..55  active-subtask mask
 *   bit  56      valid
 *
 * Zero is "not written yet": a sequence number is taken before the descriptor is
 * stored, and a promoting lane uses the valid bit to tell a reserved ring
 * position from a published one.
 */
inline constexpr uint64_t PERCOUNTER_SYNC_DESC_VALID = uint64_t{1} << 56;

inline __aicore__ uint64_t percounter_sync_desc(int64_t task_id, uint32_t block_num, uint32_t active_mask) {
    return PERCOUNTER_SYNC_DESC_VALID | (static_cast<uint64_t>(active_mask & 0xffU) << 48) |
           (static_cast<uint64_t>(block_num & 0xffffU) << 32) | (static_cast<uint64_t>(task_id) & 0xffffffffULL);
}

inline __aicore__ int64_t percounter_sync_desc_task(uint64_t desc) {
    return static_cast<int64_t>(desc & 0xffffffffULL);
}

inline __aicore__ uint32_t percounter_sync_desc_blocks(uint64_t desc) {
    return static_cast<uint32_t>((desc >> 32) & 0xffffULL);
}

inline __aicore__ uint32_t percounter_sync_desc_mask(uint64_t desc) {
    return static_cast<uint32_t>((desc >> 48) & 0xffULL);
}

/**
 * The three lists a Graph body places entries in: one per core type, then MIX.
 * Indexes the per-list arrays of the two records below.
 */
inline constexpr uint32_t PERCOUNTER_LIST_COUNT = PERCOUNTER_CORE_TYPE_COUNT + 1;
inline constexpr uint32_t PERCOUNTER_LIST_MIX = PERCOUNTER_CORE_TYPE_COUNT;

/** One body task of a Graph template. */
struct PercounterBodyTask {
    // Subtasks the task runs in all, `block_num * popcount(active_mask)`: what
    // its counter is preset against.
    uint32_t subtask_total;
    uint32_t flags;  // PERCOUNTER_BODY_*
};
static_assert(sizeof(PercounterBodyTask) == 8, "percounter body task is a wire struct");
static_assert(std::is_trivially_copyable_v<PercounterBodyTask> && std::is_standard_layout_v<PercounterBodyTask>);

// No producer inside the body: it waits on the shell's own fanin instead.
inline constexpr uint32_t PERCOUNTER_BODY_ROOT = 1U << 0;
// No consumer inside the body: each of its subtasks also counts toward the shell.
inline constexpr uint32_t PERCOUNTER_BODY_SINK = 1U << 1;
inline constexpr uint32_t PERCOUNTER_BODY_PREDICATE = 1U << 2;

/**
 * What percounter keeps of one Graph Definition, built once per Definition per
 * bind and shared by every invocation of it.
 *
 * The body's tasks are numbered 0..task_count-1 as in the Definition. The order
 * entries carry those body indices as their task id; an invocation's entries are
 * the same words with the invocation's first percounter id added.
 *
 * Every `*_offset` is from the region base.
 */
struct alignas(64) PercounterGraphTemplate {
    uint64_t task_count;
    uint64_t edge_count;
    // Entries in each list, AIC, AIV and MIX, and where the template's own copy
    // of each sits: the body's tasks by descending level inside the body.
    uint64_t order_count[PERCOUNTER_LIST_COUNT];
    uint64_t orders_offset[PERCOUNTER_LIST_COUNT];
    uint64_t tasks_offset;  // PercounterBodyTask[task_count]
    // The body's fanin rows: int32 offsets[task_count + 1], then uint64 producer
    // body indices[edge_count], which a lane reads with a 64-bit device load. A
    // producer always has the lower index.
    uint64_t fanin_offsets_offset;
    uint64_t fanin_indices_offset;
    // Subtasks of the sink tasks, summed: the shell's counter preset.
    uint64_t sink_subtask_total;
    // Blocks lanes run, and MIX rendezvous they count, for one invocation.
    uint64_t executable_blocks;
    uint64_t mix_rendezvous_count;
    uint64_t has_predicates;
    // The longest chain of body tasks, in tasks.
    uint64_t depth;
};
static_assert(sizeof(PercounterGraphTemplate) == 128, "percounter graph template is a wire struct");
static_assert(
    std::is_trivially_copyable_v<PercounterGraphTemplate> && std::is_standard_layout_v<PercounterGraphTemplate>
);

/**
 * One Graph invocation: an outer task of kind GRAPH and the body it runs.
 *
 * The host writes everything but `storage_address`, which exists only once the
 * AICPU has localized the invocation: the body's ChipTaskStorage array sits in
 * the shell's heap, and the host never sees that heap's real address.
 */
struct alignas(64) PercounterGraphInvocation {
    uint64_t template_offset;
    // The shell's percounter id, which is also its task-table id. Its fanin is
    // what the body's roots wait on, and its counter is what the sinks publish.
    uint64_t shell_task_id;
    // Percounter id of body task 0; body task i is `body_base + i`.
    uint64_t body_base;
    // First position of the invocation's range in each list. The range is the
    // template's `order_count[l]` long and reads zero until the AICPU fills it.
    uint64_t order_base[PERCOUNTER_LIST_COUNT];
    uint64_t storage_address;
    uint64_t reserved;
};
static_assert(sizeof(PercounterGraphInvocation) == 64, "a graph invocation owns exactly one cache line");
static_assert(
    std::is_trivially_copyable_v<PercounterGraphInvocation> && std::is_standard_layout_v<PercounterGraphInvocation>
);

/**
 * The AICPU threads' shared state while they materialize the invocations.
 *
 * Only AICPU threads touch it, and their caches are coherent with each other, so
 * its fields take plain atomics. The invocation records are in priority order,
 * so drawing `next` hands out the most urgent body first.
 */
struct alignas(64) PercounterGraphPrepare {
    uint64_t next;  // invocations claimed
    uint8_t reserved0[56];
    alignas(64) uint64_t prepared;  // invocations whose every body task is published
    uint64_t first_publish_cycles;  // when the first body tasks were published; 0 = not yet
    uint64_t prepared_cycles;       // when the last invocation was finished
    uint64_t error;                 // first failure, a SchedulerGraphResult
    uint64_t error_invocation;
    uint8_t reserved1[24];
};
static_assert(sizeof(PercounterGraphPrepare) == 128, "the prepare record is two cache lines");
static_assert(
    std::is_trivially_copyable_v<PercounterGraphPrepare> && std::is_standard_layout_v<PercounterGraphPrepare>
);

/**
 * Body tasks one AICPU draw materializes before it publishes them. Smaller
 * gets the first body task to a lane sooner; larger flushes and publishes less
 * often.
 */
inline constexpr int32_t PERCOUNTER_GRAPH_SLICE_TASKS = 16;

/** Where a task's producers live in the flat fanin-address array. */
struct PercounterFanin {
    int32_t count;
    int32_t begin;
};
static_assert(sizeof(PercounterFanin) == 8, "percounter fanin record is a wire struct");
static_assert(std::is_trivially_copyable_v<PercounterFanin> && std::is_standard_layout_v<PercounterFanin>);

/**
 * An order-list entry names one *block* of one task and carries everything the
 * lane needs about it in one 64-bit load:
 *
 *   bits  0..31  task id
 *   bits 32..47  block index
 *   bits 48..63  block count
 *
 * A task with `block_num == N` occupies N consecutive entries with the same task
 * id and block indices 0..N-1. A side table indexed by task id would put a
 * second device load on the readiness scan.
 *
 * The block count is carried here because the descriptor a lane can reach holds
 * the task id and the kernel ids only, and `logical_block_num` lives on the
 * AICPU-side slot state, which an AICore should not read by offset. For a
 * single-slot task the block count is also the subtask total the counter is
 * preset against.
 *
 * The orchestrator refuses any task whose `block_num * popcount(active_mask)`
 * exceeds INT16_MAX, so both halves fit 16 bits unsigned, and a task id is
 * non-negative and bounded by the graph's task count.
 */
inline constexpr uint32_t PERCOUNTER_ORDER_BLOCK_IDX_SHIFT = 32;
inline constexpr uint32_t PERCOUNTER_ORDER_BLOCK_NUM_SHIFT = 48;
inline constexpr uint64_t PERCOUNTER_ORDER_TASK_MASK = 0xffffffffULL;
inline constexpr uint64_t PERCOUNTER_ORDER_BLOCK_MASK = 0xffffULL;

inline __aicore__ uint64_t percounter_order_entry(int64_t task_id, int32_t block_idx, int32_t block_num) {
    return ((static_cast<uint64_t>(block_num) & PERCOUNTER_ORDER_BLOCK_MASK) << PERCOUNTER_ORDER_BLOCK_NUM_SHIFT) |
           ((static_cast<uint64_t>(block_idx) & PERCOUNTER_ORDER_BLOCK_MASK) << PERCOUNTER_ORDER_BLOCK_IDX_SHIFT) |
           (static_cast<uint64_t>(task_id) & PERCOUNTER_ORDER_TASK_MASK);
}

inline __aicore__ int64_t percounter_order_task_id(uint64_t entry) {
    return static_cast<int64_t>(entry & PERCOUNTER_ORDER_TASK_MASK);
}

inline __aicore__ int32_t percounter_order_block_idx(uint64_t entry) {
    return static_cast<int32_t>((entry >> PERCOUNTER_ORDER_BLOCK_IDX_SHIFT) & PERCOUNTER_ORDER_BLOCK_MASK);
}

inline __aicore__ int32_t percounter_order_block_num(uint64_t entry) {
    return static_cast<int32_t>((entry >> PERCOUNTER_ORDER_BLOCK_NUM_SHIFT) & PERCOUNTER_ORDER_BLOCK_MASK);
}

/**
 * The block index an *arming entry* carries. A sync-start task holds exactly one
 * list entry, and it is not a block: running it arms the cohort record, from
 * which the blocks are handed out.
 *
 * The INT16_MAX cap on `block_num * popcount(active_mask)` makes this a value no
 * real block index takes, so a lane recognises an arming duty from the entry it
 * has already unpacked.
 */
inline constexpr int32_t PERCOUNTER_ORDER_BLOCK_IDX_ARM = 0xffff;

inline __aicore__ bool percounter_order_is_arming(uint64_t entry) {
    return percounter_order_block_idx(entry) == PERCOUNTER_ORDER_BLOCK_IDX_ARM;
}

/**
 * The device-visible header: where every other segment is, and how many entries
 * each order list holds.
 */
struct alignas(128) PercounterControl {
    // Every task, including inline-completed ones and Graph body tasks; a lane
    // bounds-checks the task id it unpacks from an entry against this.
    uint64_t task_count;
    // Blocks a lane will actually run and fold into its progress. Under SPMD this
    // exceeds the task count, since a task with `block_num == N` runs N blocks;
    // inline-completed and off-lane tasks contribute none.
    uint64_t executable_block_count;
    uint64_t edge_count;
    uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT];
    uint64_t orders_offset[PERCOUNTER_CORE_TYPE_COUNT];
    uint64_t tickets_offset;
    // The MIX list and its draw point. `mix_order_count == 0` is what makes a
    // MIX-free graph cost nothing: no cluster ever arms its signal.
    uint64_t mix_order_count;
    uint64_t mix_order_offset;
    uint64_t mix_ticket_offset;
    // MIX rendezvous the done ticket will count. Not the list length: a
    // sync-start MIX task holds one arming entry but runs `block_num`
    // rendezvous, which report to the cohort record instead.
    uint64_t mix_rendezvous_count;
    // Sync-start tasks in the graph and where the queue lives; both zero when
    // there are none.
    uint64_t sync_cohort_count;
    uint64_t sync_cohort_offset;
    // Every (sync task, block) pair: what the queue's `completed` is waited on
    // against.
    uint64_t sync_block_total;
    // Non-zero when any task carries a dispatch predicate. A whole-graph flag:
    // the predicate lives on the task payload, a different line from the
    // descriptor a lane already reads, so a per-task test would put one more
    // device read on every task of every graph.
    uint64_t has_predicates;
    // Tasks in the task table, which take percounter ids 0..table_task_count-1.
    // Graph body tasks take the ids above, up to `task_count`; their storage is
    // in their shell's heap, not in the table.
    uint64_t table_task_count;
    uint64_t graph_invocation_count;
    uint64_t graph_invocations_offset;
    uint64_t graph_prepare_offset;
    // The ring of queued cohort descriptors, one word per sync-start task, so a
    // sequence number never wraps and no enqueue finds it full.
    uint64_t sync_pending_offset;
    // One word per cluster, set by its AIV0 once it has cleared the cluster's
    // SSBUF line and read by all three lanes before they first look at it. In
    // GM because GM is what the bind zeroes: a ready flag in SSBUF could read
    // back set from an earlier launch.
    uint64_t mix_signal_ready_offset;
    uint64_t fanin_offset;
    uint64_t fanin_addr_offset;
    uint64_t counters_offset;
    // Written by the AICPU after the handshake, since the cluster count is
    // discovered at bring-up. These two are all a lane needs to place itself:
    // scheduler_cluster_coordinate_from_worker is a pure function of them and
    // the block index.
    uint64_t cluster_count;
    uint64_t aiv_per_cluster;
    // First failure wins; the supervisor stops on any non-zero value.
    uint64_t lane_error;
    // Absolute device addresses resolved at bind, outside this region.
    uint64_t graph_storage_address;
    // Entries in the task table `graph_storage_address` points at:
    // `table_task_count`.
    uint64_t graph_storage_count;
    uint64_t callable_addresses_address;
    uint64_t callable_addresses_count;
    // PERCOUNTER_PAYLOAD_SLOT_COUNT DispatchPayloads per lane: ordinary
    // candidates, a MIX subtask, and a cohort block have separate lifetimes.
    uint64_t dispatch_payloads_offset;
    // Prefilled by the host so a lane can bound its startup wait; the AICPU
    // overwrites it with any override when it publishes the topology.
    uint64_t scheduler_timeout_cycles;
    // Pads the progress counter below onto its own 64 B line.
    uint8_t reserved[320 - 34 * sizeof(uint64_t)];

    // **Alone on the last line.** Lanes batch their completions into this, and
    // the AICPU supervisor polls it against `executable_block_count`. It is the
    // block's only writer during a run, so any field sharing its line would make
    // every reader of that field queue behind its atomics: `lane_error` read
    // beside it cost 919 cycles a task on paged_attention Case1, against 94
    // uncontended.
    //
    // Not volatile, and nothing in this struct is: a wire struct has to stay
    // trivially copyable. Cross-core visibility uses the runtime's explicit
    // discipline -- an atomic add to write, ld_dev or an invalidate to read.
    alignas(64) uint64_t completed_block_count;
    uint8_t resolved_reserved[56];
};
static_assert(sizeof(PercounterControl) == 384, "percounter control is three 128 B lines");
static_assert(
    offsetof(PercounterControl, completed_block_count) % 64 == 0,
    "the one field lanes write must start a cache line, or every reader of its neighbours pays for it"
);
static_assert(
    offsetof(PercounterControl, lane_error) / 64 != offsetof(PercounterControl, completed_block_count) / 64,
    "lane_error is read every task; it must not share a line with the counter lanes increment"
);
static_assert(std::is_trivially_copyable_v<PercounterControl> && std::is_standard_layout_v<PercounterControl>);

inline constexpr uint64_t PERCOUNTER_TICKET_STRIPE = 4;

/**
 * The order-list position of index `index` of group `group`'s share.
 *
 * Stripes of PERCOUNTER_TICKET_STRIPE positions belong to successive groups.
 * A group's indices run through its stripes in list order. Each ticket draw
 * claims one index, independently of the stripe width.
 */
inline __aicore__ uint64_t percounter_group_position(uint64_t index, uint32_t group, uint32_t group_count) {
    return ((index / PERCOUNTER_TICKET_STRIPE) * group_count + group) * PERCOUNTER_TICKET_STRIPE +
           index % PERCOUNTER_TICKET_STRIPE;
}

/** How many of a list's `order_count` positions are group `group`'s share. */
inline __aicore__ uint64_t percounter_group_size(uint64_t order_count, uint32_t group, uint32_t group_count) {
    const uint64_t blocks = (order_count + PERCOUNTER_TICKET_STRIPE - 1) / PERCOUNTER_TICKET_STRIPE;
    if (group >= blocks) return 0;
    const uint64_t own = (blocks - group + group_count - 1) / group_count;
    uint64_t size = own * PERCOUNTER_TICKET_STRIPE;
    // The list's last block, short when the list does not fill it.
    if ((blocks - 1) % group_count == group) size -= blocks * PERCOUNTER_TICKET_STRIPE - order_count;
    return size;
}

}  // namespace simpler::percounter
