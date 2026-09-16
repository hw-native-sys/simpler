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
 * The orchestrator writes a reservation-pitched shared-memory mirror; the image that
 * ships is pitched to the submitted count so its four live prefixes are
 * contiguous and travel as one copy. compact_live_image is the restack, and it
 * owes the device three things: the live payloads, a header carrying no host
 * addresses, and slot-state bindings that resolve inside the image rather than
 * back into the mirror.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_id.h"

namespace {

constexpr uint64_t WINDOW = 64;  // stands in for the mirror's task capacity
constexpr uint64_t SUBMITTED = 5;

// The per-task argument shape these tests write. Deliberately far below every cap, so
// the compacted pools are a small fraction of the mirror's and the difference is
// visible in the shipped byte count. The fanin stride is what the orchestrator
// advances by — a whole number of cache lines, since a region has to start on one.
constexpr int32_t TENSORS_PER_TASK = 2;
constexpr int32_t SCALARS_PER_TASK = 3;
constexpr int32_t FANIN_PER_TASK = 4;
constexpr int32_t FANIN_STRIDE = static_cast<int32_t>(ARG_POOL_ALIGN / sizeof(int32_t));

// A buffer aligned the way both the arena mirror and the device SM base are;
// ChipTaskSlotState is alignas(64) and every segment offset is a multiple of
// CHIP_ALIGN_SIZE.
class AlignedImage {
public:
    explicit AlignedImage(uint64_t bytes, uint8_t fill = 0) :
        storage_(bytes + CHIP_ALIGN_SIZE, std::byte{0}) {
        base_ = reinterpret_cast<char *>(
            (reinterpret_cast<uintptr_t>(storage_.data()) + CHIP_ALIGN_SIZE - 1) &
            ~static_cast<uintptr_t>(CHIP_ALIGN_SIZE - 1)
        );
        if (fill != 0) std::memset(base_, fill, bytes);
    }

    char *base() { return base_; }
    const char *base() const { return base_; }

private:
    std::vector<std::byte> storage_;
    char *base_{nullptr};
};

// A mirror in the state the orchestrator leaves it: SUBMITTED slots bound to
// their own payload and descriptor, distinguishable per-slot content, and header
// pointers naming the mirror's own arrays.
class Mirror {
public:
    Mirror() :
        image_(sm_layout::segment_offsets(WINDOW).end) {
        const auto off = sm_layout::segment_offsets(WINDOW);
        auto *header = reinterpret_cast<SharedMemoryHeader *>(image_.base());
        auto &tasks = header->tasks;
        tasks.total_tasks = static_cast<int32_t>(SUBMITTED);
        storage_ = reinterpret_cast<ChipTaskStorage *>(image_.base() + off.storage);
        tasks.task_storage = storage_;
        tasks.task_states = task_states();

        // Each live slot takes a packed region in each pool, exactly as the
        // orchestrator's bump cursors hand them out, and gets content that identifies
        // the slot so the compaction can be checked element by element.
        for (uint64_t i = 0; i < SUBMITTED; ++i) {
            ChipTaskStorage &entry = storage_[i];
            entry.task.task_id = TaskId::make_global(static_cast<int32_t>(i));
            entry.payload.tensor_count = TENSORS_PER_TASK;
            entry.payload.scalar_count = SCALARS_PER_TASK;
            entry.payload.fanin_count = FANIN_PER_TASK;
            entry.payload.bind_regions(
                tensor_pool() + i * TENSORS_PER_TASK, scalar_pool() + i * SCALARS_PER_TASK,
                fanin_pool() + i * FANIN_STRIDE
            );
            for (int32_t j = 0; j < TENSORS_PER_TASK; ++j) {
                entry.payload.tensor_data()[j].buffer.addr = 0x1000 + i * 0x10 + j;
            }
            for (int32_t j = 0; j < SCALARS_PER_TASK; ++j) {
                entry.payload.scalar_data()[j] = 0x3000 + i * 0x10 + j;
            }
            for (int32_t j = 0; j < FANIN_PER_TASK; ++j) {
                entry.payload.fanin_data()[j] = static_cast<int32_t>(0x50 + i * 0x10 + j);
            }
            entry.slot.in_graph_local_id = static_cast<int32_t>(200 + i);
            task_states()[i].store(i & 1 ? CHIP_TASK_COMPLETED : CHIP_TASK_PENDING, std::memory_order_relaxed);
        }
        // A slot past the submitted prefix, to prove it does not travel.
        storage_[SUBMITTED].task.task_id = TaskId::make_global(0xBEEF);
    }

    // Overwrite the three fields that can hold a graph-heap address with ones out
    // of the virtual window, as the orchestrator leaves them. Tensor 0 of each task
    // becomes a heap output; tensor 1 keeps the real address it already had, which
    // is what makes "only heap addresses move" checkable on the same image.
    void plant_heap_addresses() {
        for (uint64_t i = 0; i < SUBMITTED; ++i) {
            ChipTaskStorage &entry = storage_[i];
            const uint64_t packed = HEAP_VIRTUAL_BASE + i * kPackedStride;
            entry.task.packed_buffer_base = reinterpret_cast<void *>(packed);
            entry.task.packed_buffer_end = reinterpret_cast<void *>(packed + kPackedStride);
            entry.payload.predicate.addr = packed + 8;
            entry.payload.tensor_data()[0].buffer.addr = packed;
        }
    }

    static constexpr uint64_t kPackedStride = 512;
    // What plant_heap_addresses handed out, i.e. what the allocator's
    // heap_used_bytes() would report for it.
    static constexpr uint64_t kHeapUsed = SUBMITTED * kPackedStride;

    const char *base() const { return image_.base(); }

    // A task's three records share one storage entry, so each is reached by
    // indexing this array and naming the member — never by striding an array of
    // that record's own type, which the merged layout no longer has.
    ChipTaskStorage *storage() { return storage_; }
    simpler::hbg::Tensor *tensor_pool() {
        return reinterpret_cast<simpler::hbg::Tensor *>(image_.base() + sm_layout::segment_offsets(WINDOW).tensor_pool);
    }
    uint64_t *scalar_pool() {
        return reinterpret_cast<uint64_t *>(image_.base() + sm_layout::segment_offsets(WINDOW).scalar_pool);
    }
    int32_t *fanin_pool() {
        return reinterpret_cast<int32_t *>(image_.base() + sm_layout::segment_offsets(WINDOW).fanin_pool);
    }
    std::atomic<ChipTaskState> *task_states() {
        return reinterpret_cast<std::atomic<ChipTaskState> *>(
            image_.base() + sm_layout::segment_offsets(WINDOW).task_states
        );
    }

private:
    AlignedImage image_;
    ChipTaskStorage *storage_{nullptr};
};

// What a bind of `submitted` tasks put in the pools, given the per-task shape the
// Mirror writes. This is the same arithmetic the orchestrator's cursors perform.
inline sm_layout::BindUsage usage_for(uint64_t submitted) {
    return sm_layout::BindUsage{
        submitted,
        submitted * static_cast<uint64_t>(FANIN_STRIDE),
        submitted * static_cast<uint64_t>(TENSORS_PER_TASK),
        submitted * static_cast<uint64_t>(SCALARS_PER_TASK),
    };
}

// A bind that allocated nothing out of the graph heap, so the restack has no
// address to move. Every test that is not about the rebase passes this.
inline constexpr sm_layout::HeapRebase kNoHeapAddresses{HEAP_VIRTUAL_BASE, 0};

struct Compacted {
    AlignedImage image;
    uint64_t bytes;
    sm_layout::BindUsage used;
    // Resolved at construction, like the three above it. The three records of a
    // task are this array plus a member, so nothing here hands out a per-record
    // accessor — that is the shape the merged layout removed.
    ChipTaskStorage *storage{nullptr};

    explicit Compacted(
        Mirror &mirror, uint64_t submitted = SUBMITTED, const sm_layout::HeapRebase &rebase = kNoHeapAddresses
    ) :
        image(sm_layout::segment_offsets(sm_layout::image_extents(usage_for(submitted))).end, 0xAA),
        bytes(0),
        used(usage_for(submitted)) {
        bytes = sm_layout::compact_live_image(mirror.base(), WINDOW, used, rebase, image.base());
        storage = reinterpret_cast<ChipTaskStorage *>(image.base() + off().storage);
    }

    sm_layout::SegmentOffsets off(uint64_t submitted = SUBMITTED) const {
        return sm_layout::segment_offsets(sm_layout::image_extents(usage_for(submitted)));
    }

    simpler::hbg::Tensor *tensor_pool() {
        return reinterpret_cast<simpler::hbg::Tensor *>(image.base() + off().tensor_pool);
    }
    std::atomic<ChipTaskState> *task_states() {
        return reinterpret_cast<std::atomic<ChipTaskState> *>(image.base() + off().task_states);
    }
};

}  // namespace

// The point of the restack: the whole image is one range, and it is far smaller
// than the mirror it came from.
TEST(HbgSmCompaction, ShipsOnlyTheLivePrefix) {
    Mirror mirror;
    Compacted compacted(mirror);

    EXPECT_EQ(compacted.bytes, compacted.off().end);
    EXPECT_LT(compacted.bytes, sm_layout::segment_offsets(WINDOW).end);
}

TEST(HbgSmCompaction, CarriesEveryLiveSlotsContent) {
    Mirror mirror;
    Compacted compacted(mirror);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        const ChipTaskStorage &entry = compacted.storage[i];
        EXPECT_EQ(entry.task.task_id.local_id(), static_cast<int32_t>(i)) << "slot " << i;
        EXPECT_EQ(entry.payload.tensor_count, TENSORS_PER_TASK) << "slot " << i;
        EXPECT_EQ(entry.payload.tensor_data()[0].buffer.addr, 0x1000 + i * 0x10) << "slot " << i;
        EXPECT_EQ(entry.slot.in_graph_local_id, static_cast<int32_t>(200 + i)) << "slot " << i;
        const ChipTaskState expected_state = i & 1 ? CHIP_TASK_COMPLETED : CHIP_TASK_PENDING;
        EXPECT_EQ(compacted.task_states()[i].load(std::memory_order_relaxed), expected_state) << "slot " << i;
    }
    // The header's pitch-independent fields come across; the mirror slot past the
    // prefix does not.
    auto &tasks = reinterpret_cast<const SharedMemoryHeader *>(compacted.image.base())->tasks;
    // The device bounds its slot walks with this, and the restack is
    // the only thing that carries it there.
    EXPECT_EQ(tasks.total_tasks, static_cast<int32_t>(SUBMITTED));
}

// A slot state's siblings are ChipTaskStorage's own layout, so the restack's change
// of pitch cannot move them apart: each record of a shipped entry resolves to that
// same entry, with nothing re-taken.
TEST(HbgSmCompaction, SlotBindingsSurviveTheChangeOfPitch) {
    Mirror mirror;
    Compacted compacted(mirror);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        const ChipTaskStorage &entry = compacted.storage[i];
        EXPECT_EQ(&entry.slot.to_payload(), &entry.payload) << "slot " << i;
        EXPECT_EQ(&entry.slot.to_descriptor(), &entry.task) << "slot " << i;
    }
}

// A delta is invariant under a whole-image move, which is what makes the single
// copy to the device safe.
TEST(HbgSmCompaction, BindingsSurviveTheCopyToTheDevice) {
    Mirror mirror;
    Compacted compacted(mirror);

    AlignedImage landed(compacted.bytes);
    std::memcpy(landed.base(), compacted.image.base(), compacted.bytes);
    // The image's own layout, not the mirror's. The two slot-pitched offsets happen
    // to agree between the two, but reading them off the mirror overload here would
    // stop being true the moment a pool moved ahead of them.
    const auto off = compacted.off();
    auto *storage = reinterpret_cast<ChipTaskStorage *>(landed.base() + off.storage);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(&storage[i].slot.to_payload(), &storage[i].payload) << "slot " << i;
        EXPECT_EQ(&storage[i].slot.to_descriptor(), &storage[i].task) << "slot " << i;
    }
}

// The header's data pointers name the mirror. The device resolves its own in
// attach_populated, so shipping them would only put host addresses in device
// memory.
TEST(HbgSmCompaction, LeavesNoHostPointerInTheHeader) {
    Mirror mirror;
    Compacted compacted(mirror);

    auto &tasks = reinterpret_cast<const SharedMemoryHeader *>(compacted.image.base())->tasks;
    EXPECT_EQ(tasks.task_storage, nullptr);
    EXPECT_EQ(tasks.task_states, nullptr);
}

// A bind that submits nothing still ships its header and still attaches. Its pools
// are empty, so what travels is the header plus one slot's worth of pitch — far below
// the mirror, whose pools are dimensioned for the whole window.
TEST(HbgSmCompaction, ZeroSubmittedShipsTheHeaderAlone) {
    Mirror mirror;
    Compacted compacted(mirror, /*submitted=*/0);

    EXPECT_EQ(compacted.bytes, compacted.off(0).end);
    EXPECT_LT(compacted.bytes, sm_layout::segment_offsets(1).end);
    auto &tasks = reinterpret_cast<const SharedMemoryHeader *>(compacted.image.base())->tasks;
    EXPECT_EQ(tasks.total_tasks, static_cast<int32_t>(SUBMITTED));
    EXPECT_EQ(tasks.task_storage, nullptr);
}

// The pools ship what the bind used, not what the mirror is dimensioned for. That
// is the same property the payload stride used to carry, moved to the pools: fewer
// bytes, and every argument still resolves.
TEST(HbgSmCompaction, PackedPoolsShipFewerBytesAndStillResolve) {
    Mirror mirror;
    Compacted compacted(mirror);

    // The mirror's pools cover WINDOW tasks at their full caps; the image's cover
    // SUBMITTED tasks at the shape they actually used.
    EXPECT_LT(compacted.bytes, sm_layout::segment_offsets(WINDOW).end);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        const ChipTaskStorage &entry = compacted.storage[i];
        const TaskPayload *shipped = &entry.payload;
        EXPECT_EQ(shipped->tensor_count, TENSORS_PER_TASK) << "slot " << i;
        EXPECT_EQ(shipped->scalar_count, SCALARS_PER_TASK) << "slot " << i;
        EXPECT_EQ(shipped->fanin_count, FANIN_PER_TASK) << "slot " << i;
        // Every element resolves through the re-taken delta, and lands on this slot's
        // own region rather than a neighbour's.
        for (int32_t j = 0; j < TENSORS_PER_TASK; ++j) {
            EXPECT_EQ(shipped->tensor_data()[j].buffer.addr, 0x1000 + i * 0x10 + j) << "slot " << i << " arg " << j;
        }
        for (int32_t j = 0; j < SCALARS_PER_TASK; ++j) {
            EXPECT_EQ(shipped->scalar_data()[j], 0x3000 + i * 0x10 + j) << "slot " << i << " arg " << j;
        }
        for (int32_t j = 0; j < FANIN_PER_TASK; ++j) {
            EXPECT_EQ(shipped->fanin_data()[j], static_cast<int32_t>(0x50 + i * 0x10 + j))
                << "slot " << i << " arg " << j;
        }
        EXPECT_EQ(&entry.slot.to_payload(), shipped) << "slot " << i;
    }
}

// A region's delta is only correct for the layout it was taken in, so the restack has
// to re-take it. Proof: every shipped region lies inside the image's own pools, which
// the mirror's addresses could not satisfy.
TEST(HbgSmCompaction, RebindsEveryArgumentRegionInsideTheImage) {
    Mirror mirror;
    Compacted compacted(mirror);

    const auto off = compacted.off();
    const char *base = compacted.image.base();
    const char *tensor_begin = base + off.tensor_pool;
    const char *tensor_end = tensor_begin + compacted.used.tensor_elems * sizeof(simpler::hbg::Tensor);
    const char *scalar_begin = base + off.scalar_pool;
    const char *scalar_end = scalar_begin + compacted.used.scalar_elems * sizeof(uint64_t);
    const char *fanin_begin = base + off.fanin_pool;
    const char *fanin_end = fanin_begin + compacted.used.fanin_elems * sizeof(int32_t);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        const TaskPayload *shipped = &compacted.storage[i].payload;
        const char *t = reinterpret_cast<const char *>(shipped->tensor_data());
        const char *s = reinterpret_cast<const char *>(shipped->scalar_data());
        const char *f = reinterpret_cast<const char *>(shipped->fanin_data());
        EXPECT_GE(t, tensor_begin) << "slot " << i;
        EXPECT_LT(t, tensor_end) << "slot " << i;
        EXPECT_GE(s, scalar_begin) << "slot " << i;
        EXPECT_LT(s, scalar_end) << "slot " << i;
        EXPECT_GE(f, fanin_begin) << "slot " << i;
        EXPECT_LT(f, fanin_end) << "slot " << i;
    }
}

// An outer GRAPH task binds only a fanin region: its boundary arguments travel inside
// the submission image and both argument counts are 0. The restack must carry that
// unbound state through rather than resolving it to the pool base, which is what the
// consumers' count guards assume.
TEST(HbgSmCompaction, UnboundRegionsStayUnbound) {
    Mirror mirror;
    // Slot 2 stands in for the outer GRAPH task.
    constexpr uint64_t GRAPH_SLOT = 2;
    TaskPayload &graph_payload = mirror.storage()[GRAPH_SLOT].payload;
    graph_payload.tensor_count = 0;
    graph_payload.scalar_count = 0;
    graph_payload.bind_regions(nullptr, nullptr, mirror.fanin_pool() + GRAPH_SLOT * FANIN_STRIDE);

    Compacted compacted(mirror);

    const TaskPayload *shipped = &compacted.storage[GRAPH_SLOT].payload;
    EXPECT_EQ(shipped->tensor_data(), nullptr);
    EXPECT_EQ(shipped->scalar_data(), nullptr);
    // Its fanin region still resolves, and still inside the image.
    const char *f = reinterpret_cast<const char *>(shipped->fanin_data());
    EXPECT_GE(f, compacted.image.base() + compacted.off().fanin_pool);
    EXPECT_LT(f, compacted.image.base() + compacted.off().tensor_pool);
    for (int32_t j = 0; j < FANIN_PER_TASK; ++j) {
        EXPECT_EQ(shipped->fanin_data()[j], static_cast<int32_t>(0x50 + GRAPH_SLOT * 0x10 + j)) << "arg " << j;
    }
    // Its neighbours are untouched by the unbound slot.
    EXPECT_EQ(compacted.storage[1].payload.tensor_data()[0].buffer.addr, 0x1000 + 1 * 0x10);
    EXPECT_EQ(compacted.storage[3].payload.tensor_data()[0].buffer.addr, 0x1000 + 3 * 0x10);
}

// A slot names its payload, and a payload names its regions, by an int32 delta, so
// neither the mirror nor the image may span more than that reach. `init_per_ring` and
// `attach_populated` each reject a layout that does; what is checked here is the
// arithmetic they test — that the default capacity is comfortably inside the bound, and
// that the layout is what decides where the bound falls. Growing a segment enough to
// put the default capacity out of reach fails here.
TEST(HbgSmCompaction, LayoutStaysWithinDeltaReach) {
    Mirror mirror;
    Compacted compacted(mirror);
    EXPECT_LT(compacted.bytes, static_cast<uint64_t>(INT32_MAX));

    constexpr uint64_t REACH = static_cast<uint64_t>(INT32_MAX);
    EXPECT_LE(sm_layout::segment_offsets(CHIP_DEFAULT_GRAPH_TASKS).end, REACH);

    // The bound is a property of the layout, not of the capacity alone: there is a
    // capacity past which the mirror no longer fits, and it is above the default.
    uint64_t window = CHIP_DEFAULT_GRAPH_TASKS;
    while (window <= (UINT64_MAX / 2) && sm_layout::segment_offsets(window).end <= REACH) {
        window *= 2;
    }
    EXPECT_GT(window, static_cast<uint64_t>(CHIP_DEFAULT_GRAPH_TASKS));
    EXPECT_GT(sm_layout::segment_offsets(window).end, REACH);
}

// A task id indexes its slot directly, so no capacity is masked and a bind may pass
// any positive runtime_env.ring_task_window through. The layout walk therefore has to
// hold for a slot count that is not a power of two: every segment stays
// CHIP_ALIGN_SIZE-aligned, and each one leaves room for its own array at that pitch.
TEST(HbgSmCompaction, SegmentLayoutHoldsForANonPowerOfTwoCapacity) {
    for (uint64_t capacity : {uint64_t{1}, uint64_t{3}, uint64_t{10}, uint64_t{1000}, uint64_t{16383}}) {
        const sm_layout::ImageExtents e = sm_layout::mirror_extents(capacity);
        const sm_layout::SegmentOffsets off = sm_layout::segment_offsets(e);

        // The storage sits right after the padded header, whatever the pitch.
        EXPECT_EQ(off.storage, CHIP_ALIGN_UP(sizeof(SharedMemoryHeader), CHIP_ALIGN_SIZE)) << "capacity " << capacity;

        const uint64_t starts[] = {off.storage,     off.task_states, off.fanin_pool,
                                   off.tensor_pool, off.scalar_pool, off.end};
        const uint64_t spans[] = {
            capacity * sizeof(ChipTaskStorage), capacity * sizeof(std::atomic<ChipTaskState>),
            e.fanin_elems * sizeof(int32_t),    e.tensor_elems * sizeof(simpler::hbg::Tensor),
            e.scalar_elems * sizeof(uint64_t),
        };
        for (size_t i = 0; i < std::size(spans); ++i) {
            EXPECT_EQ(starts[i] % CHIP_ALIGN_SIZE, 0u) << "capacity " << capacity << " segment " << i;
            EXPECT_GE(starts[i + 1] - starts[i], spans[i]) << "capacity " << capacity << " segment " << i;
        }
        EXPECT_EQ(off.end % CHIP_ALIGN_SIZE, 0u) << "capacity " << capacity;
    }
}

// The graph heap's device region is committed after orchestration, so the
// orchestrator allocates out of the HEAP_VIRTUAL_BASE window and the restack is
// what puts the real base into the image. Three fields carry such an address: a
// descriptor's packed buffer bounds, a payload's dispatch predicate, and a
// tensor's buffer address.
TEST(HbgSmCompaction, MovesEveryHeapAddressOntoTheRealBase) {
    constexpr uint64_t REAL_BASE = 0x7F0000000000ULL;
    Mirror mirror;
    mirror.plant_heap_addresses();
    Compacted compacted(mirror, SUBMITTED, {REAL_BASE, Mirror::kHeapUsed});

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        const ChipTaskStorage &entry = compacted.storage[i];
        const uint64_t packed = REAL_BASE + i * Mirror::kPackedStride;
        EXPECT_EQ(reinterpret_cast<uint64_t>(entry.task.packed_buffer_base), packed) << "slot " << i;
        EXPECT_EQ(reinterpret_cast<uint64_t>(entry.task.packed_buffer_end), packed + Mirror::kPackedStride)
            << "slot " << i;
        EXPECT_EQ(entry.payload.predicate.addr, packed + 8) << "slot " << i;
        EXPECT_EQ(entry.payload.tensor_data()[0].buffer.addr, packed) << "slot " << i;
    }
}

// A caller's tensor is a real device address the bind did not mint, and it shares
// the pools with the heap ones. It must come through the restack unchanged, or a
// boundary tensor would be pointed at the graph heap.
TEST(HbgSmCompaction, LeavesNonHeapAddressesAlone) {
    constexpr uint64_t REAL_BASE = 0x7F0000000000ULL;
    Mirror mirror;
    mirror.plant_heap_addresses();
    Compacted compacted(mirror, SUBMITTED, {REAL_BASE, Mirror::kHeapUsed});

    // Tensor 1 of every task kept the address the Mirror gave it.
    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(compacted.storage[i].payload.tensor_data()[1].buffer.addr, 0x1000 + i * 0x10 + 1) << "slot " << i;
    }
}

// A bind whose outputs are all caller-owned puts nothing in the window, so the
// restack must leave every address exactly as the mirror had it — including the
// zero a payload's unset predicate carries.
TEST(HbgSmCompaction, RebaseIsANoOpWhenNothingCameFromTheHeap) {
    constexpr uint64_t REAL_BASE = 0x7F0000000000ULL;
    Mirror mirror;
    Compacted rebased(mirror, SUBMITTED, {REAL_BASE, 0});
    Compacted plain(mirror);

    ASSERT_EQ(rebased.bytes, plain.bytes);
    EXPECT_EQ(std::memcmp(rebased.image.base(), plain.image.base(), rebased.bytes), 0);
    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(rebased.storage[i].payload.predicate.addr, 0u) << "slot " << i;
    }
}

// Every boundary of an outer GRAPH task is rebased, not just the first. A Graph's boundary
// is this invocation's actual arguments, held as simpler::hbg::Tensors in the same pool and
// at the same stride as any other task's, so one walk covers both kinds -- a boundary the
// walk skipped would keep a virtual address the device then dereferences.
TEST(HbgSmCompaction, MovesEveryGraphBoundaryAddressOntoTheRealBase) {
    constexpr uint64_t REAL_BASE = 0x7F0000000000ULL;
    constexpr uint32_t BOUNDARIES = 2;
    static_assert(BOUNDARIES <= TENSORS_PER_TASK, "the boundaries must fit the slot's own tensor region");

    Mirror mirror;
    // One GRAPH task whose boundaries all live in the graph heap.
    constexpr uint64_t GRAPH_SLOT = 0;
    ChipTaskStorage &graph_entry = mirror.storage()[GRAPH_SLOT];
    graph_entry.slot.task_kind = TaskKind::GRAPH;
    graph_entry.payload.tensor_count = static_cast<int32_t>(BOUNDARIES);
    auto *boundaries = graph_entry.payload.tensor_data();
    for (uint32_t j = 0; j < BOUNDARIES; ++j) {
        boundaries[j] = simpler::hbg::Tensor{};
        boundaries[j].buffer.addr = HEAP_VIRTUAL_BASE + 0x1000 * (j + 1);
        boundaries[j].buffer.size = 0x40;
        // A caller-owned tensor: the restack asserts nothing a recording addressed is here.
        boundaries[j].owner_task_id = TaskId::invalid();
    }

    Compacted compacted(mirror, SUBMITTED, {REAL_BASE, 1ULL << 30});
    const auto *shipped = compacted.storage[GRAPH_SLOT].payload.tensor_data();
    for (uint32_t j = 0; j < BOUNDARIES; ++j) {
        EXPECT_EQ(shipped[j].buffer.addr, REAL_BASE + 0x1000 * (j + 1)) << "boundary " << j;
        EXPECT_EQ(shipped[j].buffer.size, 0x40u) << "boundary " << j << " had a neighbouring field rewritten";
    }
}
