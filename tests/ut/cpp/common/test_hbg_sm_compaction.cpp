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
 * The orchestrator writes a ring-pitched shared-memory mirror; the image that
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

#include "pto_shared_memory.h"

namespace {

constexpr uint64_t WINDOW = 64;  // stands in for the ring capacity
constexpr uint64_t SUBMITTED = 5;

// A buffer aligned the way both the arena mirror and the device SM base are;
// PTO2TaskSlotState is alignas(64) and every segment offset is a multiple of
// PTO2_ALIGN_SIZE.
class AlignedImage {
public:
    explicit AlignedImage(uint64_t bytes, uint8_t fill = 0) :
        storage_(bytes + PTO2_ALIGN_SIZE, std::byte{0}) {
        base_ = reinterpret_cast<char *>(
            (reinterpret_cast<uintptr_t>(storage_.data()) + PTO2_ALIGN_SIZE - 1) &
            ~static_cast<uintptr_t>(PTO2_ALIGN_SIZE - 1)
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
        image_(pto2_sm_layout::ring_segment_offsets(WINDOW).end) {
        const auto off = pto2_sm_layout::ring_segment_offsets(WINDOW);
        auto *header = reinterpret_cast<PTO2SharedMemoryHeader *>(image_.base());
        auto &ring = header->ring;
        ring.task_capacity = WINDOW;
        ring.task_capacity_mask = static_cast<int32_t>(WINDOW - 1);
        ring.fc.current_task_index.store(static_cast<int32_t>(SUBMITTED), std::memory_order_relaxed);
        ring.task_descriptors = descriptors();
        ring.task_payloads = payloads();
        ring.slot_states = slot_states();
        ring.completion_flags = completion_flags();
        (void)off;

        for (uint64_t i = 0; i < SUBMITTED; ++i) {
            descriptors()[i].task_id = PTO2TaskId::make(0, static_cast<uint32_t>(i));
            payloads()[i].tensor_count = static_cast<int32_t>(100 + i);
            slot_states()[i].last_consumer_local_id = static_cast<int32_t>(i);
            slot_states()[i].graph_node_index = static_cast<int32_t>(200 + i);
            slot_states()[i].bind_buffers(&payloads()[i], &descriptors()[i]);
            completion_flags()[i].store(static_cast<uint8_t>(i & 1), std::memory_order_relaxed);
        }
        // A slot past the submitted prefix, to prove it does not travel.
        descriptors()[SUBMITTED].task_id = PTO2TaskId::make(0, 0xBEEF);
    }

    const char *base() const { return image_.base(); }

    PTO2TaskDescriptor *descriptors() {
        return reinterpret_cast<PTO2TaskDescriptor *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).descriptors
        );
    }
    PTO2TaskPayload *payloads() {
        return reinterpret_cast<PTO2TaskPayload *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).payloads
        );
    }
    PTO2TaskSlotState *slot_states() {
        return reinterpret_cast<PTO2TaskSlotState *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).slot_states
        );
    }
    std::atomic<uint8_t> *completion_flags() {
        return reinterpret_cast<std::atomic<uint8_t> *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).completion_flags
        );
    }

private:
    AlignedImage image_;
};

struct Compacted {
    AlignedImage image;
    uint64_t bytes;
    uint64_t stride;

    explicit Compacted(
        Mirror &mirror, uint64_t submitted = SUBMITTED, uint64_t payload_stride = sizeof(PTO2TaskPayload)
    ) :
        image(
            pto2_sm_layout::ring_segment_offsets(pto2_sm_layout::live_slot_pitch(submitted), payload_stride).end, 0xAA
        ),
        bytes(0),
        stride(payload_stride) {
        bytes = pto2_sm_layout::compact_live_image(mirror.base(), WINDOW, submitted, payload_stride, image.base());
    }

    pto2_sm_layout::PTO2RingSegmentOffsets off(uint64_t submitted = SUBMITTED) const {
        return pto2_sm_layout::ring_segment_offsets(pto2_sm_layout::live_slot_pitch(submitted), stride);
    }

    PTO2TaskPayload *payload_at(uint64_t i) {
        return reinterpret_cast<PTO2TaskPayload *>(image.base() + off().payloads + i * stride);
    }
    PTO2TaskDescriptor *descriptors() {
        return reinterpret_cast<PTO2TaskDescriptor *>(image.base() + off().descriptors);
    }
    PTO2TaskPayload *payloads() { return payload_at(0); }
    PTO2TaskSlotState *slot_states() { return reinterpret_cast<PTO2TaskSlotState *>(image.base() + off().slot_states); }
    std::atomic<uint8_t> *completion_flags() {
        return reinterpret_cast<std::atomic<uint8_t> *>(image.base() + off().completion_flags);
    }
};

}  // namespace

// The point of the restack: the whole image is one range, and it is far smaller
// than the mirror it came from.
TEST(HbgSmCompaction, ShipsOnlyTheLivePrefix) {
    Mirror mirror;
    Compacted compacted(mirror);

    EXPECT_EQ(compacted.bytes, pto2_sm_layout::ring_segment_offsets(SUBMITTED).end);
    EXPECT_LT(compacted.bytes, pto2_sm_layout::ring_segment_offsets(WINDOW).end);
}

TEST(HbgSmCompaction, CarriesEveryLiveSlotsContent) {
    Mirror mirror;
    Compacted compacted(mirror);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(compacted.descriptors()[i].task_id.local(), i) << "slot " << i;
        EXPECT_EQ(compacted.payload_at(i)->tensor_count, static_cast<int32_t>(100 + i)) << "slot " << i;
        EXPECT_EQ(compacted.slot_states()[i].last_consumer_local_id, static_cast<int32_t>(i)) << "slot " << i;
        EXPECT_EQ(compacted.slot_states()[i].graph_node_index, static_cast<int32_t>(200 + i)) << "slot " << i;
        EXPECT_EQ(compacted.completion_flags()[i].load(std::memory_order_relaxed), static_cast<uint8_t>(i & 1))
            << "slot " << i;
    }
    // The header's pitch-independent fields come across; the mirror slot past the
    // prefix does not.
    auto &ring = reinterpret_cast<const PTO2SharedMemoryHeader *>(compacted.image.base())->ring;
    EXPECT_EQ(ring.task_capacity, WINDOW);
    EXPECT_EQ(ring.fc.current_task_index.load(std::memory_order_relaxed), static_cast<int32_t>(SUBMITTED));
}

// The load-bearing one. Restacking changes the distance between a slot state and
// its payload, so a binding copied verbatim would resolve to the mirror — a host
// address, in device memory.
TEST(HbgSmCompaction, RebindsEverySlotInsideTheImage) {
    Mirror mirror;
    Compacted compacted(mirror);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(compacted.slot_states()[i].payload.get(), compacted.payload_at(i)) << "slot " << i;
        EXPECT_EQ(compacted.slot_states()[i].task.get(), &compacted.descriptors()[i]) << "slot " << i;
    }
}

// A delta is invariant under a whole-image move, which is what makes the single
// copy to the device safe.
TEST(HbgSmCompaction, BindingsSurviveTheCopyToTheDevice) {
    Mirror mirror;
    Compacted compacted(mirror);

    AlignedImage landed(compacted.bytes);
    std::memcpy(landed.base(), compacted.image.base(), compacted.bytes);
    const auto off = pto2_sm_layout::ring_segment_offsets(SUBMITTED);
    auto *slots = reinterpret_cast<PTO2TaskSlotState *>(landed.base() + off.slot_states);
    auto *payloads = reinterpret_cast<PTO2TaskPayload *>(landed.base() + off.payloads);
    auto *descriptors = reinterpret_cast<PTO2TaskDescriptor *>(landed.base() + off.descriptors);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(slots[i].payload.get(), &payloads[i]) << "slot " << i;
        EXPECT_EQ(slots[i].task.get(), &descriptors[i]) << "slot " << i;
    }
}

// The header's data pointers name the mirror. The device resolves its own in
// attach_populated, so shipping them would only put host addresses in device
// memory.
TEST(HbgSmCompaction, LeavesNoHostPointerInTheHeader) {
    Mirror mirror;
    Compacted compacted(mirror);

    auto &ring = reinterpret_cast<const PTO2SharedMemoryHeader *>(compacted.image.base())->ring;
    EXPECT_EQ(ring.task_descriptors, nullptr);
    EXPECT_EQ(ring.task_payloads, nullptr);
    EXPECT_EQ(ring.slot_states, nullptr);
    EXPECT_EQ(ring.completion_flags, nullptr);
}

// A bind that submits nothing still ships its header and still attaches.
TEST(HbgSmCompaction, ZeroSubmittedShipsTheHeaderAlone) {
    Mirror mirror;
    Compacted compacted(mirror, /*submitted=*/0);

    EXPECT_EQ(compacted.bytes, pto2_sm_layout::ring_segment_offsets(1).end);
    auto &ring = reinterpret_cast<const PTO2SharedMemoryHeader *>(compacted.image.base())->ring;
    EXPECT_EQ(ring.task_capacity, WINDOW);
    EXPECT_EQ(ring.task_descriptors, nullptr);
}

// The image strides its payload array by the widest task in the bind, not by the
// type. Everything a task reads is a prefix of its payload, so a narrower stride
// carries the same information in fewer bytes.
TEST(HbgSmCompaction, CompactedStrideShipsFewerBytesAndStillResolves) {
    Mirror mirror;
    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        mirror.payloads()[i].tensor_count = 2;
        mirror.payloads()[i].scalar_count = 1;
        mirror.payloads()[i].tensors[0].buffer.addr = 0x1000 + i;
        mirror.payloads()[i].tensors[1].buffer.addr = 0x2000 + i;
        mirror.payloads()[i].scalars[0] = 0x3000 + i;
    }

    const uint64_t stride = pto2_sm_layout::shipped_payload_stride(2);
    ASSERT_LT(stride, sizeof(PTO2TaskPayload));
    ASSERT_EQ(stride % PTO2_ALIGN_SIZE, 0u);

    Compacted compacted(mirror, SUBMITTED, stride);
    EXPECT_LT(compacted.bytes, pto2_sm_layout::ring_segment_offsets(SUBMITTED).end);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        PTO2TaskPayload *shipped = compacted.payload_at(i);
        EXPECT_EQ(shipped->tensor_count, 2) << "slot " << i;
        EXPECT_EQ(shipped->scalar_count, 1) << "slot " << i;
        EXPECT_EQ(shipped->tensors[0].buffer.addr, 0x1000 + i) << "slot " << i;
        EXPECT_EQ(shipped->tensors[1].buffer.addr, 0x2000 + i) << "slot " << i;
        EXPECT_EQ(shipped->scalars[0], 0x3000 + i) << "slot " << i;
        // The binding has to follow the compacted stride, not the type's.
        EXPECT_EQ(compacted.slot_states()[i].payload.get(), shipped) << "slot " << i;
    }
}

// A stride covering the whole array is the identity case, and must not be exceeded:
// the type is the widest an image can ever be.
TEST(HbgSmCompaction, StrideNeverExceedsTheType) {
    EXPECT_EQ(pto2_sm_layout::shipped_payload_stride(MAX_TENSOR_ARGS), sizeof(PTO2TaskPayload));
    EXPECT_EQ(pto2_sm_layout::shipped_payload_stride(MAX_TENSOR_ARGS * 4), sizeof(PTO2TaskPayload));
    EXPECT_EQ(pto2_sm_layout::shipped_payload_stride(0), offsetof(PTO2TaskPayload, tensors));
    EXPECT_EQ(pto2_sm_layout::shipped_payload_stride(-1), offsetof(PTO2TaskPayload, tensors));
}
