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
 * The host accessor's two reasons for refusing an access, which must not be confused — and the
 * difference between a refusal and the run's published cause.
 *
 * A read the orchestrator cannot be served is a fatal either way — it stops the run rather than
 * proceeding on a value it was denied. What decides which failure it is is *why*: a caller device
 * buffer another live run has declared it produces has no content yet, while an address no region
 * covers is an invalid argument. This case pins that split at the accessor, which is where it is
 * made — the reason belongs to the one access that was refused — and pins that merely sharing an
 * allocation is not a reason to refuse anything. Which refusal becomes the *run's* cause is not
 * decided here: it is published by the access whose own fatal report latched the orchestration's
 * fatal field, and once published nothing clears it.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "common/host_api.h"
#include "host_build_graph/host_tensor_access.h"

namespace {

// The accessor reaches the platform only through HostApiOps, so a fake table is the whole
// environment it needs. Addresses are indices into `backing`, never dereferenced by the accessor
// on the paths under test.
struct FakePlatform {
    static constexpr uint64_t kDeviceBase = 0x4000;
    static constexpr uint64_t kBytes = 64;

    std::vector<unsigned char> backing{std::vector<unsigned char>(kBytes, 0u)};
    // What `caller_device_span_written_by_other_run` answers, and what it was asked.
    bool written_by_other_run{false};
    int held_queries{0};
    // Whether a host view is available for a child-memory allocation, and how often each service
    // was reached — an access that must be refused has to reach neither.
    bool host_view_available{true};
    int view_acquisitions{0};
    int device_reads{0};

    static FakePlatform *self;

    static void *acquire_child_memory_host_view(void *, void *dev_ptr, size_t) {
        ++self->view_acquisitions;
        if (!self->host_view_available) return nullptr;
        const uint64_t offset = reinterpret_cast<uint64_t>(dev_ptr) - kDeviceBase;
        return self->backing.data() + offset;
    }

    static int copy_from_device(void *, void *host_ptr, const void *dev_ptr, size_t size) {
        ++self->device_reads;
        const uint64_t offset = reinterpret_cast<uint64_t>(dev_ptr) - kDeviceBase;
        std::memcpy(host_ptr, self->backing.data() + offset, size);
        return 0;
    }

    static int copy_to_device(void *, void *dev_ptr, const void *host_ptr, size_t size) {
        const uint64_t offset = reinterpret_cast<uint64_t>(dev_ptr) - kDeviceBase;
        std::memcpy(self->backing.data() + offset, host_ptr, size);
        return 0;
    }

    static int caller_device_span_written_by_other_run(void *, uint64_t, uint64_t, uint64_t) {
        ++self->held_queries;
        return self->written_by_other_run ? 1 : 0;
    }
};

FakePlatform *FakePlatform::self = nullptr;

class HostTensorAccessDeferral : public ::testing::Test {
protected:
    void SetUp() override {
        platform_ = FakePlatform{};
        FakePlatform::self = &platform_;
        ops_ = HostApiOps{};
        ops_.acquire_child_memory_host_view = &FakePlatform::acquire_child_memory_host_view;
        ops_.copy_from_device = &FakePlatform::copy_from_device;
        ops_.copy_to_device = &FakePlatform::copy_to_device;
        ops_.caller_device_span_written_by_other_run = &FakePlatform::caller_device_span_written_by_other_run;
        api_.emplace(&platform_, 0u, 0u, 0u, &ops_);
    }

    void TearDown() override { FakePlatform::self = nullptr; }

    const HostApi *api() { return &api_.value(); }

    FakePlatform platform_{};
    HostApiOps ops_{};
    std::optional<HostApi> api_{};
};

constexpr uint64_t kBase = FakePlatform::kDeviceBase;
constexpr uint64_t kBytes = FakePlatform::kBytes;

}  // namespace

TEST_F(HostTensorAccessDeferral, AProducedChildMemoryReadIsDeferredAndServesNothing) {
    HostTensorAccessor accessor(api());
    ASSERT_TRUE(accessor.add_child_memory(kBase, kBytes));
    platform_.written_by_other_run = true;

    uint64_t value = 0xdeadbeef;
    EXPECT_FALSE(accessor.read(kBase, &value, sizeof(uint32_t)));
    EXPECT_TRUE(host_tensor_refusal_was_dependency());
    // The refusal happens before any means is chosen, so no mapping is installed and no device
    // copy is paid for bytes the access was not allowed to observe.
    EXPECT_EQ(platform_.view_acquisitions, 0);
    EXPECT_EQ(platform_.device_reads, 0);
    EXPECT_EQ(value, 0xdeadbeefu) << "a deferred read must leave the destination untouched";
}

TEST_F(HostTensorAccessDeferral, AProducedChildMemoryWriteIsDeferredToo) {
    HostTensorAccessor accessor(api());
    ASSERT_TRUE(accessor.add_child_memory(kBase, kBytes));
    platform_.written_by_other_run = true;

    const uint32_t value = 0x11223344;
    EXPECT_FALSE(accessor.write(kBase, &value, sizeof(value)));
    EXPECT_TRUE(host_tensor_refusal_was_dependency());
    EXPECT_EQ(platform_.view_acquisitions, 0);
    // Nothing reached the backing: two writers with no order between them is what this avoids.
    EXPECT_EQ(platform_.backing[0], 0u);
}

TEST_F(HostTensorAccessDeferral, AnAddressNoRegionCoversIsNotADeferral) {
    HostTensorAccessor accessor(api());
    ASSERT_TRUE(accessor.add_child_memory(kBase, kBytes));
    platform_.written_by_other_run = true;

    // This is the invalid-argument case the deferral must not absorb: a runtime-created buffer has
    // no region, so the refusal is final and the caller's own status must survive it.
    uint64_t value = 0;
    EXPECT_FALSE(accessor.read(kBase + 0x100000, &value, sizeof(uint32_t)));
    EXPECT_FALSE(host_tensor_refusal_was_dependency());
    EXPECT_EQ(platform_.held_queries, 0) << "an unresolvable address is refused before the declaration is consulted";

    // Nor is a span that starts inside the region but runs past its end.
    EXPECT_FALSE(accessor.read(kBase + kBytes - 2, &value, sizeof(uint64_t)));
    EXPECT_FALSE(host_tensor_refusal_was_dependency());
}

TEST_F(HostTensorAccessDeferral, AChildMemoryReadWithNoDeclaredProducerIsServedNormally) {
    HostTensorAccessor accessor(api());
    ASSERT_TRUE(accessor.add_child_memory(kBase, kBytes));
    platform_.backing[0] = 0x5a;
    platform_.written_by_other_run = false;

    uint32_t value = 0;
    EXPECT_TRUE(accessor.read(kBase, &value, sizeof(value)));
    EXPECT_EQ(value & 0xffu, 0x5au);
    EXPECT_FALSE(host_tensor_refusal_was_dependency());
    EXPECT_EQ(platform_.view_acquisitions, 1);
}

TEST_F(HostTensorAccessDeferral, ADeviceCopyChildMemoryRegionStillDefers) {
    // A backend with no host-map path serves each access by copy, and that region keeps a null
    // host view — so the deferral has to be decided on the declaration, not on whether a mapping exists.
    HostTensorAccessor accessor(api());
    ASSERT_TRUE(accessor.add_child_memory(kBase, kBytes));
    platform_.host_view_available = false;
    platform_.backing[0] = 0x27;

    uint32_t value = 0;
    ASSERT_TRUE(accessor.read(kBase, &value, sizeof(value)));
    ASSERT_EQ(platform_.device_reads, 1);
    ASSERT_FALSE(host_tensor_refusal_was_dependency());

    platform_.written_by_other_run = true;
    EXPECT_FALSE(accessor.read(kBase, &value, sizeof(value)));
    EXPECT_TRUE(host_tensor_refusal_was_dependency());
    EXPECT_EQ(platform_.device_reads, 1) << "the second read must not reach the device";
}

TEST_F(HostTensorAccessDeferral, ACallerBufferRegionIsNeverDeferred) {
    // A host-memory tensor's region is backed by the buffer this bind just copied in. Its content
    // is the caller's, so the declaration is irrelevant and never asked.
    HostTensorAccessor accessor(api());
    std::vector<unsigned char> caller_buffer(kBytes, 0x7e);
    ASSERT_TRUE(accessor.add(kBase, kBytes, caller_buffer.data()));
    platform_.written_by_other_run = true;

    uint32_t value = 0;
    EXPECT_TRUE(accessor.read(kBase, &value, sizeof(value)));
    EXPECT_EQ(value & 0xffu, 0x7eu);
    EXPECT_FALSE(host_tensor_refusal_was_dependency());
    EXPECT_EQ(platform_.held_queries, 0);
}

TEST_F(HostTensorAccessDeferral, ThePublishedCauseSurvivesCloseAndIsNeverCleared) {
    HostTensorAccessor accessor(api());
    ASSERT_TRUE(accessor.add_child_memory(kBase, kBytes));
    platform_.written_by_other_run = true;
    uint32_t value = 0;
    ASSERT_FALSE(accessor.read(kBase, &value, sizeof(value)));
    ASSERT_TRUE(host_tensor_refusal_was_dependency());
    // A refusal alone is not the run's cause: the access whose own fatal report latches the field
    // publishes it, which here is the accessor's caller.
    ASSERT_FALSE(accessor.dependency_wait_is_this_runs_cause());
    accessor.note_dependency_wait_cause();
    ASSERT_TRUE(accessor.dependency_wait_is_this_runs_cause());

    // The caller reads the cause after closing its window, so the close must not clear it — and
    // nothing later may either: not an access that is no longer refused, and not a second
    // publication, which is how two refused accesses racing each other still leave one wait.
    platform_.written_by_other_run = false;
    ASSERT_TRUE(accessor.read(kBase, &value, sizeof(value)));
    EXPECT_FALSE(host_tensor_refusal_was_dependency());
    accessor.note_dependency_wait_cause();
    accessor.close();
    EXPECT_TRUE(accessor.dependency_wait_is_this_runs_cause());
}

TEST_F(HostTensorAccessDeferral, APlatformWithoutTheQueryNeverDefers) {
    // A backend that does not publish the query answers "no declared producer" by absence, which
    // is what keeps a platform without it on exactly its old behaviour.
    ops_.caller_device_span_written_by_other_run = nullptr;
    HostApi api(&platform_, 0, 0, 0, &ops_);
    HostTensorAccessor accessor(&api);
    ASSERT_TRUE(accessor.add_child_memory(kBase, kBytes));
    platform_.written_by_other_run = true;

    uint32_t value = 0;
    EXPECT_TRUE(accessor.read(kBase, &value, sizeof(value)));
    EXPECT_FALSE(host_tensor_refusal_was_dependency());
}
