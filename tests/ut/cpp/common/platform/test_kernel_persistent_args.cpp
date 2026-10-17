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
 * PersistentKernelArgs lifecycle, driven through a fake PersistentArgsOps so
 * allocation balance, partial-failure rollback and the exact bytes that reach
 * the device are all observable without a device.
 *
 * Compiled once per runtime variant. Both builds copy only their device descriptor,
 * leaving host-only state out of the image.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "kernel_persistent_args.h"
#include "persistent_args_fixture.h"
#include "host/runtime_launch_image.h"
#include "host/kernel_execution_state.h"
#include "runtime_c_api.h"

namespace {

using namespace persistent_args_test;

struct FakeLifecycleOps {
    FakeArgsOps memory;
    uintptr_t next_handle{1};
    std::set<void *> handles;
    std::vector<std::string> &calls() { return memory.calls; }

    static int create(void *context, void **handle, const char *kind) {
        auto &fake = *static_cast<FakeLifecycleOps *>(context);
        *handle = reinterpret_cast<void *>(fake.next_handle++);
        fake.handles.insert(*handle);
        fake.calls().push_back(std::string("create_") + kind);
        return 0;
    }
    static int destroy(void *context, void *handle, const char *kind) {
        auto &fake = *static_cast<FakeLifecycleOps *>(context);
        EXPECT_EQ(fake.handles.erase(handle), 1u);
        fake.calls().push_back(
            std::string("destroy_") + kind + ":" + std::to_string(reinterpret_cast<uintptr_t>(handle))
        );
        return 0;
    }
    KernelContextOps context_ops() {
        return {
            this,
            [](void *context, int *device) noexcept {
                static_cast<FakeLifecycleOps *>(context)->calls().push_back("get_device");
                *device = kDeviceId;
                return 0;
            },
            [](void *context, void **current) noexcept {
                static_cast<FakeLifecycleOps *>(context)->calls().push_back("get_context");
                *current = context;
                return 0;
            },
            [](void *context, void **stream) noexcept {
                return create(context, stream, "stream");
            },
            [](void *context, void *stream) noexcept {
                return destroy(context, stream, "stream");
            },
            [](void *context, void **event) noexcept {
                return create(context, event, "event");
            },
            [](void *context, void *event) noexcept {
                return destroy(context, event, "event");
            },
        };
    }

    // Test-only downstream consumer; this does not implement or qualify the
    // future binder's event protocol. It consumes the real owners' resources.
    void launch(KernelExecutionState &state, const PersistentKernelArgs &args) {
        ASSERT_TRUE(state.accepts_dispatch());
        ASSERT_TRUE(args.is_prepared());
        const KernelArgs *image = args.device_k_args();
        ASSERT_NE(image, nullptr);
        EXPECT_EQ(image->runtime_args, args.args().runtime_args);
        EXPECT_EQ(image->regs, args.args().regs);
        for (auto kind : {KernelStreamKind::Aicpu, KernelStreamKind::Aicore}) {
            EXPECT_EQ(handles.count(state.hidden_stream(kind)), 1u);
        }
        for (size_t i = 0; i < static_cast<size_t>(KernelEventKind::Count); ++i) {
            EXPECT_EQ(handles.count(state.event(static_cast<KernelEventKind>(i))), 1u);
        }
        calls().push_back("fake_launch");
    }
};

class KernelResourceLifecycle : public ::testing::TestWithParam<int> {};

TEST_P(KernelResourceLifecycle, HasExactResourceCallOrder) {
    FakeLifecycleOps fake;
    std::vector<std::string> expected{"get_device", "get_context", "create_stream", "create_stream"};
    expected.insert(expected.end(), static_cast<size_t>(KernelEventKind::Count), "create_event");
    {
        KernelExecutionState state;
        Runtime runtime;
        PersistentKernelArgs args;
        ASSERT_EQ(state.initialize(kDeviceId, fake.context_ops()), 0);
        EXPECT_EQ(fake.calls(), expected);
        ASSERT_EQ(args.prepare_once(runtime, fake.memory.table(), kDeviceId), 0);
        ASSERT_EQ(state.mark_ready_enqueued(), 0);
        expected.insert(expected.end(), {"alloc", "copy", "arch", "alloc", "alloc", "copy"});
        EXPECT_EQ(fake.calls(), expected);
        const auto *device_args = args.device_k_args();
        const auto *runtime_args = args.args().runtime_args;
        const auto regs = args.args().regs;
        for (int i = 0; i < GetParam(); ++i) {
            SCOPED_TRACE(i);
            ASSERT_EQ(args.prepare_once(runtime, fake.memory.table(), kDeviceId), 0);
            fake.launch(state, args);
            expected.push_back("fake_launch");
            EXPECT_EQ(fake.calls(), expected);
            EXPECT_EQ(args.device_k_args(), device_args);
            EXPECT_EQ(args.args().runtime_args, runtime_args);
            EXPECT_EQ(args.args().regs, regs);
        }
        ASSERT_EQ(args.finalize_once(), 0);
        expected.insert(expected.end(), {"free", "free", "free"});
        EXPECT_EQ(fake.calls(), expected);
        ASSERT_EQ(state.close(), 0);
        expected.insert(expected.end(), {"get_device", "get_context"});
        for (uintptr_t id = 2 + static_cast<size_t>(KernelEventKind::Count); id > 2; --id)
            expected.push_back("destroy_event:" + std::to_string(id));
        expected.insert(expected.end(), {"destroy_stream:2", "destroy_stream:1"});
        EXPECT_EQ(fake.calls(), expected);
        EXPECT_TRUE(fake.handles.empty());
        EXPECT_EQ(fake.memory.live_blocks(), 0u);
        EXPECT_EQ(args.finalize_once(), 0);
        EXPECT_EQ(state.close(), 0);
    }
    EXPECT_EQ(fake.calls(), expected);
}

INSTANTIATE_TEST_SUITE_P(LaunchCounts, KernelResourceLifecycle, ::testing::Values(0, 1, 128));

// ---------------------------------------------------------------------------
// Prepare once, then reuse.
// ---------------------------------------------------------------------------

TEST(PersistentKernelArgs, ReusesTheSameAddressesForEveryLaunch) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    ASSERT_TRUE(args.is_prepared());
    EXPECT_EQ(ops.last_device_id, kDeviceId);

    KernelArgs *const device_args = args.device_k_args();
    Runtime *const runtime_args = args.args().runtime_args;
    const uint64_t regs = args.args().regs;
    ASSERT_NE(device_args, nullptr);
    ASSERT_NE(runtime_args, nullptr);
    ASSERT_NE(regs, 0u);

    const int allocs_after_prepare = ops.alloc_calls;
    const int copies_after_prepare = ops.copy_calls;
    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(args.device_k_args(), device_args);
        EXPECT_EQ(args.args().runtime_args, runtime_args);
        EXPECT_EQ(args.args().regs, regs);
    }
    EXPECT_EQ(ops.alloc_calls, allocs_after_prepare);
    EXPECT_EQ(ops.copy_calls, copies_after_prepare);

    EXPECT_EQ(args.finalize_once(), 0);
}

TEST(PersistentKernelArgs, PrepareOnceIsIdempotent) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    const int allocs = ops.alloc_calls;
    const int copies = ops.copy_calls;
    const int fills = ops.fill_calls;
    KernelArgs *const device_args = args.device_k_args();

    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);

    EXPECT_EQ(ops.alloc_calls, allocs);
    EXPECT_EQ(ops.copy_calls, copies);
    EXPECT_EQ(ops.fill_calls, fills);
    EXPECT_EQ(args.device_k_args(), device_args);

    EXPECT_EQ(args.finalize_once(), 0);
}

TEST(PersistentKernelArgs, IncompleteOpsTableIsRejectedBeforeAnyAllocation) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    PersistentArgsOps incomplete = ops.table();
    EXPECT_TRUE(incomplete.valid());
    incomplete.fill_arch_fields = nullptr;
    EXPECT_FALSE(incomplete.valid());

    EXPECT_EQ(args.prepare_once(runtime, incomplete, kDeviceId), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(args.is_prepared());
    EXPECT_EQ(ops.alloc_calls, 0);
}

// ---------------------------------------------------------------------------
// Every failing step rolls back to "never prepared".
// ---------------------------------------------------------------------------

struct RollbackCase {
    const char *name;
    int fail_alloc_on;
    int fail_copy_on;
    int fail_fill_on;
};

class PersistentKernelArgsRollback : public ::testing::TestWithParam<RollbackCase> {};

TEST_P(PersistentKernelArgsRollback, ReleasesEverythingItAllocated) {
    const RollbackCase &c = GetParam();
    FakeArgsOps ops;
    ops.fail_alloc_on = c.fail_alloc_on;
    ops.fail_copy_on = c.fail_copy_on;
    ops.fail_fill_on = c.fail_fill_on;

    Runtime runtime;
    PersistentKernelArgs args;

    const int rc = args.prepare_once(runtime, ops.table(), kDeviceId);
    EXPECT_NE(rc, 0) << c.name;
    EXPECT_FALSE(args.is_prepared()) << c.name;
    EXPECT_EQ(args.device_k_args(), nullptr) << c.name;
    EXPECT_EQ(args.args().runtime_args, nullptr) << c.name;
    EXPECT_EQ(args.args().regs, 0u) << c.name;
    EXPECT_EQ(ops.live_blocks(), 0u) << c.name;

    // A second attempt on a rolled-back owner starts from scratch and succeeds.
    ops.fail_alloc_on = 0;
    ops.fail_copy_on = 0;
    ops.fail_fill_on = 0;
    EXPECT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0) << c.name;
    EXPECT_TRUE(args.is_prepared()) << c.name;
    EXPECT_EQ(args.finalize_once(), 0) << c.name;
    EXPECT_EQ(ops.live_blocks(), 0u) << c.name;
}

INSTANTIATE_TEST_SUITE_P(
    EveryAllocatingStep, PersistentKernelArgsRollback,
    ::testing::Values(
        RollbackCase{"runtime_args alloc", 1, 0, 0}, RollbackCase{"runtime_args copy", 0, 1, 0},
        RollbackCase{"arch fields", 0, 0, 1}, RollbackCase{"arch fields alloc", 2, 0, 0},
        RollbackCase{"device KernelArgs alloc", 3, 0, 0}, RollbackCase{"device KernelArgs copy", 0, 2, 0}
    ),
    [](const ::testing::TestParamInfo<RollbackCase> &info) {
        std::string name(info.param.name);
        for (char &ch : name) {
            if (ch == ' ') ch = '_';
        }
        return name;
    }
);

TEST(PersistentKernelArgs, FailedRollbackRetainsOwnershipUntilFinalizeRetry) {
    FakeArgsOps ops;
    ops.fail_copy_on = 2;
    ops.fail_free_on = 1;
    Runtime runtime;
    PersistentKernelArgs args;

    EXPECT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), kInjectedRc);
    EXPECT_FALSE(args.is_prepared());
    EXPECT_NE(args.device_k_args(), nullptr);
    EXPECT_EQ(args.args().runtime_args, nullptr);
    EXPECT_EQ(args.args().regs, 0u);
    EXPECT_EQ(ops.live_blocks(), 1u);

    const int allocs = ops.alloc_calls;
    EXPECT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(ops.alloc_calls, allocs);

    EXPECT_EQ(args.finalize_once(), 0);
    EXPECT_EQ(args.device_k_args(), nullptr);
    EXPECT_EQ(ops.live_blocks(), 0u);
}

// ---------------------------------------------------------------------------
// Release, abandon, retry.
// ---------------------------------------------------------------------------

TEST(PersistentKernelArgs, FinalizeOnceBalancesAndIsIdempotent) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    KernelArgs *const device_args = args.device_k_args();
    Runtime *const runtime_args = args.args().runtime_args;
    const uint64_t regs = args.args().regs;
    ASSERT_EQ(args.finalize_once(), 0);

    // Reverse allocation order: the device copy that names the other two
    // blocks is released before either of them.
    ASSERT_EQ(ops.free_order.size(), 3u);
    EXPECT_EQ(ops.free_order[0], device_args);
    EXPECT_EQ(ops.free_order[1], reinterpret_cast<void *>(regs));
    EXPECT_EQ(ops.free_order[2], runtime_args);

    EXPECT_EQ(ops.free_calls, ops.alloc_calls);
    EXPECT_EQ(ops.live_blocks(), 0u);
    EXPECT_FALSE(args.is_prepared());
    EXPECT_EQ(args.device_k_args(), nullptr);
    EXPECT_EQ(args.args().runtime_args, nullptr);
    EXPECT_EQ(args.args().regs, 0u);

    const int frees = ops.free_calls;
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(args.finalize_once(), 0);
    EXPECT_EQ(ops.free_calls, frees);
}

TEST(PersistentKernelArgs, FinalizeRetryRedoesOnlyTheRemainder) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);

    // The device KernelArgs block is released first; failing that one release
    // still releases the remaining two and keeps only the failed address.
    ops.fail_free_on = ops.free_calls + 1;
    EXPECT_EQ(args.finalize_once(), kInjectedRc);
    EXPECT_NE(args.device_k_args(), nullptr);
    EXPECT_EQ(args.args().runtime_args, nullptr);
    EXPECT_EQ(args.args().regs, 0u);
    EXPECT_EQ(ops.live_blocks(), 1u);
    EXPECT_TRUE(args.is_prepared());

    ops.fail_free_on = 0;
    const int frees_before_retry = ops.free_calls;
    EXPECT_EQ(args.finalize_once(), 0);
    EXPECT_EQ(ops.free_calls, frees_before_retry + 1);
    EXPECT_EQ(ops.live_blocks(), 0u);
    EXPECT_EQ(args.device_k_args(), nullptr);
    EXPECT_FALSE(args.is_prepared());
}

TEST(PersistentKernelArgs, AbandonNeverReachesTheOpsTable) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    const int frees = ops.free_calls;

    args.abandon();

    EXPECT_EQ(ops.free_calls, frees);
    EXPECT_FALSE(args.is_prepared());
    EXPECT_EQ(args.device_k_args(), nullptr);
    EXPECT_EQ(args.args().runtime_args, nullptr);
    EXPECT_EQ(args.args().regs, 0u);

    // A later release attempt on an abandoned owner still reaches no table.
    EXPECT_EQ(args.finalize_once(), 0);
    EXPECT_EQ(ops.free_calls, frees);
}

TEST(PersistentKernelArgs, DestructionReleasesNothing) {
    FakeArgsOps ops;
    Runtime runtime;
    {
        PersistentKernelArgs args;
        ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    }
    EXPECT_EQ(ops.free_calls, 0);
    EXPECT_NE(ops.live_blocks(), 0u);
}

// ---------------------------------------------------------------------------
// What actually crosses to the device.
// ---------------------------------------------------------------------------

TEST(PersistentKernelArgs, CopiesExactlyTheRuntimeDeviceImage) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    ASSERT_EQ(ops.copies.size(), 2u);

    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    EXPECT_EQ(extent_bytes, sizeof(DeviceRuntimeLaunchDesc));
    // The upload is a prefix of the extent. It is shorter on a variant with a
    // device-initialized tail and equal on one without; both descriptors this
    // file is built against have one.
    EXPECT_LE(image_bytes, extent_bytes);
    EXPECT_LT(extent_bytes, sizeof(Runtime));

    // The allocation covers whatever the copy does not: the device addresses the
    // tail inside this block, so sizing it to the uploaded prefix would put those
    // reads past its end.
    EXPECT_EQ(ops.block_size(args.args().runtime_args), extent_bytes);
    EXPECT_EQ(ops.copies[0].src_bytes, image_bytes);
    EXPECT_EQ(ops.copies[0].dst_bytes, image_bytes);
    EXPECT_EQ(ops.copies[0].dst, args.args().runtime_args);
    EXPECT_EQ(std::memcmp(args.args().runtime_args, &runtime, image_bytes), 0);

    EXPECT_EQ(ops.copies[1].src_bytes, sizeof(KernelArgs));
    EXPECT_EQ(ops.copies[1].dst, args.device_k_args());
    EXPECT_EQ(std::memcmp(args.device_k_args(), &args.args(), sizeof(KernelArgs)), 0);

    EXPECT_EQ(args.finalize_once(), 0);
}

TEST(PersistentKernelArgs, LeavesEveryDfxFieldZero) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);

    const KernelArgs *const uploaded = args.device_k_args();
    EXPECT_EQ(uploaded->dump_data_base, 0u);
    EXPECT_EQ(uploaded->chip_swimlane_data_base, 0u);
    EXPECT_EQ(uploaded->pmu_data_base, 0u);
    EXPECT_EQ(uploaded->dep_gen_data_base, 0u);
    EXPECT_EQ(uploaded->scope_stats_data_base, 0u);
    EXPECT_EQ(uploaded->chip_swimlane_aicore_rotation_table, 0u);
    EXPECT_EQ(uploaded->device_wall_data_base, 0u);
    EXPECT_EQ(uploaded->enable_profiling_flag, 0u);

    EXPECT_EQ(args.finalize_once(), 0);
}

// The gate tail is device-read storage the host never uploads. Two extents, one
// block: a copy that reached the tail would overwrite what the device put there,
// and an allocation sized to the copy would leave the device addressing past its
// end.
//
// The mark is seeded by the allocator, so it is already in the block when the
// real copy runs. Seeding it afterwards would paint over exactly the evidence
// this case exists to read.
TEST(PersistentKernelArgs, LeavesTheDeviceInitializedTailUntouched) {
    constexpr unsigned char kDeviceMark = 0x5A;
    FakeArgsOps ops;
    ops.fill_byte = kDeviceMark;
    Runtime runtime;
    PersistentKernelArgs args;

    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    // Both descriptors this file is built against declare the gate array, so the
    // shortfall is a property of the type, asserted rather than skipped: a change
    // that widened the copy back to the extent must fail here, not opt out.
    ASSERT_EQ(extent_bytes, sizeof(DeviceRuntimeLaunchDesc));
    ASSERT_EQ(image_bytes, offsetof(DeviceRuntimeLaunchDesc, teardown_gates))
        << "the upload must stop where host-initialized storage ends, before the gate tail";
    ASSERT_LT(image_bytes, extent_bytes);

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    void *const block = args.args().runtime_args;
    ASSERT_NE(block, nullptr);

    // Bounds first: if an allocation shrank to the uploaded prefix, say so here
    // rather than reading past the block below.
    ASSERT_EQ(ops.block_size(block), extent_bytes) << "the allocation does not cover the device-read tail";
    ASSERT_EQ(ops.copies[0].dst, block);
    ASSERT_EQ(ops.copies[0].dst_bytes, image_bytes) << "the copy was offered more than the host-initialized prefix";

    // The mark the allocator seeded survives across the tail: the copy stopped
    // where host-initialized storage ends. `Runtime()` never writes that tail,
    // so a copy reaching it would push indeterminate host bytes to the device.
    // The prefix itself is checked against the source elsewhere.
    const auto *const bytes = reinterpret_cast<const unsigned char *>(block);
    for (size_t i = image_bytes; i < extent_bytes; ++i) {
        ASSERT_EQ(bytes[i], kDeviceMark) << "the upload reached the host-uninitialized tail at byte " << i;
    }

    EXPECT_EQ(args.finalize_once(), 0);
}

// Prepare-once means one allocation and one metadata copy no matter how often it
// is called, so a second prepare cannot re-touch the tail either.
TEST(PersistentKernelArgs, RepeatedPrepareNeitherReallocatesNorRecopies) {
    constexpr unsigned char kDeviceMark = 0x5A;
    FakeArgsOps ops;
    ops.fill_byte = kDeviceMark;
    Runtime runtime;
    PersistentKernelArgs args;

    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    ASSERT_EQ(image_bytes, offsetof(DeviceRuntimeLaunchDesc, teardown_gates));
    ASSERT_LT(image_bytes, extent_bytes);

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    const int allocs_after_first = ops.alloc_calls;
    const size_t copies_after_first = ops.copies.size();
    void *const block = args.args().runtime_args;
    ASSERT_NE(block, nullptr);

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);
    EXPECT_EQ(ops.alloc_calls, allocs_after_first);
    EXPECT_EQ(ops.copies.size(), copies_after_first);
    EXPECT_EQ(args.args().runtime_args, block);

    ASSERT_EQ(ops.block_size(block), extent_bytes);
    const auto *const bytes = reinterpret_cast<const unsigned char *>(block);
    for (size_t i = image_bytes; i < extent_bytes; ++i) {
        ASSERT_EQ(bytes[i], kDeviceMark) << "a repeated prepare reached the tail at byte " << i;
    }

    EXPECT_EQ(args.finalize_once(), 0);
}

}  // namespace

TEST(RuntimeLaunchImage, SnapshotIsIndependentOfLaterHostMutationAndConsumedOnce) {
    Runtime runtime;
    runtime.dev.worker_count = 7;
    RuntimeLaunchImage image;
    image.prepare(runtime, runtime_device_copy_size(runtime));
    runtime.dev.worker_count = 19;
    int copies = 0;
    EXPECT_EQ(
        image.publish(
            [&](const void *source, size_t bytes) {
                ++copies;
                // The snapshot is the uploaded prefix, which may be shorter than the
                // descriptor: reconstruct into a zeroed one and take only what the
                // snapshot holds, or this reads past the source.
                EXPECT_EQ(bytes, runtime_device_copy_size(runtime));
                EXPECT_LE(bytes, runtime_device_extent_size(runtime));
                DeviceRuntimeLaunchDesc descriptor{};
                std::memcpy(&descriptor, source, bytes);
                EXPECT_EQ(descriptor.worker_count, 7);
                return 0;
            },
            runtime_device_copy_size(runtime)
        ),
        0
    );
    EXPECT_NE(
        image.publish(
            [&](const void *, size_t) {
                ++copies;
                return 0;
            },
            runtime_device_copy_size(runtime)
        ),
        0
    );
    EXPECT_EQ(copies, 1);
}

TEST(RuntimeLaunchImage, FailedPublicationConsumesSourceAndFreshPrepareReplacesIt) {
    Runtime runtime;
    RuntimeLaunchImage image;
    image.prepare(runtime, runtime_device_copy_size(runtime));
    EXPECT_EQ(
        image.publish(
            [](const void *, size_t) {
                return -91;
            },
            runtime_device_copy_size(runtime)
        ),
        -91
    );
    EXPECT_NE(
        image.publish(
            [](const void *, size_t) {
                return 0;
            },
            runtime_device_copy_size(runtime)
        ),
        0
    );
    runtime.dev.worker_count = 3;
    image.prepare(runtime, runtime_device_copy_size(runtime));
    EXPECT_EQ(
        image.publish(
            [](const void *source, size_t bytes) {
                DeviceRuntimeLaunchDesc descriptor;
                std::memcpy(&descriptor, source, bytes);
                EXPECT_EQ(descriptor.worker_count, 3);
                return 0;
            },
            runtime_device_copy_size(runtime)
        ),
        0
    );
}
