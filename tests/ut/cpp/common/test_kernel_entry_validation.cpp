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

#include <gtest/gtest.h>

#include <cstdlib>

#include "host/kernel_entry_validation.h"
#include "utils/device_arena.h"

namespace {

int dummy_ctx_storage = 0;
void *const kCtx = &dummy_ctx_storage;
const uint8_t kBinary[4] = {1, 2, 3, 4};
const int kConfigStorage = 0;
const void *const kConfig = &kConfigStorage;
int dummy_stream_storage = 0;
void *const kStream = &dummy_stream_storage;
alignas(ChipCallable) const unsigned char kCallableImage[sizeof(ChipCallable)] = {};

TEST(KernelEntryValidation, BinarySpanRequiresPointerAndSizeTogether) {
    EXPECT_TRUE(kernel_binary_span_is_consistent(nullptr, 0));
    EXPECT_TRUE(kernel_binary_span_is_consistent(kBinary, sizeof(kBinary)));
    EXPECT_FALSE(kernel_binary_span_is_consistent(nullptr, 4));
    EXPECT_FALSE(kernel_binary_span_is_consistent(kBinary, 0));
}

TEST(KernelEntryValidation, InitAcceptsConsistentArgs) {
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        0
    );
}

TEST(KernelEntryValidation, InitRejectsEachStructuralViolation) {
    EXPECT_EQ(
        validate_kernel_init_args(
            nullptr, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1
        ),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, nullptr, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, -1, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 0),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    // One inconsistent span per position, in both directions.
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, nullptr, 4, kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, 0, kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), nullptr, 4, nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, 0, nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), kBinary, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 4, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelEntryValidation, PrepareCallableChecksIdRangeAndImageSize) {
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, 0, kCallableImage, sizeof(ChipCallable), kStream), 0);
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(
            kCtx, MAX_REGISTERED_CALLABLE_IDS - 1, kCallableImage, sizeof(ChipCallable), kStream
        ),
        0
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(nullptr, 0, kCallableImage, sizeof(ChipCallable), kStream),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, 0, nullptr, sizeof(ChipCallable), kStream),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, 0, kCallableImage, sizeof(ChipCallable), nullptr),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, -1, kCallableImage, sizeof(ChipCallable), kStream),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(
            kCtx, MAX_REGISTERED_CALLABLE_IDS, kCallableImage, sizeof(ChipCallable), kStream
        ),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, 0, kCallableImage, sizeof(ChipCallable) - 1, kStream),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    // storage_ sits at a CALLABLE_CHILD_ALIGN offset from the header, so an
    // image the caller placed off-alignment puts every child off-alignment.
    static_assert(alignof(ChipCallable) > 1, "a misaligned image must be expressible");
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, 0, kCallableImage + 1, sizeof(ChipCallable), kStream),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelEntryValidation, LaunchChecksPointersAndIdRange) {
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 0, kCallableImage, kStream), 0);
    EXPECT_EQ(validate_kernel_launch_args(nullptr, 0, kCallableImage, kStream), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 0, nullptr, kStream), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 0, kCallableImage, nullptr), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, -1, kCallableImage, kStream), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(
        validate_kernel_launch_args(kCtx, MAX_REGISTERED_CALLABLE_IDS, kCallableImage, kStream),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelArenaGuard, UncommittedRegionAcceptsAnyRequest) {
    // Kernel mode must be able to establish capacity once, so nothing about an
    // uncommitted region is refusable.
    EXPECT_FALSE(kernel_arena_change_is_forbidden(false, 0, 0));
    EXPECT_FALSE(kernel_arena_change_is_forbidden(false, 0, 4096));
    EXPECT_FALSE(kernel_arena_change_is_forbidden(false, 4096, 8192));
}

TEST(KernelArenaGuard, CommittedRegionRefusesGrowAndRelease) {
    EXPECT_TRUE(kernel_arena_change_is_forbidden(true, 4096, 4097));
    EXPECT_TRUE(kernel_arena_change_is_forbidden(true, 4096, 0));
}

TEST(KernelArenaGuard, CommittedRegionAllowsSameSizeAndShrink) {
    // A shrink keeps the base and the committed size, so a captured graph's
    // addresses survive it.
    EXPECT_FALSE(kernel_arena_change_is_forbidden(true, 4096, 4096));
    EXPECT_FALSE(kernel_arena_change_is_forbidden(true, 4096, 2048));
    // Committed with nothing cached: asking for nothing changes nothing.
    EXPECT_FALSE(kernel_arena_change_is_forbidden(true, 0, 0));
}

struct CountingBackend {
    int frees = 0;
    static void *alloc(void * /*ctx*/, size_t size) { return std::malloc(size); }
    static void free_fn(void *ctx, void *ptr) {
        static_cast<CountingBackend *>(ctx)->frees++;
        std::free(ptr);
    }
};

TEST(KernelArenaGuard, RefusalScanOverCommittedRegionsReleasesNothing) {
    // setup_static_arena's three-region shape. The guard is evaluated over
    // every region before any of them is touched, so reaching a refusal must
    // leave each committed peer's base, size and committed flag intact —
    // the arena commit sequence's rollback would free exactly these.
    CountingBackend backend;
    DeviceArena gm_heap(&CountingBackend::alloc, &CountingBackend::free_fn, &backend);
    DeviceArena gm_sm(&CountingBackend::alloc, &CountingBackend::free_fn, &backend);
    DeviceArena runtime_pool(&CountingBackend::alloc, &CountingBackend::free_fn, &backend);

    const size_t cached[] = {4096, 2048, 1024};
    DeviceArena *arenas[] = {&gm_heap, &gm_sm, &runtime_pool};
    for (size_t i = 0; i < 3; ++i) {
        arenas[i]->reserve(cached[i], 64);
        ASSERT_NE(arenas[i]->commit(), nullptr);
    }
    const int frees_after_commit = backend.frees;
    void *const bases[] = {gm_heap.base(), gm_sm.base(), runtime_pool.base()};

    // The third region asks to grow; the first two are unchanged requests.
    const size_t requested[] = {4096, 2048, 8192};
    bool refused = false;
    for (size_t i = 0; i < 3 && !refused; ++i) {
        refused = kernel_arena_change_is_forbidden(arenas[i]->is_committed(), cached[i], requested[i]);
    }

    EXPECT_TRUE(refused);
    EXPECT_EQ(backend.frees, frees_after_commit);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(arenas[i]->is_committed());
        EXPECT_EQ(arenas[i]->base(), bases[i]);
    }
}

}  // namespace
