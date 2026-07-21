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
// Host-side fake HostApi tests for a2a3 TRB bind/validate tensor leases.
//
// The retained temporary buffer's grow/pack/slice logic lives entirely in
// runtime_maker.cpp (file-local RetainedTempBump). The platform side is just a
// {addr, size} slot exposed via get/set_retained_temp_buffer, and the buffer
// is grown through the ordinary device_malloc/device_free callbacks. So these
// end-to-end bind/validate tests exercise the real grow/reuse logic while the
// fake only remembers the slot and records malloc/copy counts.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "arg_direction.h"
#include "common/host_api.h"
#include "pto_runtime2_types.h"
#include "runtime.h"
#include "task_args.h"

extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api);

namespace {

// 1024-byte aligned device pointers are required by TRB kernels; RetainedTempBump
// packs and slices at this alignment, so the test's expected sizes use it too.
constexpr size_t kAlign = 1024;

size_t align_up(size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

struct FakeHostApi {
    int device_malloc_count = 0;
    int device_free_count = 0;
    int copy_to_count = 0;
    int copy_from_count = 0;
    int device_memset_count = 0;
    int setup_static_arena_count = 0;
    int fail_copy_to_on_call = 0;
    int fail_device_malloc_on_call = 0;
    // The retained temporary-buffer slot the platform remembers across runs.
    void *retained_addr = nullptr;
    size_t retained_size = 0;
    std::unordered_set<void *> live_mallocs;
    std::vector<uint8_t> gm_heap;
    std::vector<uint8_t> gm_sm;
    std::vector<uint8_t> runtime_arena;

    ~FakeHostApi() { release_all(); }

    void release_all() {
        for (void *ptr : live_mallocs) {
            std::free(ptr);
        }
        live_mallocs.clear();
        retained_addr = nullptr;
        retained_size = 0;
    }

    void reset() {
        release_all();
        *this = FakeHostApi();
    }
};

FakeHostApi *g_fake = nullptr;

void *fake_device_malloc(size_t size) {
    if (g_fake->fail_device_malloc_on_call != 0 &&
        g_fake->device_malloc_count + 1 == g_fake->fail_device_malloc_on_call) {
        ++g_fake->device_malloc_count;
        return nullptr;
    }
    // Over-align so a retained-buffer base satisfies the 1024-byte requirement
    // the same way the real device_malloc does.
    void *ptr = nullptr;
    if (posix_memalign(&ptr, kAlign, std::max<size_t>(size, 1)) != 0) {
        return nullptr;
    }
    ++g_fake->device_malloc_count;
    g_fake->live_mallocs.insert(ptr);
    return ptr;
}

void fake_device_free(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    ++g_fake->device_free_count;
    EXPECT_EQ(g_fake->live_mallocs.count(ptr), 1u);
    g_fake->live_mallocs.erase(ptr);
    std::free(ptr);
}

int fake_copy_to_device(void *dev_ptr, const void *host_ptr, size_t size) {
    ++g_fake->copy_to_count;
    if (g_fake->fail_copy_to_on_call != 0 && g_fake->copy_to_count == g_fake->fail_copy_to_on_call) {
        return -7;
    }
    std::memcpy(dev_ptr, host_ptr, size);
    return 0;
}

int fake_copy_from_device(void *host_ptr, const void *dev_ptr, size_t size) {
    ++g_fake->copy_from_count;
    std::memcpy(host_ptr, dev_ptr, size);
    return 0;
}

void *fake_register_device_memory_to_host(void *dev_ptr, size_t /* bytes */) { return dev_ptr; }

void fake_unregister_device_memory_from_host(void * /* dev_ptr */) {}

int fake_device_memset(void *dev_ptr, int value, size_t size) {
    ++g_fake->device_memset_count;
    std::memset(dev_ptr, value, size);
    return 0;
}

void fake_get_retained_temp_buffer(void **addr, size_t *size) {
    if (addr != nullptr) *addr = g_fake->retained_addr;
    if (size != nullptr) *size = g_fake->retained_size;
}

void fake_set_retained_temp_buffer(void *addr, size_t size) {
    g_fake->retained_addr = addr;
    g_fake->retained_size = size;
}

int fake_setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size) {
    ++g_fake->setup_static_arena_count;
    g_fake->gm_heap.assign(gm_heap_size, 0);
    g_fake->gm_sm.assign(gm_sm_size, 0);
    g_fake->runtime_arena.assign(runtime_arena_size, 0);
    return 0;
}

void *fake_acquire_pooled_gm_heap() { return g_fake->gm_heap.empty() ? nullptr : g_fake->gm_heap.data(); }
void *fake_acquire_pooled_gm_sm() { return g_fake->gm_sm.empty() ? nullptr : g_fake->gm_sm.data(); }
void *fake_acquire_pooled_runtime_arena() {
    return g_fake->runtime_arena.empty() ? nullptr : g_fake->runtime_arena.data();
}
bool fake_lookup_prebuilt_runtime_arena_cache(
    uint64_t /* hash */, const void * /* key_data */, size_t /* key_size */, void ** /* gm_heap_base */,
    void ** /* sm_base */, void ** /* runtime_arena_base */, size_t * /* runtime_off */, const void ** /* image_data */,
    size_t * /* image_size */
) {
    return false;
}
void fake_mark_prebuilt_runtime_arena_cached(
    uint64_t /* hash */, const void * /* key_data */, size_t /* key_size */, void * /* gm_heap_base */,
    void * /* sm_base */, void * /* runtime_arena_base */, size_t /* runtime_off */, const void * /* image_data */,
    size_t /* image_size */
) {}
uint64_t fake_upload_chip_callable_buffer(const void * /* callable */) { return 0; }

// with_temporary_buffer=false leaves the retained-slot callbacks null, which
// makes trb bind fall back to a per-tensor device_malloc for each tensor.
HostApi make_host_api(bool with_temporary_buffer = true) {
    return HostApi{
        fake_device_malloc,
        fake_device_free,
        fake_copy_to_device,
        fake_copy_from_device,
        fake_register_device_memory_to_host,
        fake_unregister_device_memory_from_host,
        fake_device_memset,
        with_temporary_buffer ? fake_get_retained_temp_buffer : nullptr,
        with_temporary_buffer ? fake_set_retained_temp_buffer : nullptr,
        fake_setup_static_arena,
        fake_acquire_pooled_gm_heap,
        fake_acquire_pooled_gm_sm,
        fake_acquire_pooled_runtime_arena,
        fake_lookup_prebuilt_runtime_arena_cache,
        fake_mark_prebuilt_runtime_arena_cached,
        fake_upload_chip_callable_buffer,
    };
}

Tensor make_tensor(std::vector<uint8_t> &storage, bool child_memory = false) {
    Tensor tensor;
    uint32_t shape[1] = {static_cast<uint32_t>(storage.size())};
    tensor.init_external(storage.data(), storage.size(), shape, 1, DataType::UINT8, 0, false, child_memory ? 1 : 0);
    return tensor;
}

ChipStorageTaskArgs make_args(std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(input));
    args.add_tensor(make_tensor(output));
    return args;
}

int bind_runtime(
    Runtime &runtime, const HostApi &api, const ChipStorageTaskArgs &args, const ArgDirection *signature, int sig_count
) {
    uint64_t ring_task_window[PTO2_MAX_RING_DEPTH] = {4, 4, 4, 4};
    uint64_t ring_heap[PTO2_MAX_RING_DEPTH] = {1024, 1024, 1024, 1024};
    uint64_t ring_dep_pool[PTO2_MAX_RING_DEPTH] = {4, 4, 4, 4};
    return bind_callable_to_runtime_impl(
        &runtime, &api, &args, nullptr, signature, sig_count, ring_task_window, ring_heap, ring_dep_pool
    );
}

class TrbRuntimeTempBufferTest : public ::testing::Test {
protected:
    void SetUp() override { g_fake = &fake_; }
    void TearDown() override {
        fake_.release_all();
        g_fake = nullptr;
    }

    Runtime make_runtime() { return Runtime{}; }

    FakeHostApi fake_;
    HostApi api_ = make_host_api();
    // Temp-buffer slot callbacks left null: trb bind falls back to device_malloc.
    HostApi malloc_api_ = make_host_api(false);
};

}  // namespace

// The retained buffer is malloc'd once for the run and sliced (not per-tensor
// malloc'd); copies/memsets are unchanged from the fallback path.
TEST_F(TrbRuntimeTempBufferTest, TemporaryBufferSlicesWithoutChangingCopies) {
    std::vector<uint8_t> input(64, 7);
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args = make_args(input, output);
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    // Fallback path (null slot callbacks): one device_malloc per tensor.
    fake_.reset();
    Runtime malloc_runtime = make_runtime();
    ASSERT_EQ(bind_runtime(malloc_runtime, malloc_api_, args, signature, 2), 0);
    EXPECT_EQ(fake_.device_malloc_count, 2);
    EXPECT_EQ(fake_.copy_to_count, 2);
    EXPECT_EQ(fake_.device_memset_count, 0);
    ASSERT_EQ(validate_runtime_impl(&malloc_runtime, &malloc_api_), 0);
    EXPECT_EQ(fake_.device_free_count, 2);
    EXPECT_EQ(fake_.copy_from_count, 2);

    // Retained-buffer path: single device_malloc for the whole run (two
    // 64-byte tensors pack to 2 * 1024-aligned = 2048 bytes), sliced in place.
    fake_.reset();
    Runtime buffer_runtime = make_runtime();
    ASSERT_EQ(bind_runtime(buffer_runtime, api_, args, signature, 2), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) * 2);
    EXPECT_EQ(fake_.copy_to_count, 2);
    EXPECT_EQ(fake_.device_memset_count, 0);
    ASSERT_EQ(validate_runtime_impl(&buffer_runtime, &api_), 0);
    // Retained buffer is NOT freed at end of run — it lives on the slot.
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_EQ(fake_.copy_from_count, 2);
    EXPECT_NE(fake_.retained_addr, nullptr);
}

TEST_F(TrbRuntimeTempBufferTest, SecondSameShapeRunReusesRetainedBuffer) {
    std::vector<uint8_t> input(64, 7);
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args = make_args(input, output);
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    fake_.reset();
    Runtime run1 = make_runtime();
    ASSERT_EQ(bind_runtime(run1, api_, args, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run1, &api_), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    void *first_addr = fake_.retained_addr;

    Runtime run2 = make_runtime();
    ASSERT_EQ(bind_runtime(run2, api_, args, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run2, &api_), 0);
    // Same shape → no new allocation, same retained buffer.
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_EQ(fake_.retained_addr, first_addr);
}

TEST_F(TrbRuntimeTempBufferTest, LargerRunGrowsSmallerRunKeepsBuffer) {
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    fake_.reset();
    std::vector<uint8_t> small_in(64, 1);
    std::vector<uint8_t> small_out(64, 0);
    ChipStorageTaskArgs small = make_args(small_in, small_out);
    Runtime run1 = make_runtime();
    ASSERT_EQ(bind_runtime(run1, api_, small, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run1, &api_), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) * 2);

    // Larger run: free old + malloc new.
    std::vector<uint8_t> big_in(4096, 1);
    std::vector<uint8_t> big_out(4096, 0);
    ChipStorageTaskArgs big = make_args(big_in, big_out);
    Runtime run2 = make_runtime();
    ASSERT_EQ(bind_runtime(run2, api_, big, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run2, &api_), 0);
    EXPECT_EQ(fake_.device_malloc_count, 2);
    EXPECT_EQ(fake_.device_free_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(4096, kAlign) * 2);
    size_t after_grow_mallocs = fake_.device_malloc_count;

    // Smaller run again: retained buffer is big enough, no free/malloc.
    Runtime run3 = make_runtime();
    ASSERT_EQ(bind_runtime(run3, api_, small, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run3, &api_), 0);
    EXPECT_EQ(fake_.device_malloc_count, static_cast<int>(after_grow_mallocs));
    EXPECT_EQ(fake_.device_free_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(4096, kAlign) * 2);
}

TEST_F(TrbRuntimeTempBufferTest, ChildMemoryIsPassThroughAndPureOutSkipsStaging) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> child(64, 3);
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(child, true));
    args.add_tensor(make_tensor(output));
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 2), 0);
    // The pure-OUT tensor still gets a retained slice (one 1024-aligned slot,
    // no per-tensor malloc), but its buffer is handed to the kernel with no
    // staging; the child is passed through.
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign));
    // The pure-OUT tensor is neither copied nor memset and the child is passed
    // through, so no tensor copy-in and no memset — the single copy_to is the
    // runtime arena image upload that every bind performs.
    EXPECT_EQ(fake_.copy_to_count, 1);
    EXPECT_EQ(fake_.device_memset_count, 0);
    ASSERT_EQ(validate_runtime_impl(&runtime, &api_), 0);
    EXPECT_EQ(fake_.device_free_count, 0);
}

TEST_F(TrbRuntimeTempBufferTest, GrowAllocationFailureFailsBindWithoutLeak) {
    fake_.reset();
    fake_.fail_device_malloc_on_call = 1;  // fail the retained-buffer grow
    Runtime runtime = make_runtime();
    std::vector<uint8_t> input(64, 1);
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args = make_args(input, output);
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    EXPECT_EQ(bind_runtime(runtime, api_, args, signature, 2), -1);
    EXPECT_EQ(fake_.retained_addr, nullptr);
    EXPECT_EQ(fake_.retained_size, 0u);
    EXPECT_TRUE(fake_.live_mallocs.empty());
    EXPECT_TRUE(runtime.tensor_leases_.empty());
}

TEST_F(TrbRuntimeTempBufferTest, FailedCopyReleasesRecordedFreeLease) {
    fake_.reset();
    fake_.fail_copy_to_on_call = 1;
    Runtime runtime = make_runtime();
    std::vector<uint8_t> input(64, 9);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(input));
    ArgDirection signature[1] = {ArgDirection::IN};

    // Fallback path: the per-tensor device_malloc is freed on the copy-failure
    // error path, leaving no lease and no leak.
    EXPECT_EQ(bind_runtime(runtime, malloc_api_, args, signature, 1), -1);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.device_free_count, 1);
    EXPECT_TRUE(runtime.tensor_leases_.empty());
}

TEST_F(TrbRuntimeTempBufferTest, FailedCopyOnTemporaryPathDoesNotFreeRetainedBuffer) {
    fake_.reset();
    fake_.fail_copy_to_on_call = 1;
    Runtime runtime = make_runtime();
    std::vector<uint8_t> input(64, 9);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(input));
    ArgDirection signature[1] = {ArgDirection::IN};

    EXPECT_EQ(bind_runtime(runtime, api_, args, signature, 1), -1);
    // Retained buffer was allocated once for the grow and is NOT freed on the
    // error path (it lives on the slot for the next run); the slice lease is a
    // no-op, so no device_free happens here.
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_NE(fake_.retained_addr, nullptr);
    EXPECT_TRUE(runtime.tensor_leases_.empty());
}
