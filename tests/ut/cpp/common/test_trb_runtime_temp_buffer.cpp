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
// Host-side fake HostApi tests for TRB bind/validate tensor leases.
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
#include "runtime_status.h"
#include "runtime_types.h"
#include "shared_memory.h"
#include "runtime.h"
#include "task_args.h"
#include "worker/runtime_c_api.h"

extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc);
extern "C" int concurrent_native_prepare_supported_impl(void);
extern "C" int prepared_run_config_compatible_impl(
    const HostApi *api, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
);

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
    // The context's execution identity, as the platform reports it. A
    // kernel-mode context's retained buffer keeps its address for the
    // context's life.
    bool kernel_mode = false;
    // The retained temporary-buffer slot the platform remembers across runs.
    void *retained_addr = nullptr;
    size_t retained_size = 0;
    std::unordered_set<void *> live_mallocs;
    std::vector<uint8_t> gm_heap;
    std::vector<uint8_t> gm_sm;
    std::vector<uint8_t> runtime_arena;
    bool compatibility_key_valid = false;
    uint64_t compatibility_hash = 0;
    std::vector<uint8_t> compatibility_key;
    uint64_t observed_hash = 0;
    std::vector<uint8_t> observed_key;

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

void *fake_device_malloc(void * /*runner_ctx*/, size_t size) {
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

void fake_device_free(void * /*runner_ctx*/, void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    ++g_fake->device_free_count;
    EXPECT_EQ(g_fake->live_mallocs.count(ptr), 1u);
    g_fake->live_mallocs.erase(ptr);
    std::free(ptr);
}

int fake_copy_to_device(void * /*runner_ctx*/, void *dev_ptr, const void *host_ptr, size_t size) {
    ++g_fake->copy_to_count;
    if (g_fake->fail_copy_to_on_call != 0 && g_fake->copy_to_count == g_fake->fail_copy_to_on_call) {
        return -7;
    }
    std::memcpy(dev_ptr, host_ptr, size);
    return 0;
}

int fake_copy_from_device(void * /*runner_ctx*/, void *host_ptr, const void *dev_ptr, size_t size) {
    ++g_fake->copy_from_count;
    std::memcpy(host_ptr, dev_ptr, size);
    return 0;
}

bool fake_is_kernel_mode(void * /*runner_ctx*/) { return g_fake->kernel_mode; }

void *fake_register_device_memory_to_host(void * /*runner_ctx*/, void *dev_ptr, size_t /* bytes */) { return dev_ptr; }

void fake_unregister_device_memory_from_host(void * /*runner_ctx*/, void * /* dev_ptr */) {}

int fake_device_memset(void * /*runner_ctx*/, void *dev_ptr, int value, size_t size) {
    ++g_fake->device_memset_count;
    std::memset(dev_ptr, value, size);
    return 0;
}

void fake_get_retained_temp_buffer(void * /*runner_ctx*/, uint32_t /*pipeline_slot*/, void **addr, size_t *size) {
    if (addr != nullptr) *addr = g_fake->retained_addr;
    if (size != nullptr) *size = g_fake->retained_size;
}

void fake_set_retained_temp_buffer(void * /*runner_ctx*/, uint32_t /*pipeline_slot*/, void *addr, size_t size) {
    g_fake->retained_addr = addr;
    g_fake->retained_size = size;
}

int fake_setup_static_arena(
    void * /*runner_ctx*/, uint32_t /*arena_bank*/, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size
) {
    ++g_fake->setup_static_arena_count;
    g_fake->gm_heap.assign(gm_heap_size, 0);
    g_fake->gm_sm.assign(gm_sm_size, 0);
    g_fake->runtime_arena.assign(runtime_arena_size, 0);
    return 0;
}

void *fake_acquire_pooled_gm_heap(void * /*runner_ctx*/, uint32_t /*arena_bank*/) {
    return g_fake->gm_heap.empty() ? nullptr : g_fake->gm_heap.data();
}
void *fake_acquire_pooled_gm_sm(void * /*runner_ctx*/, uint32_t /*arena_bank*/) {
    return g_fake->gm_sm.empty() ? nullptr : g_fake->gm_sm.data();
}
void *fake_acquire_pooled_runtime_arena(void * /*runner_ctx*/, uint32_t /*arena_bank*/) {
    return g_fake->runtime_arena.empty() ? nullptr : g_fake->runtime_arena.data();
}
bool fake_lookup_prebuilt_runtime_arena_cache(
    void * /*runner_ctx*/, uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size,
    void **gm_heap_base, void **sm_base, void **runtime_arena_base, size_t *runtime_off, const void **image_data,
    size_t *image_size
) {
    const auto *key = static_cast<const uint8_t *>(key_data);
    g_fake->observed_hash = hash;
    g_fake->observed_key.assign(key, key + key_size);
    const bool hit = arena_bank == 0 && g_fake->compatibility_key_valid && hash == g_fake->compatibility_hash &&
                     g_fake->observed_key == g_fake->compatibility_key;
    if (hit) {
        *gm_heap_base = reinterpret_cast<void *>(1);
        *sm_base = reinterpret_cast<void *>(2);
        *runtime_arena_base = reinterpret_cast<void *>(3);
        *runtime_off = 4;
        *image_data = reinterpret_cast<const void *>(5);
        *image_size = 6;
    }
    return hit;
}
void fake_mark_prebuilt_runtime_arena_cached(
    void * /*runner_ctx*/, uint32_t /*arena_bank*/, uint64_t /* hash */, const void * /* key_data */,
    size_t /* key_size */, void * /* gm_heap_base */, void * /* sm_base */, void * /* runtime_arena_base */,
    size_t /* runtime_off */, const void * /* image_data */, size_t /* image_size */
) {}
uint64_t fake_upload_chip_callable_buffer(void * /*runner_ctx*/, const void * /* callable */) { return 0; }

HostApi make_host_api() {
    static const HostApiOps ops = {
        .device_malloc = fake_device_malloc,
        .device_free = fake_device_free,
        .copy_to_device = fake_copy_to_device,
        .copy_from_device = fake_copy_from_device,
        .register_device_memory_to_host = fake_register_device_memory_to_host,
        .unregister_device_memory_from_host = fake_unregister_device_memory_from_host,
        .device_memset = fake_device_memset,
        .get_retained_temp_buffer = fake_get_retained_temp_buffer,
        .set_retained_temp_buffer = fake_set_retained_temp_buffer,
        .setup_static_arena = fake_setup_static_arena,
        .acquire_pooled_gm_heap = fake_acquire_pooled_gm_heap,
        .acquire_pooled_gm_sm = fake_acquire_pooled_gm_sm,
        .acquire_pooled_runtime_arena = fake_acquire_pooled_runtime_arena,
        .lookup_prebuilt_runtime_arena_cache = fake_lookup_prebuilt_runtime_arena_cache,
        .mark_prebuilt_runtime_arena_cached = fake_mark_prebuilt_runtime_arena_cached,
        .upload_chip_callable_buffer = fake_upload_chip_callable_buffer,
        .is_kernel_mode = fake_is_kernel_mode,
    };
    return HostApi(nullptr, 0, 0, &ops);
}

ChipTensor make_tensor(std::vector<uint8_t> &storage, bool child_memory = false) {
    ChipTensor tensor;
    uint32_t shape[1] = {static_cast<uint32_t>(storage.size())};
    tensor.init_external(
        storage.data(), storage.size(), shape, 1, DataType::UINT8,
        child_memory ? AddressSpace::DEVICE : AddressSpace::HOST
    );
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
    uint64_t ring_task_window[CHIP_MAX_RING_DEPTH] = {4, 4, 4, 4};
    uint64_t ring_heap[CHIP_MAX_RING_DEPTH] = {1024, 1024, 1024, 1024};
    uint64_t ring_dep_pool[CHIP_MAX_RING_DEPTH] = {4, 4, 4, 4};
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
};

}  // namespace

TEST_F(TrbRuntimeTempBufferTest, SuccessfulValidateCopiesOnlyOutputTensor) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(output));
    ArgDirection signature[1] = {ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 1u);
    std::memset(runtime.tensor_leases_[0].dev_ptr, 0x2a, output.size());

    ASSERT_EQ(validate_runtime_impl(&runtime, &api_, 0), 0);
    EXPECT_EQ(fake_.copy_from_count, 1);
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint8_t value) {
        return value == 0x2a;
    }));
}

TEST_F(TrbRuntimeTempBufferTest, FailedExecutionCopiesRuntimeStatus) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(output));
    ArgDirection signature[1] = {ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    auto *header = static_cast<SharedMemoryHeader *>(runtime.get_gm_sm_ptr());
    ASSERT_NE(header, nullptr);
    header->orch_error_code.store(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, std::memory_order_relaxed);

    EXPECT_EQ(validate_runtime_impl(&runtime, &api_, -1), -SIMPLER_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(fake_.copy_from_count, 1);
}

TEST_F(TrbRuntimeTempBufferTest, FailedExecutionWithoutDeviceStatusSkipsTensorCopyBack) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(output));
    ArgDirection signature[1] = {ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 1u);
    std::memset(runtime.tensor_leases_[0].dev_ptr, 0x2a, output.size());

    // A stream/bind failure may happen before the device publishes a
    // status. The one D2H is the diagnostic header; tensor data stays untouched.
    EXPECT_EQ(validate_runtime_impl(&runtime, &api_, -1), 0);
    EXPECT_EQ(fake_.copy_from_count, 1);
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint8_t value) {
        return value == 0;
    }));
}

// The retained buffer is malloc'd once for the run and sliced, not per tensor.
TEST_F(TrbRuntimeTempBufferTest, TemporaryBufferSlicesWithoutChangingCopies) {
    std::vector<uint8_t> input(64, 7);
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args = make_args(input, output);
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    // A single device_malloc backs the whole run (two
    // 64-byte tensors pack to 2 * 1024-aligned = 2048 bytes), sliced in place.
    fake_.reset();
    Runtime buffer_runtime = make_runtime();
    ASSERT_EQ(bind_runtime(buffer_runtime, api_, args, signature, 2), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) * 2);
    EXPECT_EQ(fake_.copy_to_count, 2);
    EXPECT_EQ(fake_.device_memset_count, 0);
    ASSERT_EQ(validate_runtime_impl(&buffer_runtime, &api_, 0), 0);
    // Retained buffer is NOT freed at end of run — it lives on the slot.
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_EQ(fake_.copy_from_count, 1);
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
    ASSERT_EQ(validate_runtime_impl(&run1, &api_, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    void *first_addr = fake_.retained_addr;

    Runtime run2 = make_runtime();
    ASSERT_EQ(bind_runtime(run2, api_, args, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run2, &api_, 0), 0);
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
    ASSERT_EQ(validate_runtime_impl(&run1, &api_, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) * 2);

    // Larger run: free old + malloc new.
    std::vector<uint8_t> big_in(4096, 1);
    std::vector<uint8_t> big_out(4096, 0);
    ChipStorageTaskArgs big = make_args(big_in, big_out);
    Runtime run2 = make_runtime();
    ASSERT_EQ(bind_runtime(run2, api_, big, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run2, &api_, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, 2);
    EXPECT_EQ(fake_.device_free_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(4096, kAlign) * 2);
    size_t after_grow_mallocs = fake_.device_malloc_count;

    // Smaller run again: retained buffer is big enough, no free/malloc.
    Runtime run3 = make_runtime();
    ASSERT_EQ(bind_runtime(run3, api_, small, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&run3, &api_, 0), 0);
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
    ASSERT_EQ(validate_runtime_impl(&runtime, &api_, 0), 0);
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

    EXPECT_EQ(bind_runtime(runtime, api_, args, signature, 2), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fake_.retained_addr, nullptr);
    EXPECT_EQ(fake_.retained_size, 0u);
    EXPECT_TRUE(fake_.live_mallocs.empty());
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

    EXPECT_EQ(bind_runtime(runtime, api_, args, signature, 1), PTO_RUNTIME_ERR_INTERNAL);
    // Retained buffer was allocated once for the grow and is NOT freed on the
    // error path (it lives on the slot for the next run); the slice lease is a
    // no-op, so no device_free happens here.
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_NE(fake_.retained_addr, nullptr);
    EXPECT_TRUE(runtime.tensor_leases_.empty());
}

TEST_F(TrbRuntimeTempBufferTest, PreparedRuntimeEnvRequiresTheActiveArenaKey) {
    fake_.reset();
    HostApi compatibility_api = make_host_api();
    uint64_t task_window[CHIP_MAX_RING_DEPTH] = {4, 4, 4, 4};
    uint64_t heap[CHIP_MAX_RING_DEPTH] = {1024, 1024, 1024, 1024};
    uint64_t dep_pool[CHIP_MAX_RING_DEPTH] = {4, 4, 4, 4};

    EXPECT_EQ(concurrent_native_prepare_supported_impl(), 1);
    EXPECT_EQ(prepared_run_config_compatible_impl(&compatibility_api, task_window, heap, dep_pool), 0);
    fake_.compatibility_key_valid = true;
    fake_.compatibility_hash = fake_.observed_hash;
    fake_.compatibility_key = fake_.observed_key;

    EXPECT_EQ(prepared_run_config_compatible_impl(&compatibility_api, task_window, heap, dep_pool), 1);
    heap[2] = 2048;
    EXPECT_EQ(prepared_run_config_compatible_impl(&compatibility_api, task_window, heap, dep_pool), 0);
    EXPECT_NE(fake_.observed_key, fake_.compatibility_key);
}

// ---------------------------------------------------------------------------
// Kernel-mode capacity: the retained buffer is context-static, so a run that
// would grow it is refused rather than served. Growing is free + malloc, which
// re-bases every slice the buffer has handed out, and a captured graph replays
// the addresses of the run it captured. An empty slot holds no such address, so
// the context's first allocation is not a re-base and is taken normally.
//
// The latch is write-once, so a real context is in kernel mode from its very
// first bind -- KernelModeAllocatesOnceThenRefusesToGrow is that shape. The
// cases that warm up in program mode and then flip the flag isolate the guard
// itself against a slot that already holds a buffer.
// ---------------------------------------------------------------------------

// A run whose tensors are all device memory stages nothing: the packed
// temporary size is zero, so the retained slot is never reached and no grow is
// considered. This is the shape a kernel bind is meant to have.
TEST_F(TrbRuntimeTempBufferTest, KernelModeDeviceTensorRunTouchesNoAllocator) {
    fake_.reset();
    fake_.kernel_mode = true;
    Runtime runtime = make_runtime();
    std::vector<uint8_t> first(64, 1);
    std::vector<uint8_t> second(64, 2);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(first, true));
    args.add_tensor(make_tensor(second, true));
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 2), 0);

    EXPECT_EQ(fake_.device_malloc_count, 0);
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_EQ(fake_.retained_addr, nullptr);
    EXPECT_EQ(fake_.retained_size, 0u);
}

// The shape a real kernel context takes. Its first bind finds an empty slot and
// must be allowed to allocate, since no captured graph can hold a slice of a
// buffer that never existed; every later bind that would outgrow that buffer is
// refused, because by then the address is one a replay may reference.
TEST_F(TrbRuntimeTempBufferTest, KernelModeAllocatesOnceThenRefusesToGrow) {
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};
    fake_.reset();
    fake_.kernel_mode = true;

    std::vector<uint8_t> sized_in(64, 1);
    std::vector<uint8_t> sized_out(64, 0);
    ChipStorageTaskArgs sized = make_args(sized_in, sized_out);
    Runtime first_run = make_runtime();
    ASSERT_EQ(bind_runtime(first_run, api_, sized, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&first_run, &api_, 0), 0);
    void *pinned_addr = fake_.retained_addr;
    const size_t pinned_size = fake_.retained_size;
    ASSERT_NE(pinned_addr, nullptr);
    EXPECT_EQ(pinned_size, align_up(64, kAlign) * 2);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.device_free_count, 0);

    // Past that one allocation the slot is frozen.
    std::vector<uint8_t> big_in(64, 1);
    std::vector<uint8_t> big_out(kAlign + 1, 0);
    ChipStorageTaskArgs big = make_args(big_in, big_out);
    Runtime big_run = make_runtime();
    EXPECT_EQ(bind_runtime(big_run, api_, big, signature, 2), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fake_.retained_addr, pinned_addr);
    EXPECT_EQ(fake_.retained_size, pinned_size);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.device_free_count, 0);

    // The run the context was sized for still binds, from the same address.
    Runtime again = make_runtime();
    EXPECT_EQ(bind_runtime(again, api_, sized, signature, 2), 0);
    EXPECT_EQ(validate_runtime_impl(&again, &api_, 0), 0);
    EXPECT_EQ(fake_.retained_addr, pinned_addr);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    EXPECT_EQ(fake_.device_free_count, 0);
}

// A run larger than the retained buffer is refused, and the refusal happens
// ahead of the free: the slot still holds the address and size the previous
// run left there.
TEST_F(TrbRuntimeTempBufferTest, KernelModeRefusesToGrowTheRetainedBuffer) {
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};
    fake_.reset();

    // The buffer the context was prepared with.
    std::vector<uint8_t> warm_in(64, 1);
    std::vector<uint8_t> warm_out(64, 0);
    ChipStorageTaskArgs warm = make_args(warm_in, warm_out);
    Runtime warm_run = make_runtime();
    ASSERT_EQ(bind_runtime(warm_run, api_, warm, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&warm_run, &api_, 0), 0);
    void *pinned_addr = fake_.retained_addr;
    const size_t pinned_size = fake_.retained_size;
    ASSERT_EQ(pinned_size, align_up(64, kAlign) * 2);
    const int mallocs = fake_.device_malloc_count;
    const int frees = fake_.device_free_count;

    fake_.kernel_mode = true;
    // One alignment unit past what the buffer holds.
    std::vector<uint8_t> big_in(64, 1);
    std::vector<uint8_t> big_out(kAlign + 1, 0);
    ChipStorageTaskArgs big = make_args(big_in, big_out);
    Runtime refused = make_runtime();

    EXPECT_EQ(bind_runtime(refused, api_, big, signature, 2), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fake_.retained_addr, pinned_addr);
    EXPECT_EQ(fake_.retained_size, pinned_size);
    EXPECT_EQ(fake_.device_malloc_count, mallocs);
    EXPECT_EQ(fake_.device_free_count, frees);
    EXPECT_TRUE(refused.tensor_leases_.empty());
}

// The refused run leaves the plan the context was prepared with intact: a run
// that fits still binds, from the same address, without entering the allocator.
TEST_F(TrbRuntimeTempBufferTest, KernelModeRefusalLeavesTheRetainedBufferUsable) {
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};
    fake_.reset();

    std::vector<uint8_t> warm_in(64, 1);
    std::vector<uint8_t> warm_out(64, 0);
    ChipStorageTaskArgs warm = make_args(warm_in, warm_out);
    Runtime warm_run = make_runtime();
    ASSERT_EQ(bind_runtime(warm_run, api_, warm, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&warm_run, &api_, 0), 0);
    void *pinned_addr = fake_.retained_addr;
    const size_t pinned_size = fake_.retained_size;

    fake_.kernel_mode = true;
    std::vector<uint8_t> big_in(64, 1);
    std::vector<uint8_t> big_out(kAlign + 1, 0);
    ChipStorageTaskArgs big = make_args(big_in, big_out);
    Runtime refused = make_runtime();
    ASSERT_EQ(bind_runtime(refused, api_, big, signature, 2), PTO_RUNTIME_ERR_INTERNAL);
    const int mallocs = fake_.device_malloc_count;
    const int frees = fake_.device_free_count;

    Runtime after = make_runtime();
    EXPECT_EQ(bind_runtime(after, api_, warm, signature, 2), 0);
    EXPECT_EQ(validate_runtime_impl(&after, &api_, 0), 0);
    EXPECT_EQ(fake_.retained_addr, pinned_addr);
    EXPECT_EQ(fake_.retained_size, pinned_size);
    EXPECT_EQ(fake_.device_malloc_count, mallocs);
    EXPECT_EQ(fake_.device_free_count, frees);
}

// Address and capacity hold across repeated runs, including a run that fills
// the buffer exactly and runs whose tensors arrive in a different order.
TEST_F(TrbRuntimeTempBufferTest, KernelModeHoldsOneAddressAcrossRepeatedRuns) {
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};
    fake_.reset();

    std::vector<uint8_t> warm_in(kAlign * 2, 1);
    std::vector<uint8_t> warm_out(kAlign * 2, 0);
    ChipStorageTaskArgs warm = make_args(warm_in, warm_out);
    Runtime warm_run = make_runtime();
    ASSERT_EQ(bind_runtime(warm_run, api_, warm, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&warm_run, &api_, 0), 0);
    void *pinned_addr = fake_.retained_addr;
    const size_t pinned_size = fake_.retained_size;
    ASSERT_EQ(pinned_size, kAlign * 4);

    fake_.kernel_mode = true;
    const int mallocs = fake_.device_malloc_count;
    const int frees = fake_.device_free_count;

    // A run that exactly fills the buffer, four times over.
    for (int run = 0; run < 4; ++run) {
        Runtime exact = make_runtime();
        EXPECT_EQ(bind_runtime(exact, api_, warm, signature, 2), 0);
        EXPECT_EQ(validate_runtime_impl(&exact, &api_, 0), 0);
        EXPECT_EQ(fake_.retained_addr, pinned_addr);
        EXPECT_EQ(fake_.retained_size, pinned_size);
    }

    // A run that packs to strictly less than the retained size, and one whose
    // tensors are the same pair reversed.
    std::vector<uint8_t> small_in(64, 1);
    std::vector<uint8_t> small_out(64, 0);
    ChipStorageTaskArgs small = make_args(small_in, small_out);
    ASSERT_LT(align_up(64, kAlign) * 2, pinned_size);
    Runtime small_run = make_runtime();
    EXPECT_EQ(bind_runtime(small_run, api_, small, signature, 2), 0);
    EXPECT_EQ(validate_runtime_impl(&small_run, &api_, 0), 0);

    ChipStorageTaskArgs reordered = make_args(warm_out, warm_in);
    Runtime reordered_run = make_runtime();
    EXPECT_EQ(bind_runtime(reordered_run, api_, reordered, signature, 2), 0);
    EXPECT_EQ(validate_runtime_impl(&reordered_run, &api_, 0), 0);

    EXPECT_EQ(fake_.retained_addr, pinned_addr);
    EXPECT_EQ(fake_.retained_size, pinned_size);
    EXPECT_EQ(fake_.device_malloc_count, mallocs);
    EXPECT_EQ(fake_.device_free_count, frees);
}

// Program mode still grows, so the rule is selected by the context's identity
// rather than applied to every caller.
TEST_F(TrbRuntimeTempBufferTest, ProgramModeStillGrowsTheRetainedBuffer) {
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};
    fake_.reset();

    std::vector<uint8_t> small_in(64, 1);
    std::vector<uint8_t> small_out(64, 0);
    ChipStorageTaskArgs small = make_args(small_in, small_out);
    Runtime small_run = make_runtime();
    ASSERT_EQ(bind_runtime(small_run, api_, small, signature, 2), 0);
    ASSERT_EQ(validate_runtime_impl(&small_run, &api_, 0), 0);

    std::vector<uint8_t> big_in(64, 1);
    std::vector<uint8_t> big_out(kAlign + 1, 0);
    ChipStorageTaskArgs big = make_args(big_in, big_out);
    Runtime big_run = make_runtime();
    EXPECT_EQ(bind_runtime(big_run, api_, big, signature, 2), 0);
    EXPECT_EQ(validate_runtime_impl(&big_run, &api_, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, 2);
    EXPECT_EQ(fake_.device_free_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) + align_up(kAlign + 1, kAlign));
}
