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
// Host-side fake HostApi tests for TRB tensor leases: what the bind records,
// what each half of the run's tensor IO then does with it.
//
// The retained temporary buffer's grow/pack/slice logic lives entirely in
// runtime_maker.cpp (file-local RetainedTempBump). The platform side is just a
// {addr, size} slot exposed via get/set_retained_temp_buffer, and the buffer
// is grown through the ordinary device_malloc/device_free callbacks. So these
// end-to-end tests exercise the real grow/reuse logic while the fake only
// remembers the slot and records malloc/copy counts.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "arg_direction.h"
#include "call_config.h"
#include "common/host_api.h"
#include "host/kernel_pipeline_contract.h"
#include "runtime_status.h"
#include "runtime_types.h"
#include "shared_memory.h"
#include "runtime.h"
#include "task_args.h"
#include "worker/runtime_c_api.h"
#include "worker/pipeline_contract.h"

extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);
extern "C" int copy_in_run_inputs_impl(const Runtime *runtime, const HostApi *api);
extern "C" int copy_back_run_outputs_impl(const Runtime *runtime, const HostApi *api, int execution_rc, int launched);
extern "C" int release_run_bindings_impl(Runtime *runtime, const HostApi *api);
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
    // Deliberately NOT over-aligned: the sim backend's device_malloc is
    // std::malloc, and RetainedTempBump is what aligns the base it hands out.
    void *ptr = std::malloc(std::max<size_t>(size, 1));
    if (ptr == nullptr) {
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

// The grow the platform now owns, as the sequence the bump used to run itself.
int fake_acquire_retained_temp(
    void *runner_ctx, uint32_t pipeline_slot, size_t bytes, void **addr_out, size_t *size_out
) {
    fake_get_retained_temp_buffer(runner_ctx, pipeline_slot, addr_out, size_out);
    if (bytes == 0 || bytes <= *size_out) return 0;
    if (*addr_out != nullptr) fake_device_free(runner_ctx, *addr_out);
    void *grown = fake_device_malloc(runner_ctx, bytes);
    fake_set_retained_temp_buffer(runner_ctx, pipeline_slot, grown, grown == nullptr ? 0 : bytes);
    if (grown == nullptr) {
        *addr_out = nullptr;
        *size_out = 0;
        return -1;
    }
    *addr_out = grown;
    *size_out = bytes;
    return 0;
}

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
        .acquire_retained_temp = fake_acquire_retained_temp,
        .setup_static_arena = fake_setup_static_arena,
        .acquire_pooled_gm_heap = fake_acquire_pooled_gm_heap,
        .acquire_pooled_gm_sm = fake_acquire_pooled_gm_sm,
        .acquire_pooled_runtime_arena = fake_acquire_pooled_runtime_arena,
        .lookup_prebuilt_runtime_arena_cache = fake_lookup_prebuilt_runtime_arena_cache,
        .mark_prebuilt_runtime_arena_cached = fake_mark_prebuilt_runtime_arena_cached,
        .upload_chip_callable_buffer = fake_upload_chip_callable_buffer,
    };
    return HostApi(nullptr, 0, 0, 0, &ops);
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

// A HOST tensor with a size and no address. `init_external` accepts it and the
// bind gives it a real device slice, so it is reachable through the native API.
ChipTensor null_source_tensor(size_t bytes) {
    ChipTensor tensor;
    uint32_t shape[1] = {static_cast<uint32_t>(bytes)};
    tensor.init_external(nullptr, bytes, shape, 1, DataType::UINT8, AddressSpace::HOST);
    return tensor;
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

    int stage_inputs(Runtime &runtime) { return copy_in_run_inputs_impl(&runtime, &api_); }

    // The two halves the c_api calls back to back for a run it is finalizing:
    // read the results, then end the bindings they came back through.
    int finish_run(Runtime &runtime, int execution_rc, int launched = 1) {
        const int rc = copy_back_run_outputs_impl(&runtime, &api_, execution_rc, launched);
        const int release_rc = release_run_bindings_impl(&runtime, &api_);
        return rc != 0 ? rc : release_rc;
    }

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

    ASSERT_EQ(finish_run(runtime, 0), 0);
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

    EXPECT_EQ(finish_run(runtime, -1), -SIMPLER_ERROR_EXPLICIT_ORCH_FATAL);
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
    EXPECT_EQ(finish_run(runtime, -1), 0);
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
    // Over-sized by the headroom RetainedTempBump may spend aligning its base.
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) * 2 + kAlign - 1);
    // One H2D, and it is the runtime arena image: the bind names the tensors'
    // buffers and moves none of their bytes.
    EXPECT_EQ(fake_.copy_to_count, 1);
    EXPECT_EQ(fake_.device_memset_count, 0);
    ASSERT_EQ(stage_inputs(buffer_runtime), 0);
    EXPECT_EQ(fake_.copy_to_count, 2);
    ASSERT_EQ(finish_run(buffer_runtime, 0), 0);
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
    ASSERT_EQ(finish_run(run1, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    void *first_addr = fake_.retained_addr;

    Runtime run2 = make_runtime();
    ASSERT_EQ(bind_runtime(run2, api_, args, signature, 2), 0);
    ASSERT_EQ(finish_run(run2, 0), 0);
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
    ASSERT_EQ(finish_run(run1, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, 1);
    // Over-sized by the headroom RetainedTempBump may spend aligning its base.
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) * 2 + kAlign - 1);

    // Larger run: free old + malloc new.
    std::vector<uint8_t> big_in(4096, 1);
    std::vector<uint8_t> big_out(4096, 0);
    ChipStorageTaskArgs big = make_args(big_in, big_out);
    Runtime run2 = make_runtime();
    ASSERT_EQ(bind_runtime(run2, api_, big, signature, 2), 0);
    ASSERT_EQ(finish_run(run2, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, 2);
    EXPECT_EQ(fake_.device_free_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(4096, kAlign) * 2 + kAlign - 1);
    size_t after_grow_mallocs = fake_.device_malloc_count;

    // Smaller run again: retained buffer is big enough, no free/malloc.
    Runtime run3 = make_runtime();
    ASSERT_EQ(bind_runtime(run3, api_, small, signature, 2), 0);
    ASSERT_EQ(finish_run(run3, 0), 0);
    EXPECT_EQ(fake_.device_malloc_count, static_cast<int>(after_grow_mallocs));
    EXPECT_EQ(fake_.device_free_count, 1);
    EXPECT_EQ(fake_.retained_size, align_up(4096, kAlign) * 2 + kAlign - 1);
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
    EXPECT_EQ(fake_.retained_size, align_up(64, kAlign) + kAlign - 1);
    // The pure-OUT tensor is neither copied nor memset and the child is passed
    // through, so no tensor copy-in and no memset — the single copy_to is the
    // runtime arena image upload that every bind performs.
    EXPECT_EQ(fake_.copy_to_count, 1);
    EXPECT_EQ(fake_.device_memset_count, 0);
    // Staging has nothing to do either: an OUT tensor has no host content worth
    // moving and a child-memory tensor was never given a slice.
    ASSERT_EQ(stage_inputs(runtime), 0);
    EXPECT_EQ(fake_.copy_to_count, 1);
    ASSERT_EQ(finish_run(runtime, 0), 0);
    EXPECT_EQ(fake_.device_free_count, 0);
}

// The bind settles which device buffer each tensor gets; the run's own staging
// step settles what is in it. If the bind still moved the bytes, the copy count
// after it would already be 2.
TEST_F(TrbRuntimeTempBufferTest, BindNamesTheBufferAndStagingMovesTheBytes) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> input(64, 7);
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args = make_args(input, output);
    ArgDirection signature[2] = {ArgDirection::IN, ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 2), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 2u);
    EXPECT_TRUE(runtime.tensor_leases_[0].needs_copy_in);
    EXPECT_FALSE(runtime.tensor_leases_[0].needs_copy_back);
    EXPECT_FALSE(runtime.tensor_leases_[1].needs_copy_in);
    EXPECT_TRUE(runtime.tensor_leases_[1].needs_copy_back);
    EXPECT_EQ(fake_.copy_to_count, 1);

    // A sentinel the staging has to overwrite, so the assertion below cannot
    // pass on bytes that were already there.
    std::memset(runtime.tensor_leases_[0].dev_ptr, 0xab, input.size());
    ASSERT_EQ(stage_inputs(runtime), 0);
    EXPECT_EQ(fake_.copy_to_count, 2);
    EXPECT_EQ(std::memcmp(runtime.tensor_leases_[0].dev_ptr, input.data(), input.size()), 0);
}

// Staging reads the caller's buffer at the point the run owns it, so a caller
// that rewrites its inputs between two runs of one bind gets the new values.
TEST_F(TrbRuntimeTempBufferTest, StagingMovesWhateverTheCallerHoldsNow) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> input(64, 1);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(input));
    ArgDirection signature[1] = {ArgDirection::IN};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 1u);
    void *slice = runtime.tensor_leases_[0].dev_ptr;
    ASSERT_EQ(stage_inputs(runtime), 0);
    EXPECT_EQ(static_cast<const uint8_t *>(slice)[0], 1);

    std::fill(input.begin(), input.end(), 2);
    ASSERT_EQ(stage_inputs(runtime), 0);
    EXPECT_EQ(static_cast<const uint8_t *>(slice)[0], 2);
    EXPECT_EQ(slice, runtime.tensor_leases_[0].dev_ptr) << "staging must not re-place the buffer";
}

// An input the kernel will read, whose device buffer no host bytes can reach,
// must fail staging rather than be passed over. Skipping it would leave the
// slice holding whatever it held before and report success, so the run would
// consume those bytes as its input.
TEST_F(TrbRuntimeTempBufferTest, StagingRejectsAnInputWithNoHostSource) {
    fake_.reset();
    Runtime runtime = make_runtime();
    ChipStorageTaskArgs args;
    args.add_tensor(null_source_tensor(64));
    ArgDirection signature[1] = {ArgDirection::IN};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 1u);
    ASSERT_NE(runtime.tensor_leases_[0].dev_ptr, nullptr) << "the bind gave this input a real slice";
    ASSERT_EQ(runtime.tensor_leases_[0].host_ptr, nullptr);
    EXPECT_TRUE(runtime.tensor_leases_[0].needs_copy_in);
    // A sentinel the staging would otherwise leave for the kernel to read.
    std::memset(runtime.tensor_leases_[0].dev_ptr, 0xab, 64);
    const int copies_before = fake_.copy_to_count;

    EXPECT_NE(stage_inputs(runtime), 0) << "a null-source input was staged as a success";
    EXPECT_EQ(fake_.copy_to_count, copies_before) << "no H2D can have been attempted from a null source";
    // The bindings stay the caller's to release, as after any staging failure.
    EXPECT_EQ(runtime.tensor_leases_.size(), 1u);
    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    EXPECT_TRUE(runtime.tensor_leases_.empty());
}

// A pure OUTPUT tensor carries no input, so the same missing address is not an
// error there: the kernel defines every byte it writes.
TEST_F(TrbRuntimeTempBufferTest, StagingIgnoresAnOutputWithNoHostSource) {
    fake_.reset();
    Runtime runtime = make_runtime();
    ChipStorageTaskArgs args;
    args.add_tensor(null_source_tensor(64));
    ArgDirection signature[1] = {ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 1u);
    EXPECT_FALSE(runtime.tensor_leases_[0].needs_copy_in);
    const int copies_before = fake_.copy_to_count;

    EXPECT_EQ(stage_inputs(runtime), 0);
    EXPECT_EQ(fake_.copy_to_count, copies_before);
    ASSERT_EQ(finish_run(runtime, 0), 0);
}

// Reading a run's results and retiring the memory behind them are separate
// steps. A free during the copy-back would mean the two are still welded, and a
// partially submitted run would have no way to keep its bindings.
TEST_F(TrbRuntimeTempBufferTest, CopyBackLeavesTheBindingsForTheReleaseToEnd) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> output(64, 0);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(output));
    ArgDirection signature[1] = {ArgDirection::OUT};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    // An owned allocation alongside the retained slice, so the release has
    // something to actually free.
    std::vector<uint8_t> owned_host(32, 0);
    void *owned = fake_device_malloc(nullptr, owned_host.size());
    ASSERT_NE(owned, nullptr);
    runtime.tensor_leases_.push_back(
        {owned_host.data(), owned, owned_host.size(), false, false, TensorReleaseKind::Free}
    );
    const size_t lease_count = runtime.tensor_leases_.size();

    ASSERT_EQ(copy_back_run_outputs_impl(&runtime, &api_, 0, 1), 0);
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_EQ(runtime.tensor_leases_.size(), lease_count);

    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    EXPECT_EQ(fake_.device_free_count, 1) << "the owned allocation is the only lease the release frees";
    EXPECT_TRUE(runtime.tensor_leases_.empty());
    // The retained buffer is the slot's, not the run's.
    EXPECT_NE(fake_.retained_addr, nullptr);
}

// A run that never reached a stream has no device-side status, and the shared
// memory it would be read from belongs to whoever ran there last. `launched`
// carries that fact, so nothing has to null a pointer in the image to say it.
TEST_F(TrbRuntimeTempBufferTest, AnUnlaunchedRunReadsNoDeviceStatus) {
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
    void *const gm_sm_before = runtime.get_gm_sm_ptr();

    EXPECT_EQ(copy_back_run_outputs_impl(&runtime, &api_, PTO_RUNTIME_ERR_INTERNAL, /*launched=*/0), 0);
    EXPECT_EQ(fake_.copy_from_count, 0) << "an unlaunched run read device state";
    EXPECT_EQ(runtime.get_gm_sm_ptr(), gm_sm_before) << "the image must survive a run that never launched";
    EXPECT_EQ(runtime.tensor_leases_.size(), 1u);
    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
}

// A staging failure leaves the run releasable: the bindings are the bind's, and
// a partially staged input set is exactly when the caller must be able to end
// them without having launched.
TEST_F(TrbRuntimeTempBufferTest, FailedInputStagingKeepsTheRunsBindings) {
    fake_.reset();
    Runtime runtime = make_runtime();
    std::vector<uint8_t> input(64, 9);
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor(input));
    ArgDirection signature[1] = {ArgDirection::IN};

    ASSERT_EQ(bind_runtime(runtime, api_, args, signature, 1), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 1u);
    // The next H2D is this run's only input.
    fake_.fail_copy_to_on_call = fake_.copy_to_count + 1;
    EXPECT_EQ(stage_inputs(runtime), -7);

    // The retained buffer lives on the slot and the lease is still the run's to
    // release; the slice release is a no-op, so nothing is freed.
    EXPECT_NE(fake_.retained_addr, nullptr);
    EXPECT_EQ(runtime.tensor_leases_.size(), 1u);
    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    EXPECT_EQ(fake_.device_free_count, 0);
    EXPECT_TRUE(runtime.tensor_leases_.empty());
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

TEST_F(TrbRuntimeTempBufferTest, FailedImageUploadDoesNotFreeRetainedBuffer) {
    fake_.reset();
    // The bind's own H2D is the runtime arena image, and it is the first one a
    // bind performs now that no tensor bytes move here.
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
    // The lease the walk recorded before the failure is the caller's to release.
    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
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

namespace {

CallConfig small_kernel_config() {
    CallConfig config;
    for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
        config.runtime_env.ring_task_window[i] = 4;
        config.runtime_env.ring_heap[i] = 1024;
        config.runtime_env.ring_dep_pool[i] = 4;
    }
    return config;
}

uint64_t required_bytes(const PipelineContract &contract, PipelineResourceKind kind) {
    for (uint32_t i = 0; i < contract.resource_count; ++i) {
        if (contract.resources[i].kind == kind) return contract.resources[i].bytes_per_copy;
    }
    ADD_FAILURE() << "Missing resource " << kind;
    return 0;
}

void expect_same_contract(const PipelineContract &a, const PipelineContract &b) {
    EXPECT_EQ(a.abi_version, b.abi_version);
    EXPECT_EQ(a.pipeline_depth, b.pipeline_depth);
    ASSERT_EQ(a.resource_count, b.resource_count);
    for (uint32_t i = 0; i < a.resource_count; ++i) {
        EXPECT_EQ(a.resources[i].kind, b.resources[i].kind);
        EXPECT_EQ(a.resources[i].resource_class, b.resources[i].resource_class);
        EXPECT_EQ(a.resources[i].bytes_per_copy, b.resources[i].bytes_per_copy);
    }
}

}  // namespace

TEST(KernelPipelineBuilder, DefaultAndPackedInputsPreserveProgramContract) {
    const PipelineContract program_before = *get_pipeline_contract();
    CallConfig defaults;
    PipelineContract contract{};
    ASSERT_EQ(build_kernel_pipeline_contract_impl(&defaults, &contract), 0);
    EXPECT_TRUE(is_valid_tmr_kernel_pipeline_contract(&contract));
    EXPECT_EQ(contract.pipeline_depth, 2u);
    // Per-run args are a pipelined host buffer: one copy per slot, of the size a
    // launch actually hands over.
    EXPECT_EQ(required_bytes(contract, PTO_PIPELINE_TASK_ARGS), sizeof(ChipStorageTaskArgs));

    // CallConfig is packed and may start at any byte; use a genuinely unaligned input.
    alignas(uint64_t) std::array<unsigned char, sizeof(CallConfig) + 1> packed{};
    ASSERT_NE(
        reinterpret_cast<uintptr_t>(packed.data() + 1 + offsetof(CallConfig, runtime_env)) % alignof(uint64_t), 0u
    );
    const CallConfig small = small_kernel_config();
    std::memcpy(packed.data() + 1, &small, sizeof(small));
    const auto before = packed;
    PipelineContract expected{};
    ASSERT_EQ(build_kernel_pipeline_contract_impl(&small, &expected), 0);
    ASSERT_EQ(
        build_kernel_pipeline_contract_impl(reinterpret_cast<const CallConfig *>(packed.data() + 1), &contract), 0
    );
    expect_same_contract(contract, expected);
    EXPECT_EQ(packed, before);
    expect_same_contract(*get_pipeline_contract(), program_before);
    EXPECT_TRUE(is_valid_pipeline_contract(get_pipeline_contract()));
    EXPECT_EQ(get_pipeline_contract()->pipeline_depth, 2u);
}

TEST(KernelPipelineBuilder, InvalidSizesLeaveOutputUntouched) {
    PipelineContract output;
    std::memset(&output, 0x5a, sizeof(output));
    std::array<unsigned char, sizeof(output)> original{};
    std::memcpy(original.data(), &output, sizeof(output));
    auto reject = [&](const CallConfig *config) {
        EXPECT_EQ(build_kernel_pipeline_contract_impl(config, &output), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
        EXPECT_EQ(std::memcmp(&output, original.data(), sizeof(output)), 0);
    };
    reject(nullptr);
    auto config = small_kernel_config();
    EXPECT_EQ(build_kernel_pipeline_contract_impl(&config, nullptr), PTO_RUNTIME_ERR_INTERNAL);
    for (uint64_t bad : {uint64_t{1}, uint64_t{3}, uint64_t{6}, uint64_t{1} << 31}) {
        config = small_kernel_config();
        config.runtime_env.ring_task_window[0] = bad;
        reject(&config);
    }
    config = small_kernel_config();
    config.runtime_env.ring_task_window[0] = uint64_t{1} << 30;
    config.runtime_env.ring_task_window[1] = uint64_t{1} << 30;
    reject(&config);
    config = small_kernel_config();
    config.runtime_env.ring_heap[0] = 1023;
    reject(&config);
    config.runtime_env.ring_heap[0] = std::numeric_limits<uint64_t>::max();
    reject(&config);
    for (uint64_t bad : {uint64_t{3}, uint64_t{INT32_MAX} + 1}) {
        config = small_kernel_config();
        config.runtime_env.ring_dep_pool[0] = bad;
        reject(&config);
    }
    // Sum fits uint64_t but adding DeviceArena base-alignment slack would overflow.
    config = small_kernel_config();
    config.runtime_env.ring_heap[0] = std::numeric_limits<uint64_t>::max() - 3 * 1024;
    reject(&config);
    // Last usable byte before that alignment limit is legal; reserve must not allocate it.
    config.runtime_env.ring_heap[0] -= 1023;
    ASSERT_EQ(build_kernel_pipeline_contract_impl(&config, &output), 0);
    EXPECT_EQ(required_bytes(output, PTO_PIPELINE_GM_HEAP), std::numeric_limits<uint64_t>::max() - 1023);
}

TEST_F(TrbRuntimeTempBufferTest, KernelRequirementsMatchRealBindWithoutQuerySideEffects) {
    auto config = small_kernel_config();
    PipelineContract contract{};
    ASSERT_EQ(build_kernel_pipeline_contract_impl(&config, &contract), 0);
    EXPECT_EQ(fake_.setup_static_arena_count, 0);
    EXPECT_EQ(fake_.device_malloc_count, 0);
    EXPECT_EQ(fake_.copy_to_count, 0);
    Runtime runtime = make_runtime();
    ChipStorageTaskArgs args;
    ASSERT_EQ(bind_runtime(runtime, api_, args, nullptr, 0), 0);
    EXPECT_EQ(required_bytes(contract, PTO_PIPELINE_GM_HEAP), fake_.gm_heap.size());
    EXPECT_EQ(required_bytes(contract, PTO_PIPELINE_GM_SM), fake_.gm_sm.size());
    EXPECT_EQ(required_bytes(contract, PTO_PIPELINE_RUNTIME_IMAGE), fake_.runtime_arena.size());
    ASSERT_EQ(finish_run(runtime, 0), 0);
}

TEST_F(TrbRuntimeTempBufferTest, LargestRingCountsOnlyReserveLayout) {
    auto config = small_kernel_config();
    PipelineContract small{};
    ASSERT_EQ(build_kernel_pipeline_contract_impl(&config, &small), 0);
    config.runtime_env.ring_task_window[0] = uint64_t{1} << 30;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; ++r) {
        config.runtime_env.ring_dep_pool[r] = INT32_MAX;
    }
    PipelineContract large{};
    ASSERT_EQ(build_kernel_pipeline_contract_impl(&config, &large), 0);
    EXPECT_TRUE(is_valid_tmr_kernel_pipeline_contract(&large));
    EXPECT_EQ(required_bytes(large, PTO_PIPELINE_GM_HEAP), required_bytes(small, PTO_PIPELINE_GM_HEAP));
    EXPECT_GT(required_bytes(large, PTO_PIPELINE_GM_SM), required_bytes(small, PTO_PIPELINE_GM_SM));
    EXPECT_GT(required_bytes(large, PTO_PIPELINE_RUNTIME_IMAGE), required_bytes(small, PTO_PIPELINE_RUNTIME_IMAGE));
    EXPECT_EQ(fake_.setup_static_arena_count, 0);
    EXPECT_EQ(fake_.device_malloc_count, 0);
    EXPECT_EQ(fake_.copy_to_count, 0);
}

// Sizing reads only its own config and writes only its own output: no static or
// thread-local state backs it, so concurrent calls cannot interfere.
TEST(KernelPipelineBuilder, SizingKeepsNoSharedState) {
    constexpr size_t count = 4;
    std::array<CallConfig, count> configs;
    std::array<PipelineContract, count> expected{};
    std::array<std::thread, count> threads;
    for (size_t i = 0; i < count; ++i) {
        configs[i] = small_kernel_config();
        configs[i].runtime_env.ring_task_window[0] = uint64_t{4} << i;
        configs[i].runtime_env.ring_heap[0] = 1024 * (i + 1);
        configs[i].runtime_env.ring_dep_pool[0] = 4 + i;
        ASSERT_EQ(build_kernel_pipeline_contract_impl(&configs[i], &expected[i]), 0);
    }
    for (size_t i = 0; i < count; ++i) {
        threads[i] = std::thread([&, i] {
            const CallConfig config = configs[i];
            for (int iteration = 0; iteration < 32; ++iteration) {
                PipelineContract actual{};
                EXPECT_EQ(build_kernel_pipeline_contract_impl(&config, &actual), 0);
                expect_same_contract(actual, expected[i]);
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
}

TEST_F(TrbRuntimeTempBufferTest, RejectsUnsupportedTransferBeforeReadingEarlierArguments) {
    Runtime runtime = make_runtime();
    const uint32_t shape[] = {16};
    // Any copy of the first input is a fault, not a weak copy-count assertion.
    const ChipTensor first = make_tensor_external(reinterpret_cast<void *>(1), shape, 1, DataType::UINT8);
    for (auto transfer : {TensorTransfer::NONE, TensorTransfer::D2H, static_cast<TensorTransfer>(255)}) {
        SCOPED_TRACE(static_cast<int>(transfer));
        ChipTensor invalid = first;
        invalid.transfer = transfer;
        ChipStorageTaskArgs args;
        args.add_tensor(first);
        args.add_tensor(invalid);
        const ArgDirection sig[] = {ArgDirection::IN, ArgDirection::IN};
        EXPECT_NE(bind_runtime(runtime, api_, args, sig, 2), 0);
        EXPECT_EQ(fake_.copy_to_count, 0);
        EXPECT_EQ(fake_.device_malloc_count, 0);
    }
}
