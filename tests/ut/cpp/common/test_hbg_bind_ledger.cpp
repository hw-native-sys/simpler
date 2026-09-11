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
// host_build_graph bind: the tensor-lease ledger belongs to one run.
//
// validate_runtime_impl is the only consumer of the ledger and the only other
// place that clears it, so a run whose validate never executes — a finalize
// whose attach_current_thread failed — leaves its leases behind. Since the
// staging slices come from a buffer the next bind re-slices from offset zero,
// an inherited lease names a byte range that now belongs to a different tensor,
// and validate copies those bytes back to the earlier run's host pointer.
//
// bind therefore clears the ledger on entry. These tests drive the real
// bind_callable_to_runtime_impl against a fake HostApi; no orchestration .so is
// needed because bind takes the resolved host-orch entry points as a parameter.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "arg_direction.h"
#include "common/host_api.h"
#include "runtime.h"
#include "runtime_core.h"
#include "task_args.h"
#include "worker/runtime_c_api.h"

extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc);

namespace {

constexpr size_t kAlign = 1024;

// Mirrors the file-local HostOrchEntryPoints in runtime_maker.cpp: bind reads
// exactly these two function pointers out of host_orch_func_ptr.
using TestOrchEntryFunc = void (*)(const ChipTaskArgs &);
using TestOrchBindFunc = void (*)(RuntimeContext *);
struct TestHostOrchEntryPoints {
    TestOrchEntryFunc entry{nullptr};
    TestOrchBindFunc bind{nullptr};
};

// An orchestration that submits nothing: this suite is about the ledger, and a
// graph with no tasks exercises bind end to end all the same.
void empty_orch_entry(const ChipTaskArgs & /*args*/) {}
void empty_orch_bind(RuntimeContext * /*rt*/) {}

struct FakeHostApi {
    void *retained_addr = nullptr;
    size_t retained_size = 0;
    std::unordered_set<void *> live;
    std::vector<uint8_t> gm_heap;
    std::vector<uint8_t> runtime_arena;
    std::vector<uint8_t> sm_mirror;
    std::vector<uint8_t> definition_device;
    std::vector<uint8_t> definition_staging;

    ~FakeHostApi() { release_all(); }
    void release_all() {
        for (void *p : live)
            std::free(p);
        live.clear();
        retained_addr = nullptr;
        retained_size = 0;
    }
};

FakeHostApi *g_fake = nullptr;

void *fake_device_malloc(void *, size_t size) {
    // Plain malloc, like the sim backend: the bump is what aligns its base.
    void *p = std::malloc(std::max<size_t>(size, 1));
    if (p != nullptr) g_fake->live.insert(p);
    return p;
}
void fake_device_free(void *, void *p) {
    if (p == nullptr) return;
    EXPECT_EQ(g_fake->live.count(p), 1u);
    g_fake->live.erase(p);
    std::free(p);
}
int fake_copy_to_device(void *, void *dev, const void *host, size_t n) {
    std::memcpy(dev, host, n);
    return 0;
}
int fake_copy_from_device(void *, void *host, const void *dev, size_t n) {
    std::memcpy(host, dev, n);
    return 0;
}
void fake_get_retained(void *, uint32_t, void **addr, size_t *size) {
    if (addr != nullptr) *addr = g_fake->retained_addr;
    if (size != nullptr) *size = g_fake->retained_size;
}
void fake_set_retained(void *, uint32_t, void *addr, size_t size) {
    g_fake->retained_addr = addr;
    g_fake->retained_size = size;
}
int fake_setup_static_arena(void *, uint32_t, size_t heap, size_t, size_t arena) {
    g_fake->gm_heap.assign(std::max<size_t>(heap, kAlign) + kAlign, 0);
    g_fake->runtime_arena.assign(std::max<size_t>(arena, kAlign) + kAlign, 0);
    return 0;
}
void *aligned_in(std::vector<uint8_t> &v) {
    auto raw = reinterpret_cast<uintptr_t>(v.data());
    return reinterpret_cast<void *>((raw + kAlign - 1) & ~static_cast<uintptr_t>(kAlign - 1));
}
void *fake_acquire_gm_heap(void *, uint32_t) { return aligned_in(g_fake->gm_heap); }
void *fake_acquire_runtime_arena(void *, uint32_t) { return aligned_in(g_fake->runtime_arena); }
int fake_acquire_sm_mirror(void *, uint32_t, size_t bytes, size_t alignment, void **out) {
    g_fake->sm_mirror.assign(bytes + alignment, 0);
    auto raw = reinterpret_cast<uintptr_t>(g_fake->sm_mirror.data());
    *out = reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    return 0;
}
int fake_acquire_graph_definition_block(void *, uint32_t, size_t bytes, size_t alignment, void **dev, void **stage) {
    g_fake->definition_device.assign(bytes + alignment, 0);
    g_fake->definition_staging.assign(bytes + alignment, 0);
    auto align = [alignment](std::vector<uint8_t> &v) {
        auto raw = reinterpret_cast<uintptr_t>(v.data());
        return reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    };
    *dev = align(g_fake->definition_device);
    *stage = align(g_fake->definition_staging);
    return 0;
}
void fake_get_graph_definition_staging(void *, uint32_t, void **addr, size_t *size) {
    if (g_fake->definition_staging.empty()) {
        if (addr != nullptr) *addr = nullptr;
        if (size != nullptr) *size = 0;
        return;
    }
    if (addr != nullptr) *addr = g_fake->definition_staging.data();
    if (size != nullptr) *size = g_fake->definition_staging.size();
}

const HostApiOps &fake_ops() {
    static const HostApiOps ops = []() {
        HostApiOps r{};
        r.device_malloc = fake_device_malloc;
        r.device_free = fake_device_free;
        r.copy_to_device = fake_copy_to_device;
        r.copy_from_device = fake_copy_from_device;
        r.get_retained_temp_buffer = fake_get_retained;
        r.set_retained_temp_buffer = fake_set_retained;
        r.setup_static_arena = fake_setup_static_arena;
        r.acquire_pooled_gm_heap = fake_acquire_gm_heap;
        r.acquire_pooled_runtime_arena = fake_acquire_runtime_arena;
        r.acquire_sm_mirror = fake_acquire_sm_mirror;
        r.acquire_graph_definition_block = fake_acquire_graph_definition_block;
        r.get_graph_definition_staging = fake_get_graph_definition_staging;
        return r;
    }();
    return ops;
}

ChipTensor host_tensor(std::vector<uint8_t> &storage) {
    ChipTensor t;
    uint32_t shape[1] = {static_cast<uint32_t>(storage.size())};
    t.init_external(storage.data(), storage.size(), shape, 1, DataType::UINT8, AddressSpace::HOST);
    return t;
}

class HbgBindLedgerTest : public ::testing::Test {
protected:
    void SetUp() override { g_fake = &fake_; }
    void TearDown() override {
        fake_.release_all();
        g_fake = nullptr;
    }

    // A Runtime as init leaves it: bind derives block_dim as
    // worker_count / PLATFORM_CORES_PER_BLOCKDIM and fails before it reaches the
    // staging ledger if that is zero.
    static void init_runtime(Runtime &rt) {
        rt.worker_count = PLATFORM_CORES_PER_BLOCKDIM;
        for (int i = 0; i < rt.worker_count; ++i) {
            rt.workers[i].core_type = CoreType::AIV;
            rt.workers[i].physical_core_id = static_cast<uint32_t>(i);
        }
    }

    int bind(Runtime &rt, const ChipStorageTaskArgs &args, const ArgDirection *sig, int n) {
        // hbg reads only entry 0, through resolve_graph_task_capacity; the four
        // entries match RuntimeEnv's per-ring array width.
        uint64_t win[4] = {8, 0, 0, 0};
        return bind_callable_to_runtime_impl(&rt, &api_, &args, &eps_, sig, n, win, nullptr, nullptr);
    }

    FakeHostApi fake_;
    HostApi api_{nullptr, 0, 0, &fake_ops()};
    TestHostOrchEntryPoints eps_{empty_orch_entry, empty_orch_bind};
};

}  // namespace

// The regression barrier: a bind whose validate never ran must not leak its
// leases into the next bind's ledger.
TEST_F(HbgBindLedgerTest, SecondBindDoesNotInheritTheFirstBindsLeases) {
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> first(64, 0x11);
    ChipStorageTaskArgs args_a;
    args_a.add_tensor(host_tensor(first));
    ArgDirection sig[1] = {ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args_a, sig, 1), 0);
    ASSERT_EQ(runtime.tensor_leases_.size(), 1u);

    // No validate_runtime_impl here: this is the finalize-attach-failure shape.
    std::vector<uint8_t> second(64, 0x22);
    ChipStorageTaskArgs args_b;
    args_b.add_tensor(host_tensor(second));

    ASSERT_EQ(bind(runtime, args_b, sig, 1), 0);
    EXPECT_EQ(runtime.tensor_leases_.size(), 1u) << "the first bind's lease survived into the second bind";
    EXPECT_EQ(runtime.tensor_leases_[0].host_ptr, second.data());
}

// What the stale lease would actually do: validate copies every recorded slice
// back, so an inherited lease writes this run's bytes into the previous run's
// caller buffer.
TEST_F(HbgBindLedgerTest, ValidateAfterARebindLeavesTheEarlierRunsBufferAlone) {
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> first(64, 0x11);
    const std::vector<uint8_t> first_before = first;
    ChipStorageTaskArgs args_a;
    args_a.add_tensor(host_tensor(first));
    ArgDirection sig[1] = {ArgDirection::INOUT};
    ASSERT_EQ(bind(runtime, args_a, sig, 1), 0);

    std::vector<uint8_t> second(64, 0x22);
    ChipStorageTaskArgs args_b;
    args_b.add_tensor(host_tensor(second));
    ASSERT_EQ(bind(runtime, args_b, sig, 1), 0);

    // Stand in for the kernel writing the second run's output. Written through
    // every recorded slice, so the assertion below still has something to catch
    // when a stale lease is present rather than aborting on the count.
    for (const TensorLease &lease : runtime.tensor_leases_) {
        std::memset(lease.dev_ptr, 0x5a, 64);
    }

    ASSERT_EQ(validate_runtime_impl(&runtime, &api_, 0), 0);
    EXPECT_EQ(second, std::vector<uint8_t>(64, 0x5a));
    EXPECT_EQ(first, first_before) << "the earlier run's host buffer was overwritten by this run's bytes";
}
