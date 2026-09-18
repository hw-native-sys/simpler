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
// copy_back_run_outputs_impl is the only consumer of the ledger, and
// release_run_bindings_impl the only other place that clears it, so a run whose
// finalize never runs either — one whose attach_current_thread failed — leaves
// its leases behind. Since the staging slices come from a buffer the next bind
// re-slices from offset zero, an inherited lease names a byte range that now
// belongs to a different tensor, and the copy-back sends those bytes to the
// earlier run's host pointer.
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
#include "host/host_phase_records.h"
#include "worker/runtime_c_api.h"

extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);
extern "C" int copy_in_run_inputs_impl(const Runtime *runtime, const HostApi *api);
extern "C" int copy_back_run_outputs_impl(const Runtime *runtime, const HostApi *api, int execution_rc, int launched);
extern "C" int release_run_bindings_impl(Runtime *runtime, const HostApi *api);
extern "C" int publish_run_image_impl(Runtime *runtime, const HostApi *api);

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
    // Retained across binds, like the runner's block: what the bind assembles has
    // to still be readable when the publication reads it.
    std::vector<uint8_t> image_staging;
    // A real record pool, so what the prepare path records is observable. Armed
    // unconditionally: whether the runner would offer one is the platform's
    // decision and not what these tests are about.
    std::vector<HostPhaseRecordBuffer> phase_buffers;
    HostPhaseRecordPool phase_pool{};
    uint64_t phase_finish_calls = 0;

    HostPhaseRecordPool *arm_phase_pool() {
        phase_buffers.assign(PLATFORM_HOST_PHASE_BUFFERS, HostPhaseRecordBuffer{});
        phase_pool.buffers = phase_buffers.data();
        phase_pool.buffer_count = static_cast<uint32_t>(phase_buffers.size());
        phase_pool.next_buffer.store(0);
        phase_pool.generation.fetch_add(1);
        phase_pool.dropped.store(0);
        return &phase_pool;
    }

    size_t phase_records_of(HostPhaseKind kind, uint64_t *payload_out = nullptr) const {
        size_t found = 0;
        for (const HostPhaseRecordBuffer &buffer : phase_buffers) {
            const uint32_t count = buffer.count;
            for (uint32_t i = 0; i < count && i < PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER; ++i) {
                if (buffer.records[i].kind != static_cast<uint32_t>(kind)) continue;
                if (payload_out != nullptr) *payload_out = buffer.records[i].payload;
                ++found;
            }
        }
        return found;
    }
    size_t phase_records_total() const {
        size_t found = 0;
        for (const HostPhaseRecordBuffer &buffer : phase_buffers) {
            found += buffer.count;
        }
        return found;
    }
    struct H2D {
        void *dst;
        const void *src;
        size_t bytes;
    };
    std::vector<H2D> copies;

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
    g_fake->copies.push_back({dev, host, n});
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
int fake_acquire_run_image_staging(void *, uint32_t, size_t bytes, size_t alignment, void **out) {
    // Grow-only and never re-seated once large enough, so an address handed to one
    // bind stays valid — the property the publication depends on.
    if (g_fake->image_staging.size() < bytes + alignment) g_fake->image_staging.assign(bytes + alignment, 0);
    auto raw = reinterpret_cast<uintptr_t>(g_fake->image_staging.data());
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
        r.acquire_run_image_staging = fake_acquire_run_image_staging;
        r.host_phase_pool_arm = [](void *, uint32_t, int) -> void * {
            return g_fake->arm_phase_pool();
        };
        r.host_phase_pool_finish = [](void *, uint32_t, uint64_t, uint64_t) {
            ++g_fake->phase_finish_calls;
        };
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
        rt.dev.worker_count = PLATFORM_CORES_PER_BLOCKDIM;
        for (int i = 0; i < rt.dev.worker_count; ++i) {
            rt.dev.workers[i].core_type = CoreType::AIV;
            rt.dev.workers[i].physical_core_id = static_cast<uint32_t>(i);
        }
    }

    int bind(Runtime &rt, const ChipStorageTaskArgs &args, const ArgDirection *sig, int n) {
        // hbg reads only entry 0, through resolve_graph_task_capacity; the four
        // entries match RuntimeEnv's per-ring array width.
        uint64_t win[4] = {8, 0, 0, 0};
        return bind_callable_to_runtime_impl(&rt, &api_, &args, &eps_, sig, n, win, nullptr, nullptr);
    }

    // The two halves the c_api calls back to back for a run it is finalizing.
    int finish_run(Runtime &rt, int execution_rc, int launched = 1) {
        const int rc = copy_back_run_outputs_impl(&rt, &api_, execution_rc, launched);
        const int release_rc = release_run_bindings_impl(&rt, &api_);
        return rc != 0 ? rc : release_rc;
    }

    FakeHostApi fake_;
    HostApi api_{nullptr, 0, 0, 0, &fake_ops()};
    TestHostOrchEntryPoints eps_{empty_orch_entry, empty_orch_bind};
};

}  // namespace

// An empty caller tensor is accepted and passed through with a null address.
// hbg used to reject it: the staging loop handed it to HostTensorAccessor::add,
// which refuses an empty region, and the bind failed with "no host view". TRB
// has always passed it through, and hbg now matches.
TEST_F(HbgBindLedgerTest, AnEmptyTensorIsPassedThroughAndTakesNoSlice) {
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> empty;
    std::vector<uint8_t> real(64, 0x11);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(empty));
    args.add_tensor(host_tensor(real));
    ArgDirection sig[2] = {ArgDirection::INOUT, ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args, sig, 2), 0);
    // Only the non-empty tensor takes a slice and a lease.
    ASSERT_EQ(runtime.tensor_leases().size(), 1u);
    EXPECT_EQ(runtime.tensor_leases()[0].host_ptr, real.data());
    EXPECT_EQ(runtime.tensor_leases()[0].size, 64u);
}

// Preparation and publication are two steps. The bind assembles the run's device
// image into staging that outlives it and records where the bytes go; the write
// happens afterwards, on the caller's schedule. Asserted by which bytes moved
// when, because that is what a later ordered or captured write depends on — a
// bind that still performed the copy itself would fail the first assertion.
TEST_F(HbgBindLedgerTest, PreparingTheRunImageDoesNotPublishIt) {
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> payload(64, 0x11);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(payload));
    ArgDirection sig[1] = {ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args, sig, 1), 0);

    const auto &publication = runtime.pending_publication();
    ASSERT_NE(publication.bytes, 0u) << "the bind recorded no image to publish";
    ASSERT_NE(publication.device_target, nullptr);
    ASSERT_NE(publication.source, nullptr);

    void *const target = publication.device_target;
    const uint64_t bytes = publication.bytes;
    const auto copies_to_target = [this, target]() {
        size_t n = 0;
        for (const auto &copy : fake_.copies) {
            if (copy.dst == target) ++n;
        }
        return n;
    };
    EXPECT_EQ(copies_to_target(), 0u) << "the bind published the image itself";

    // The staging the slot owns survives the bind, so the source is still
    // readable here — which is what lets the write happen on the caller's
    // schedule. Stamp it and watch the stamp arrive.
    auto *source = const_cast<uint8_t *>(static_cast<const uint8_t *>(publication.source));
    source[0] = 0xa5;
    source[bytes - 1] = 0x5a;

    ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);
    ASSERT_EQ(copies_to_target(), 1u) << "the publication did not perform exactly one write";
    for (const auto &copy : fake_.copies) {
        if (copy.dst != target) continue;
        EXPECT_EQ(copy.bytes, bytes);
    }
    const auto *published = static_cast<const uint8_t *>(target);
    EXPECT_EQ(published[0], 0xa5);
    EXPECT_EQ(published[bytes - 1], 0x5a);

    // Consumed: a second publication is an error, not a second write.
    EXPECT_EQ(runtime.pending_publication().bytes, 0u);
    EXPECT_NE(publish_run_image_impl(&runtime, &api_), 0);
    EXPECT_EQ(copies_to_target(), 1u);

    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
}

// One trace spans preparation and publication. The image's H2D segment is the
// publication's to record, and the trace the bind armed has to still be open when
// it does — a trace that ended at the bind's return would drop the segment
// silently, leaving the timeline with no host-to-device handover for the run's
// largest transfer. Asserted against a real record pool: the kind has to be
// present with the published byte count, and other segments have to be there too,
// so a recorder that was never armed cannot pass this by recording nothing.
TEST_F(HbgBindLedgerTest, ThePublicationRecordsTheArenaH2dSegment) {
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> payload(64, 0x33);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(payload));
    ArgDirection sig[1] = {ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    const uint64_t published_bytes = runtime.pending_publication().bytes;
    ASSERT_NE(published_bytes, 0u);

    // The bind recorded its own segments, so the pool is live and armed.
    EXPECT_GT(fake_.phase_records_total(), 0u) << "no phase records at all: the recorder was never armed";
    EXPECT_EQ(fake_.phase_records_of(HostPhaseKind::BindArenaH2d), 0u) << "the bind recorded a copy it did not perform";

    ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);

    uint64_t recorded_bytes = 0;
    EXPECT_EQ(fake_.phase_records_of(HostPhaseKind::BindArenaH2d, &recorded_bytes), 1u)
        << "the publication's host-to-device segment was dropped";
    EXPECT_EQ(recorded_bytes, published_bytes) << "the segment's byte count is not the one published";
    // The trace closes once, over preparation and publication together.
    EXPECT_EQ(fake_.phase_finish_calls, 1u);

    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    EXPECT_EQ(fake_.phase_finish_calls, 1u) << "releasing a published run closed a second trace";
}

// A preparation nobody publishes still closes its trace exactly once, at the
// point the run's bindings are released — so an abandoned bind reports its
// breakdown rather than leaving the trace open for the next one to inherit.
TEST_F(HbgBindLedgerTest, AnAbandonedPreparationClosesItsTraceOnRelease) {
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> payload(64, 0x44);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(payload));
    ArgDirection sig[1] = {ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    ASSERT_NE(runtime.pending_publication().bytes, 0u);
    EXPECT_EQ(fake_.phase_finish_calls, 0u) << "the bind closed a trace the publication still owns";

    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    EXPECT_EQ(fake_.phase_finish_calls, 1u);
    EXPECT_EQ(runtime.pending_publication().bytes, 0u) << "the abandoned record outlived the run";
}

// A bind whose publication never runs leaves the execution image unwritten: its
// device target holds none of these bytes, and the record and its staging are
// still there, so the caller may publish later or abandon the run. Other device
// writes the bind performs — an INOUT tensor's input copy, for one — are not in
// question here.
TEST_F(HbgBindLedgerTest, AnUnpublishedBindLeavesTheImageTargetUnwritten) {
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> payload(64, 0x22);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(payload));
    ArgDirection sig[1] = {ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    const auto &publication = runtime.pending_publication();
    ASSERT_NE(publication.bytes, 0u);
    for (const auto &copy : fake_.copies) {
        EXPECT_NE(copy.dst, publication.device_target) << "an unpublished bind wrote to the image target";
    }
    // Still publishable, which is what makes abandoning a choice rather than a
    // loss.
    EXPECT_EQ(publish_run_image_impl(&runtime, &api_), 0);
    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
}

// What the device image is allowed to contain. The boundary is already a
// grouped host-only tail; naming the other side makes it checkable by content:
// the uploaded bytes are the descriptor and nothing behind it, so a value only
// the host reads cannot ride along. Found by searching for it, not by comparing
// a size — a size assertion passes even when a member is on the wrong side.
TEST_F(HbgBindLedgerTest, TheDeviceImageCarriesNoHostOnlyBytes) {
    Runtime runtime;
    init_runtime(runtime);

    // Three host-only members, each given a recognisable value: the callable
    // stamp, an orchestration scalar, and a lease.
    constexpr int32_t kStamp = 0x5ea15ea1;
    runtime.set_active_callable_id(kStamp);
    std::vector<uint8_t> payload(64, 0x11);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(payload));
    constexpr uint64_t kScalar = 0xfeedfacecafebeedULL;
    args.add_scalar(kScalar);
    runtime.set_orch_args(args);
    ArgDirection sig[1] = {ArgDirection::INOUT};
    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    ASSERT_FALSE(runtime.tensor_leases().empty()) << "the bind must have recorded the host-only ledger";

    const size_t image_bytes = runtime_device_copy_size(runtime);
    ASSERT_EQ(image_bytes, sizeof(DeviceRuntimeLaunchDesc));
    ASSERT_LT(image_bytes, sizeof(Runtime)) << "the whole object is still crossing to the device";

    // The copy the platform performs: `image_bytes` from offset 0.
    std::vector<uint8_t> image(image_bytes);
    std::memcpy(image.data(), &runtime, image_bytes);

    auto contains = [&image](const void *needle, size_t bytes) {
        const auto *first = static_cast<const uint8_t *>(needle);
        return std::search(image.begin(), image.end(), first, first + bytes) != image.end();
    };
    EXPECT_FALSE(contains(&kStamp, sizeof(kStamp))) << "the callable stamp reached the device image";
    EXPECT_FALSE(contains(&kScalar, sizeof(kScalar))) << "an orchestration scalar reached the device image";
    const void *const lease_dev_ptr = runtime.tensor_leases()[0].dev_ptr;
    EXPECT_FALSE(contains(&lease_dev_ptr, sizeof(lease_dev_ptr))) << "the tensor ledger reached the device image";

    // The device-read fields are in it, so the exclusions above are not vacuous.
    // Read through an aligned object rather than a cast over the byte buffer:
    // std::vector<uint8_t> carries no 64-byte guarantee and the descriptor is
    // alignas(64), so the cast would be a misaligned access.
    DeviceRuntimeLaunchDesc uploaded;
    std::memcpy(&uploaded, image.data(), sizeof(uploaded));
    EXPECT_EQ(uploaded.worker_count, runtime.get_worker_count());
    EXPECT_EQ(uploaded.sm_image_bytes, runtime.dev.sm_image_bytes);
    EXPECT_EQ(uploaded.gm_sm_ptr_, runtime.get_gm_sm_ptr());
}

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
    ASSERT_EQ(runtime.tensor_leases().size(), 1u);

    // No copy-back here: this is the finalize-attach-failure shape.
    std::vector<uint8_t> second(64, 0x22);
    ChipStorageTaskArgs args_b;
    args_b.add_tensor(host_tensor(second));

    ASSERT_EQ(bind(runtime, args_b, sig, 1), 0);
    EXPECT_EQ(runtime.tensor_leases().size(), 1u) << "the first bind's lease survived into the second bind";
    EXPECT_EQ(runtime.tensor_leases()[0].host_ptr, second.data());
}

// What the stale lease would actually do: the copy-back walks every recorded
// slice, so an inherited lease writes this run's bytes into the previous run's
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
    for (const TensorLease &lease : runtime.tensor_leases()) {
        std::memset(lease.dev_ptr, 0x5a, 64);
    }

    ASSERT_EQ(finish_run(runtime, 0), 0);
    EXPECT_EQ(second, std::vector<uint8_t>(64, 0x5a));
    EXPECT_EQ(first, first_before) << "the earlier run's host buffer was overwritten by this run's bytes";
}
