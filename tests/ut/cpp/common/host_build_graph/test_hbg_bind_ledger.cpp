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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "arg_direction.h"
#include "callable.h"
#include "common/host_api.h"
#include "host_build_graph/graph_host_state.h"
#include "host/raii_scope_guard.h"
#include "host_build_graph/runtime_status.h"
#include "runtime.h"
#include "runtime_core.h"
#include "host_build_graph/orchestrator.h"
#include "task_args.h"
#include "host/host_phase_records.h"
#include "host/platform_compile_info.h"
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

// The wait callback releases a real recorder job only while its borrowed build
// state is still alive. The test's boundary outlives bind even on the negative
// path, so a missing wait is an assertion failure rather than a use-after-free.
struct RecordingLifetime {
    RuntimeOps ops{};
    GraphTaskArgs boundary;
    std::mutex mutex;
    std::condition_variable cv;
    bool release_job{false};
    bool job_finished{false};
    bool throw_from_entry{false};
    int wait_calls{0};
};

RecordingLifetime *g_recording = nullptr;
RuntimeContext *g_recording_runtime = nullptr;

void recording_wait(RuntimeContext *rt) {
    EXPECT_NE(rt->orchestrator, nullptr);
    EXPECT_NE(rt->orchestrator->graph_host_state, nullptr);
    EXPECT_NE(rt->tensor_access, nullptr);
    ++g_recording->wait_calls;
    {
        std::lock_guard<std::mutex> lock(g_recording->mutex);
        g_recording->release_job = true;
    }
    g_recording->cv.notify_all();
    graph_record_wait_impl(rt);
    EXPECT_TRUE(g_recording->job_finished);
}

void recording_bind(RuntimeContext *rt) {
    g_recording_runtime = rt;
    g_recording->ops = *rt->ops;
    g_recording->ops.graph_record_wait = recording_wait;
    rt->ops = &g_recording->ops;
}

void recording_entry(const ChipTaskArgs &) {
    std::function<void(const GraphTaskArgs &)> job = [](const GraphTaskArgs &) {
        std::unique_lock<std::mutex> lock(g_recording->mutex);
        g_recording->cv.wait(lock, [] {
            return g_recording->release_job;
        });
        g_recording->job_finished = true;
    };
    if (!graph_record_start_impl(g_recording_runtime, g_recording->boundary, &job)) {
        throw std::runtime_error("could not start recorder");
    }
    if (g_recording->throw_from_entry) throw std::runtime_error("host orchestration failed");
}

enum class InputProducer { None, Allocated, Overlapping, Disjoint };

struct HostAccessProbe {
    RuntimeContext *runtime{nullptr};
    InputProducer producer{InputProducer::None};
    bool write{false};
    std::vector<uint64_t> reads;
    int32_t error{0};
};

HostAccessProbe *g_access = nullptr;

void access_bind(RuntimeContext *rt) { g_access->runtime = rt; }

void access_entry(const ChipTaskArgs &args) {
    RuntimeContext *rt = g_access->runtime;
    for (int i = 0; i < args.tensor_count(); ++i) {
        simpler::hbg::Tensor tensor = args.tensor(i).ref();
        if (g_access->producer != InputProducer::None) {
            CoreTaskArgs task_args;
            MixedKernels kernels{};
            kernels.aiv0_kernel_id = 0;
            if (g_access->producer == InputProducer::Allocated) {
                const uint32_t shape[] = {1};
                TensorCreateInfo output(shape, 1, DataType::UINT8);
                task_args.add_output(output);
                const TaskOutputTensors result = rt->orchestrator->submit_task(kernels, task_args);
                ASSERT_EQ(result.size(), 1u);
                tensor = result.get_ref(0);
                ASSERT_TRUE(tensor.owner_task_id.is_valid());
            } else {
                const simpler::hbg::Tensor written = tensor.slice(0, 0, 1);
                task_args.add_output(written);
                const TaskOutputTensors result = rt->orchestrator->submit_task(kernels, task_args);
                ASSERT_TRUE(result.task_id().is_valid());
                // The input alias has no owner id; rejection must come from the
                // TensorMap overlap rather than the runtime-allocation branch.
                ASSERT_FALSE(tensor.owner_task_id.is_valid());
                tensor = tensor.slice(0, g_access->producer == InputProducer::Disjoint ? 1 : 0, 2);
            }
        }
        const uint32_t index[] = {0};
        if (g_access->write) {
            set_tensor_data(rt, tensor, 1, index, 0x5a);
        } else {
            g_access->reads.push_back(get_tensor_data(rt, tensor, 1, index));
        }
        g_access->error = rt->orchestrator->fatal_code.load(std::memory_order_acquire);
    }
}

struct FakeHostApi {
    void *retained_addr = nullptr;
    size_t retained_size = 0;
    std::unordered_set<void *> live;
    std::vector<uint8_t> gm_heap;
    std::vector<uint8_t> runtime_arena;
    std::vector<uint8_t> sm_mirror;
    std::vector<uint8_t> definition_device;
    std::vector<uint8_t> definition_staging;
    size_t definition_bytes{0};
    size_t definition_offset{0};
    int copy_count{0};
    int fail_copy_on{0};
    int orchestration_count{0};
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

RuntimeContext *g_orch_runtime = nullptr;
void capture_orch_bind(RuntimeContext *rt) { g_orch_runtime = rt; }
void recording_orch_entry(const ChipTaskArgs &) {
    ++g_fake->orchestration_count;
    auto &orch = *g_orch_runtime->orchestrator;
    uint32_t data[16]{};
    uint32_t shape[] = {16};
    GraphTaskArgs boundary;
    auto input = simpler::hbg::make_tensor_external(data, shape, 1);
    boundary.add_input(input);
    auto graph = orch.graph_begin(0x521, boundary, 0x523);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary));
    CoreTaskArgs task;
    task.add_input(graph.params->tensor(0).ref());
    TensorCreateInfo output(shape, 1, DataType::UINT32);
    task.add_output(output);
    ASSERT_TRUE(orch.submit_dummy_task(task).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();
}

void ordinary_orch_entry(const ChipTaskArgs &) {
    CoreTaskArgs args;
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    ASSERT_TRUE(g_orch_runtime->orchestrator->submit_task(kernels, args).task_id().is_valid());
}

void mixed_orch_entry(const ChipTaskArgs &) {
    CoreTaskArgs args;
    MixedKernels kernels{};
    kernels.aic_kernel_id = 0;
    kernels.aiv0_kernel_id = 0;
    ASSERT_TRUE(g_orch_runtime->orchestrator->submit_task(kernels, args).task_id().is_valid());
}

void host_get_set_orch_entry(const ChipTaskArgs &args) {
    uint32_t index[] = {0};
    EXPECT_EQ(get_tensor_data(g_orch_runtime, args.tensor(0).ref(), 1, index), 0x37u);
    set_tensor_data(g_orch_runtime, args.tensor(0).ref(), 1, index, 0x52);
    EXPECT_EQ(get_tensor_data(g_orch_runtime, args.tensor(0).ref(), 1, index), 0x52u);
}

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
    ++g_fake->copy_count;
    if (g_fake->fail_copy_on == g_fake->copy_count) return -17;
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
    auto align = [alignment](std::vector<uint8_t> &v) {
        auto raw = reinterpret_cast<uintptr_t>(v.data());
        return reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    };
    if (bytes > g_fake->definition_bytes) {
        std::vector<uint8_t> staging(bytes + alignment, 0);
        void *new_base = align(staging);
        if (g_fake->definition_bytes != 0) {
            std::memcpy(
                new_base, g_fake->definition_staging.data() + g_fake->definition_offset, g_fake->definition_bytes
            );
        }
        g_fake->definition_offset = static_cast<uint8_t *>(new_base) - staging.data();
        g_fake->definition_staging = std::move(staging);
        g_fake->definition_device.assign(bytes + alignment, 0);
        g_fake->definition_bytes = bytes;
    }
    *dev = align(g_fake->definition_device);
    *stage = g_fake->definition_staging.data() + g_fake->definition_offset;
    return 0;
}
void fake_get_graph_definition_staging(void *, uint32_t, void **addr, size_t *size) {
    if (addr != nullptr) {
        *addr = g_fake->definition_bytes == 0 ? nullptr : g_fake->definition_staging.data() + g_fake->definition_offset;
    }
    if (size != nullptr) *size = g_fake->definition_bytes;
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

    // a5 keeps scheduler allocations in a runtime-address keyed owner table.
    // Release that ownership while the runtime and its fake bank still exist;
    // freeing the fake allocations alone leaves a stale owner for a later test.
    auto cleanup_runtime(Runtime &rt) {
        return RAIIScopeGuard([this, &rt, bank = g_fake]() {
            FakeHostApi *saved = g_fake;
            g_fake = bank;
            EXPECT_EQ(release_run_bindings_impl(&rt, &api_), 0);
            g_fake = saved;
        });
    }

    // `recording` is a stack object the pool's worker reads through a global, so
    // every exit from the test body — including an assertion or an exception the
    // body does not catch — has to release the job, drain the pool, and drop the
    // global before that object dies.
    auto enter_recording(RecordingLifetime &recording) {
        g_recording = &recording;
        eps_ = {recording_entry, recording_bind};
        return RAIIScopeGuard([&recording]() {
            {
                std::lock_guard<std::mutex> lock(recording.mutex);
                recording.release_job = true;
            }
            recording.cv.notify_all();
            graph_record_wait_impl(nullptr);
            g_recording_runtime = nullptr;
            g_recording = nullptr;
        });
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

class HbgHostAccessContractTest : public HbgBindLedgerTest {
protected:
    void SetUp() override {
        HbgBindLedgerTest::SetUp();
        g_access = &access_;
        eps_ = {access_entry, access_bind};
    }
    void TearDown() override {
        g_access = nullptr;
        HbgBindLedgerTest::TearDown();
    }
    HostAccessProbe access_;
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
    EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
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

    // Three boundaries, each with its own reason to exist. A steady-state run
    // stops before the handshake region, whose words the device writes; the
    // first publication onto an allocation adds that region and stops before
    // the gate tail, whose host storage is never initialized; the allocation
    // covers everything, because the device addresses the tail inside it.
    const size_t steady_bytes = runtime_device_copy_size(runtime);
    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    ASSERT_EQ(extent_bytes, sizeof(DeviceRuntimeLaunchDesc));
    ASSERT_EQ(steady_bytes, offsetof(DeviceRuntimeLaunchDesc, workers))
        << "a steady-state run must stop before the handshake region";
    ASSERT_EQ(image_bytes, offsetof(DeviceRuntimeLaunchDesc, teardown_gates))
        << "no publication may reach the device-initialized gate tail";
    ASSERT_LT(steady_bytes, image_bytes) << "the handshake region is still in the steady upload";
    ASSERT_LT(image_bytes, extent_bytes) << "the gate tail is still crossing to the device";
    ASSERT_LT(image_bytes, sizeof(Runtime)) << "the whole object is still crossing to the device";

    // Search the longest thing the host ever publishes — the initializing
    // prefix — so the exclusions below cover every byte that can cross, not
    // just the ones a steady-state run re-sends.
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
    // alignas(64), so the cast would be a misaligned access. The image holds only
    // the uploaded prefix, so reconstruct into a zeroed descriptor and copy just
    // that many bytes — reading `sizeof(uploaded)` from it would run off the end.
    DeviceRuntimeLaunchDesc uploaded{};
    std::memcpy(&uploaded, image.data(), image_bytes);
    EXPECT_EQ(uploaded.worker_count, runtime.get_worker_count());
    EXPECT_EQ(uploaded.sm_image_bytes, runtime.dev.sm_image_bytes);
    EXPECT_EQ(uploaded.gm_sm_ptr_, runtime.get_gm_sm_ptr());
    EXPECT_EQ(uploaded.prebuilt_runtime_offset_, runtime.get_prebuilt_runtime_offset())
        << "the last device-read field ahead of the handshake region must be inside the upload";
    EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
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
    ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);
    ASSERT_EQ(runtime.tensor_leases().size(), 1u);

    // No copy-back here: this is the finalize-attach-failure shape.
    std::vector<uint8_t> second(64, 0x22);
    ChipStorageTaskArgs args_b;
    args_b.add_tensor(host_tensor(second));

    ASSERT_EQ(bind(runtime, args_b, sig, 1), 0);
    EXPECT_EQ(runtime.tensor_leases().size(), 1u) << "the first bind's lease survived into the second bind";
    EXPECT_EQ(runtime.tensor_leases()[0].host_ptr, second.data());
    EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
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
    ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);

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

TEST_F(HbgBindLedgerTest, AllMetadataSourcesSurviveBindAndPublishInOrder) {
    Runtime runtime;
    init_runtime(runtime);
    eps_ = {recording_orch_entry, capture_orch_bind};
    ChipStorageTaskArgs args;
    ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
    EXPECT_EQ(fake_.copy_count, 0);
    EXPECT_EQ(fake_.orchestration_count, 1);
    const auto &pending = runtime.pending_publication();
    ASSERT_EQ(pending.prerequisites.size(), 1u);
    EXPECT_EQ(pending.prerequisites[0].phase, HostPhaseKind::BindGraphUpload);
    auto *definition_target = pending.prerequisites[0].device_target;
    auto *image_target = pending.device_target;
    const auto definition_bytes = pending.prerequisites[0].bytes;
    const auto image_bytes = pending.bytes;
    ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);
    ASSERT_EQ(fake_.copies.size(), 2u);
    EXPECT_EQ(fake_.copies[0].dst, definition_target);
    EXPECT_EQ(fake_.copies[1].dst, image_target);
    EXPECT_EQ(fake_.copies[0].bytes, definition_bytes);
    EXPECT_EQ(fake_.copies[1].bytes, image_bytes);
    uint64_t recorded_bytes = 0;
    EXPECT_EQ(fake_.phase_records_of(HostPhaseKind::BindGraphUpload, &recorded_bytes), 1u);
    EXPECT_EQ(recorded_bytes, definition_bytes);
    EXPECT_NE(publish_run_image_impl(&runtime, &api_), 0);
    EXPECT_EQ(fake_.copies.size(), 2u);
    EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
}

TEST_F(HbgBindLedgerTest, SchedulerModeChangesPublishOnlyThisRunsSources) {
    Runtime runtime;
    init_runtime(runtime);
    auto cleanup = cleanup_runtime(runtime);
    auto callable = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    reinterpret_cast<CoreCallable *>(callable.data())->set_resolved_addr(0x1000);
    runtime.replay_function_bin_addr(0, reinterpret_cast<uint64_t>(callable.data()));
    const bool a5 = std::strcmp(get_platform(), "a5sim") == 0;
    uint32_t resident_mode = 0;
    uint32_t graph_mode = 0;
    // A5 selects resident for ordinary tasks and legacy for GRAPH/MIX. A2/A3
    // uses AICPU scheduling throughout, with no resident scheduler region.
    for (TestOrchEntryFunc entry : {ordinary_orch_entry, recording_orch_entry, mixed_orch_entry, ordinary_orch_entry}) {
        SCOPED_TRACE(entry == recording_orch_entry ? "graph" : entry == mixed_orch_entry ? "mixed" : "ordinary");
        const bool resident = a5 && entry == ordinary_orch_entry;
        const bool definitions = entry == recording_orch_entry;
        fake_.copy_count = 0;
        fake_.copies.clear();
        eps_ = {entry, capture_orch_bind};
        ChipStorageTaskArgs args;
        ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
        EXPECT_EQ(fake_.copy_count, 0);
        const auto &pending = runtime.pending_publication();
        ASSERT_EQ(pending.prerequisites.size(), static_cast<size_t>(resident) + static_cast<size_t>(definitions));
        std::vector<void *> destinations;
        std::vector<std::vector<uint8_t>> snapshots;
        for (const auto &region : pending.prerequisites) {
            EXPECT_EQ(region.phase, definitions ? HostPhaseKind::BindGraphUpload : HostPhaseKind::Count);
            destinations.push_back(region.device_target);
            const auto *source = static_cast<const uint8_t *>(region.source);
            snapshots.emplace_back(source, source + region.bytes);
        }
        if (a5) {
            // The bootstrap inputs are host-authored and per run, not per worker:
            // workers[i].aicpu_ready and .task now carry the AICPU's handshake alone.
            // Worker i derives its own context from the base, which the AICore does
            // through scheduler_worker_context_address; that derivation is covered by
            // test_a5_hbg_scheduler_bootstrap and not re-asserted here.
            const uint32_t mode = runtime.dev.scheduler_bootstrap.runtime_mode;
            const uint64_t base = runtime.dev.scheduler_bootstrap.worker_context_base;
            EXPECT_NE(mode, 0u);
            if (resident) {
                if (resident_mode == 0) resident_mode = mode;
                EXPECT_EQ(mode, resident_mode);
                const auto &scheduler = pending.prerequisites.front();
                EXPECT_GE(base, reinterpret_cast<uint64_t>(scheduler.device_target));
                EXPECT_LT(base, reinterpret_cast<uint64_t>(scheduler.device_target) + scheduler.bytes);
            } else {
                EXPECT_EQ(base, 0u) << "legacy launch must not borrow a previous scheduler";
                EXPECT_NE(mode, resident_mode);
                if (definitions) {
                    if (graph_mode == 0) graph_mode = mode;
                    EXPECT_EQ(mode, graph_mode);
                } else {
                    EXPECT_NE(mode, graph_mode);
                }
            }
            for (int i = 0; i < runtime.get_worker_count(); ++i) {
                EXPECT_EQ(runtime.dev.workers[i].task, 0u) << "handshake state is the AICPU's to write";
                EXPECT_EQ(runtime.dev.workers[i].aicpu_ready, 0u);
            }
        }
        destinations.push_back(pending.device_target);
        const auto *source = static_cast<const uint8_t *>(pending.source);
        snapshots.emplace_back(source, source + pending.bytes);
        ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);
        ASSERT_EQ(fake_.copies.size(), snapshots.size());
        for (size_t i = 0; i < snapshots.size(); ++i) {
            EXPECT_EQ(fake_.copies[i].dst, destinations[i]);
            EXPECT_EQ(fake_.copies[i].bytes, snapshots[i].size());
            EXPECT_EQ(std::memcmp(destinations[i], snapshots[i].data(), snapshots[i].size()), 0);
        }
        EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
        EXPECT_TRUE(fake_.live.empty());
        EXPECT_TRUE(runtime.pending_publication().prerequisites.empty());
        if (a5) {
            // Release clears the host-authored selection, so the next upload cannot
            // inherit a mode whose allocation this release already freed.
            EXPECT_EQ(runtime.dev.scheduler_bootstrap.runtime_mode, 0u);
            EXPECT_EQ(runtime.dev.scheduler_bootstrap.worker_context_base, 0u);
        }
    }
}

TEST_F(HbgBindLedgerTest, SchedulerPublicationFailureAllowsFreshModeSelection) {
    Runtime runtime;
    init_runtime(runtime);
    auto cleanup = cleanup_runtime(runtime);
    auto callable = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    reinterpret_cast<CoreCallable *>(callable.data())->set_resolved_addr(0x1000);
    runtime.replay_function_bin_addr(0, reinterpret_cast<uint64_t>(callable.data()));
    const bool a5 = std::strcmp(get_platform(), "a5sim") == 0;
    for (TestOrchEntryFunc entry : {ordinary_orch_entry, recording_orch_entry, mixed_orch_entry}) {
        SCOPED_TRACE(entry == recording_orch_entry ? "graph" : entry == mixed_orch_entry ? "mixed" : "ordinary");
        const size_t regions = 1 + static_cast<size_t>(entry == recording_orch_entry) +
                               static_cast<size_t>(a5 && entry == ordinary_orch_entry);
        for (size_t failure = 1; failure <= regions; ++failure) {
            SCOPED_TRACE(failure);
            fake_.copy_count = 0;
            fake_.copies.clear();
            eps_ = {entry, capture_orch_bind};
            ChipStorageTaskArgs args;
            ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
            ASSERT_EQ(runtime.pending_publication().prerequisites.size() + 1, regions);
            const uint32_t failed_mode = runtime.dev.scheduler_bootstrap.runtime_mode;
            fake_.fail_copy_on = static_cast<int>(failure);
            EXPECT_NE(publish_run_image_impl(&runtime, &api_), 0);
            EXPECT_EQ(fake_.copy_count, failure);
            EXPECT_EQ(fake_.copies.size(), failure - 1);
            EXPECT_EQ(runtime.pending_publication().bytes, 0u);
            EXPECT_TRUE(runtime.pending_publication().prerequisites.empty());
            EXPECT_NE(publish_run_image_impl(&runtime, &api_), 0);
            EXPECT_EQ(fake_.copy_count, failure);
            ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
            EXPECT_TRUE(fake_.live.empty());
            if (a5) {
                // A failed publication must leave no selection behind for the next one.
                EXPECT_EQ(runtime.dev.scheduler_bootstrap.runtime_mode, 0u);
                EXPECT_EQ(runtime.dev.scheduler_bootstrap.worker_context_base, 0u);
            }
            fake_.fail_copy_on = 0;
            const bool definitions = entry == ordinary_orch_entry;
            const bool resident = a5 && !definitions;
            eps_ = {definitions ? recording_orch_entry : ordinary_orch_entry, capture_orch_bind};
            ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
            const auto &replacement = runtime.pending_publication();
            ASSERT_EQ(
                replacement.prerequisites.size(), static_cast<size_t>(resident) + static_cast<size_t>(definitions)
            );
            for (const auto &region : replacement.prerequisites) {
                EXPECT_EQ(region.phase, definitions ? HostPhaseKind::BindGraphUpload : HostPhaseKind::Count);
            }
            if (a5) {
                const uint32_t mode = runtime.dev.scheduler_bootstrap.runtime_mode;
                const uint64_t base = runtime.dev.scheduler_bootstrap.worker_context_base;
                EXPECT_NE(mode, 0u);
                EXPECT_NE(mode, failed_mode);
                if (resident) {
                    const auto &scheduler = replacement.prerequisites.front();
                    EXPECT_GE(base, reinterpret_cast<uint64_t>(scheduler.device_target));
                    EXPECT_LT(base, reinterpret_cast<uint64_t>(scheduler.device_target) + scheduler.bytes);
                } else {
                    EXPECT_EQ(base, 0u) << "legacy call must not reuse the failed scheduler";
                }
            }
            ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);
            ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
            EXPECT_TRUE(fake_.live.empty());
        }
    }
}

TEST_F(HbgBindLedgerTest, HostGetSetCompletesBeforeMetadataPublication) {
    eps_ = {host_get_set_orch_entry, capture_orch_bind};
    Runtime runtime;
    init_runtime(runtime);
    std::vector<uint8_t> input(64, 0x37);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(input));
    ArgDirection sig[] = {ArgDirection::INOUT};
    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    EXPECT_EQ(fake_.copy_count, 2);
    EXPECT_EQ(input[0], 0x52);
    ASSERT_EQ(runtime.tensor_leases().size(), 1u);
    EXPECT_EQ(std::memcmp(runtime.tensor_leases()[0].dev_ptr, input.data(), input.size()), 0);
    for (const auto &copy : fake_.copies)
        EXPECT_NE(copy.dst, runtime.pending_publication().device_target);
    EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    EXPECT_EQ(fake_.copy_count, 2);
    EXPECT_TRUE(runtime.tensor_leases().empty());
}

TEST_F(HbgBindLedgerTest, EachMetadataFailureConsumesTheRecordAndStopsLaterWrites) {
    for (bool graph : {false, true}) {
        eps_ = {graph ? recording_orch_entry : empty_orch_entry, capture_orch_bind};
        size_t region_count = 0;
        {
            Runtime runtime;
            init_runtime(runtime);
            ChipStorageTaskArgs args;
            ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
            region_count = runtime.pending_publication().prerequisites.size() + 1;
            EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
            EXPECT_TRUE(fake_.live.empty());
        }
        for (size_t failure = 1; failure <= region_count; ++failure) {
            Runtime runtime;
            init_runtime(runtime);
            ChipStorageTaskArgs args;
            fake_.copy_count = 0;
            ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
            EXPECT_EQ(fake_.copy_count, 0);
            fake_.fail_copy_on = static_cast<int>(failure);
            EXPECT_NE(publish_run_image_impl(&runtime, &api_), 0);
            EXPECT_EQ(fake_.copy_count, failure);
            EXPECT_EQ(runtime.pending_publication().bytes, 0u);
            EXPECT_TRUE(runtime.pending_publication().prerequisites.empty());
            EXPECT_NE(publish_run_image_impl(&runtime, &api_), 0);
            EXPECT_EQ(fake_.copy_count, failure);
            EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
            EXPECT_TRUE(fake_.live.empty());
            fake_.fail_copy_on = 0;
        }
    }
}

TEST_F(HbgBindLedgerTest, RepeatedArgumentsStillOrchestrateEveryInvocation) {
    eps_ = {recording_orch_entry, capture_orch_bind};
    for (uint64_t value : {11, 22, 11}) {
        Runtime runtime;
        init_runtime(runtime);
        ChipStorageTaskArgs args;
        args.add_scalar(value);
        ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
        ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);
        ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    }
    EXPECT_EQ(fake_.orchestration_count, 3);
}

TEST_F(HbgBindLedgerTest, RejectedRebindPreservesTheUnpublishedImage) {
    Runtime runtime;
    init_runtime(runtime);
    ChipStorageTaskArgs args;
    ASSERT_EQ(bind(runtime, args, nullptr, 0), 0);
    const void *source = runtime.pending_publication().source;
    const auto bytes = runtime.pending_publication().bytes;
    EXPECT_NE(bind(runtime, args, nullptr, 0), 0);
    EXPECT_EQ(runtime.pending_publication().source, source);
    EXPECT_EQ(runtime.pending_publication().bytes, bytes);
    ASSERT_EQ(publish_run_image_impl(&runtime, &api_), 0);
    EXPECT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
}

TEST_F(HbgBindLedgerTest, OldReleaseDoesNotCloseSuccessorTrace) {
    Runtime predecessor;
    Runtime successor;
    init_runtime(predecessor);
    init_runtime(successor);
    ChipStorageTaskArgs args;
    ASSERT_EQ(bind(predecessor, args, nullptr, 0), 0);
    ASSERT_EQ(publish_run_image_impl(&predecessor, &api_), 0);
    HostApi successor_api{nullptr, 1, 1, 0, &fake_ops()};
    uint64_t win[4] = {8, 0, 0, 0};
    ASSERT_EQ(
        bind_callable_to_runtime_impl(&successor, &successor_api, &args, &eps_, nullptr, 0, win, nullptr, nullptr), 0
    );
    const auto finishes_before = fake_.phase_finish_calls;
    ASSERT_EQ(release_run_bindings_impl(&predecessor, &api_), 0);
    EXPECT_EQ(fake_.phase_finish_calls, finishes_before);
    ASSERT_EQ(publish_run_image_impl(&successor, &successor_api), 0);
    EXPECT_EQ(fake_.phase_records_of(HostPhaseKind::BindArenaH2d), 1u);
    EXPECT_EQ(release_run_bindings_impl(&successor, &successor_api), 0);
}

// Normal completion and exception unwinding must both join recorder jobs before
// the local GraphHostState/OrchestratorState and tensor views leave scope.
TEST_F(HbgBindLedgerTest, BindDrainsRecordersBeforeReleasingBuildState) {
    RecordingLifetime recording;
    auto recording_scope = enter_recording(recording);
    Runtime runtime;
    init_runtime(runtime);
    auto runtime_cleanup = cleanup_runtime(runtime);
    ChipStorageTaskArgs args;

    EXPECT_EQ(bind(runtime, args, nullptr, 0), 0);
    EXPECT_EQ(recording.wait_calls, 1);
}

TEST_F(HbgBindLedgerTest, ThrowingBindDrainsRecordersBeforeReleasingBuildState) {
    RecordingLifetime recording;
    recording.throw_from_entry = true;
    auto recording_scope = enter_recording(recording);
    Runtime runtime;
    init_runtime(runtime);
    auto runtime_cleanup = cleanup_runtime(runtime);
    ChipStorageTaskArgs args;

    try {
        (void)bind(runtime, args, nullptr, 0);
        ADD_FAILURE() << "orchestration exception was swallowed";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ(error.what(), "host orchestration failed");
        EXPECT_EQ(recording.wait_calls, 1);
    }

    // A later bind can use the same runtime after the failed build is drained.
    eps_ = {empty_orch_entry, empty_orch_bind};
    EXPECT_EQ(bind(runtime, args, nullptr, 0), 0);
}

TEST_F(HbgBindLedgerTest, BindDrainsItsRecordersWithoutJoiningAnotherRuntime) {
    for (bool throws : {false, true}) {
        SCOPED_TRACE(throws);
        RecordingLifetime recording;
        recording.throw_from_entry = throws;
        auto recording_scope = enter_recording(recording);
        Runtime runtime;
        init_runtime(runtime);
        auto runtime_cleanup = cleanup_runtime(runtime);
        ChipStorageTaskArgs args;
        RuntimeContext other{};
        GraphTaskArgs boundary;
        std::promise<void> entered, release;
        auto released = release.get_future().share();
        std::function<void(const GraphTaskArgs &)> job = [&](const GraphTaskArgs &) {
            entered.set_value();
            released.wait();
        };
        ASSERT_TRUE(graph_record_start_impl(&other, boundary, &job));
        auto job_cleanup = RAIIScopeGuard([&]() {
            release.set_value();
            graph_record_wait_impl(&other);
        });
        ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
        auto binding = std::async(std::launch::async, [&]() {
            if (throws) {
                EXPECT_THROW(bind(runtime, args, nullptr, 0), std::runtime_error);
            } else {
                EXPECT_EQ(bind(runtime, args, nullptr, 0), 0);
            }
            EXPECT_EQ(recording.wait_calls, 1);
        });
        const auto status = binding.wait_for(std::chrono::seconds(5));
        release.set_value();
        graph_record_wait_impl(&other);
        job_cleanup.dismiss();
        binding.get();
        EXPECT_EQ(status, std::future_status::ready) << "bind waited for another runtime's recorder";
    }
}

TEST_F(HbgHostAccessContractTest, HostInputIsReadableDuringBindAndInoutWritesReachBothCopies) {
    Runtime runtime;
    init_runtime(runtime);
    auto runtime_cleanup = cleanup_runtime(runtime);
    std::vector<uint8_t> input(4, 0x17);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(input));
    ArgDirection sig[] = {ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    ASSERT_EQ(access_.reads.size(), 1u);
    EXPECT_EQ(access_.reads[0], 0x17u);
    EXPECT_EQ(access_.error, 0);
    ASSERT_EQ(runtime.tensor_leases().size(), 1u);
    EXPECT_EQ(*static_cast<uint8_t *>(runtime.tensor_leases()[0].dev_ptr), 0x17);
    // An unpublished record belongs to the run that prepared it, so this run ends
    // before the one below binds.
    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);

    access_.write = true;
    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    EXPECT_EQ(input[0], 0x5a);
    ASSERT_EQ(runtime.tensor_leases().size(), 1u);
    EXPECT_EQ(*static_cast<uint8_t *>(runtime.tensor_leases()[0].dev_ptr), 0x5a);
}

TEST_F(HbgHostAccessContractTest, ChildMemoryInputUsesItsCurrentDeviceBytesDuringBind) {
    Runtime runtime;
    init_runtime(runtime);
    auto runtime_cleanup = cleanup_runtime(runtime);
    std::vector<uint8_t> device_bytes(4, 0x29);
    ChipTensor child = host_tensor(device_bytes);
    child.address_space = AddressSpace::DEVICE;
    ChipStorageTaskArgs args;
    args.add_tensor(child);
    ArgDirection sig[] = {ArgDirection::INOUT};

    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    ASSERT_EQ(access_.reads.size(), 1u);
    EXPECT_EQ(access_.reads[0], 0x29u);
    EXPECT_TRUE(runtime.tensor_leases().empty()) << "child memory must not acquire host staging";
    ASSERT_EQ(release_run_bindings_impl(&runtime, &api_), 0);
    access_.write = true;
    ASSERT_EQ(bind(runtime, args, sig, 1), 0);
    EXPECT_EQ(device_bytes[0], 0x5a);
}

TEST_F(HbgHostAccessContractTest, PureHostOutputRejectsGetAndSet) {
    for (bool write : {false, true}) {
        SCOPED_TRACE(write);
        access_.write = write;
        access_.error = 0;
        Runtime runtime;
        init_runtime(runtime);
        auto cleanup = cleanup_runtime(runtime);
        std::vector<uint8_t> output(4, 0x39);
        ChipStorageTaskArgs args;
        args.add_tensor(host_tensor(output));
        ArgDirection signature[] = {ArgDirection::OUT};
        EXPECT_EQ(bind(runtime, args, signature, 1), runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS));
        EXPECT_EQ(access_.error, SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_EQ(output, std::vector<uint8_t>(4, 0x39));
    }
}

TEST_F(HbgHostAccessContractTest, SuccessorReadsPredecessorOutputAfterExplicitCopyback) {
    Runtime predecessor;
    init_runtime(predecessor);
    auto predecessor_cleanup = cleanup_runtime(predecessor);
    std::vector<uint8_t> output(4, 0x11);
    ChipStorageTaskArgs args;
    args.add_tensor(host_tensor(output));
    ArgDirection sig[] = {ArgDirection::INOUT};
    ASSERT_EQ(bind(predecessor, args, sig, 1), 0);
    ASSERT_EQ(predecessor.tensor_leases().size(), 1u);

    // Model bytes from a completed kernel. A device fence alone would leave
    // the caller's host buffer stale; the explicit finalize copy-back is needed.
    *static_cast<uint8_t *>(predecessor.tensor_leases()[0].dev_ptr) = 0x42;
    EXPECT_EQ(output[0], 0x11);
    ASSERT_EQ(finish_run(predecessor, 0), 0);
    ASSERT_EQ(output[0], 0x42);

    Runtime successor;
    init_runtime(successor);
    auto successor_cleanup = cleanup_runtime(successor);
    ASSERT_EQ(bind(successor, args, sig, 1), 0);
    ASSERT_EQ(access_.reads.size(), 2u);
    EXPECT_EQ(access_.reads.back(), 0x42u);
}

TEST_F(HbgHostAccessContractTest, IndependentAndSharedReadOnlyInputsCanPrepareBeforeCopyback) {
    Runtime predecessor;
    init_runtime(predecessor);
    auto predecessor_cleanup = cleanup_runtime(predecessor);
    std::vector<uint8_t> output(4, 0x11);
    std::vector<uint8_t> shared_input(4, 0x27);
    ChipStorageTaskArgs first_args;
    first_args.add_tensor(host_tensor(output));
    first_args.add_tensor(host_tensor(shared_input));
    ArgDirection first_sig[] = {ArgDirection::INOUT, ArgDirection::IN};
    ASSERT_EQ(bind(predecessor, first_args, first_sig, 2), 0);
    ASSERT_EQ(predecessor.tensor_leases().size(), 2u);
    *static_cast<uint8_t *>(predecessor.tensor_leases()[0].dev_ptr) = 0x42;

    // A separate fake bank retains the predecessor's staging, as the runner's
    // leased slot does. No wait/copy-back is called before this bind.
    FakeHostApi successor_bank;
    g_fake = &successor_bank;
    Runtime successor;
    init_runtime(successor);
    auto successor_cleanup = cleanup_runtime(successor);
    std::vector<uint8_t> independent(4, 0x38);
    ChipStorageTaskArgs second_args;
    second_args.add_tensor(host_tensor(shared_input));
    second_args.add_tensor(host_tensor(independent));
    ArgDirection second_sig[] = {ArgDirection::IN, ArgDirection::IN};
    EXPECT_EQ(bind(successor, second_args, second_sig, 2), 0);
    EXPECT_EQ(access_.reads, (std::vector<uint64_t>{0x11, 0x27, 0x27, 0x38}));
    EXPECT_EQ(output[0], 0x11);
    EXPECT_EQ(finish_run(successor, 0), 0);

    g_fake = &fake_;
    EXPECT_EQ(finish_run(predecessor, 0), 0);
    EXPECT_EQ(output[0], 0x42);
    EXPECT_EQ(shared_input[0], 0x27);
}

TEST_F(HbgHostAccessContractTest, GetAndSetRejectCurrentGraphOutputsAndOverlappingWriters) {
    for (InputProducer producer : {InputProducer::Allocated, InputProducer::Overlapping}) {
        for (bool write : {false, true}) {
            SCOPED_TRACE(static_cast<int>(producer));
            SCOPED_TRACE(write);
            access_.producer = producer;
            access_.write = write;
            access_.error = 0;
            Runtime runtime;
            init_runtime(runtime);
            auto runtime_cleanup = cleanup_runtime(runtime);
            std::vector<uint8_t> input(4, 0x17);
            ChipStorageTaskArgs args;
            args.add_tensor(host_tensor(input));
            ArgDirection sig[] = {ArgDirection::INOUT};

            EXPECT_EQ(bind(runtime, args, sig, 1), runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS));
            EXPECT_EQ(access_.error, SIMPLER_ERROR_INVALID_ARGS);
            EXPECT_EQ(input[0], 0x17) << "a rejected write must not reach the caller's buffer";
        }
    }
}

TEST_F(HbgHostAccessContractTest, DisjointWriterDoesNotPreventReadyInputAccess) {
    for (bool write : {false, true}) {
        access_.producer = InputProducer::Disjoint;
        access_.write = write;
        access_.error = 0;
        Runtime runtime;
        init_runtime(runtime);
        auto runtime_cleanup = cleanup_runtime(runtime);
        // a5sim resolves the callable metadata while constructing its scheduler
        // image, even though this memory backend never launches the kernel.
        auto callable = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
        reinterpret_cast<CoreCallable *>(callable.data())->set_resolved_addr(0x1000);
        runtime.replay_function_bin_addr(0, reinterpret_cast<uint64_t>(callable.data()));
        std::vector<uint8_t> input(4, 0x17);
        ChipStorageTaskArgs args;
        args.add_tensor(host_tensor(input));
        ArgDirection sig[] = {ArgDirection::INOUT};

        ASSERT_EQ(bind(runtime, args, sig, 1), 0);
        EXPECT_EQ(access_.error, 0);
        EXPECT_EQ(input[0], 0x17);
        EXPECT_EQ(input[1], write ? 0x5a : 0x17);
        if (!write) EXPECT_EQ(access_.reads.back(), 0x17u);
    }
}
