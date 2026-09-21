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
 * Registration owns a callable's function tables, and a run only references
 * them — through the real `SimDeviceRunnerBase` pool.
 *
 * The sim runner is the one variant whose registration needs no SDK: its
 * "upload" is a host scratch plus a real `dlopen` of each child kernel, so
 * these cases drive `upload_chip_callable_buffer`,
 * `record_host_orch_callable`, `bind_callable_to_runtime`,
 * `unregister_callable` and `release_chip_callable_buffer` as production code,
 * with a genuine loadable child binary. Two collaborators are supplied rather
 * than mirrored: `bind_callable_to_runtime_impl`, whose per-run orchestration
 * is a separate contract, and
 * `runtime_uses_callable_entry_table_impl`, which is the runtime property one
 * binary has to be able to answer both ways.
 *
 * What is under test is the ownership and reference contract: a registration
 * that fails publishes nothing, a published entry's tables are already
 * complete, callable_ids that dedup onto one entry share exactly one table,
 * and rebinding A -> B -> A installs each callable's own reference with no
 * table rebuilt at bind time.
 */

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "callable.h"
#include "chip_callable_layout.h"
#include "device_runner_base.h"
#include "runtime.h"

namespace {

// The runtime property the registration consults. Production defines this in
// each runtime_maker.cpp — true only for a5 host_build_graph; a single test
// binary has to reach both shapes, so it is settable here.
bool g_uses_entry_table = false;

// Per-run orchestration is a separate contract with its own coverage; what
// matters here is that `bind_callable_to_runtime` installs the table reference
// before it delegates, and that a delegate failure is reported.
int g_bind_impl_rc = 0;
int g_bind_impl_calls = 0;
uint64_t g_bind_impl_seen_object_addr = 0;
uint32_t g_bind_impl_seen_len = 0;

}  // namespace

extern "C" bool runtime_uses_callable_entry_table_impl() { return g_uses_entry_table; }

extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi * /*api*/, const ChipStorageTaskArgs * /*orch_args*/, void * /*host_orch_func_ptr*/,
    const ArgDirection * /*signature*/, int /*sig_count*/, const uint64_t * /*ring_task_window*/,
    const uint64_t * /*ring_heap*/, const uint64_t * /*ring_dep_pool*/
) {
    ++g_bind_impl_calls;
    // Read the reference through the descriptor the delegate is handed, which
    // is how the real a5 hbg maker reaches it.
    g_bind_impl_seen_object_addr = runtime->dev.callable_table_addr_;
    g_bind_impl_seen_len = runtime->dev.callable_table_len_;
    return g_bind_impl_rc;
}

namespace {

// A concrete runner: the pure virtuals below are the execution surface, which
// no case here drives. The accessors read the production pool the registration
// publishes into.
class PoolRunner : public SimDeviceRunnerBase {
public:
    int prepare_execution(
        Runtime & /*runtime*/, const CallConfig & /*config*/, uint32_t /*pipeline_slot*/,
        const NativeRunIdentity & /*identity*/, std::unique_ptr<PreparedExecution> * /*prepared*/
    ) override {
        return -1;
    }
    LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> /*prepared*/, LaunchPermit /*permit*/) override {
        return LaunchOutcome{};
    }
    void abandon_prepared_execution(PreparedExecution & /*prepared*/) noexcept override {}
    int poll_execution(const ActiveExecution & /*active*/) override { return 0; }
    int drain_execution(ActiveExecution & /*active*/) override { return 0; }
    int finalize() override { return 0; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs & /*reg_args*/) override { return 0; }

    // The bulk-free finalize() performs, so each case leaves no scratch or
    // handle behind.
    void release_all() { release_callable_state(); }

    size_t pool_size() const { return chip_callable_buffers_.size(); }

    bool pool_has(uint64_t hash) const { return chip_callable_buffers_.count(hash) != 0; }

    // Everything a published entry must already carry, read straight off it.
    struct PoolView {
        uint64_t chip_dev{0};
        uint64_t object_table_dev{0};
        uint64_t entry_table_dev{0};
        uint32_t table_len{0};
        int refcount{0};
        size_t object_table_size{0};
        size_t entry_table_size{0};
        size_t dlopen_handles{0};
        const uint64_t *object_table{nullptr};
        const uint64_t *entry_table{nullptr};
    };

    PoolView view(uint64_t hash) const {
        PoolView out;
        auto it = chip_callable_buffers_.find(hash);
        if (it == chip_callable_buffers_.end()) return out;
        out.chip_dev = it->second.chip_dev;
        out.object_table_dev = it->second.object_table_dev;
        out.entry_table_dev = it->second.entry_table_dev;
        out.table_len = it->second.table_len;
        out.refcount = it->second.refcount;
        out.object_table_size = it->second.object_table.size();
        out.entry_table_size = it->second.entry_table.size();
        out.dlopen_handles = it->second.dlopen_handles.size();
        out.object_table = it->second.object_table.data();
        out.entry_table = it->second.entry_table.data();
        return out;
    }
};

// The child kernel's bytes. A real loadable module built beside this test, so
// the registration's own dlopen / dlsym("kernel_entry") path runs for real
// instead of being skipped.
std::vector<uint8_t> loadable_kernel_bytes() {
    std::FILE *f = std::fopen(FAKE_KERNEL_SO_PATH, "rb");
    if (f == nullptr) return {};
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(size > 0 ? static_cast<size_t>(size) : 0);
    if (!bytes.empty() && std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
    std::fclose(f);
    return bytes;
}

std::vector<uint8_t> make_child(const std::vector<uint8_t> &binary) {
    return make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary.data(), static_cast<uint32_t>(binary.size()));
}

// A ChipCallable whose children claim `func_ids`, each carrying `binary`.
// `orch_tag` makes two otherwise-identical callables differ in content, which
// is what decides whether they dedup onto one pool entry.
std::vector<uint8_t>
make_chip(const std::vector<int32_t> &func_ids, const std::vector<uint8_t> &binary, uint8_t orch_tag) {
    std::vector<std::vector<uint8_t>> children;
    children.reserve(func_ids.size());
    for (size_t i = 0; i < func_ids.size(); ++i)
        children.push_back(make_child(binary));
    const uint8_t orch[64] = {orch_tag};
    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch_entry", orch, sizeof(orch), func_ids.data(), children.data(),
        static_cast<int32_t>(func_ids.size()), "orch_config"
    );
}

const ChipCallable *as_chip(const std::vector<uint8_t> &buffer) {
    return reinterpret_cast<const ChipCallable *>(buffer.data());
}

uint64_t hash_of(const std::vector<uint8_t> &buffer) {
    return compute_chip_callable_layout(as_chip(buffer)).content_hash;
}

class SimCallablePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_uses_entry_table = false;
        g_bind_impl_rc = 0;
        g_bind_impl_calls = 0;
        g_bind_impl_seen_object_addr = 0;
        g_bind_impl_seen_len = 0;
        kernel_ = loadable_kernel_bytes();
        ASSERT_FALSE(kernel_.empty()) << "the fake child kernel module did not load from " << FAKE_KERNEL_SO_PATH;
    }

    void TearDown() override { runner_.release_all(); }

    // Registration as `record_callable_on_runner` performs it on the hbg path:
    // upload, then record the id against the retained block. The orchestration
    // handle is a real dlopen handle because the runner owns it from here on
    // and dlcloses it when the id goes away.
    int register_host_orch(int32_t callable_id, const std::vector<uint8_t> &chip, uint64_t *chip_dev_out) {
        const uint64_t chip_dev = runner_.upload_chip_callable_buffer(as_chip(chip));
        if (chip_dev_out != nullptr) *chip_dev_out = chip_dev;
        if (chip_dev == 0) return -1;
        void *orch_handle = dlopen(FAKE_KERNEL_SO_PATH, RTLD_NOW | RTLD_LOCAL);
        if (orch_handle == nullptr) return -1;
        void *orch_entry = dlsym(orch_handle, "kernel_entry");
        if (orch_entry == nullptr) {
            dlclose(orch_handle);
            return -1;
        }
        return runner_.record_host_orch_callable(callable_id, hash_of(chip), orch_handle, orch_entry, {});
    }

    PoolRunner runner_;
    std::vector<uint8_t> kernel_;
};

// A registration whose child kernel cannot be loaded must publish nothing: no
// pool entry, and therefore no chip_dev any caller could reach. The successful
// registration that follows proves the refused one left the pool usable rather
// than merely empty.
TEST_F(SimCallablePoolTest, AFailedRegistrationPublishesNothing) {
    const std::vector<uint8_t> unloadable = make_chip({0}, std::vector<uint8_t>(256, 0x7f), 0x11);

    EXPECT_EQ(runner_.upload_chip_callable_buffer(as_chip(unloadable)), 0u);
    EXPECT_EQ(runner_.pool_size(), 0u);
    EXPECT_FALSE(runner_.pool_has(hash_of(unloadable)));

    const std::vector<uint8_t> good = make_chip({0}, kernel_, 0x22);
    EXPECT_NE(runner_.upload_chip_callable_buffer(as_chip(good)), 0u);
    EXPECT_EQ(runner_.pool_size(), 1u);
}

// A func_id the runtime's tables cannot address is refused before anything is
// allocated, so it too publishes nothing.
TEST_F(SimCallablePoolTest, AnUnaddressableFuncIdPublishesNothing) {
    const std::vector<uint8_t> chip = make_chip({RUNTIME_MAX_FUNC_ID}, kernel_, 0x33);
    EXPECT_EQ(runner_.upload_chip_callable_buffer(as_chip(chip)), 0u);
    EXPECT_EQ(runner_.pool_size(), 0u);
}

// A published entry is already complete: the tables are sized, their addresses
// are set, and every handle the dlopen loop opened belongs to it. Nothing here
// is deferred to a first bind.
TEST_F(SimCallablePoolTest, APublishedEntryCarriesCompleteTables) {
    g_uses_entry_table = true;
    const std::vector<uint8_t> chip = make_chip({0, 3}, kernel_, 0x44);

    const uint64_t chip_dev = runner_.upload_chip_callable_buffer(as_chip(chip));
    ASSERT_NE(chip_dev, 0u);

    const auto v = runner_.view(hash_of(chip));
    EXPECT_EQ(v.chip_dev, chip_dev);
    EXPECT_EQ(v.refcount, 1);
    EXPECT_EQ(v.table_len, 4u) << "one past the largest func_id";
    EXPECT_EQ(v.object_table_size, 4u);
    EXPECT_EQ(v.entry_table_size, 4u);
    EXPECT_EQ(v.dlopen_handles, 2u) << "the entry owns every handle the registration opened";
    EXPECT_EQ(v.object_table_dev, reinterpret_cast<uint64_t>(v.object_table));
    EXPECT_EQ(v.entry_table_dev, reinterpret_cast<uint64_t>(v.entry_table));

    // The object view addresses the CoreCallable inside the retained scratch;
    // the entry view carries the host function pointer its dlopen resolved.
    const ChipCallableLayout layout = compute_chip_callable_layout(as_chip(chip));
    for (int32_t i = 0; i < as_chip(chip)->child_count(); ++i) {
        const uint32_t func_id = static_cast<uint32_t>(as_chip(chip)->child_func_id(i));
        const uint64_t object = chip_dev + layout.header_size + as_chip(chip)->child_offset(i);
        EXPECT_EQ(v.object_table[func_id], object) << "func_id=" << func_id;
        EXPECT_EQ(v.entry_table[func_id], reinterpret_cast<const CoreCallable *>(object)->resolved_addr())
            << "func_id=" << func_id;
        EXPECT_NE(v.entry_table[func_id], 0u);
    }
    EXPECT_EQ(v.object_table[1], 0u) << "a func_id no child claims stays unmapped";
    EXPECT_EQ(v.entry_table[1], 0u);
}

// A runtime whose device consumers resolve the entry from the object asks for
// no entry view, and must then retain none.
TEST_F(SimCallablePoolTest, TheEntryViewIsBuiltOnlyWhereARuntimeReadsIt) {
    g_uses_entry_table = false;
    const std::vector<uint8_t> chip = make_chip({2}, kernel_, 0x55);
    ASSERT_NE(runner_.upload_chip_callable_buffer(as_chip(chip)), 0u);

    const auto v = runner_.view(hash_of(chip));
    EXPECT_EQ(v.table_len, 3u);
    EXPECT_EQ(v.object_table_size, 3u);
    EXPECT_EQ(v.entry_table_size, 0u);
    EXPECT_EQ(v.entry_table_dev, 0u);
    EXPECT_NE(v.object_table_dev, 0u);
}

// Two callable_ids registered from identical bytes share one pool entry, so
// they share one table. That is what makes the entry a consistent owner: the
// content hash covers the child func_ids and offsets the tables are derived
// from, so there is no second derivation to disagree with.
TEST_F(SimCallablePoolTest, DedupedCallableIdsShareOneTable) {
    g_uses_entry_table = true;
    const std::vector<uint8_t> chip = make_chip({1}, kernel_, 0x66);

    uint64_t first_dev = 0;
    uint64_t second_dev = 0;
    ASSERT_EQ(register_host_orch(7, chip, &first_dev), 0);
    ASSERT_EQ(register_host_orch(9, chip, &second_dev), 0);
    EXPECT_EQ(first_dev, second_dev);
    EXPECT_EQ(runner_.pool_size(), 1u);

    const auto v = runner_.view(hash_of(chip));
    EXPECT_EQ(v.refcount, 2);
    EXPECT_EQ(v.dlopen_handles, 1u) << "the second registration dedups rather than loading again";

    Runtime runtime;
    ASSERT_EQ(runner_.bind_callable_to_runtime(runtime, 7, nullptr, nullptr, nullptr, nullptr, nullptr), 0);
    const uint64_t from_first = runtime.dev.callable_table_addr_;
    const uint64_t entry_from_first = runtime.callable_entry_table_addr();
    ASSERT_EQ(runner_.bind_callable_to_runtime(runtime, 9, nullptr, nullptr, nullptr, nullptr, nullptr), 0);
    EXPECT_EQ(runtime.dev.callable_table_addr_, from_first);
    EXPECT_EQ(runtime.callable_entry_table_addr(), entry_from_first);
    EXPECT_EQ(v.object_table_dev, from_first);

    // One id going away leaves the other's table standing.
    ASSERT_EQ(runner_.unregister_callable(7), 0);
    EXPECT_EQ(runner_.pool_size(), 1u);
    EXPECT_EQ(runner_.view(hash_of(chip)).refcount, 1);
    ASSERT_EQ(runner_.unregister_callable(9), 0);
    EXPECT_EQ(runner_.pool_size(), 0u);
}

// A -> B -> A. Each bind installs that callable's own reference, so the third
// bind resolves A's func_ids again and never B's — and no table is built,
// compared or copied at bind time, because the reference is all a bind writes.
TEST_F(SimCallablePoolTest, RebindingInstallsEachCallablesOwnReference) {
    g_uses_entry_table = true;
    const std::vector<uint8_t> a = make_chip({0, 1}, kernel_, 0xa1);
    const std::vector<uint8_t> b = make_chip({0}, kernel_, 0xb2);
    ASSERT_NE(hash_of(a), hash_of(b));
    ASSERT_EQ(register_host_orch(1, a, nullptr), 0);
    ASSERT_EQ(register_host_orch(2, b, nullptr), 0);
    ASSERT_EQ(runner_.pool_size(), 2u);

    const auto va = runner_.view(hash_of(a));
    const auto vb = runner_.view(hash_of(b));
    ASSERT_EQ(va.table_len, 2u);
    ASSERT_EQ(vb.table_len, 1u);

    Runtime runtime;
    for (int round = 0; round < 2; ++round) {
        SCOPED_TRACE(round);

        ASSERT_EQ(runner_.bind_callable_to_runtime(runtime, 1, nullptr, nullptr, nullptr, nullptr, nullptr), 0);
        EXPECT_EQ(runtime.dev.callable_table_addr_, va.object_table_dev);
        EXPECT_EQ(runtime.dev.callable_table_len_, 2u);
        EXPECT_EQ(runtime.callable_entry_table_addr(), va.entry_table_dev);
        EXPECT_EQ(runtime.get_function_bin_addr(0), va.object_table[0]);
        EXPECT_EQ(runtime.get_function_bin_addr(1), va.object_table[1]);
        EXPECT_EQ(runtime.get_active_callable_id(), 1);

        ASSERT_EQ(runner_.bind_callable_to_runtime(runtime, 2, nullptr, nullptr, nullptr, nullptr, nullptr), 0);
        EXPECT_EQ(runtime.dev.callable_table_addr_, vb.object_table_dev);
        EXPECT_EQ(runtime.dev.callable_table_len_, 1u);
        EXPECT_EQ(runtime.callable_entry_table_addr(), vb.entry_table_dev);
        EXPECT_EQ(runtime.get_function_bin_addr(0), vb.object_table[0]);
        EXPECT_EQ(runtime.get_function_bin_addr(1), 0u) << "func_id 1 still resolves into the block A registered";
        EXPECT_EQ(runtime.get_active_callable_id(), 2);
    }

    // The reference the delegate is handed is the one the descriptor carries,
    // so the a5 hbg maker's read of it sees the bound callable's table.
    EXPECT_EQ(g_bind_impl_seen_object_addr, vb.object_table_dev);
    EXPECT_EQ(g_bind_impl_seen_len, 1u);
    EXPECT_EQ(g_bind_impl_calls, 4);
}

// An unregistered id has no block, so a bind against it installs nothing and
// leaves no reference to whatever the Runtime last carried.
TEST_F(SimCallablePoolTest, ABindAgainstAnUnregisteredIdLeavesNoReference) {
    const std::vector<uint8_t> chip = make_chip({0}, kernel_, 0xc3);
    ASSERT_EQ(register_host_orch(4, chip, nullptr), 0);

    Runtime runtime;
    ASSERT_EQ(runner_.bind_callable_to_runtime(runtime, 4, nullptr, nullptr, nullptr, nullptr, nullptr), 0);
    ASSERT_NE(runtime.dev.callable_table_addr_, 0u);

    ASSERT_EQ(runner_.unregister_callable(4), 0);
    EXPECT_EQ(runner_.pool_size(), 0u);
    EXPECT_NE(runner_.bind_callable_to_runtime(runtime, 4, nullptr, nullptr, nullptr, nullptr, nullptr), 0);
    EXPECT_EQ(runtime.dev.callable_table_addr_, 0u)
        << "the reference into the freed block outlived the bind that failed";
    EXPECT_EQ(runtime.dev.callable_table_len_, 0u);
    EXPECT_EQ(runtime.callable_entry_table_addr(), 0u);
    EXPECT_EQ(runtime.get_function_bin_addr(0), 0u);
}

// A delegate failure is reported, and the reference it was handed is the bound
// callable's — a bind that fails inside the runtime's own half has already
// replaced any predecessor's reference rather than leaving one standing.
TEST_F(SimCallablePoolTest, ADelegateFailureIsReportedAfterTheReferenceIsInstalled) {
    const std::vector<uint8_t> chip = make_chip({5}, kernel_, 0xd4);
    ASSERT_EQ(register_host_orch(6, chip, nullptr), 0);
    const auto v = runner_.view(hash_of(chip));

    Runtime runtime;
    g_bind_impl_rc = -7;
    EXPECT_EQ(runner_.bind_callable_to_runtime(runtime, 6, nullptr, nullptr, nullptr, nullptr, nullptr), -7);
    EXPECT_EQ(g_bind_impl_seen_object_addr, v.object_table_dev);
    EXPECT_EQ(runtime.dev.callable_table_addr_, v.object_table_dev);
}

// Re-registering the same bytes after the last id went away rebuilds the block
// and its tables from scratch, so a second life gets its own addresses rather
// than the freed ones.
TEST_F(SimCallablePoolTest, ReRegisteringRebuildsTheTables) {
    g_uses_entry_table = true;
    const std::vector<uint8_t> chip = make_chip({0}, kernel_, 0xe5);

    ASSERT_EQ(register_host_orch(3, chip, nullptr), 0);
    const auto first = runner_.view(hash_of(chip));
    ASSERT_NE(first.object_table_dev, 0u);
    ASSERT_EQ(runner_.unregister_callable(3), 0);
    ASSERT_EQ(runner_.pool_size(), 0u);

    ASSERT_EQ(register_host_orch(3, chip, nullptr), 0);
    const auto second = runner_.view(hash_of(chip));
    EXPECT_EQ(second.refcount, 1);
    EXPECT_EQ(second.table_len, 1u);
    EXPECT_EQ(second.object_table_size, 1u);
    EXPECT_EQ(second.entry_table_size, 1u);
    EXPECT_EQ(second.object_table_dev, reinterpret_cast<uint64_t>(second.object_table));
    EXPECT_NE(second.object_table[0], 0u);
    EXPECT_NE(second.entry_table[0], 0u);
}

}  // namespace
