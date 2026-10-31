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

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "orchestration_api.h"

static_assert(offsetof(RuntimeOps, submit_task) == 0);
static_assert(offsetof(RuntimeOps, scope_set_site) == 17 * sizeof(void *));
static_assert(sizeof(RuntimeOps) == 18 * sizeof(void *));
static_assert(offsetof(RuntimeContext, pending_scope_mode) == sizeof(void *));

namespace {

struct FakeRuntime : RuntimeContext {
    bool fatal{false};
    int ordinary_calls{0};
    int wait_calls{0};
    int fatal_calls{0};
    int32_t fatal_code{SIMPLER_ERROR_NONE};
    std::string fatal_message;
    const CoreTaskArgs *last_args{nullptr};
    const BlockingWait *last_wait{nullptr};
    MixedKernels last_kernels{};
    std::vector<TaskId> dependencies;
    std::vector<DepFlags> dependency_kinds;
};

constexpr TaskId kResult = TaskId::make(0, 77);

void record_args(FakeRuntime &runtime, const MixedKernels &kernels, const CoreTaskArgs &args) {
    runtime.last_args = &args;
    runtime.last_kernels = kernels;
    runtime.dependencies.clear();
    runtime.dependency_kinds.clear();
    for (uint32_t i = 0; i < args.explicit_dep_count(); ++i) {
        runtime.dependencies.push_back(args.explicit_dep(i));
        runtime.dependency_kinds.push_back(args.explicit_dep_kind(i));
    }
}

TaskOutputTensors ordinary_submit(RuntimeContext *rt, const MixedKernels &kernels, const CoreTaskArgs &args) {
    auto &runtime = *static_cast<FakeRuntime *>(rt);
    ++runtime.ordinary_calls;
    runtime.last_wait = nullptr;
    record_args(runtime, kernels, args);
    TaskOutputTensors result;
    result.set_task_id(kResult);
    return result;
}

TaskOutputTensors
wait_submit(RuntimeContext *rt, const MixedKernels &kernels, const CoreTaskArgs &args, const BlockingWait &wait) {
    auto &runtime = *static_cast<FakeRuntime *>(rt);
    ++runtime.wait_calls;
    runtime.last_wait = &wait;
    record_args(runtime, kernels, args);
    TaskOutputTensors result;
    result.set_task_id(kResult);
    return result;
}

bool is_fatal(RuntimeContext *rt) { return static_cast<FakeRuntime *>(rt)->fatal; }

void report_fatal(RuntimeContext *rt, int32_t code, const char *, const char *format, ...) {
    auto &runtime = *static_cast<FakeRuntime *>(rt);
    runtime.fatal = true;
    ++runtime.fatal_calls;
    runtime.fatal_code = code;
    char message[256]{};
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    runtime.fatal_message = message;
}

const BlockingWaitOps kWaitOps{BLOCKING_WAIT_OPS_VERSION, sizeof(BlockingWaitOps), wait_submit};

class BlockingWaitBinding : public testing::Test {
protected:
    RuntimeOps ordinary_ops{};
    FakeRuntime runtime{};

    void SetUp() override {
        ordinary_ops.submit_task = ordinary_submit;
        ordinary_ops.is_fatal = is_fatal;
        ordinary_ops.report_fatal = report_fatal;
        runtime.ops = &ordinary_ops;
        framework_bind_runtime(&runtime);
    }

    void TearDown() override { framework_bind_runtime(nullptr); }

    void expect_rejected() {
        EXPECT_EQ(runtime.fatal_calls, 1);
        EXPECT_EQ(runtime.fatal_code, SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_NE(runtime.fatal_message.find("blocking-wait admission"), std::string::npos);
        EXPECT_EQ(runtime.ordinary_calls, 0);
        EXPECT_EQ(runtime.wait_calls, 0);
    }
};

TEST_F(BlockingWaitBinding, OrdinaryOverloadsUseTheLegacyTable) {
    CoreTaskArgs args;
    MixedKernels kernels;
    kernels.aic_kernel_id = 11;
    EXPECT_EQ(rt_submit_task(kernels, args).task_id(), kResult);
    EXPECT_EQ(runtime.last_args, &args);
    EXPECT_EQ(rt_submit_aic_task(12, args).task_id(), kResult);
    EXPECT_EQ(runtime.last_kernels.aic_kernel_id, 12);
    EXPECT_EQ(rt_submit_aiv_task(13, args).task_id(), kResult);
    EXPECT_EQ(runtime.last_kernels.aiv0_kernel_id, 13);
    EXPECT_EQ(runtime.ordinary_calls, 3);
    EXPECT_EQ(runtime.wait_calls, 0);
    EXPECT_EQ(runtime.fatal_calls, 0);
    EXPECT_EQ(framework_current_blocking_wait_ops(), nullptr);
}

TEST_F(BlockingWaitBinding, MissingCapabilityRejectsInsteadOfFallingBackToOrdinarySubmit) {
    CoreTaskArgs args;
    const TaskId producers[] = {TaskId::make(0, 1)};
    const auto wait = BlockingWait::after_producers(producers, 1);
    EXPECT_FALSE(rt_submit_aiv_task(3, args, wait).task_id().is_valid());
    expect_rejected();
}

TEST_F(BlockingWaitBinding, InvalidVersionAndNullFunctionAreRejected) {
    CoreTaskArgs args;
    const auto wait = BlockingWait::resident();
    const BlockingWaitOps invalid[] = {
        {0, sizeof(BlockingWaitOps), wait_submit},
        {BLOCKING_WAIT_OPS_VERSION + 1, sizeof(BlockingWaitOps), wait_submit},
        {BLOCKING_WAIT_OPS_VERSION, sizeof(BlockingWaitOps), nullptr},
    };
    for (const auto &ops : invalid) {
        SCOPED_TRACE(ops.version);
        runtime.fatal = false;
        runtime.fatal_calls = 0;
        framework_bind_runtime_with_wait(&runtime, &ops);
        EXPECT_FALSE(rt_submit_aic_task(4, args, wait).task_id().is_valid());
        expect_rejected();
    }
}

TEST_F(BlockingWaitBinding, ExtendedOverloadsPassTheIndependentDescriptor) {
    framework_bind_runtime_with_wait(&runtime, &kWaitOps);
    CoreTaskArgs args;
    const TaskId ordinary[] = {TaskId::make(0, 2)};
    const DepFlags kinds[] = {DEP_RETAIN};
    args.set_dependencies_with_kinds(ordinary, kinds, 1);
    const TaskId producers[] = {TaskId::make(0, 3), TaskId::make(1, 4)};
    const auto wait = BlockingWait::after_producers(producers, 2);
    MixedKernels kernels;
    kernels.aic_kernel_id = 5;
    EXPECT_EQ(rt_submit_task(kernels, args, wait).task_id(), kResult);
    EXPECT_EQ(runtime.last_args, &args);
    EXPECT_EQ(runtime.last_wait, &wait);
    EXPECT_EQ(runtime.last_wait->producers, producers);
    EXPECT_EQ(runtime.last_wait->producer_count, 2U);
    EXPECT_EQ(runtime.last_wait->mode, BlockingWaitMode::ProducersComplete);
    ASSERT_EQ(runtime.dependencies.size(), 1U);
    EXPECT_EQ(runtime.dependencies[0], ordinary[0]);
    EXPECT_EQ(runtime.dependency_kinds[0], DEP_RETAIN);
    EXPECT_EQ(rt_submit_aic_task(6, args, wait).task_id(), kResult);
    EXPECT_EQ(runtime.last_kernels.aic_kernel_id, 6);
    EXPECT_EQ(rt_submit_aiv_task(7, args, wait).task_id(), kResult);
    EXPECT_EQ(runtime.last_kernels.aiv0_kernel_id, 7);
    EXPECT_EQ(runtime.wait_calls, 3);
    EXPECT_EQ(rt_submit_aiv_task(8, args).task_id(), kResult);
    EXPECT_EQ(runtime.last_wait, nullptr);
    EXPECT_EQ(runtime.ordinary_calls, 1);
    EXPECT_EQ(runtime.fatal_calls, 0);
}

TEST_F(BlockingWaitBinding, LegacyRebindAndNewBinderCleanupClearTheCapabilityAcrossRuns) {
    CoreTaskArgs args;
    const TaskId producers[] = {TaskId::make(0, 1)};
    const auto wait = BlockingWait::after_producers(producers, 1);
    for (int run = 0; run < 3; ++run) {
        SCOPED_TRACE(run);
        framework_bind_runtime_with_wait(&runtime, &kWaitOps);
        EXPECT_EQ(framework_current_runtime(), &runtime);
        EXPECT_EQ(framework_current_blocking_wait_ops(), &kWaitOps);
        EXPECT_EQ(rt_submit_aiv_task(1, args, wait).task_id(), kResult);
        framework_bind_runtime_with_wait(nullptr, nullptr);
        EXPECT_EQ(framework_current_runtime(), nullptr);
        EXPECT_EQ(framework_current_blocking_wait_ops(), nullptr);
        framework_bind_runtime_with_wait(&runtime, &kWaitOps);
        framework_bind_runtime(&runtime);
        EXPECT_EQ(framework_current_runtime(), &runtime);
        EXPECT_EQ(framework_current_blocking_wait_ops(), nullptr);
        EXPECT_EQ(rt_submit_aiv_task(1, args).task_id(), kResult);
    }
    EXPECT_EQ(runtime.wait_calls, 3);
    EXPECT_EQ(runtime.ordinary_calls, 3);
    EXPECT_EQ(runtime.fatal_calls, 0);
}

TEST_F(BlockingWaitBinding, WithDepsFinalizationAndResetDoNotConsumeTheWaitDescriptor) {
    framework_bind_runtime_with_wait(&runtime, &kWaitOps);
    CoreTaskArgsWithDeps<2> args;
    const TaskId dependency = TaskId::make(0, 5);
    args.add_dep_wait(dependency);
    const TaskId producers[] = {TaskId::make(0, 6)};
    const auto wait = BlockingWait::after_producers(producers, 1);
    EXPECT_EQ(rt_submit_aiv_task(2, args, wait).task_id(), kResult);
    EXPECT_EQ(runtime.last_wait, &wait);
    ASSERT_EQ(runtime.dependencies.size(), 1U);
    EXPECT_EQ(runtime.dependencies[0], dependency);
    EXPECT_EQ(runtime.dependency_kinds[0], DEP_WAIT);
    MixedKernels kernels;
    kernels.aic_kernel_id = 3;
    EXPECT_EQ(rt_submit_task(kernels, args, wait).task_id(), kResult);
    EXPECT_EQ(runtime.last_wait, &wait);
    EXPECT_EQ(runtime.dependencies.size(), 1U);
    args.reset();
    EXPECT_EQ(rt_submit_aic_task(4, args, wait).task_id(), kResult);
    EXPECT_EQ(runtime.last_wait, &wait);
    EXPECT_TRUE(runtime.dependencies.empty());
    EXPECT_EQ(rt_submit_aiv_task(5, args).task_id(), kResult);
    EXPECT_EQ(runtime.last_wait, nullptr);
    EXPECT_EQ(runtime.wait_calls, 3);
    EXPECT_EQ(runtime.ordinary_calls, 1);
    EXPECT_EQ(runtime.fatal_calls, 0);
}

TEST_F(BlockingWaitBinding, FatalRuntimeShortCircuitsBothSubmitInterfaces) {
    framework_bind_runtime_with_wait(&runtime, &kWaitOps);
    runtime.fatal = true;
    CoreTaskArgs args;
    const auto wait = BlockingWait::resident();
    EXPECT_FALSE(rt_submit_aiv_task(1, args).task_id().is_valid());
    EXPECT_FALSE(rt_submit_aiv_task(1, args, wait).task_id().is_valid());
    EXPECT_EQ(runtime.ordinary_calls, 0);
    EXPECT_EQ(runtime.wait_calls, 0);
    EXPECT_EQ(runtime.fatal_calls, 0);
}

#if defined(__linux__) || defined(__APPLE__)
class GuardedWaitOps {
public:
    GuardedWaitOps() = default;
    GuardedWaitOps(const GuardedWaitOps &) = delete;
    GuardedWaitOps &operator=(const GuardedWaitOps &) = delete;

    ~GuardedWaitOps() {
        framework_bind_runtime(nullptr);
        if (mapping_ != nullptr) munmap(mapping_, 2 * page_size_);
    }

    bool init(uint32_t version, uint32_t size) {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) return false;
        page_size_ = static_cast<size_t>(page_size);
        void *mapping = mmap(nullptr, 2 * page_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) return false;
        mapping_ = static_cast<unsigned char *>(mapping);
        ops_ = new (mapping_ + page_size_ - offsetof(BlockingWaitOps, submit_task))
            BlockingWaitOps{version, size, wait_submit};
        return mprotect(mapping_ + page_size_, page_size_, PROT_NONE) == 0;
    }

    const BlockingWaitOps *header() const { return ops_; }
    const BlockingWaitOps *inaccessible() const {
        return reinterpret_cast<const BlockingWaitOps *>(mapping_ + page_size_);
    }

private:
    unsigned char *mapping_{nullptr};
    size_t page_size_{0};
    BlockingWaitOps *ops_{nullptr};
};
#endif

TEST_F(BlockingWaitBinding, ShortTableIsRejectedBeforeReadingTheGuardedFunctionPointer) {
#if defined(__linux__) || defined(__APPLE__)
    GuardedWaitOps table;
    ASSERT_TRUE(table.init(BLOCKING_WAIT_OPS_VERSION, offsetof(BlockingWaitOps, submit_task)));
    framework_bind_runtime_with_wait(&runtime, table.header());
    CoreTaskArgs args;
    const auto wait = BlockingWait::resident();
    EXPECT_FALSE(rt_submit_aiv_task(1, args, wait).task_id().is_valid());
    expect_rejected();
#else
    GTEST_SKIP() << "requires mmap/mprotect guard pages";
#endif
}

TEST_F(BlockingWaitBinding, OrdinarySubmitDoesNotReadAnInaccessibleExtension) {
#if defined(__linux__) || defined(__APPLE__)
    GuardedWaitOps table;
    ASSERT_TRUE(table.init(BLOCKING_WAIT_OPS_VERSION, sizeof(BlockingWaitOps)));
    framework_bind_runtime_with_wait(&runtime, table.inaccessible());
    CoreTaskArgs args;
    EXPECT_EQ(rt_submit_aiv_task(1, args).task_id(), kResult);
    EXPECT_EQ(runtime.ordinary_calls, 1);
    EXPECT_EQ(runtime.wait_calls, 0);
    EXPECT_EQ(runtime.fatal_calls, 0);
#else
    GTEST_SKIP() << "requires mmap/mprotect guard pages";
#endif
}

}  // namespace
