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

#include <chrono>
#include <atomic>
#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <vector>

#include "orchestration_api.h"

namespace {
struct ScopeTransfer {
    const OrchOps *ops;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread::id delayed_thread;
    bool entered{false};
    bool released{false};
    ScopeMode observed{ScopeMode::AUTO};
    std::atomic<bool> fatal{false};
    std::atomic<int> begins{0};
    std::atomic<int> ends{0};
    std::atomic<int> wrong_modes{0};
    std::vector<ScopeMode> modes;
};
OrchContext *bound_context = nullptr;
extern "C" OrchContext *framework_current_orch_context() { return bound_context; }
extern "C" void framework_bind_orch_context(OrchContext *ctx) { bound_context = ctx; }

thread_local ScopeMode requested_mode = ScopeMode::AUTO;
bool is_fatal(OrchContext *ctx) { return reinterpret_cast<ScopeTransfer *>(ctx)->fatal.load(); }
void scope_end(OrchContext *ctx) { reinterpret_cast<ScopeTransfer *>(ctx)->ends++; }
void scope_begin(OrchContext *ctx, ScopeMode mode) {
    auto &state = *reinterpret_cast<ScopeTransfer *>(ctx);
    const bool delayed = std::this_thread::get_id() == state.delayed_thread;
    if (delayed) {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.entered = true;
        state.cv.notify_all();
        state.cv.wait(lock, [&] {
            return state.released;
        });
    }
    if (delayed) state.observed = mode;
    if (mode != requested_mode) state.wrong_modes++;
    state.begins++;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.modes.push_back(mode);
}
const OrchOps ops = {.scope_begin = scope_begin, .scope_end = scope_end, .is_fatal = is_fatal};

void check_interleaving(bool explicit_begin, ScopeMode competing_mode) {
    ScopeTransfer state{};
    state.ops = &ops;
    framework_bind_orch_context(reinterpret_cast<OrchContext *>(&state));
    std::thread worker([&] {
        state.delayed_thread = std::this_thread::get_id();
        requested_mode = ScopeMode::MANUAL;
        if (explicit_begin) {
            rt_scope_begin(ScopeMode::MANUAL);
            rt_scope_end();
        } else {
            ScopeGuard guard(ScopeMode::MANUAL);
        }
    });
    bool entered;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        entered = state.cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return state.entered;
        });
    }
    if (entered) {
        requested_mode = competing_mode;
        ScopeGuard competing(competing_mode);
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.released = true;
    }
    state.cv.notify_all();
    worker.join();
    framework_bind_orch_context(nullptr);
    ASSERT_TRUE(entered);
    EXPECT_EQ(state.observed, ScopeMode::MANUAL);
    EXPECT_EQ(state.wrong_modes.load(), 0);
    EXPECT_EQ(state.begins.load(), 2);
    EXPECT_EQ(state.ends.load(), 2);
}
}  // namespace

TEST(HbgScopeMode, GuardPreservesManualAcrossConcurrentAuto) { check_interleaving(false, ScopeMode::AUTO); }
TEST(HbgScopeMode, ExplicitBeginPreservesManualAcrossConcurrentAuto) { check_interleaving(true, ScopeMode::AUTO); }
TEST(HbgScopeMode, GuardPreservesManualAcrossConcurrentManual) { check_interleaving(false, ScopeMode::MANUAL); }

TEST(HbgScopeMode, ConcurrentRecordersKeepTheirRequestedModes) {
    ScopeTransfer state{};
    state.ops = &ops;
    framework_bind_orch_context(reinterpret_cast<OrchContext *>(&state));
    std::array<std::thread, 4> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] {
            for (int j = 0; j < 256; ++j) {
                requested_mode = (i + j) % 2 == 0 ? ScopeMode::AUTO : ScopeMode::MANUAL;
                if (j % 2 == 0) {
                    ScopeGuard guard(requested_mode);
                } else {
                    rt_scope_begin(requested_mode);
                    rt_scope_end();
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
    framework_bind_orch_context(nullptr);
    EXPECT_EQ(state.wrong_modes.load(), 0);
    EXPECT_EQ(state.begins.load(), 1024);
    EXPECT_EQ(state.ends.load(), 1024);
}

TEST(HbgScopeMode, NestedGuardsAndExplicitBeginForwardEachMode) {
    ScopeTransfer state{};
    state.ops = &ops;
    framework_bind_orch_context(reinterpret_cast<OrchContext *>(&state));
    requested_mode = ScopeMode::AUTO;
    {
        ScopeGuard outer;
        requested_mode = ScopeMode::MANUAL;
        {
            ScopeGuard manual(ScopeMode::MANUAL);
            ScopeGuard nested(ScopeMode::MANUAL);
        }
        requested_mode = ScopeMode::AUTO;
        rt_scope_begin();
        rt_scope_end();
    }
    framework_bind_orch_context(nullptr);
    EXPECT_EQ(
        state.modes, (std::vector<ScopeMode>{ScopeMode::AUTO, ScopeMode::MANUAL, ScopeMode::MANUAL, ScopeMode::AUTO})
    );
    EXPECT_EQ(state.wrong_modes.load(), 0);
    EXPECT_EQ(state.begins.load(), 4);
    EXPECT_EQ(state.ends.load(), 4);
}

TEST(HbgScopeMode, ExceptionUnwindingClosesTheEnteredScope) {
    ScopeTransfer state{};
    state.ops = &ops;
    framework_bind_orch_context(reinterpret_cast<OrchContext *>(&state));
    requested_mode = ScopeMode::MANUAL;
    EXPECT_THROW(
        {
            ScopeGuard guard(ScopeMode::MANUAL);
            throw std::runtime_error("body failed");
        },
        std::runtime_error
    );
    framework_bind_orch_context(nullptr);
    EXPECT_EQ(state.begins.load(), 1);
    EXPECT_EQ(state.ends.load(), 1);
    EXPECT_EQ(state.wrong_modes.load(), 0);
}

TEST(HbgScopeMode, FatalBeforeEntrySuppressesGuardAndExplicitCallbacks) {
    ScopeTransfer state{};
    state.ops = &ops;
    state.fatal = true;
    framework_bind_orch_context(reinterpret_cast<OrchContext *>(&state));
    {
        ScopeGuard guard(ScopeMode::MANUAL);
        rt_scope_begin(ScopeMode::MANUAL);
        rt_scope_end();
    }
    framework_bind_orch_context(nullptr);
    EXPECT_EQ(state.begins.load(), 0);
    EXPECT_EQ(state.ends.load(), 0);
}

TEST(HbgScopeMode, FatalInBodyKeepsScopeTeardownSuppressed) {
    ScopeTransfer state{};
    state.ops = &ops;
    framework_bind_orch_context(reinterpret_cast<OrchContext *>(&state));
    requested_mode = ScopeMode::MANUAL;
    {
        ScopeGuard guard(ScopeMode::MANUAL);
        state.fatal = true;
    }
    framework_bind_orch_context(nullptr);
    EXPECT_EQ(state.begins.load(), 1);
    EXPECT_EQ(state.ends.load(), 0);
}

namespace {
struct ScopeExchange {
    const OrchOps *ops;
    std::mutex mutex;
    std::condition_variable cv;
    std::array<bool, 2> entered{};
    std::array<bool, 2> released{};
    std::array<ScopeMode, 2> observed{ScopeMode::MANUAL, ScopeMode::AUTO};
    std::atomic<int> ends{0};
};

bool exchange_is_fatal(OrchContext *) { return false; }
void exchange_end(OrchContext *ctx) { reinterpret_cast<ScopeExchange *>(ctx)->ends++; }
void exchange_begin(OrchContext *ctx, ScopeMode mode) {
    auto &state = *reinterpret_cast<ScopeExchange *>(ctx);
    const size_t slot = requested_mode == ScopeMode::AUTO ? 0 : 1;
    std::unique_lock<std::mutex> lock(state.mutex);
    state.entered[slot] = true;
    state.cv.notify_all();
    state.cv.wait(lock, [&] {
        return state.released[slot];
    });
    state.observed[slot] = mode;
}
const OrchOps exchange_ops = {.scope_begin = exchange_begin, .scope_end = exchange_end, .is_fatal = exchange_is_fatal};
}  // namespace

TEST(HbgScopeMode, BothModesSurviveWhenAutoConsumesBeforeManual) {
    ScopeExchange state{};
    state.ops = &exchange_ops;
    framework_bind_orch_context(reinterpret_cast<OrchContext *>(&state));
    std::thread automatic([&] {
        requested_mode = ScopeMode::AUTO;
        rt_scope_begin(ScopeMode::AUTO);
        rt_scope_end();
    });
    bool auto_entered;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        auto_entered = state.cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return state.entered[0];
        });
    }
    std::thread manual([&] {
        requested_mode = ScopeMode::MANUAL;
        ScopeGuard guard(ScopeMode::MANUAL);
    });
    bool manual_entered;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        manual_entered = state.cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return state.entered[1];
        });
        state.released[0] = true;
    }
    state.cv.notify_all();
    automatic.join();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.released[1] = true;
    }
    state.cv.notify_all();
    manual.join();
    framework_bind_orch_context(nullptr);
    ASSERT_TRUE(auto_entered);
    ASSERT_TRUE(manual_entered);
    EXPECT_EQ(state.observed[0], ScopeMode::AUTO);
    EXPECT_EQ(state.observed[1], ScopeMode::MANUAL);
    EXPECT_EQ(state.ends.load(), 2);
}
