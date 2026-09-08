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

#include "aicpu/profiler_device_engine.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace {

constexpr uint32_t kReadyQueueSize = 4;
constexpr uint32_t kSlotCount = 4;

struct FakeBuffer {
    volatile uint32_t count = 0;
};

struct FakeFreeQueue {
    volatile uint64_t buffer_ptrs[kSlotCount] = {};
    volatile uint32_t head = 0;
    volatile uint32_t tail = 0;
};

struct FakeHeader {
    struct Entry {
        uint64_t buffer_ptr = 0;
        uint32_t buffer_seq = 0;
    };
    Entry queues[PLATFORM_MAX_AICPU_THREADS][kReadyQueueSize] = {};
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS] = {};
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS] = {};
};

struct FakeState {
    FakeFreeQueue free_queue;
    uint64_t current_ptr = 0;
    uint32_t current_seq = 0;
    uint32_t dropped = 0;
};

struct FakeModule {
    struct Context {
        FakeHeader *header;
        int thread_idx;
        FakeBuffer **current_buf;
    };

    using DataHeader = FakeHeader;
    using State = FakeState;
    using FreeQueue = FakeFreeQueue;
    using Buffer = FakeBuffer;

    static constexpr uint32_t kReadyQueueSize = ::kReadyQueueSize;
    static constexpr uint32_t kSlotCount = ::kSlotCount;
    static constexpr uint64_t kBackpressureWaitCycles = 1000;

    static DataHeader *header(Context ctx) { return ctx.header; }
    static int ready_thread(Context ctx) { return ctx.thread_idx; }
    static FreeQueue *free_queue(State *state) { return &state->free_queue; }
    static uint64_t current_ptr(State *state) { return state->current_ptr; }
    static void set_current_ptr(State *state, uint64_t ptr) { state->current_ptr = ptr; }
    static uint32_t current_seq(State *state) { return state->current_seq; }
    static void set_current_seq(State *state, uint32_t seq) { state->current_seq = seq; }
    static uint32_t count(Buffer *buffer) { return buffer->count; }
    static void set_count(Buffer *buffer, uint32_t count) { buffer->count = count; }

    static void write_ready_entry(Context ctx, uint32_t tail, uint64_t ptr, uint32_t seq) {
        ctx.header->queues[ctx.thread_idx][tail].buffer_ptr = ptr;
        ctx.header->queues[ctx.thread_idx][tail].buffer_seq = seq;
    }
    static void account_dropped(Context, State *state, uint32_t count) { state->dropped += count; }
    static void on_pop_success(Context ctx, State *, Buffer *buffer) { *ctx.current_buf = buffer; }
    static void on_current_cleared(Context ctx, State *) { *ctx.current_buf = nullptr; }
    static void on_no_replacement(Context, State *) {}
    static void on_null_free_slot(Context, State *) {}
    static void on_enqueue_failed(Context, State *, Buffer *) {}
    static void on_switch_complete(Context, State *, Buffer *) {}
};

using Engine = profiling_device::DeviceProfilerEngine<FakeModule>;

FakeModule::Context context(FakeHeader *header, FakeBuffer **current) {
    return FakeModule::Context{header, 0, current};
}

TEST(ProfilerDeviceEngineTest, SwitchPublishesOldBufferAndPopsReplacement) {
    FakeHeader header;
    FakeState state;
    FakeBuffer old_buffer;
    FakeBuffer new_buffer;
    FakeBuffer *current = &old_buffer;
    state.current_ptr = reinterpret_cast<uint64_t>(&old_buffer);
    state.current_seq = 7;
    old_buffer.count = 3;
    state.free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(&new_buffer);
    state.free_queue.tail = 1;

    Engine::switch_buffer(context(&header, &current), &state);

    EXPECT_EQ(header.queue_tails[0], 1u);
    EXPECT_EQ(header.queues[0][0].buffer_ptr, reinterpret_cast<uint64_t>(&old_buffer));
    EXPECT_EQ(header.queues[0][0].buffer_seq, 7u);
    EXPECT_EQ(state.free_queue.head, 1u);
    EXPECT_EQ(state.current_ptr, reinterpret_cast<uint64_t>(&new_buffer));
    EXPECT_EQ(state.current_seq, 8u);
    EXPECT_EQ(current, &new_buffer);
    EXPECT_EQ(new_buffer.count, 0u);
}

// A full ready queue is the only way the push gate can fail, and it fails
// BEFORE publishing: the tail never advances, so the caller still owns the
// buffer and must account its records dropped rather than hand it to the host.
TEST(ProfilerDeviceEngineTest, SwitchDropsWithoutPublishingWhenReadyQueueStaysFull) {
    FakeHeader header;
    FakeState state;
    FakeBuffer old_buffer;
    FakeBuffer new_buffer;
    FakeBuffer *current = &old_buffer;
    state.current_ptr = reinterpret_cast<uint64_t>(&old_buffer);
    state.current_seq = 7;
    old_buffer.count = 3;
    // head == tail + 1 leaves no slot: next_tail wraps onto head forever.
    header.queue_tails[0] = 0;
    header.queue_heads[0] = 1;
    state.free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(&new_buffer);
    state.free_queue.tail = 1;

    Engine::switch_buffer(context(&header, &current), &state);

    EXPECT_EQ(header.queue_tails[0], 0u);
    EXPECT_EQ(header.queues[0][0].buffer_ptr, 0u);
    EXPECT_EQ(state.dropped, 3u);
    EXPECT_EQ(old_buffer.count, 0u);
    // The replacement was never claimed: switch_buffer returns on the enqueue
    // failure, so the buffer the caller holds is the one it came in with.
    EXPECT_EQ(state.free_queue.head, 0u);
    EXPECT_EQ(state.current_ptr, reinterpret_cast<uint64_t>(&old_buffer));
    EXPECT_EQ(state.current_seq, 7u);
}

TEST(ProfilerDeviceEngineTest, SwitchClearsCurrentBufferWhenReplacementIsUnavailable) {
    FakeHeader header;
    FakeState state;
    FakeBuffer old_buffer;
    FakeBuffer *current = &old_buffer;
    state.current_ptr = reinterpret_cast<uint64_t>(&old_buffer);
    state.current_seq = 7;
    old_buffer.count = 3;
    // Free queue left empty and nothing refills it, so the pop gate exhausts its
    // own budget — the wait is bounded even with no host handshake at all.

    Engine::switch_buffer(context(&header, &current), &state);

    EXPECT_EQ(header.queue_tails[0], 1u);
    EXPECT_EQ(state.current_ptr, 0u);
    EXPECT_EQ(state.current_seq, 8u);
    EXPECT_EQ(current, nullptr);
}

TEST(ProfilerDeviceEngineTest, PopFreeSupportsRecoveryAfterSwitchFailure) {
    FakeHeader header;
    FakeState state;
    FakeBuffer new_buffer;
    FakeBuffer *current = nullptr;
    state.current_seq = 8;
    state.free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(&new_buffer);
    state.free_queue.tail = 1;

    auto buffer = Engine::pop_free(context(&header, &current), &state, state.current_seq);

    EXPECT_EQ(buffer, &new_buffer);
    EXPECT_EQ(current, &new_buffer);
    EXPECT_EQ(state.current_ptr, reinterpret_cast<uint64_t>(&new_buffer));
    EXPECT_EQ(state.current_seq, 8u);
    EXPECT_EQ(new_buffer.count, 0u);
}

TEST(ProfilerDeviceEngineTest, PopFreeReturnsAtOnceWhenAFreeSlotIsAvailable) {
    FakeHeader header;
    FakeState state;
    FakeBuffer new_buffer;
    FakeBuffer *current = nullptr;
    state.free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(&new_buffer);
    state.free_queue.tail = 1;

    auto buffer = Engine::pop_free(context(&header, &current), &state, 0);

    EXPECT_EQ(buffer, &new_buffer);
    EXPECT_EQ(state.free_queue.head, 1u);
}

// The pop gate must give up on its own deadline when nothing ever refills the
// queue — the property `docs/dfx/backpressure-design.md` states as "each gate is
// bounded by Module::kBackpressureWaitCycles". One budget spans every iteration,
// so a host that stops making progress cannot hold the lane forever.
//
// Every object the worker touches is static on purpose: a regression leaves it
// spinning forever, and a detached thread must not reference a freed frame. That
// storage therefore outlives the test body, so each invocation has to establish
// the state it asserts on rather than inherit it from the previous one.
TEST(ProfilerDeviceEngineTest, PopGateGivesUpWhenNothingEverRefillsTheQueue) {
    static FakeFreeQueue free_queue;
    static std::atomic<bool> finished{false};
    static std::atomic<bool> acquired{true};

    free_queue.head = 0;
    free_queue.tail = 0;
    finished.store(false);
    acquired.store(true);

    ASSERT_EQ(free_queue.head, free_queue.tail);  // the gate finds no slot

    std::thread worker([] {
        uint32_t head = 0;
        uint32_t tail = 0;
        acquired.store(Engine::wait_for_free_queue_entry(&free_queue, &head, &tail));
        finished.store(true);
    });
    worker.detach();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!finished.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(finished.load()) << "pop gate spun with no deadline of its own";
    EXPECT_FALSE(acquired.load());
}

TEST(ProfilerDeviceEngineTest, TryPopFreeReturnsImmediatelyWhenStartupQueueIsEmpty) {
    FakeHeader header;
    FakeState state;
    FakeBuffer *current = nullptr;

    auto buffer = Engine::try_pop_free(context(&header, &current), &state, 0);

    EXPECT_EQ(buffer, nullptr);
    EXPECT_EQ(current, nullptr);
    EXPECT_EQ(state.current_ptr, 0u);
    EXPECT_EQ(state.current_seq, 0u);
}

}  // namespace
