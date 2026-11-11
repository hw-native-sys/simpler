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

#pragma once

#define __CPU_SIM 1
#define PTO_URMA_SUPPORTED 1

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace pto::comm {
enum class DmaEngine : uint8_t;
}

namespace {
uint32_t total_wait_calls = 0;
struct TestSession {
    bool valid{true};
    pto::comm::DmaEngine engine{static_cast<pto::comm::DmaEngine>(1)};
    uint8_t *contextGm{};
    uint32_t destRankId{1};
    uint32_t qpIdxBase{2};
    struct {
        struct {
            uint32_t qpCount{2};
        } urma;
    } cfg;
};

struct TestEvent {
    uint64_t handle{(uint64_t{1} << 32) | 4};
    pto::comm::DmaEngine engine{static_cast<pto::comm::DmaEngine>(1)};
    uint32_t urmaTargetCqe{};
    std::array<uint32_t, 8> urmaTargetBbPerJetty{4, 9};
    std::array<uint32_t, 8> urmaTargetCqePerJetty{3, 7};
    uint16_t urmaJettyBase{2};
    uint16_t urmaJettyCount{2};
    bool postFailed{};
    mutable uint32_t wait_calls{};

    bool Wait(const TestSession &) const {
        ++wait_calls;
        ++total_wait_calls;
        return false;
    }
};

}  // namespace

// Host-only doubles replace the ISA session and posting boundary. Adapter
// validation, slicing, posting and completion registration remain real.
namespace pto {
inline constexpr int64_t DYNAMIC = -1;

template <int64_t... Dims>
struct Shape {
    std::array<int64_t, 5> values;
    Shape(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e) :
        values{a, b, c, d, e} {}
};

template <int64_t... Dims>
using Stride = Shape<Dims...>;

template <typename T>
struct TestAddress {
    uint64_t bytes;
    TestAddress operator+(uint64_t elements) const { return {bytes + elements * sizeof(T)}; }
};

template <typename T, typename ShapeT, typename StrideT>
struct GlobalTensor {
    using RawDType = T;
    TestAddress<T> address;
    ShapeT shape;
    StrideT stride;
    GlobalTensor(TestAddress<T> ptr, ShapeT shp, StrideT steps) :
        address(ptr),
        shape(shp),
        stride(steps) {}
    TestAddress<T> data() const { return address; }
    int64_t GetShape(int dim) const { return shape.values[dim]; }
    int64_t GetStride(int dim) const { return stride.values[dim]; }
};

namespace comm {
using AsyncSession = TestSession;
using AsyncEvent = TestEvent;

template <DmaEngine Engine>
bool BuildAsyncSession(uint8_t *, uint32_t, AsyncSession &) = delete;
namespace detail {
template <DmaEngine Engine>
void ActivateEngineState(AsyncSession &session);
}  // namespace detail
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TGET_ASYNC(Dst dst, Src src, const AsyncSession &session);
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TPUT_ASYNC(Dst dst, Src src, const AsyncSession &session);
namespace urma {
inline uint64_t UrmaPeerMrBaseAddr(uint8_t *, uint32_t) { return 0; }
}  // namespace urma
}  // namespace comm
}  // namespace pto

// The adapter must not depend on the implicit CCE block identity.
uint32_t get_block_idx() = delete;

#include "urma/urma_request.h"
#include <pto/comm/async/urma/urma_types.hpp>

namespace {
struct Submission {
    bool put;
    uint64_t dst;
    uint64_t src;
    std::array<int64_t, 5> shape;
    std::array<int64_t, 5> stride;
};
struct PostScript {
    std::vector<TestEvent> events;
    std::vector<Submission> submissions;
    uint32_t activation_calls{};
    bool post_from_queue{};
};
PostScript *active_script = nullptr;
}  // namespace

namespace pto::comm {
namespace detail {
template <DmaEngine Engine>
void ActivateEngineState(AsyncSession &session) {
    static_assert(Engine == DmaEngine::URMA);
    ++active_script->activation_calls;
    session.engine = Engine;
    session.cfg.urma.qpCount = 0;
}
}  // namespace detail
template <typename Dst, typename Src>
AsyncEvent post_for_test(bool put, Dst dst, Src src, const AsyncSession &session) {
    auto &script = *active_script;
    script.submissions.push_back({put, dst.data().bytes, src.data().bytes, dst.shape.values, dst.stride.values});
    EXPECT_EQ(src.shape.values, dst.shape.values);
    EXPECT_EQ(src.stride.values, dst.stride.values);
    if (script.post_from_queue) {
        auto *info = reinterpret_cast<urma::UrmaInfo *>(session.contextGm);
        auto *queues = reinterpret_cast<urma::UrmaWQCtx *>(info->sqPtr);
        TestEvent event{};
        event.urmaJettyBase = session.qpIdxBase;
        event.urmaJettyCount = session.cfg.urma.qpCount;
        const uint64_t bytes = static_cast<uint64_t>(src.shape.values[4]) * sizeof(typename Src::RawDType);
        const uint64_t wqes = simpler::urma::chunk_count(bytes);
        for (uint32_t j = 0; j < event.urmaJettyCount; ++j) {
            uint32_t row = 0;
            EXPECT_TRUE(
                simpler::urma::queue_row_for(session.contextGm, session.destRankId, session.qpIdxBase + j, row)
            );
            auto &queue = queues[row];
            auto *head = reinterpret_cast<uint32_t *>(queue.headAddr);
            // Non-rail ISA posts rotate within one call and restart at jetty zero on the next call.
            const uint32_t posted = wqes / event.urmaJettyCount + (j < wqes % event.urmaJettyCount ? 1 : 0);
            if (posted != 0) {
                *head += posted;
                queue.submittedWqeCount += posted;
                *reinterpret_cast<uint32_t *>(queue.dbAddr) = *head;
            }
            event.urmaTargetBbPerJetty[j] = *head;
            event.urmaTargetCqePerJetty[j] = queue.submittedWqeCount;
        }
        event.handle = (static_cast<uint64_t>(session.destRankId) << 32) | event.urmaTargetBbPerJetty[0];
        return event;
    }
    const auto index = script.submissions.size() - 1;
    EXPECT_LT(index, script.events.size());
    return script.events.at(index);
}
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TGET_ASYNC(Dst dst, Src src, const AsyncSession &session) {
    static_assert(Engine == DmaEngine::URMA);
    return post_for_test(false, dst, src, session);
}
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TPUT_ASYNC(Dst dst, Src src, const AsyncSession &session) {
    static_assert(Engine == DmaEngine::URMA);
    return post_for_test(true, dst, src, session);
}
}  // namespace pto::comm
