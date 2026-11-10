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

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <map>

#include "host/ccu_device.h"
#include "host/ccu_execution_registry.h"

namespace {
using Wire = std::array<uint64_t, 32>;
struct Owner {
    std::atomic<int> refs{0};
    bool fail_release{};
};
int Retain(void *p) {
    ++static_cast<Owner *>(p)->refs;
    return 0;
}
int Release(void *p) {
    auto &o = *static_cast<Owner *>(p);
    if (o.fail_release) return -1;
    --o.refs;
    return 0;
}
struct Endpoint;
struct Rendezvous {
    std::array<Wire, 2> wires{};
    std::array<bool, 2> present{};
    uint32_t readers{};
};
struct Group {
    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::pair<uint64_t, uint32_t>, Rendezvous> rounds;
    int fail_phase{};
    int wrong_rank_phase{};
    bool reenter{};
};
struct Endpoint : Owner {
    Group *group{};
    uint32_t rank{};
    CcuContextId context{};
    int reentry_status{};
    CcuDeviceObservation observation;
    std::array<uint64_t, 8> args{};
    uint32_t launches{}, triggers{}, closes{};
    bool fail_launch{}, fail_trigger{}, fail_close{}, fail_token{}, fail_open{};
};
int Exchange(
    void *p, uint64_t, uint64_t collective, uint32_t phase, const void *local, uint32_t size, void *ranked,
    uint32_t stride, uint32_t ranks, uint32_t timeout
) {
    auto &e = *static_cast<Endpoint *>(p);
    auto &g = *e.group;
    if (g.reenter && phase == 2) {
        CcuHostResultV1 nested{};
        e.reentry_status = ccu_context_close(e.context, 0, &nested);
    }
    if (g.fail_phase == static_cast<int>(phase)) return -1;
    if (size != sizeof(Wire) || stride != sizeof(Wire) || ranks != 2) return -1;
    std::unique_lock<std::mutex> lock(g.mutex);
    const auto key = std::make_pair(collective, phase);
    auto &r = g.rounds[key];
    if (r.present[e.rank]) return -1;
    std::memcpy(r.wires[e.rank].data(), local, size);
    r.present[e.rank] = true;
    g.cv.notify_all();
    if (!g.cv.wait_for(lock, std::chrono::milliseconds(timeout), [&] {
            return r.present[0] && r.present[1];
        }))
        return -1;
    auto wires = r.wires;
    if (g.wrong_rank_phase == static_cast<int>(phase)) wires[1][5] = 0;
    std::memcpy(ranked, wires.data(), sizeof(wires));
    if (++r.readers == 2) g.rounds.erase(key);
    return 0;
}
int Abort(void *, uint64_t, uint64_t, uint32_t, int32_t) { return 0; }

// This backend models SDK outcomes only; it does not execute or validate CCU hardware instructions.
// 测试后端仅注入 SDK 返回和完成观测，不模拟或证明 CCU 硬件执行。
class TestDevice final : public CcuDevice {
public:
    void Open(const CcuCreateOptionsV1 &o, uint64_t context, uint64_t generation) override {
        e = static_cast<Endpoint *>(o.bootstrap_owner);
        if (e->fail_open) throw CcuSdkError(71, "injected Open");
        auto &r = e->observation.record;
        r.abi_version = 1;
        r.struct_size = sizeof(r);
        r.context_id = context;
        r.context_generation = generation;
    }
    uint64_t Token(uint64_t base, uint64_t) override {
        if (e->fail_token) throw CcuSdkError(72, "injected Token");
        return base ^ 0xabcdef;
    }
    uint64_t Program(uint32_t root) const override { return 100 + root; }
    void Launch(uint32_t, const std::array<uint64_t, 8> &args) override {
        ++e->launches;
        e->args = args;
        if (e->fail_launch) throw CcuSdkError(73, "injected Launch");
        e->observation.trigger_drained = e->observation.ccu_drained = false;
    }
    void Trigger() override {
        ++e->triggers;
        if (e->fail_trigger) throw CcuSdkError(74, "injected Trigger");
    }
    CcuDeviceObservation Observe() override { return e->observation; }
    uint64_t Gate() const override { return 0x800000; }
    uint64_t Completion() const override { return 0x900000; }
    void Close() override {
        if (e->fail_close) throw CcuSdkError(75, "injected Close");
        ++e->closes;
    }

private:
    Endpoint *e{};
};

struct Pair {
    Group group;
    std::array<Endpoint, 2> endpoints;
    std::array<Owner, 2> inputs, outputs;
    std::array<CcuContextId, 2> contexts{};
    std::array<CcuWindowId, 2> in{}, out{};
    std::array<CcuRequestId, 2> requests{};
    std::array<CcuTriggerInfoV1, 2> triggers{};
    std::array<CcuHostResultV1, 2> results{};
    std::array<uint64_t, 2> root{};
    CcuOwnerOpsV1 owners{1, sizeof(CcuOwnerOpsV1), Retain, Release};
    CcuBootstrapOpsV1 bootstrap{1, sizeof(CcuBootstrapOpsV1), Exchange, Abort};

    void Create() {
        for (uint32_t rank = 0; rank < 2; ++rank) {
            auto &e = endpoints[rank];
            e.group = &group;
            e.rank = rank;
            CcuCreateOptionsV1 options{1,
                                       sizeof(CcuCreateOptionsV1),
                                       static_cast<int32_t>(rank),
                                       rank,
                                       2,
                                       16,
                                       17,
                                       root.data(),
                                       nullptr,
                                       &bootstrap,
                                       &e,
                                       &owners};
            ASSERT_EQ(ccu_context_create(&options, &contexts[rank], &results[rank]), 0);
            e.context = contexts[rank];
            for (uint32_t which = 0; which < 2; ++which) {
                CcuWindowImportV1 window{};
                window.abi_version = 1;
                window.struct_size = sizeof(window);
                window.device_id = rank;
                window.allocation_id = 1000 + rank * 2 + which;
                window.allocation_generation = 1;
                window.allocation_base_va = (rank + 1) * 0x100000 + which * 0x10000;
                window.allocation_size = 8192;
                window.window_offset = 128;
                window.window_size = 4096;
                window.allocation_owner = which == 0 ? &inputs[rank] : &outputs[rank];
                window.allocation_owner_ops = &owners;
                ASSERT_EQ(
                    ccu_window_register(contexts[rank], &window, which == 0 ? &in[rank] : &out[rank], &results[rank]), 0
                );
            }
        }
    }
    template <typename Fn>
    std::array<int, 2> Both(Fn fn) {
        auto first = std::async(std::launch::async, [&] {
            return fn(0);
        });
        auto second = std::async(std::launch::async, [&] {
            return fn(1);
        });
        return {first.get(), second.get()};
    }
    void Bind() {
        auto rc = Both([&](uint32_t rank) {
            CcuWindowBindingV1 binding{1, sizeof(CcuWindowBindingV1), in[rank], out[rank], 1, {0, 0, 0}};
            return ccu_bind_windows(contexts[rank], &binding, &results[rank]);
        });
        ASSERT_EQ(rc, (std::array<int, 2>{0, 0}));
    }
    std::array<int, 2> Prepare(uint64_t collective = 1, bool mismatch = false, uint32_t root_rank = 0) {
        return Both([&](uint32_t rank) {
            CcuRequestSpecV1 spec{
                1,
                sizeof(CcuRequestSpecV1),
                CCU_OPERATION_BROADCAST,
                CCU_DATA_TYPE_FLOAT32,
                collective,
                0,
                rank * 64,
                mismatch ? 32 + rank : 32,
                root_rank,
                0
            };
            return ccu_prepare(contexts[rank], &spec, &requests[rank], &results[rank]);
        });
    }
    void Arm() {
        auto rc = Both([&](uint32_t rank) {
            return ccu_arm(requests[rank], &triggers[rank], &results[rank]);
        });
        ASSERT_EQ(rc, (std::array<int, 2>{0, 0}));
    }
    void Trigger() {
        for (uint32_t rank = 0; rank < 2; ++rank)
            ASSERT_EQ(ccu_trigger_host(requests[rank], &results[rank]), 0);
    }
    void Complete() {
        for (auto &e : endpoints) {
            e.observation.record.completed_seq = e.args[7];
            e.observation.trigger_drained = e.observation.ccu_drained = true;
        }
    }
    void Retire() {
        auto rc = Both([&](uint32_t rank) {
            return ccu_retire(requests[rank], 1000, &results[rank]);
        });
        ASSERT_EQ(rc, (std::array<int, 2>{0, 0}));
    }
    void Close() {
        for (uint32_t rank = 0; rank < 2; ++rank) {
            ASSERT_EQ(ccu_context_close(contexts[rank], 100, &results[rank]), 0);
            EXPECT_EQ(inputs[rank].refs, 0);
            EXPECT_EQ(outputs[rank].refs, 0);
            EXPECT_EQ(endpoints[rank].refs, 0);
        }
    }
};
// Poisoned submissions intentionally keep callback owners alive until process exit.
std::vector<std::shared_ptr<Pair>> retained_pairs;
}  // namespace

std::shared_ptr<CcuDevice> ccu_make_device() { return std::make_shared<TestDevice>(); }
bool ccu_device_available() { return true; }
uint32_t ccu_device_root_info_size() { return 16; }
void ccu_device_root_info(int, void *bytes) { std::memset(bytes, 0, 16); }

TEST(CcuExecutionApi, TwoRanksUseIndependentTokensAndRemoteOffsetAcrossRootsAndRounds) {
    Pair p;
    p.Create();
    p.Bind();
    for (uint32_t root = 0; root < 2; ++root) {
        ASSERT_EQ(p.Prepare(root + 1, false, root), (std::array<int, 2>{0, 0}));
        p.Arm();
        EXPECT_EQ(p.endpoints[0].args[4], 0x210000U + 128 + 64);
        EXPECT_EQ(p.endpoints[1].args[4], 0x110000U + 128);
        EXPECT_NE(p.endpoints[0].args[2], p.endpoints[0].args[3]);
        EXPECT_EQ(p.endpoints[0].args[7], root + 1);
        p.Trigger();
        p.Complete();
        p.Retire();
    }
    p.Close();
}
TEST(CcuExecutionApi, DataReadyDoesNotMeanDrainedAndTimeoutRetainsOwnersThenRecovers) {
    Pair p;
    p.Create();
    p.Bind();
    ASSERT_EQ(p.Prepare(), (std::array<int, 2>{0, 0}));
    p.Arm();
    p.Trigger();
    auto &e = p.endpoints[0];
    e.observation.record.completed_seq = e.args[7];
    EXPECT_EQ(ccu_poll(p.requests[0], &p.results[0]), CCU_STATUS_PENDING);
    EXPECT_EQ(p.results[0].data_ready, 1U);
    EXPECT_EQ(p.results[0].drained, 0U);
    EXPECT_EQ(ccu_wait(p.requests[0], 0, &p.results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(ccu_context_close(p.contexts[0], 0, &p.results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(p.inputs[0].refs, 1);
    EXPECT_EQ(p.outputs[0].refs, 1);
    p.Complete();
    p.Retire();
    p.Close();
}
TEST(CcuExecutionApi, CancelConsumesSequenceAndInvalidatesOldHandle) {
    Pair p;
    p.Create();
    p.Bind();
    ASSERT_EQ(p.Prepare(), (std::array<int, 2>{0, 0}));
    const auto old = p.requests;
    EXPECT_EQ(
        p.Both([&](uint32_t rank) {
            return ccu_cancel_prepared(p.requests[rank], &p.results[rank]);
        }),
        (std::array<int, 2>{0, 0})
    );
    EXPECT_EQ(ccu_trigger_host(old[0], &p.results[0]), CCU_STATUS_STALE_HANDLE);
    ASSERT_EQ(p.Prepare(2), (std::array<int, 2>{0, 0}));
    p.Arm();
    EXPECT_EQ(p.endpoints[0].args[7], 2U);
    p.Trigger();
    p.Complete();
    p.Retire();
    p.Close();
}
TEST(CcuExecutionApi, RejectsMismatchedCountAndWrongRankBeforeLaunch) {
    Pair p;
    p.Create();
    p.Bind();
    EXPECT_EQ(p.Prepare(1, true), (std::array<int, 2>{CCU_STATUS_PROTOCOL_ERROR, CCU_STATUS_PROTOCOL_ERROR}));
    p.group.wrong_rank_phase = 2;
    EXPECT_EQ(p.Prepare(2), (std::array<int, 2>{CCU_STATUS_PROTOCOL_ERROR, CCU_STATUS_PROTOCOL_ERROR}));
    EXPECT_EQ(p.endpoints[0].launches, 0U);
    p.Close();
}
TEST(CcuExecutionApi, PartialLaunchNeverAllowsTriggerOrRelease) {
    auto p = std::make_shared<Pair>();
    retained_pairs.push_back(p);
    p->Create();
    p->Bind();
    ASSERT_EQ(p->Prepare(), (std::array<int, 2>{0, 0}));
    p->endpoints[1].fail_launch = true;
    auto rc = p->Both([&](uint32_t rank) {
        return ccu_arm(p->requests[rank], &p->triggers[rank], &p->results[rank]);
    });
    EXPECT_EQ(rc, (std::array<int, 2>{CCU_STATUS_RETAINED, CCU_STATUS_RETAINED}));
    EXPECT_EQ(ccu_trigger_host(p->requests[0], &p->results[0]), CCU_STATUS_INVALID_STATE);
    EXPECT_EQ(p->endpoints[0].triggers, 0U);
    EXPECT_EQ(ccu_context_close(p->contexts[0], 0, &p->results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(p->inputs[0].refs, 1);
}
TEST(CcuExecutionApi, RejectsDuplicateTriggerAndKeepsRequestUntilBothRetired) {
    Pair p;
    p.Create();
    p.Bind();
    ASSERT_EQ(p.Prepare(), (std::array<int, 2>{0, 0}));
    p.Arm();
    p.Trigger();
    EXPECT_EQ(ccu_trigger_host(p.requests[0], &p.results[0]), CCU_STATUS_INVALID_STATE);
    p.Complete();
    p.group.fail_phase = 5;
    EXPECT_EQ(
        p.Both([&](uint32_t rank) {
            return ccu_retire(p.requests[rank], 10, &p.results[rank]);
        }),
        (std::array<int, 2>{CCU_STATUS_RETAINED, CCU_STATUS_RETAINED})
    );
    EXPECT_EQ(p.inputs[0].refs, 1);
    p.group.fail_phase = 0;
    p.Retire();
    p.Close();
}
TEST(CcuExecutionApi, CleanupAndAllocationReleaseFailuresAreRetryable) {
    Pair p;
    p.Create();
    p.Bind();
    p.endpoints[0].fail_close = true;
    EXPECT_EQ(ccu_context_close(p.contexts[0], 0, &p.results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(p.inputs[0].refs, 1);
    p.endpoints[0].fail_close = false;
    p.inputs[0].fail_release = true;
    EXPECT_EQ(ccu_context_close(p.contexts[0], 0, &p.results[0]), CCU_STATUS_RETAINED);
    p.inputs[0].fail_release = false;
    p.Close();
}
TEST(CcuExecutionApi, CompletionHeaderMismatchDoesNotReleaseLease) {
    auto p = std::make_shared<Pair>();
    retained_pairs.push_back(p);
    p->Create();
    p->Bind();
    ASSERT_EQ(p->Prepare(), (std::array<int, 2>{0, 0}));
    p->Arm();
    p->Trigger();
    p->Complete();
    ++p->endpoints[0].observation.record.context_generation;
    EXPECT_EQ(ccu_poll(p->requests[0], &p->results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(ccu_context_close(p->contexts[0], 0, &p->results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(p->outputs[0].refs, 1);
}

TEST(CcuExecutionApi, PreparedCancellationCanRetryAfterControlPlaneFailure) {
    Pair p;
    p.Create();
    p.Bind();
    ASSERT_EQ(p.Prepare(), (std::array<int, 2>{0, 0}));
    p.group.fail_phase = 4;
    EXPECT_EQ(
        p.Both([&](uint32_t rank) {
            return ccu_cancel_prepared(p.requests[rank], &p.results[rank]);
        }),
        (std::array<int, 2>{CCU_STATUS_RETAINED, CCU_STATUS_RETAINED})
    );
    p.group.fail_phase = 0;
    EXPECT_EQ(
        p.Both([&](uint32_t rank) {
            return ccu_cancel_prepared(p.requests[rank], &p.results[rank]);
        }),
        (std::array<int, 2>{0, 0})
    );
    p.Close();
}

TEST(CcuExecutionApi, FutureSequencePoisonsRecoveryAndOldSequenceRemainsPending) {
    auto p = std::make_shared<Pair>();
    retained_pairs.push_back(p);
    p->Create();
    p->Bind();
    ASSERT_EQ(p->Prepare(), (std::array<int, 2>{0, 0}));
    p->Arm();
    p->Trigger();
    EXPECT_EQ(ccu_poll(p->requests[0], &p->results[0]), CCU_STATUS_PENDING);
    p->endpoints[0].observation.record.completed_seq = p->endpoints[0].args[7] + 1;
    EXPECT_EQ(ccu_poll(p->requests[0], &p->results[0]), CCU_STATUS_RETAINED);
    p->Complete();
    EXPECT_EQ(ccu_wait(p->requests[0], 0, &p->results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(p->inputs[0].refs, 1);
}

TEST(CcuExecutionApi, InvalidImportIsRejectedAndTokenFailureReleasesLease) {
    Pair p;
    p.Create();
    CcuWindowImportV1 window{
        1, sizeof(CcuWindowImportV1), 0, 0, 100, 1, 0x700000, 4096, 4000, 1024, &p.inputs[0], &p.owners
    };
    CcuWindowId id = 123;
    EXPECT_EQ(ccu_window_register(p.contexts[0], &window, &id, &p.results[0]), CCU_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(id, 0U);
    EXPECT_EQ(p.inputs[0].refs, 1);
    window.window_offset = 0;
    p.endpoints[0].fail_token = true;
    EXPECT_EQ(ccu_window_register(p.contexts[0], &window, &id, &p.results[0]), CCU_STATUS_SDK_ERROR);
    EXPECT_EQ(p.inputs[0].refs, 1);
    EXPECT_EQ(id, 0U);
    p.Close();
}

TEST(CcuExecutionApi, UnknownTriggerSubmissionIsNeverRetried) {
    auto p = std::make_shared<Pair>();
    retained_pairs.push_back(p);
    p->Create();
    p->Bind();
    ASSERT_EQ(p->Prepare(), (std::array<int, 2>{0, 0}));
    p->Arm();
    p->endpoints[0].fail_trigger = true;
    EXPECT_EQ(ccu_trigger_host(p->requests[0], &p->results[0]), CCU_STATUS_RETAINED);
    EXPECT_EQ(ccu_trigger_host(p->requests[0], &p->results[0]), CCU_STATUS_INVALID_STATE);
    EXPECT_EQ(p->endpoints[0].triggers, 1U);
    EXPECT_EQ(p->outputs[0].refs, 1);
}

TEST(CcuExecutionApi, CloseCoordinatesPreparedCancellationAndDrainedRetirement) {
    Pair p;
    p.Create();
    p.Bind();
    ASSERT_EQ(p.Prepare(), (std::array<int, 2>{0, 0}));
    EXPECT_EQ(
        p.Both([&](uint32_t rank) {
            return ccu_context_close(p.contexts[rank], 1000, &p.results[rank]);
        }),
        (std::array<int, 2>{0, 0})
    );
    EXPECT_EQ(p.inputs[0].refs, 0);
    EXPECT_EQ(p.endpoints[1].refs, 0);
    Pair q;
    q.Create();
    q.Bind();
    ASSERT_EQ(q.Prepare(), (std::array<int, 2>{0, 0}));
    q.Arm();
    q.Trigger();
    q.Complete();
    EXPECT_EQ(
        q.Both([&](uint32_t rank) {
            return ccu_context_close(q.contexts[rank], 1000, &q.results[rank]);
        }),
        (std::array<int, 2>{0, 0})
    );
    EXPECT_EQ(q.inputs[1].refs, 0);
    EXPECT_EQ(q.endpoints[0].refs, 0);
}

TEST(CcuExecutionApi, PollBeforeTriggerDoesNotPoisonPreparedCancellation) {
    Pair p;
    p.Create();
    p.Bind();
    ASSERT_EQ(p.Prepare(), (std::array<int, 2>{0, 0}));
    EXPECT_EQ(ccu_poll(p.requests[0], &p.results[0]), CCU_STATUS_INVALID_STATE);
    EXPECT_EQ(
        p.Both([&](uint32_t rank) {
            return ccu_cancel_prepared(p.requests[rank], &p.results[rank]);
        }),
        (std::array<int, 2>{0, 0})
    );
    p.Close();
}

TEST(CcuExecutionApi, InitializationRollbackKeepsHandleWhenCleanupIsUncertain) {
    Group group;
    Endpoint endpoint;
    endpoint.group = &group;
    endpoint.rank = 0;
    endpoint.fail_open = endpoint.fail_close = true;
    CcuOwnerOpsV1 owners{1, sizeof(CcuOwnerOpsV1), Retain, Release};
    CcuBootstrapOpsV1 bootstrap{1, sizeof(CcuBootstrapOpsV1), Exchange, Abort};
    std::array<uint64_t, 2> root{};
    CcuCreateOptionsV1 options{
        1, sizeof(CcuCreateOptionsV1), 0, 0, 2, 16, 17, root.data(), nullptr, &bootstrap, &endpoint, &owners
    };
    CcuContextId context{};
    CcuHostResultV1 result{};
    EXPECT_EQ(ccu_context_create(&options, &context, &result), CCU_STATUS_RETAINED);
    EXPECT_NE(context, 0U);
    EXPECT_EQ(endpoint.refs, 1);
    endpoint.fail_close = false;
    EXPECT_EQ(ccu_context_close(context, 1, &result), CCU_STATUS_SUCCESS);
    EXPECT_EQ(endpoint.refs, 0);
}

TEST(CcuExecutionApi, BootstrapCanReenterAndGetsBusyWithoutHoldingContextMutex) {
    Pair p;
    p.Create();
    p.Bind();
    p.group.reenter = true;
    ASSERT_EQ(p.Prepare(), (std::array<int, 2>{0, 0}));
    EXPECT_EQ(p.endpoints[0].reentry_status, CCU_STATUS_BUSY);
    EXPECT_EQ(p.endpoints[1].reentry_status, CCU_STATUS_BUSY);
    p.Arm();
    p.Trigger();
    p.Complete();
    p.Retire();
    p.Close();
}
