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
#include <cassert>
#include <cstdlib>
#include <string>

// The SDK CPU prefetch stub defines an always-successful Wait. Use the real
// event implementation instead; any unexpected hardware copy must fail here.
#define PTO_CPU_TPREFETCH_ASYNC_HPP
namespace pto {
struct PrefetchAsyncContext;
}
#include <pto/pto-inst.hpp>
template <typename... Args>
void copy_ubuf_to_gm_align_b32(Args...) {
    std::abort();
}
template <typename... Args>
void copy_gm_to_ubuf_align_b32(Args...) {
    std::abort();
}
#include <pto/comm/async_common/async_event_impl.hpp>
#include <pto/comm/async_common/TGetAsyncCommonDetail.hpp>
#include <pto/comm/async_common/TPutAsyncCommonDetail.hpp>

namespace sdk_boundary {
bool empty_tensor = false;
uint32_t payload[16]{};
struct Tensor {
    using RawDType = uint32_t;
    static constexpr int layout = 0;
    int GetShape(int dim) const { return dim == 4 ? (empty_tensor ? 0 : 16) : 1; }
    int GetStride(int dim) const { return dim == 4 ? 1 : GetShape(4); }
    uint32_t *data() const { return payload; }
};
struct Scratch {};
alignas(128) uint8_t records[4096]{};
int submissions = 0;
uint32_t queues = 2;
bool post_failed = false;
bool empty = false;
bool wrong_engine = false;
bool session_failed = false;
bool invalid_handle = false;
bool missing_records = false;
bool sdk_rejected = false;
alignas(
    128
) uint8_t workspace[sizeof(pto::comm::sdma::BatchWriteFlagInfo) + 2 * sizeof(pto::comm::sdma::BatchWriteChannelInfo)]{};

pto::comm::AsyncEvent rejected_post(Tensor &dst, Tensor &src, pto::comm::AsyncSession &session, bool put) {
    ++submissions;
    using namespace pto::comm;
    sdma::SdmaSession converted;
    sdma::detail::LoadSdmaSession(session, converted);
    sdma::SdmaConfig config{};
    sdma::detail::SdmaPostState state{};
    assert(sdma::detail::PrepareSdmaPostBase(sizeof(payload), converted, config, state));
    assert(config.iter_num == 1);
    // One data SQE plus its flag cannot fit the real SDK's depth-one queue.
    assert(!sdma::detail::ValidateSinglePostSqCapacity(state.channels, config, 1, 1));
    const auto event =
        put ? detail::TPUT_ASYNC_SDMA_IMPL(dst, src, session) : detail::TGET_ASYNC_SDMA_IMPL(dst, src, session);
    assert(event.handle == 0 && !event.postFailed);
    assert(state.channels[0].sq_head == 0 && state.channels[0].sq_tail == 0);
    return event;
}

pto::comm::AsyncEvent post() {
    ++submissions;
    pto::comm::AsyncEvent event{};
    event.engine = wrong_engine ? pto::comm::DmaEngine::URMA : pto::comm::DmaEngine::SDMA;
    event.postFailed = post_failed;
    if (!empty) assert(pto::comm::sdma::detail::EncodeSdmaEventHandle(7, queues, event.handle));
    if (invalid_handle) event.handle = 1;
    return event;
}
}  // namespace sdk_boundary

// Only the SDK's session creation and physical submission are replaced. The
// runtime adapter and the SDK event/record interpretation remain real.
namespace pto::comm {
bool BuildAsyncSession(sdk_boundary::Scratch &, uint8_t *, AsyncSession &session, uint32_t) {
    session.engine = DmaEngine::SDMA;
    session.valid = true;
    session.cfg.sdma.queueNum = 2;
    session.rt.sdma.postDoneBase = sdk_boundary::missing_records ? nullptr : sdk_boundary::records;
    if (sdk_boundary::sdk_rejected) {
        session.contextGm = sdk_boundary::workspace;
        session.cfg.sdma.blockBytes = sizeof(sdk_boundary::payload);
        auto *channels =
            reinterpret_cast<sdma::BatchWriteChannelInfo *>(sdk_boundary::workspace + sizeof(sdma::BatchWriteFlagInfo));
        channels[0].sq_depth = 1;
        channels[1].sq_depth = 1;
    }
    return !sdk_boundary::session_failed;
}
AsyncEvent TGET_ASYNC(sdk_boundary::Tensor &dst, sdk_boundary::Tensor &src, AsyncSession &session) {
    return sdk_boundary::sdk_rejected ? sdk_boundary::rejected_post(dst, src, session, false) : sdk_boundary::post();
}
AsyncEvent TPUT_ASYNC(sdk_boundary::Tensor &dst, sdk_boundary::Tensor &src, AsyncSession &session) {
    return sdk_boundary::sdk_rejected ? sdk_boundary::rejected_post(dst, src, session, true) : sdk_boundary::post();
}
}  // namespace pto::comm

#include "backend/sdma/sdma_completion_kernel.h"

int main(int argc, char **argv) {
    assert(argc == 3);
    const std::string scenario = argv[1];
    DeferredCompletionSlab slab{};
    AsyncCtx context{};
    context.task_token = TaskId::make(0, 1);
    context.completion_count = &slab.count;
    context.completion_error_code = &slab.error_code;
    context.completion_entries = slab.entries;
    context.completion_capacity = MAX_COMPLETIONS_PER_TASK;
    bool expected_success = false;
    int expected_posts = 0;
    int expected_error = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
    uint32_t initial_count = 0;
    uint32_t added_records = 0;
    if (scenario == "full") {
        initial_count = MAX_COMPLETIONS_PER_TASK;
    } else if (scenario == "partial") {
        initial_count = MAX_COMPLETIONS_PER_TASK - 1;
    } else if (scenario == "invalid_count") {
        initial_count = UINT32_MAX;
    } else if (scenario == "exact") {
        initial_count = MAX_COMPLETIONS_PER_TASK - 2;
        expected_success = true;
        expected_posts = 1;
        expected_error = 0;
        added_records = 2;
    } else if (scenario == "prior_error") {
        slab.error_code = 106;
        expected_error = 106;
    } else if (scenario == "missing_entries" || scenario == "missing_count" || scenario == "session_failed") {
        if (scenario == "missing_entries") context.completion_entries = nullptr;
        if (scenario == "missing_count") context.completion_count = nullptr;
        sdk_boundary::session_failed = scenario == "session_failed";
        expected_error = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    } else if (scenario == "failed_with_records") {
        initial_count = 1;
        added_records = 2;
        sdk_boundary::post_failed = true;
        expected_posts = 1;
        expected_error = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    } else if (scenario == "sdk_rejected" || scenario == "sync_sdk_rejected") {
        sdk_boundary::sdk_rejected = true;
        expected_posts = 1;
        expected_error = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        if (scenario == "sync_sdk_rejected") {
            context.task_token = TaskId::invalid();
            expected_error = 0;
        }
    } else if (scenario == "post_failed" || scenario == "sync_failed") {
        sdk_boundary::post_failed = true;
        sdk_boundary::empty = true;
        expected_posts = 1;
        expected_error = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        if (scenario == "sync_failed") {
            context.task_token = TaskId::invalid();
            expected_error = 0;
        }
    } else if (scenario == "wrong_engine" || scenario == "too_many_records" || scenario == "invalid_handle" ||
               scenario == "missing_records") {
        sdk_boundary::wrong_engine = scenario == "wrong_engine";
        sdk_boundary::invalid_handle = scenario == "invalid_handle";
        sdk_boundary::missing_records = scenario == "missing_records";
        if (scenario == "too_many_records") sdk_boundary::queues = 3;
        expected_posts = 1;
        expected_error = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    } else {
        assert(scenario == "empty" || scenario == "sync_empty");
        sdk_boundary::empty = true;
        sdk_boundary::empty_tensor = true;
        expected_success = true;
        expected_posts = 0;
        expected_error = 0;
        if (scenario == "sync_empty") context.task_token = TaskId::invalid();
    }
    slab.count = initial_count;
    slab.entries[0].backend_cookie = 91;
    auto request = SdmaTget(sdk_boundary::Tensor{}, sdk_boundary::Tensor{}, sdk_boundary::Scratch{}, nullptr);
    if (std::atoi(argv[2]) != 0) request.op = SdmaOp::TPUT;
    const bool submitted = send_request_entry(context, request);
    assert(sdk_boundary::submissions == expected_posts);
    assert(submitted == expected_success);
    assert(slab.error_code == expected_error);
    assert(slab.entries[0].backend_cookie == 91);
    assert(slab.count == initial_count + added_records);
    if (added_records != 0) {
        for (uint32_t i = initial_count; i < slab.count; ++i) {
            assert(slab.entries[i].addr != 0);
            assert(slab.entries[i].backend_cookie == 7);
            assert(slab.entries[i].completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD);
        }
        assert(slab.entries[initial_count].addr != slab.entries[initial_count + 1].addr);
    }
}
