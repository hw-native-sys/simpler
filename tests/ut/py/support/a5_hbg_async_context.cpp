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

#include <pto/pto-inst.hpp>

#include <array>
#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "async_completion_plan.h"
#include "async_kernel_api.h"
#include "backend/sdma/sdma_completion_registration.h"
#include "scheduler/scheduler_prepared_layout.h"
#include "scheduler/scheduler_args_dump.h"

namespace {
struct Session {};
struct Event {
    bool postFailed{false};
    uint64_t handle{1};
    pto::comm::DmaEngine engine{pto::comm::DmaEngine::SDMA};
    uint32_t count{2};
    pto::comm::AsyncCompletionRecord records[2]{};

    Event() {
        for (uint32_t i = 0; i < 2; ++i) {
            records[i].addr = reinterpret_cast<__gm__ uint8_t *>(UINT64_C(0x10000) + i * 128);
            records[i].expected = UINT64_C(0x100000003) + i;
            records[i].kind = pto::comm::CompletionKind::SDMA_POST_DONE;
        }
    }
    uint32_t CompletionRecordCount(const Session &) const { return count; }
    pto::comm::AsyncCompletionRecord CompletionRecordAt(const Session &, uint32_t index) const {
        assert(index < 2);
        return records[index];
    }
};

struct FlatTensor {
    using RawDType = uint32_t;
    int64_t shapes[5]{1, 1, 1, 1, 16};
    int64_t strides[5]{16, 16, 16, 16, 1};
    uint32_t *pointer{reinterpret_cast<uint32_t *>(UINT64_C(0x20000))};
    int64_t GetShape(int dim) const { return shapes[static_cast<uint32_t>(dim)]; }
    int64_t GetStride(int dim) const { return strides[static_cast<uint32_t>(dim)]; }
    uint32_t *data() const { return pointer; }
};
}  // namespace

int main(int argc, char **argv) {
    assert(argc == 2);
    const int scenario = std::atoi(argv[1]);
    AsyncTaskPlanInput input;
    input.task_ref = 0;
    input.block_count = 1;
    input.active_mask = LaneMask::aiv0();
    input.capacities[1] = 2;
    AsyncCompletionHostPlan plan;
    std::string error;
    assert(plan_async_completion(sizeof(SchedulerRunControl), 9, 1, {input}, &plan, &error));
    std::vector<uint8_t> backing(plan.total_bytes + 127);
    auto *image = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(backing.data()) + 127) & ~uintptr_t(127));
    assert(initialize_async_completion(image, plan.total_bytes, plan));
    auto *data = scheduler_state_at<pto2::resident::AsyncGroupData>(image, plan.layout.groups_offset);
    SchedulerPreparedCall call{};
    SchedulerExecutionBinding binding{};
    static_assert(sizeof(SchedulerPreparedCall) == 512 && offsetof(SchedulerPreparedCall, args) == 64);
    static_assert(offsetof(SchedulerExecutionBinding, global_context) == 64 && sizeof(LocalContext) == 8);
    call.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&binding.local_context);
    call.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&binding.global_context);
    auto &global = binding.global_context;
    global.async_present = 1;
    global.async_image_address = reinterpret_cast<uint64_t>(image);
    global.async_image_bytes = plan.total_bytes;
    global.async_run_epoch = 9;
    global.async_group_offset = plan.layout.groups_offset;
    global.async_group_index = 0;
    auto *args = reinterpret_cast<int64_t *>(call.args);
    const CompletionToken token{0x10000, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    if (scenario == 0) {
        auto ctx = get_async_ctx(args);
        assert(async_ctx_is_deferred(ctx));
        assert(register_completion_condition(ctx, token));
        auto second = get_async_ctx(args);
        save_expected_notification_counter(second, reinterpret_cast<void *>(token.addr), 2);
        pto2::detail::defer_flush(ctx);
        assert(data->count == 2 && data->state == pto2::resident::AsyncPublicationState::OPEN);
        assert(pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario == 1) {
        auto ctx = get_async_ctx(args);
        assert(register_completion_condition(ctx, token));
        assert(register_completion_condition(ctx, token));
        assert(!register_completion_condition(ctx, token));
        assert(data->count == 2);
        assert(data->error_code == SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        assert(!pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario == 2) {
        ++global.async_present;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(data->error_code == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        assert(!pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario == 3) {
        global.async_output_capture_refused = 1;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(data->error_code == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    } else if (scenario == 4) {
        ++global.async_run_epoch;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(!register_completion_condition(ctx, token));
        assert(data->count == 0 && data->error_code == 0);
    } else if (scenario == 5) {
        global.async_present = 0;
        // Absence of an async group never touches a LocalContext or a slab.
        call.args[SPMD_LOCAL_CONTEXT_INDEX] = 0;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(!register_completion_condition(ctx, token));
        pto2::detail::defer_flush(ctx);
        assert(ctx.completion_count == nullptr && ctx.completion_entries == nullptr);
        assert(data->count == 0 && data->error_code == 0);
    } else if (scenario == 6) {
        auto ctx = get_async_ctx(args);
        auto invalid = token;
        invalid.addr += 1;
        assert(!register_completion_condition(ctx, invalid));
        assert(data->count == 0);
        assert(!pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario >= 7 && scenario <= 17) {
        auto ctx = get_async_ctx(args);
        const auto reservation = pto2::resident::reserve_conditions(ctx.resident, 2);
        Event event;
        if (scenario == 8) event.count = 1;
        if (scenario == 9) event.handle = 0;
        if (scenario == 10) event.postFailed = true;
        if (scenario == 11) event.count = UINT32_MAX;
        if (scenario == 12) event.count = 3;
        if (scenario == 13) event.records[1].kind = static_cast<pto::comm::CompletionKind>(255);
        if (scenario == 14) event.records[1].expected = 0;
        if (scenario == 15) event.records[0].addr += 1;
        if (scenario == 16) event.engine = pto::comm::DmaEngine::URMA;
        if (scenario == 17) event.count = 0;
        const bool registered = pto2::detail::register_pto_async_event(ctx, event, Session{}, reservation, 2);
        if (scenario < 9) {
            assert(registered);
            assert(data->count == event.count && data->reservation_limit == 0);
            auto *entries = reinterpret_cast<DeferredCompletionEntry *>(image + data->entries_offset);
            assert(entries[0].backend_cookie == UINT64_C(0x100000003));
            assert(pto2::resident::seal_conditions(ctx.resident));
        } else {
            assert(!registered && data->error_code != 0);
            assert(!pto2::resident::seal_conditions(ctx.resident));
        }
    } else if (scenario >= 18 && scenario <= 23) {
        FlatTensor tensor;
        uint64_t bytes = 0;
        if (scenario == 19) tensor.shapes[0] = 2;
        if (scenario == 20) tensor.shapes[4] = -1;
        if (scenario == 21) tensor.strides[4] = 2;
        if (scenario == 22) tensor.shapes[4] = UINT64_C(0x100000000);
        if (scenario == 23) tensor.pointer = reinterpret_cast<uint32_t *>(UINT64_MAX - 31);
        assert(pto2::detail::flat_transfer_bytes(tensor, bytes) == (scenario == 18));
        if (scenario == 18) {
            assert(bytes == 64);
            tensor.pointer = nullptr;
            tensor.shapes[4] = 0;
            for (uint32_t i = 0; i < 4; ++i)
                tensor.strides[i] = 0;
            assert(pto2::detail::flat_transfer_bytes(tensor, bytes) && bytes == 0);
        }
    } else if (scenario == 24) {
        auto ctx = get_async_ctx(args);
        assert(async_ctx_is_deferred(ctx));
        assert(ctx.completion_count == &data->count);
        assert(ctx.completion_error_code == &data->error_code);
        assert(ctx.completion_capacity == data->capacity);
        assert(
            ctx.completion_entries == reinterpret_cast<volatile DeferredCompletionEntry *>(image + data->entries_offset)
        );
        save_expected_notification_counter(ctx, reinterpret_cast<void *>(token.addr), 1);
        assert(*ctx.completion_count == 1);
        assert(ctx.completion_entries[0].addr == token.addr);
    } else if (scenario == 25) {
        for (uint32_t attempt = 0; attempt < 4; ++attempt) {
            auto ctx = get_async_ctx(args);
            assert(async_ctx_is_deferred(ctx));
            save_expected_notification_counter(ctx, reinterpret_cast<void *>(token.addr), 1);
        }
        auto ctx = get_async_ctx(args);
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        assert(data->count == 2);
        assert(data->error_code == SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        assert(!pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario == 26 || scenario == 27) {
        if (scenario == 26) data->entries_bytes += sizeof(*data);
        else data->entries_offset = plan.layout.groups_offset;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(ctx.completion_entries == nullptr);
        assert(data->count == 0);
        assert(data->error_code == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    } else if (scenario >= 28 && scenario <= 32) {
        if (scenario == 29) global.async_present = 0;
        auto ctx = get_async_ctx(args);
        const auto reservation =
            scenario == 29 ? pto2::resident::AsyncReservation{} : pto2::resident::reserve_conditions(ctx.resident, 2);
        Event event;
        event.handle = 0;
        event.count = UINT32_MAX;  // An already-completed event has no records to query.
        if (scenario == 30) event.postFailed = true;
        if (scenario == 31) event.engine = pto::comm::DmaEngine::URMA;
        if (scenario == 32) pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        const bool registered =
            pto2::detail::register_pto_async_event(ctx, event, Session{}, reservation, 2, SdmaOp::TPUT);
        if (scenario == 29) {
            assert(!registered);
            assert(ctx.completion_count == nullptr && ctx.completion_error_code == nullptr);
            assert(data->count == 0 && data->error_code == 0);
            return 0;
        }
        assert(registered == (scenario == 28));
        assert(*ctx.completion_count == 0);
        if (scenario <= 29) {
            assert(*ctx.completion_error_code == 0);
            assert(data->reservation_limit == 0);
            assert(pto2::resident::seal_conditions(ctx.resident));
        } else {
            assert(*ctx.completion_error_code != 0);
            assert(!pto2::resident::seal_conditions(ctx.resident));
            if (scenario == 32) assert(data->error_code == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        }
    } else if (scenario >= 33 && scenario <= 36) {
        SchedulerArgsDumpTask selected{};
        if (scenario == 33 || scenario == 36) selected.output_tensor_mask = 1;
        scheduler_args_dump_limit_async_output(scenario == 34 ? nullptr : &selected, global);
        const pto2::resident::AsyncWriter writer{image, plan.total_bytes, 9, plan.layout.groups_offset, 0};
        if (scenario == 36) {
            // An output snapshot of an ordinary synchronous task is still allowed.
            assert(pto2::resident::seal_conditions(writer));
        } else {
            auto ctx = get_async_ctx(args);
            assert(async_ctx_is_deferred(ctx) == (scenario != 33));
            assert(register_completion_condition(ctx, token) == (scenario != 33));
            assert(data->count == (scenario == 33 ? 0u : 1u));
            assert(pto2::resident::seal_conditions(writer) == (scenario != 33));
            assert(data->error_code == (scenario == 33 ? SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED : 0));
        }
    } else return 2;
    return 0;
}
