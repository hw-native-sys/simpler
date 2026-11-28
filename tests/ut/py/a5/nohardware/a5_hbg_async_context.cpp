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
#include <cstddef>

#include "async_kernel_api.h"
#include "backend/sdma/sdma_completion_registration.h"

namespace {
void *lane_context = nullptr;

struct AsyncImage {
    alignas(64) std::array<uint8_t, 64> control{};
    pto2::resident::AsyncGroupData group{};
    alignas(64) std::array<DeferredCompletionEntry, 2> entries{};
};

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
    using simpler::percounter::PercounterInvocationContext;
    constexpr uint64_t group_offset = offsetof(AsyncImage, group);
    AsyncImage state{};
    auto *image = reinterpret_cast<uint8_t *>(&state);
    auto *data = &state.group;
    data->run_epoch = 9;
    data->capacity = state.entries.size();
    data->entries_bytes = sizeof(state.entries);
    data->entries_offset = offsetof(AsyncImage, entries);

    PercounterInvocationContext context{};
    context.local = {2, 5};
    simpler_sim_install_invocation_context_hooks(
        [](void *value) {
            lane_context = value;
            return 0;
        },
        [] {
            return lane_context;
        }
    );
    assert(simpler::percounter::percounter_bind_sim_context(&context));
    auto &global = context.global;
    global.sub_block_id = 1;
    global.async_present = 1;
    global.async_image_address = reinterpret_cast<uint64_t>(image);
    global.async_image_bytes = sizeof(state);
    global.async_run_epoch = data->run_epoch;
    global.async_group_offset = group_offset;
    global.async_group_index = data->group_index;
    int64_t payload{};
    auto *args = &payload;
    assert(get_block_idx(args) == 2 && get_block_num(args) == 5 && get_sub_block_id(args) == 1);
    const CompletionToken token{0x10000, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    if (scenario == 0) {
        assert(context.async_touched == 0);
        auto ctx = get_async_ctx(args);
        assert(context.async_touched == 1);
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
        else data->entries_offset = group_offset;
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
    } else if (scenario == 33 || scenario == 34) {
        global.async_output_capture_refused = 1;
        const pto2::resident::AsyncWriter writer{image, sizeof(state), data->run_epoch, group_offset, 0};
        if (scenario == 34) {
            // Selecting an output snapshot alone does not register any deferred condition.
            assert(context.async_touched == 0);
            assert(pto2::resident::seal_conditions(writer));
        } else {
            auto ctx = get_async_ctx(args);
            assert(!async_ctx_is_deferred(ctx));
            assert(!register_completion_condition(ctx, token));
            assert(data->count == 0);
            assert(!pto2::resident::seal_conditions(writer));
            assert(data->error_code == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        }
    } else return 2;
    return 0;
}
