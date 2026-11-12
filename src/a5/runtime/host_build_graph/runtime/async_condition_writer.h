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

#include "async_completion_types.h"
#include "host_build_graph/completion_token.h"
#include "platform_comm/ccu/ccu_native_abi.h"

namespace pto2::resident {

// Local kernel view; only integer offsets and indices are stored in the image.
// Copies share all mutable state, including the first error and reservation.
struct AsyncWriter {
    __gm__ uint8_t *image{nullptr};
    uint64_t image_bytes{0};
    uint64_t run_epoch{0};
    uint64_t group_offset{0};
    uint32_t group_index{ASYNC_INVALID_INDEX};
};

namespace detail {

inline __aicore__ bool span_valid(const AsyncWriter &writer, uint64_t offset, uint64_t bytes, uint64_t alignment) {
    return writer.image != nullptr && writer.image_bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(writer.image) &&
           reinterpret_cast<uintptr_t>(writer.image) % ASYNC_RECORD_ALIGNMENT == 0 && offset % alignment == 0 &&
           offset <= writer.image_bytes && bytes <= writer.image_bytes - offset;
}

inline __aicore__ __gm__ AsyncGroupData *group(const AsyncWriter &writer) {
    if (writer.run_epoch == 0 || writer.group_index == ASYNC_INVALID_INDEX ||
        !span_valid(writer, writer.group_offset, sizeof(AsyncGroupData), ASYNC_RECORD_ALIGNMENT))
        return nullptr;
    auto *data = reinterpret_cast<__gm__ AsyncGroupData *>(writer.image + writer.group_offset);
    if (data->run_epoch != writer.run_epoch || data->group_index != writer.group_index) return nullptr;
    return data;
}

inline __aicore__ bool fail(const AsyncWriter &writer, int32_t error) {
    auto *data = group(writer);
    if (data != nullptr && data->state == AsyncPublicationState::OPEN && data->error_code == SIMPLER_ERROR_NONE)
        data->error_code = error;
    return false;
}

inline __aicore__ bool layout_valid(const AsyncWriter &writer, __gm__ const AsyncGroupData &data) {
    const uint32_t bytes = async_group_entry_bytes(data.capacity);
    return data.task_ref >= 0 && data.subtask_slot < 3 && bytes != 0 && data.entries_bytes == bytes &&
           data.entries_offset >= writer.group_offset + sizeof(AsyncGroupData) &&
           span_valid(writer, data.entries_offset, bytes, ASYNC_RECORD_ALIGNMENT);
}

inline __aicore__ bool valid(const AsyncWriter &writer) {
    auto *data = group(writer);
    if (data == nullptr) return false;
    if (!layout_valid(writer, *data)) return fail(writer, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    if (data->state != AsyncPublicationState::OPEN || data->error_code != SIMPLER_ERROR_NONE) return false;
    if (data->count > data->capacity) return fail(writer, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    return true;
}

// The caller has validated the group's address/identity and observed OPEN.
// One execution owns its registration state until sealing completes.
template <bool LayoutValidated = false>
inline __aicore__ bool seal_open_conditions(const AsyncWriter &writer, __gm__ AsyncGroupData &data, uint32_t count) {
    const int32_t first_error = data.error_code;
    int32_t error = first_error;
    if (error == SIMPLER_ERROR_NONE) {
        if constexpr (!LayoutValidated)
            if (!layout_valid(writer, data)) error = SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
        if (error == SIMPLER_ERROR_NONE) {
            if (count > data.capacity) error = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
            else if (data.reservation_limit != 0) error = SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
        }
    }
    if (error != SIMPLER_ERROR_NONE) {
        if (first_error == SIMPLER_ERROR_NONE) data.error_code = error;
        data.state = AsyncPublicationState::FAILED;
        return false;
    }
    data.state = AsyncPublicationState::SEALED;
    return true;
}

// Like the existing kernel API, the caller supplies a live completion address.
// The runtime validates the record format; existing run/communication owners
// retain its storage until all accepted work has retired.
inline __aicore__ bool token_valid(const CompletionToken &token) {
    if (token.addr == 0) return false;
    if (token.engine == COMPLETION_ENGINE_CCU && token.completion_type == COMPLETION_TYPE_CCU)
        return token.addr % alignof(uint64_t) == 0 && token.expected_value == 0 &&
               token.backend_cookie == SIMPLER_CCU_EVENT_MASK;
    if (token.engine != COMPLETION_ENGINE_SDMA) return false;
    if (token.completion_type == COMPLETION_TYPE_COUNTER)
        return token.addr % sizeof(uint32_t) == 0 && token.backend_cookie == 0 && token.expected_value <= INT32_MAX;
    if (token.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD)
        return token.addr % sizeof(uint64_t) == 0 && token.expected_value == 0 && token.backend_cookie != 0;
    return false;
}

inline __aicore__ bool reservation_valid(const AsyncWriter &writer, const AsyncReservation &reservation) {
    auto *data = group(writer);
    if (reservation.run_epoch != writer.run_epoch || reservation.group_index != writer.group_index ||
        reservation.serial == 0 || reservation.serial != data->reservation_serial || reservation.capacity == 0 ||
        reservation.capacity != data->reservation_limit || data->reservation_begin > data->capacity ||
        data->reservation_limit > data->capacity - data->reservation_begin ||
        data->reservation_written > data->reservation_limit ||
        data->count != data->reservation_begin + data->reservation_written)
        return fail(writer, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    return true;
}

}  // namespace detail

inline __aicore__ AsyncReservation reserve_conditions(const AsyncWriter &writer, uint32_t upper_bound) {
    AsyncReservation reservation{};
    if (!detail::valid(writer)) return reservation;
    auto *data = detail::group(writer);
    if (data->reservation_limit != 0 || upper_bound == 0 || data->reservation_serial == UINT32_MAX) {
        detail::fail(writer, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        return reservation;
    }
    if (upper_bound > data->capacity - data->count) {
        detail::fail(writer, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        return reservation;
    }
    data->reservation_begin = data->count;
    data->reservation_written = 0;
    data->reservation_limit = upper_bound;
    data->reservation_serial = data->reservation_serial + 1;
    reservation.run_epoch = writer.run_epoch;
    reservation.group_index = writer.group_index;
    reservation.serial = data->reservation_serial;
    reservation.capacity = upper_bound;
    return reservation;
}

inline __aicore__ bool
commit_condition(const AsyncWriter &writer, const AsyncReservation &reservation, const CompletionToken &token) {
    if (!detail::valid(writer) || !detail::reservation_valid(writer, reservation)) return false;
    auto *data = detail::group(writer);
    if (data->reservation_written == data->reservation_limit)
        return detail::fail(writer, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    if (!detail::token_valid(token)) return detail::fail(writer, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    const uint32_t index = data->count;
    auto *entries = reinterpret_cast<__gm__ volatile DeferredCompletionEntry *>(writer.image + data->entries_offset);
    entries[index].addr = token.addr;
    entries[index].backend_cookie = token.backend_cookie;
    entries[index].expected_value = token.expected_value;
    entries[index].engine = token.engine;
    entries[index].completion_type = token.completion_type;
    entries[index]._pad = 0;
    data->reservation_written = data->reservation_written + 1;
    data->count = index + 1;
    return true;
}

inline __aicore__ bool
finish_reservation(const AsyncWriter &writer, const AsyncReservation &reservation, uint32_t actual_count) {
    if (!detail::valid(writer) || !detail::reservation_valid(writer, reservation)) return false;
    auto *data = detail::group(writer);
    if (actual_count != data->reservation_written) return detail::fail(writer, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    data->reservation_limit = 0;
    data->reservation_begin = 0;
    data->reservation_written = 0;
    return true;
}

inline __aicore__ bool register_completion_condition(const AsyncWriter &writer, const CompletionToken &token) {
    const auto reservation = reserve_conditions(writer, 1);
    return reservation.serial != 0 && commit_condition(writer, reservation, token) &&
           finish_reservation(writer, reservation, 1);
}

// The caller publishes the changed header fields and registered entries before
// activating a nonzero group's lease/directory. This function
// never grants scheduler ownership and never discharges a join obligation.
inline __aicore__ bool seal_conditions(const AsyncWriter &writer) {
    auto *data = detail::group(writer);
    if (data == nullptr || data->state != AsyncPublicationState::OPEN) return false;
    return detail::seal_open_conditions(writer, *data, data->count);
}

}  // namespace pto2::resident
