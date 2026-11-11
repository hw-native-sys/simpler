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

#include "async_condition_writer.h"
#include "urma/urma_request.h"

namespace simpler::urma {

class HbgRequestAdapter {
public:
    explicit __aicore__ HbgRequestAdapter(const pto2::resident::AsyncWriter &writer, uint32_t sender) :
        writer_(writer),
        sender_(sender) {}

    inline __aicore__ uint32_t sender_id() const { return sender_; }

    inline __aicore__ bool capacity_available(uint64_t required) const {
        if (sender_ == UINT32_MAX || error_ != SIMPLER_ERROR_NONE || !pto2::resident::detail::valid(writer_))
            return false;
        const auto *data = pto2::resident::detail::group(writer_);
        if (reservation_.serial != 0)
            return pto2::resident::detail::reservation_valid(writer_, reservation_) &&
                   required <= reservation_.capacity - data->reservation_written;
        return data->reservation_limit == 0 && required <= data->capacity - data->count;
    }

    inline __aicore__ bool reserve(uint64_t required) {
        if (required == 0 || required > UINT32_MAX || !capacity_available(required)) return false;
        reservation_ = pto2::resident::reserve_conditions(writer_, static_cast<uint32_t>(required));
        return reservation_.serial != 0;
    }

    inline __aicore__ bool register_target(const CompletionTarget &target) {
        if (target.workspace == nullptr || reinterpret_cast<uintptr_t>(target.workspace) % alignof(UrmaInfo) != 0)
            return false;
        const auto *info = reinterpret_cast<__gm__ const UrmaInfo *>(target.workspace);
        if (target.queue_row >= info->jettyCount) return false;
        const CompletionToken token{
            (static_cast<uint64_t>(target.queue_row) << 32) | target.target_bb, target.target_cqe,
            COMPLETION_ENGINE_URMA, COMPLETION_TYPE_URMA_EVENT_HANDLE, reinterpret_cast<uint64_t>(target.workspace)
        };
        if (!pto2::resident::commit_condition(writer_, reservation_, token)) return false;
        ++written_;
        return true;
    }

    inline __aicore__ void set_error(RequestError error) {
        if (error_ == SIMPLER_ERROR_NONE)
            error_ = error == RequestError::INVALID ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID :
                                                      SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
        // An active reservation finishes its accepted prefix before publishing the error.
        if (reservation_.serial == 0) pto2::resident::detail::fail(writer_, error_);
    }

    inline __aicore__ bool finish() {
        const bool finished = pto2::resident::finish_reservation(writer_, reservation_, written_);
        if (error_ != SIMPLER_ERROR_NONE) pto2::resident::detail::fail(writer_, error_);
        return finished && error_ == SIMPLER_ERROR_NONE;
    }

private:
    pto2::resident::AsyncWriter writer_;
    uint32_t sender_;
    pto2::resident::AsyncReservation reservation_{};
    uint32_t written_{0};
    int32_t error_{SIMPLER_ERROR_NONE};
};

}  // namespace simpler::urma
