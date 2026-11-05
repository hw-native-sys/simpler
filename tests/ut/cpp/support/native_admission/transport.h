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
// Allocation and transport faults; all admission decisions stay in production code.
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
namespace native_admission_transport {
struct Control {
    size_t ceiling{std::numeric_limits<size_t>::max()};
    size_t bytes{0};
    size_t allocations{0};
    size_t successful_allocations{0};
    size_t fail_allocation{0};
    int loader_finalize_error{0};
    size_t loader_finalize_calls{0};
    int reset_error{0};
    bool retired{false};
    size_t calls_after_retire{0};
    int attach_rc{0};
    size_t device_set_calls{0};
    size_t device_reset_calls{0};
    int device_refs{0};
    size_t context_get_calls{0};
    size_t context_set_calls{0};
    int context_get_rc{0};
    int context_set_rc{0};
    size_t fail_context_set_call{0};
    bool null_context{false};
    void *last_bound_context{nullptr};
    size_t acl_init_calls{0};
    size_t acl_finalize_calls{0};
    size_t acl_set_calls{0};
    size_t acl_reset_calls{0};
    int acl_init_rc{0};
    int acl_set_rc{0};
    int acl_reset_rc{0};
    int acl_finalize_rc{0};
    int refs_at_acl_finalize{-1};
    int op_timeout_rc{0};
    bool replace_op_timeout{false};
    uint64_t actual_op_timeout_us{0};
    uint64_t requested_op_timeout_us{0};
    size_t op_timeout_calls{0};
    int copy_rc{0};
    size_t stream_create_calls{0};
    size_t fail_stream_create{0};
    size_t d2h_calls{0};
    size_t d2h_bytes{0};
    std::vector<const void *> downloads;
    int stream_sync_rc{0};
    std::vector<void *> observed_streams;
    bool events_complete{true};
    int event_wait_rc{0};
    std::vector<std::pair<void *, const void *>> uploads;
};
Control &control();
void release_provider_state();
}  // namespace native_admission_transport
