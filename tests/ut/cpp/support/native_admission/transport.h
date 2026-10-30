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
#include <functional>
#include <limits>
#include <string>
#include <vector>
namespace native_admission_transport {
struct Control {
    bool initialize{false};
    bool throw_kernel_registration{false};
    bool fail_free_once{false};
    bool fail_event_destroy_once{false};
    std::function<void()> sync_hook;
    size_t bootstrap_calls{0};
    size_t loader_init_calls{0};
    size_t aicore_launch_calls{0};
    size_t sync_calls{0};
    size_t fail_sync_call{0};
    std::vector<std::string> launched_symbols;
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
    int op_timeout_rc{0};
    bool replace_op_timeout{false};
    uint64_t actual_op_timeout_us{0};
    uint64_t requested_op_timeout_us{0};
    size_t op_timeout_calls{0};
    int copy_rc{0};
    int capture_query_rc{0};
    int capture_status{0};
    void *capture_model{nullptr};
    int capture_callback_rc{0};
    size_t capture_callback_calls{0};
    void (*capture_destroy_callback)(void *){nullptr};
    void *capture_destroy_data{nullptr};
    size_t event_creates{0};
    size_t event_destroys{0};
    size_t stream_create_calls{0};
    size_t fail_stream_create{0};
};
Control &control();
void release_provider_state();
}  // namespace native_admission_transport
