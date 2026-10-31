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
#include <vector>
namespace native_admission_transport {
enum class SubmissionOp : uint8_t { None, Record, Wait, Admission, Core, Cpu };
struct SubmissionStep {
    SubmissionOp operation{SubmissionOp::None};
    void *stream{nullptr};
    void *event{nullptr};
};
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
    int op_timeout_rc{0};
    bool replace_op_timeout{false};
    uint64_t actual_op_timeout_us{0};
    uint64_t requested_op_timeout_us{0};
    size_t op_timeout_calls{0};
    int copy_rc{0};
    int run_result_copy_rc{0};
    size_t run_result_copies{0};
    size_t stream_create_calls{0};
    size_t fail_stream_create{0};
    int stream_sync_rc{0};
    bool fail_stream_id{false};
    bool capture_active{false};
    size_t capture_info_calls{0};
    int capture_info_rc{0};
    bool *capture_refuse_after_core{nullptr};
    size_t capture_record_allocation_failures{0};
    bool capture_loader_throws{false};
    bool capture_registration_throws{false};
    size_t capture_registration_calls{0};
    size_t capture_core_calls{0};
    size_t core_payload_bytes{0};
    uint64_t core_report_epoch{0};
    size_t capture_cpu_calls{0};
    size_t capture_sync_calls{0};
    size_t event_create_calls{0};
    bool trace_submission{false};
    SubmissionOp fail_submission{SubmissionOp::None};
    int submission_rc{-4332};
    std::vector<SubmissionStep> submissions;
};
Control &control();
void release_provider_state();
int note_submission(SubmissionOp operation, void *stream, void *event = nullptr);
}  // namespace native_admission_transport
