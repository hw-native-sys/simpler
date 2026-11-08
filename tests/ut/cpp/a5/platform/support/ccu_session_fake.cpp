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
#include "ccu_session_fake.h"
#include "platform_comm/ccu/ccu_device.h"

namespace {
thread_local std::shared_ptr<ccu_test::SessionControl> next_control;
class Backend final : public CcuSessionBackend {
public:
    explicit Backend(std::shared_ptr<ccu_test::SessionControl> control) :
        state_(std::move(control)) {}
    void Initialize(const CcuSessionConfig &) override {}
    uint64_t PrepareMemory(uint64_t, uint64_t) override { return 100; }
    bool Idle() override {
        if (state_->image.empty()) return true;
        auto *ctx = reinterpret_cast<CcuDeviceContext *>(state_->image.data());
        return ccu_detail::At<CcuRequestRecord>(ctx, ctx->record_offset)->slot_state == CcuSlotState::FREE;
    }
    void PreventRequests() override {
        if (!state_->image.empty()) {
            auto *ctx = reinterpret_cast<CcuDeviceContext *>(state_->image.data());
            ccu_detail::At<CcuAdmissionRecord>(ctx, ctx->admission_offset)->closed = 1;
        }
    }
    void Publish(
        const std::vector<CcuRegisteredFunction> &, std::vector<CcuImageBlock> image, CcuDeviceContextView &view
    ) override {
        state_->image = std::move(image);
        auto *ctx = reinterpret_cast<CcuDeviceContext *>(state_->image.data());
        auto *bindings = ccu_detail::At<CcuProgramBinding>(ctx, ctx->bindings_offset);
        for (uint32_t i = 0; i < ctx->binding_count; ++i)
            bindings[i].gate_addr = reinterpret_cast<uint64_t>(&state_->gate);
        view = {reinterpret_cast<uint64_t>(ctx), ctx->image_size, CCU_DEVICE_ABI_VERSION, 0};
    }
    bool Stop(uint64_t) override { return Idle(); }
    void Close() override { state_->image.clear(); }

private:
    std::shared_ptr<ccu_test::SessionControl> state_;
};
}  // namespace
void ccu_test::QueueBackend(std::shared_ptr<SessionControl> control) { next_control = std::move(control); }
void ccu_test::SessionControl::Complete(CcuRequestRef request) {
    if (image.empty()) throw std::runtime_error("No published test context");
    auto *ctx = reinterpret_cast<CcuDeviceContext *>(image.data());
    auto *record = ccu_detail::At<CcuRequestRecord>(ctx, ctx->record_offset);
    if (request.record_addr != reinterpret_cast<uint64_t>(record) || request.sequence != record->sequence ||
        record->slot_state != CcuSlotState::SUBMITTED)
        throw std::runtime_error("No matching submitted test request");
    record->completed_epoch = ctx->session_epoch;
    record->completed_program = record->program_index;
    record->completed_state = CcuRequestState::SUCCEEDED;
    record->status = CcuStatus::OK;
    record->completed_sequence = request.sequence;
    gate = 0;
}
std::unique_ptr<CcuSessionBackend> ccu_make_session_backend() {
    if (!next_control) throw std::runtime_error("No CCU test backend queued on this thread");
    return std::make_unique<Backend>(std::move(next_control));
}
const char *ccu_session_sdk_build_id() { return "ccu-unit-test-sdk"; }
