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
#include "ccu_service_program.h"

#include <array>
#include <algorithm>
#include <stdexcept>
#include "hcomm/ccu/ccu_primitives.hpp"

namespace {
namespace ccu = AscendC::ccu;
void Check(CcuResult rc) {
    if (rc != CCU_SUCCESS) throw std::runtime_error("CCU service build failed");
}

class Builder final : public CcuBuildContext {
public:
    Builder(
        const CcuServiceBuildArgs &build, const CcuRegisteredFunction &function, ccu::Variable args,
        ccu::Variable peer_args, ccu::Variable status
    ) :
        build_(build),
        function_(function),
        args_(args),
        peer_args_(peer_args),
        status_(status) {}
    uint32_t rank() const override { return build_.rank; }
    uint32_t rank_count() const override { return 2; }
    CcuBuildTensor tensor(const char *name) override { return Tensor(name, args_); }
    CcuBuildTensor peer_tensor(const char *name) override { return Tensor(name, peer_args_); }
    CcuSymbol scalar(const char *name) override { return Scalar(name, args_); }
    CcuSymbol peer_scalar(const char *name) override { return Scalar(name, peer_args_); }
    CcuSymbol constant(uint64_t value) override {
        ccu::Variable variable;
        variable = value;
        return Save(variable);
    }
    CcuSymbol multiply(CcuSymbol value, uint16_t factor) override {
        if (factor != 0) (void)require_le(value, constant(UINT64_MAX / factor));
        ccu::Variable variable;
        variable = Get(value) * factor;
        return Save(variable);
    }
    CcuStatus require_le(CcuSymbol value, CcuSymbol bound) override {
        const auto label = Label("require_le");
        Check(CcuIfBeginVar(Get(value).handle, Get(bound).handle, CCU_CONDITION_GT, label.c_str()));
        status_ = static_cast<uint64_t>(CcuStatus::INVALID_ARGUMENT);
        Check(CcuIfEnd(label.c_str()));
        return CcuStatus::OK;
    }
    CcuStatus require_equal(CcuSymbol lhs, CcuSymbol rhs) override {
        const auto label = Label("require_equal");
        Check(CcuIfBeginVar(Get(lhs).handle, Get(rhs).handle, CCU_CONDITION_NE, label.c_str()));
        status_ = static_cast<uint64_t>(CcuStatus::INVALID_ARGUMENT);
        Check(CcuIfEnd(label.c_str()));
        return CcuStatus::OK;
    }
    CcuStatus local_copy(CcuBuildTensor destination, CcuBuildTensor source, CcuSymbol bytes) override {
        (void)require_le(bytes, source.byte_size);
        (void)require_le(bytes, destination.byte_size);
        (void)require_le(bytes, constant(4096));
        const auto label = BeginGuard();
        ccu::LocalAddr src, dst;
        src.addr = Get(source.address);
        src.token = Get(source.token);
        dst.addr = Get(destination.address);
        dst.token = Get(destination.token);
        ccu::CcuBuffer buffer;
        ccu::Event event;
        Check(ccu::LocalCopy(buffer, src, Get(bytes), event));
        Check(ccu::EventWait(event));
        Check(ccu::LocalCopy(dst, buffer, Get(bytes), event));
        Check(ccu::EventWait(event));
        Check(CcuIfEnd(label.c_str()));
        return CcuStatus::OK;
    }
    CcuStatus write_peer(CcuBuildTensor destination, CcuBuildTensor source, CcuSymbol bytes) override {
        (void)require_le(bytes, source.byte_size);
        (void)require_le(bytes, destination.byte_size);
        const auto label = BeginGuard();
        ccu::LocalAddr src;
        ccu::RemoteAddr dst;
        src.addr = Get(source.address);
        src.token = Get(source.token);
        dst.addr = Get(destination.address);
        dst.token = Get(destination.token);
        ccu::Event event;
        Check(ccu::Write(build_.channel, dst, src, Get(bytes), event));
        Check(ccu::EventWait(event));
        Check(CcuIfEnd(label.c_str()));
        return CcuStatus::OK;
    }
    CcuStatus peer_barrier() override {
        Check(ccu::NotifyRecord(build_.channel, 1, 1));
        Check(ccu::NotifyWait(build_.channel, 1, 1));
        return CcuStatus::OK;
    }

private:
    std::string Label(const char *prefix) {
        std::string key;
        static constexpr char digits[] = "0123456789abcdef";
        for (auto byte : function_.spec.program_key.bytes) {
            key += digits[byte >> 4];
            key += digits[byte & 15];
        }
        return key + "." + prefix + "." + std::to_string(next_label_++);
    }
    std::string BeginGuard() {
        auto label = Label("operation");
        Check(CcuIfBegin(status_.handle, 0, CCU_CONDITION_EQ, label.c_str()));
        return label;
    }
    CcuSymbol Save(const ccu::Variable &value) {
        variables_.push_back(value);
        return {variables_.size() - 1};
    }
    const ccu::Variable &Get(CcuSymbol value) const { return variables_.at(value.index); }
    size_t Index(const char *name, CcuArgumentKind kind) const {
        const auto it = std::find(function_.names.begin(), function_.names.end(), name);
        if (it == function_.names.end()) throw std::runtime_error("Unknown CCU parameter");
        const auto index = static_cast<size_t>(it - function_.names.begin());
        if (function_.parameters[index].kind != kind) throw std::runtime_error("CCU parameter role mismatch");
        return index;
    }
    CcuSymbol Load(ccu::Variable base, uint32_t word) {
        if (word * 8 > UINT16_MAX) throw std::runtime_error("CCU argument offset exceeds build ABI");
        ccu::Variable address, value;
        address = base + static_cast<uint16_t>(word * 8);
        Check(ccu::Load(address, value));
        return Save(value);
    }
    CcuBuildTensor Tensor(const char *name, ccu::Variable base) {
        const auto word = function_.parameters[Index(name, CcuArgumentKind::TENSOR)].word_offset;
        return {Load(base, word), Load(base, word + 1), Load(base, word + 2)};
    }
    CcuSymbol Scalar(const char *name, ccu::Variable base) {
        return Load(base, function_.parameters[Index(name, CcuArgumentKind::SCALAR)].word_offset);
    }
    const CcuServiceBuildArgs &build_;
    const CcuRegisteredFunction &function_;
    ccu::Variable args_, peer_args_, status_;
    std::vector<ccu::Variable> variables_;
    uint32_t next_label_{};
};

void BuiltinBroadcast(Builder &builder, uint32_t root) {
    const auto src = builder.tensor("src"), dst = builder.tensor("dst"), remote_dst = builder.peer_tensor("dst");
    const auto count = builder.scalar("count");
    (void)builder.require_le(builder.constant(1), count);
    (void)builder.require_le(count, builder.constant(1024));
    (void)builder.require_equal(count, builder.peer_scalar("count"));
    const auto bytes = builder.multiply(count, 4);
    (void)builder.require_le(bytes, builder.rank() == root ? src.byte_size : builder.peer_tensor("src").byte_size);
    (void)builder.require_le(bytes, dst.byte_size);
    (void)builder.require_le(bytes, remote_dst.byte_size);
    if (builder.rank() == root) {
        (void)builder.local_copy(dst, src, bytes);
        (void)builder.write_peer(remote_dst, src, bytes);
    }
}

void LoadOffset(ccu::Variable base, uint16_t offset, ccu::Variable value) {
    ccu::Variable address;
    address = base + offset;
    Check(ccu::Load(address, value));
}
void StoreOffset(ccu::Variable base, uint16_t offset, ccu::Variable value) {
    ccu::Variable address;
    address = base + offset;
    Check(ccu::Store(address, value));
}
void Transfer(
    ChannelHandle channel, ccu::Variable source, ccu::Variable source_token, ccu::Variable destination,
    ccu::Variable destination_token, uint64_t size
) {
    ccu::LocalAddr src;
    ccu::RemoteAddr dst;
    ccu::Variable bytes;
    ccu::Event copied;
    src.addr = source;
    src.token = source_token;
    dst.addr = destination;
    dst.token = destination_token;
    bytes = size;
    Check(ccu::Write(channel, dst, src, bytes, copied));
    Check(ccu::EventWait(copied));
}
}  // namespace

CcuResult CcuServiceProgram(const CcuServiceBuildArgs *build) noexcept {
    try {
        ccu::Variable record, args, context_token, mailbox, mailbox_token, peer_mailbox, peer_token, scratch,
            scratch_token;
        const std::array<ccu::Variable, 9> task{record,       args,       context_token, mailbox,      mailbox_token,
                                                peer_mailbox, peer_token, scratch,       scratch_token};
        for (uint32_t i = 0; i < task.size(); ++i)
            Check(ccu::LoadArg(task[i], i));
        ccu::Variable running, program, sequence, epoch, status, last_sequence, last_peer_sequence;
        running = 1;
        last_sequence = 0;
        last_peer_sequence = 0;
        ccu::Event gate(build->gate);
        Check(CcuWhileBegin(running.handle, 0, CCU_CONDITION_NE, "service"));
        Check(ccu::EventWait(gate, 1));
        LoadOffset(record, offsetof(CcuRequestRecord, program_index), program);
        Check(CcuIfBegin(program.handle, UINT64_MAX, CCU_CONDITION_EQ, "stop"));
        running = 0;
        Check(CcuIfElse("stop"));
        LoadOffset(record, offsetof(CcuRequestRecord, sequence), sequence);
        LoadOffset(record, offsetof(CcuRequestRecord, session_epoch), epoch);
        status = static_cast<uint64_t>(CcuStatus::OK);
        ccu::Variable peer_arguments, remote_arguments;
        peer_arguments = mailbox + uint16_t{128};
        remote_arguments = peer_mailbox + uint16_t{128};
        Transfer(build->channel, record, context_token, peer_mailbox, peer_token, 128);
        Transfer(build->channel, args, context_token, remote_arguments, peer_token, build->argument_bytes);
        Check(ccu::NotifyRecord(build->channel, 0, 1));
        Check(ccu::NotifyWait(build->channel, 0, 1));
        ccu::Variable peer_program, peer_epoch, peer_sequence, slot, peer_slot;
        LoadOffset(mailbox, offsetof(CcuRequestRecord, program_index), peer_program);
        LoadOffset(mailbox, offsetof(CcuRequestRecord, session_epoch), peer_epoch);
        LoadOffset(mailbox, offsetof(CcuRequestRecord, sequence), peer_sequence);
        LoadOffset(record, offsetof(CcuRequestRecord, slot_state), slot);
        LoadOffset(mailbox, offsetof(CcuRequestRecord, slot_state), peer_slot);
        Check(CcuIfBeginVar(sequence.handle, last_sequence.handle, CCU_CONDITION_LE, "sequence_order"));
        status = static_cast<uint64_t>(CcuStatus::STALE_HANDLE);
        Check(CcuIfEnd("sequence_order"));
        Check(CcuIfBeginVar(peer_sequence.handle, last_peer_sequence.handle, CCU_CONDITION_LE, "peer_sequence_order"));
        status = static_cast<uint64_t>(CcuStatus::STALE_HANDLE);
        Check(CcuIfEnd("peer_sequence_order"));
        Check(
            CcuIfBegin(slot.handle, static_cast<uint64_t>(CcuSlotState::SUBMITTED), CCU_CONDITION_NE, "submitted_slot")
        );
        status = static_cast<uint64_t>(CcuStatus::INVALID_ARGUMENT);
        Check(CcuIfEnd("submitted_slot"));
        Check(CcuIfBegin(
            peer_slot.handle, static_cast<uint64_t>(CcuSlotState::SUBMITTED), CCU_CONDITION_NE, "peer_submitted_slot"
        ));
        status = static_cast<uint64_t>(CcuStatus::INVALID_ARGUMENT);
        Check(CcuIfEnd("peer_submitted_slot"));
        Check(CcuIfBeginVar(program.handle, peer_program.handle, CCU_CONDITION_NE, "peer_program"));
        status = static_cast<uint64_t>(CcuStatus::INVALID_ARGUMENT);
        Check(CcuIfEnd("peer_program"));
        Check(CcuIfBegin(epoch.handle, build->epoch, CCU_CONDITION_NE, "epoch"));
        status = static_cast<uint64_t>(CcuStatus::STALE_HANDLE);
        Check(CcuIfEnd("epoch"));
        Check(CcuIfBegin(peer_epoch.handle, build->epoch, CCU_CONDITION_NE, "peer_epoch"));
        status = static_cast<uint64_t>(CcuStatus::STALE_HANDLE);
        Check(CcuIfEnd("peer_epoch"));
        Check(CcuIfBegin(program.handle, build->functions->size(), CCU_CONDITION_GE, "program_range"));
        status = static_cast<uint64_t>(CcuStatus::INVALID_ARGUMENT);
        Check(CcuIfEnd("program_range"));
        Check(CcuIfBegin(status.handle, 0, CCU_CONDITION_EQ, "valid_request"));
        for (size_t i = 0; i < build->functions->size(); ++i) {
            const auto label = "program." + std::to_string(i);
            Check(CcuIfBegin(program.handle, i, CCU_CONDITION_EQ, label.c_str()));
            const auto &function = build->functions->at(i);
            Builder builder(*build, function, args, peer_arguments, status);
            if (function.spec.library_path.empty())
                BuiltinBroadcast(builder, function.spec.entry == "builtin.broadcast.root0" ? 0 : 1);
            else if (function.entry == nullptr || function.entry(&builder) != 0)
                throw std::runtime_error("User CCU builder rejected function");
            Check(CcuIfEnd(label.c_str()));
        }
        Check(CcuIfEnd("valid_request"));
        Check(ccu::NotifyRecord(build->channel, 0, 2));
        Check(ccu::NotifyWait(build->channel, 0, 2));
        Check(ccu::Store(scratch, status));
        ccu::Variable remote_status_address, peer_status_address;
        const auto status_offset = static_cast<uint16_t>(128 + build->argument_bytes);
        remote_status_address = peer_mailbox + status_offset;
        peer_status_address = mailbox + status_offset;
        Transfer(build->channel, scratch, scratch_token, remote_status_address, peer_token, 8);
        Check(ccu::NotifyRecord(build->channel, 0, 4));
        Check(ccu::NotifyWait(build->channel, 0, 4));
        ccu::Variable peer_status;
        Check(ccu::Load(peer_status_address, peer_status));
        Check(CcuIfBegin(peer_status.handle, 0, CCU_CONDITION_NE, "peer_status"));
        status = static_cast<uint64_t>(CcuStatus::EXECUTION_FAILED);
        Check(CcuIfEnd("peer_status"));
        ccu::Variable completion_word, high, zero;
        high = uint64_t{1} << 32;
        zero = 0;
        completion_word = static_cast<uint64_t>(CcuRequestState::SUCCEEDED);
        Check(CcuIfBegin(status.handle, 0, CCU_CONDITION_NE, "failed"));
        completion_word = status * high;
        completion_word = completion_word + static_cast<uint16_t>(CcuRequestState::FAILED);
        Check(CcuIfEnd("failed"));
        StoreOffset(record, offsetof(CcuRequestRecord, completed_epoch), epoch);
        StoreOffset(record, offsetof(CcuRequestRecord, completed_program), program);
        StoreOffset(record, offsetof(CcuRequestRecord, completed_state), completion_word);
        StoreOffset(record, offsetof(CcuRequestRecord, native_error), zero);
        last_sequence = sequence;
        last_peer_sequence = peer_sequence;
        StoreOffset(record, offsetof(CcuRequestRecord, completed_sequence), sequence);
        Check(CcuIfEnd("stop"));
        Check(CcuWhileEnd("service"));
        return CCU_SUCCESS;
    } catch (...) {
        return CCU_E_INTERNAL;
    }
}
