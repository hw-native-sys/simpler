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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include "common/run_terminal_accumulator.h"

// Owned by the prepared execution, outside every replayed upload/reset range.
// Access is single-writer: entry initializer, then the last callback, then an
// observer after physical exit. A diagnostic never grants or denies execution.
class NativeInvocationEvidence {
public:
    // PublicationFailed denotes invalid producer binding/record state, not an
    // ordinary missing diagnostic or a fallible RTS/DMA transfer.
    enum class Kind : uint8_t { None, Error, MissingTerminal, InvalidTerminal, PublicationFailed };
    struct Snapshot {
        Kind kind{Kind::None};
        RunTerminalSelection terminal{};
        bool publication_succeeded{false};
        size_t payload_bytes{0};
        uint8_t payload[DEVICE_RUN_RESULT_PAYLOAD_BYTES]{};
        uint64_t fault_count{0};
    };

    void begin_after_exit() {
        settle_after_exit();
        pending_ = true;
    }

    // Physical exit must be proven independently, including on timeout. A
    // missing callback cannot write its own Unknown; the next initializer or
    // the post-exit observer records it before the coordination state is reset.
    void settle_after_exit() {
        if (pending_) complete_unknown(Kind::MissingTerminal);
    }

    void complete_unknown(Kind kind) {
        if (!pending_) return;
        pending_ = false;
        retain(kind, {}, nullptr, 0, false);
    }

    void
    complete(RunTerminalSelection terminal, const void *payload, size_t bytes, bool published, bool requested = true) {
        if (!pending_) return;
        pending_ = false;
        if (terminal.verdict == DeviceRunVerdict::Error) {
            retain(Kind::Error, terminal, payload, bytes, published);
        } else if (requested && !published) {
            retain(Kind::PublicationFailed, {}, nullptr, 0, false);
        }
    }

    Snapshot read_after_exit() {
        settle_after_exit();
        return saved_;
    }

    // A read is not an acknowledgement. The owner must retain/copy diagnostics
    // before calling this, and may only do so with exclusive post-exit access.
    bool acknowledge_after_exit() {
        if (pending_) return false;
        saved_ = {};
        return true;
    }

private:
    void retain(Kind kind, RunTerminalSelection terminal, const void *payload, size_t bytes, bool published) {
        if (saved_.fault_count != std::numeric_limits<uint64_t>::max()) ++saved_.fault_count;
        if (saved_.kind != Kind::None) return;
        saved_.kind = kind;
        saved_.terminal = terminal;
        saved_.publication_succeeded = published;
        if (payload != nullptr && bytes <= sizeof(saved_.payload)) {
            saved_.payload_bytes = bytes;
            if (bytes != 0) std::memcpy(saved_.payload, payload, bytes);
        }
    }
    bool pending_{false};
    Snapshot saved_{};
};

// The prefix keeps existing result readers and KernelArgs bindings unchanged.
// The tail must never be included in a replayed initialization upload.
struct NativeInvocationResultStorage {
    DeviceRunResultRegion result{};
    alignas(64) NativeInvocationEvidence evidence{};
};
static_assert(std::is_standard_layout_v<NativeInvocationEvidence>);
static_assert(std::is_trivially_copyable_v<NativeInvocationEvidence>);
static_assert(std::is_standard_layout_v<NativeInvocationEvidence::Snapshot>);
static_assert(std::is_trivially_copyable_v<NativeInvocationEvidence::Snapshot>);
static_assert(std::is_standard_layout_v<NativeInvocationResultStorage>);
static_assert(offsetof(NativeInvocationResultStorage, result) == 0);
static_assert(std::is_trivially_copyable_v<NativeInvocationResultStorage>);
