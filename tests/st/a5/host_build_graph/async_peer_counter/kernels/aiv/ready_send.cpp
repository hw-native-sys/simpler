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
#include "../remote_store.h"
#include "async_kernel_api.h"

using namespace peer_counter;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *packet = data<int32_t>(args[0]);
    auto *signal = data<int32_t>(args[1]);
    auto *gate = data<uint64_t>(args[2]);
    auto *diagnostic = data<uint64_t>(args[3]);
    auto *comm = reinterpret_cast<__gm__ const CommContext *>(args[4]);
    const int64_t epoch = args[5], tag = args[6], maximum_rows = args[7];
    begin(diagnostic, Role::READY_SEND, epoch, tag);
    uint64_t remote_packet = 0, remote_signal = 0;
    if (!parameters(epoch, tag) || maximum_rows != kMaxRows ||
        !peer_region(comm, 1, reinterpret_cast<uint64_t>(packet), kReadyWords * sizeof(int32_t), &remote_packet) ||
        !peer_region(comm, 1, reinterpret_cast<uint64_t>(signal), kSignalWords * sizeof(int32_t), &remote_signal)) {
        error(diagnostic, Error::BAD_COMM_CONTEXT);
        finish(diagnostic);
        return;
    }
    const uint64_t rows = observe(gate + static_cast<uint32_t>(Field::ROWS));
    const uint64_t capacity = observe(gate + static_cast<uint32_t>(Field::CAPACITY));
    const uint64_t registered = observe(gate + static_cast<uint32_t>(Field::REGISTERED));
    const uint64_t gate_error = observe(gate + static_cast<uint32_t>(Field::ERROR));
    if (rows == 0 || rows > kMaxRows || capacity == 0 || capacity > kMaxRows || registered > INT32_MAX ||
        gate_error > INT32_MAX) {
        error(diagnostic, Error::BAD_ARGUMENT);
        finish(diagnostic);
        return;
    }
    int32_t status = static_cast<int32_t>(gate_error);
    if (observe(gate + static_cast<uint32_t>(Field::BODY_DONE)) != 1 ||
        observe(gate + static_cast<uint32_t>(Field::EPOCH)) != static_cast<uint64_t>(epoch) ||
        observe(gate + static_cast<uint32_t>(Field::TAG)) != static_cast<uint64_t>(tag) || registered != rows) {
        if (status == 0) status = static_cast<int32_t>(Error::REGISTRATION_MISMATCH);
    }
    for (uint32_t index = 0; index < kReadyWords; ++index)
        packet[index] = kGuard;
    packet[0] = kReadyVersion;
    packet[1] = static_cast<int32_t>(epoch);
    packet[2] = static_cast<int32_t>(tag);
    packet[3] = static_cast<int32_t>(rows);
    packet[4] = static_cast<int32_t>(registered);
    packet[5] = 1;
    packet[6] = status;
    packet[7] = static_cast<int32_t>(capacity);
    flush_line(packet);
    flush_line(packet + 16);
    pto::Tile<pto::TileType::Vec, int32_t, 1, kDataWords> tile;
    TASSIGN(tile, 0);
    remote_store(tile, packet, reinterpret_cast<__gm__ int32_t *>(remote_packet));
    send_notification(
        reinterpret_cast<__gm__ int32_t *>(remote_signal), static_cast<int32_t>(epoch), pto::comm::NotifyOp::Set
    );
    record(diagnostic, Field::ROWS, static_cast<int64_t>(rows));
    record(diagnostic, Field::CAPACITY, static_cast<int64_t>(capacity));
    record(diagnostic, Field::REGISTERED, static_cast<int64_t>(registered));
    record(diagnostic, Field::PUBLISHED);
    if (status != 0) error(diagnostic, status);
    finish(diagnostic);
}
