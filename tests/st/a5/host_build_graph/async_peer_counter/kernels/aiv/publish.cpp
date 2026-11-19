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
    auto *source = data<int32_t>(args[0]);
    auto *payload = data<int32_t>(args[1]);
    auto *signals = data<int32_t>(args[2]);
    auto *packet = data<int32_t>(args[3]);
    auto *diagnostic = data<uint64_t>(args[4]);
    auto *comm = reinterpret_cast<__gm__ const CommContext *>(args[5]);
    const int64_t epoch = args[6], tag = args[7], maximum_rows = args[8], pressure = args[9];
    begin(diagnostic, Role::PUBLISH, epoch, tag);
    uint64_t remote_payload = 0, remote_signals = 0, unused_remote_source = 0, unused_remote_packet = 0;
    if (!parameters(epoch, tag) || maximum_rows != kMaxRows || (pressure != 0 && pressure != 1) ||
        !peer_region(
            comm, 0, reinterpret_cast<uint64_t>(source), kMaxRows * kRowWords * sizeof(int32_t), &unused_remote_source
        ) ||
        !peer_region(
            comm, 0, reinterpret_cast<uint64_t>(payload), kMaxRows * kRowWords * sizeof(int32_t), &remote_payload
        ) ||
        !peer_region(
            comm, 0, reinterpret_cast<uint64_t>(signals), kMaxRows * kSignalWords * sizeof(int32_t), &remote_signals
        ) ||
        !peer_region(
            comm, 0, reinterpret_cast<uint64_t>(packet), kReadyWords * sizeof(int32_t), &unused_remote_packet
        )) {
        error(diagnostic, Error::BAD_COMM_CONTEXT);
        finish(diagnostic);
        return;
    }

    // The runtime prepares this declared consumer input. Do not invalidate the
    // received payload here: doing so would conceal a missing consumer acquire.
    const auto *received = reinterpret_cast<volatile __gm__ const int32_t *>(packet);
    const int32_t version = received[0], received_epoch = received[1], received_tag = received[2];
    const int32_t rows = received[3], registered = received[4], receiver_rank = received[5];
    const int32_t status = received[6], capacity = received[7];
    record(diagnostic, Field::SEEN_TAG, received_tag);
    bool guard_valid = true;
    for (uint32_t index = 8; index < kReadyWords; ++index)
        guard_valid = guard_valid && received[index] == kGuard;
    if (version != kReadyVersion || received_epoch != epoch || received_tag != tag || receiver_rank != 1 || rows <= 0 ||
        rows > static_cast<int32_t>(kMaxRows) || capacity <= 0 || capacity > static_cast<int32_t>(kMaxRows) ||
        registered < 0 || status < 0 || !guard_valid || rows != (pressure == 0 ? 1 : capacity + 1) ||
        (status == 0 && registered != rows)) {
        error(diagnostic, Error::BAD_READY_PACKET);
        finish(diagnostic);
        return;
    }
    record(diagnostic, Field::ROWS, rows);
    record(diagnostic, Field::CAPACITY, capacity);
    record(diagnostic, Field::REGISTERED, registered);
    // A truthful failure packet with valid addresses and bounds still permits
    // one real publication per row to clean up the peer's pending conditions.
    // Its nonzero diagnostic remains a hard failure in the Host oracle.
    if (status != 0) error(diagnostic, status);
    // Fixture latency gives the already-registered receiver conditions a
    // resident observation window. This reads only this core's clock: it does
    // not wait for a peer event or promise a particular CHECK schedule.
    constexpr uint64_t kPublicationDelayMs = 20;
    const uint64_t delay_started = now();
    while (now() - delay_started < kPublicationDelayMs * ticks_per_ms()) {}
    auto *peer_payload = reinterpret_cast<__gm__ int32_t *>(remote_payload);
    auto *peer_signals = reinterpret_cast<__gm__ int32_t *>(remote_signals);
    pto::Tile<pto::TileType::Vec, int32_t, 1, kDataWords> tile;
    TASSIGN(tile, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(rows); ++row) {
        remote_store(tile, source + row * kRowWords, peer_payload + row * kRowWords);
        send_notification(peer_signals + row * kSignalWords, static_cast<int32_t>(epoch), pto::comm::NotifyOp::Set);
        record(diagnostic, Field::PUBLISHED);
    }
    finish(diagnostic);
}
