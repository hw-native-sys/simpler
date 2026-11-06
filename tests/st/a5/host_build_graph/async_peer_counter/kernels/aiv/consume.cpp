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
#include "../device.h"

using namespace peer_counter;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *payload = data<int32_t>(args[0]);
    auto *signal = data<int32_t>(args[1]);
    auto *output = data<int32_t>(args[2]);
    auto *diagnostic = data<uint64_t>(args[3]);
    const int64_t epoch = args[4], tag = args[5], row = args[6], source_rank = args[7];
    begin(diagnostic, Role::CONSUME, epoch, tag);
    if (!parameters(epoch, tag) || row < 0 || row >= kMaxRows || source_rank != 0) {
        error(diagnostic, Error::BAD_ARGUMENT);
        finish(diagnostic);
        return;
    }
    const int32_t seen_signal = signal_value(signal);
    record(diagnostic, Field::SEEN_SIGNAL, seen_signal);
    if (seen_signal != epoch) {
        if (seen_signal < epoch) record(diagnostic, Field::EARLY);
        error(diagnostic, Error::EARLY_SIGNAL);
    }
    // No payload DCCI here. The executing worker's declared consumer acquire
    // must make these actual peer-written values visible.
    const auto *received = reinterpret_cast<volatile __gm__ const int32_t *>(payload);
    int64_t checksum = 0;
    for (uint32_t element = 0; element < kDataWords; ++element) {
        const int32_t value = received[element];
        if (element == 0) record(diagnostic, Field::SEEN_TAG, value / 100000);
        checksum += value;
        if (value != expected_value(tag, static_cast<uint32_t>(source_rank), static_cast<uint32_t>(row), element))
            error(diagnostic, Error::BAD_PAYLOAD);
        // Preserve a diagnostic output even for arbitrary corrupt input, with
        // unsigned arithmetic avoiding signed overflow on the failure path.
        output[element] = static_cast<int32_t>(static_cast<uint32_t>(value) + UINT32_C(7));
    }
    flush_line(output);
    flush_line(output + 16);
    record(diagnostic, Field::CHECKSUM, checksum);
    record(diagnostic, Field::ROWS);
    finish(diagnostic);
}
