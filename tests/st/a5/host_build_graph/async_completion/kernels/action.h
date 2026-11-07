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

#include "device.h"

namespace async_case {

inline __aicore__ void action_task(__gm__ int64_t *args) {
    auto *signals = data<int32_t>(args[0]);
    auto *payload = data<int32_t>(args[1]);
    auto *records = data<uint64_t>(args[2]);
    auto *result = data<int32_t>(args[3]);
    auto *control = data<uint64_t>(args[4]);
    const auto action = static_cast<Action>(args[5]);
    const int64_t base = args[6], waiters = args[7], blocks = args[8], mask = args[9];
    const int64_t conditions = args[10], expected = args[11], epoch = args[12], instance = args[13];
    const int64_t milliseconds = args[14], record_index = args[15], auxiliary = args[16];
    auto *record = record_index < 0 ? nullptr : records + record_index * kObservationWords;
    if (record != nullptr) stamp(record, Observation::VISITS);

    if (action == Action::PREREQUISITE) {
        delay(milliseconds);
        add(cell(control, Control::PREREQUISITE, instance), 1);
        if (auxiliary == 3 && observe(cell(control, Control::PUBLISHED, instance)) == 0)
            add(cell(control, Control::BUSINESS_WHILE_PENDING, instance), 1);
    } else if (action == Action::WARM) {
        // Scalar cached reads are intentional; the later consumer does not invalidate payload itself.
        int64_t sum = 0;
        for (int64_t execution = base; execution < base + waiters * blocks * 3; ++execution)
            for (int64_t i = 0; i < kPayloadWords; ++i)
                sum += reinterpret_cast<volatile __gm__ int32_t *>(payload)[execution * kPayloadWords + i];
        add(cell(control, Control::WARMED, instance), 1);
        if (!await_at_least(cell(control, Control::WARMED, instance), get_block_num(args)))
            add(cell(control, Control::RESCUED, instance), 1);
        // Consume the loaded value without changing the communication payload.
        if (sum == INT64_MIN) add(cell(control, Control::RESCUED, instance), 1);
    } else if (action == Action::PUBLISH || action == Action::RESCUE) {
        if (action == Action::RESCUE) {
            if (await_at_least(cell(control, Control::PUBLISHED, instance), 1)) return;
            add(cell(control, Control::RESCUED, instance), 1);
        } else {
            add(cell(control, Control::CAPACITY, instance), auxiliary);
            const uint64_t executions = static_cast<uint64_t>(waiters * blocks * __builtin_popcount(mask));
            add(cell(control, Control::EXECUTIONS, instance), executions);
            if (!await_at_least(cell(control, Control::ARRIVED, instance), executions)) {
                if (record != nullptr) stamp(record, Observation::TIMEOUT);
            }
            delay(milliseconds);
        }
        for (int64_t local = 0; local < waiters * blocks * 3; ++local) {
            if ((mask & (int64_t{1} << (local % 3))) == 0) continue;
            const int64_t execution = base + local;
            bool pending = false;
            for (int64_t c = 0; c < conditions; ++c)
                pending = pending || signal_value(condition(signals, execution, c)) < expected;
            if (pending) {
                add(cell(control, Control::PENDING, instance), 1);
                for (int64_t i = 0; i < kPayloadWords; ++i)
                    payload[execution * kPayloadWords + i] = payload_value(epoch, execution, i);
                flush(payload + execution * kPayloadWords);
                flush(payload + execution * kPayloadWords + kPayloadWords / 2);
            }
            for (int64_t c = 0; c < conditions; ++c) {
                if (conditions > 1 && c + 1 == conditions && milliseconds != 0) delay(milliseconds);
                auto *address = condition(signals, execution, c);
                if (signal_value(address) < expected) publish_signal(address, static_cast<int32_t>(expected));
            }
        }
        add(cell(control, Control::PUBLISHED, instance), 1);
    } else if (action == Action::CONSUME) {
        if (auxiliary == 3)
            add(cell(control, Control::BUSINESS_AT_CONSUMER, instance),
                observe(cell(control, Control::PREREQUISITE, instance)));
        bool ready = auxiliary != 1 || observe(cell(control, Control::PREREQUISITE, instance)) != 0;
        for (int64_t local = 0; local < blocks * 3; ++local) {
            if ((mask & (int64_t{1} << (local % 3))) == 0) continue;
            const int64_t execution = base + local;
            if (auxiliary != 2) ready = ready && observe(records + execution * kObservationWords + 1) == 1;
            for (int64_t c = 0; c < conditions; ++c)
                ready = ready && signal_value(condition(signals, execution, c)) >= expected;
        }
        if (!ready && record != nullptr) stamp(record, Observation::EARLY);
        if (ready) {
            for (int64_t local = 0; local < blocks * 3; ++local) {
                if ((mask & (int64_t{1} << (local % 3))) == 0) continue;
                const int64_t execution = base + local;
                for (int64_t i = 0; i < kPayloadWords; ++i)
                    result[execution * kPayloadWords + i] =
                        reinterpret_cast<volatile __gm__ int32_t *>(payload)[execution * kPayloadWords + i] + 7;
                flush(result + execution * kPayloadWords);
                flush(result + execution * kPayloadWords + kPayloadWords / 2);
            }
        }
    }
    if (record != nullptr) stamp(record, Observation::BODY_DONE);
}

}  // namespace async_case
