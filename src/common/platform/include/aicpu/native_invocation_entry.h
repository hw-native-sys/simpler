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

#include "aicpu/platform_aicpu_affinity.h"

// Shared invocation protocol; platform gates still decide which callbacks work.
struct NativeInvocationEntry {
    int32_t callbacks;
    uint64_t result_base;
    uint64_t result_epoch;
    int32_t (*begin)(int32_t, uint64_t, uint64_t);
    void (*reset)();
    int32_t (*finish)(int32_t, int32_t);

    static int32_t initialize(void *opaque) {
        auto &entry = *static_cast<NativeInvocationEntry *>(opaque);
        if (entry.begin == nullptr || entry.reset == nullptr || entry.finish == nullptr) return -1;
        const int32_t rc = entry.begin(entry.callbacks, entry.result_base, entry.result_epoch);
        entry.reset();
        return rc;
    }
    AicpuEntryGateInitializer initializer() { return {initialize, this}; }

    template <typename Work>
    int32_t complete(AicpuEntryGateResult gate, Work work) const {
        if (gate.callback_index < 0) return gate.status;
        const int32_t rc = gate.execute ? work() : gate.status;
        // Work has left all local cleanup scopes before its unique contribution.
        return finish != nullptr ? finish(gate.callback_index, rc) : -1;
    }
};

// Dynamic loaders share the same required symbol set and refusal rule.
struct NativeInvocationFunctions {
    int32_t (*begin)(int32_t, uint64_t, uint64_t){nullptr};
    void (*reset)(){nullptr};
    int32_t (*finish)(int32_t, int32_t){nullptr};

    bool ready() const { return begin != nullptr && reset != nullptr && finish != nullptr; }
    template <typename Load>
    bool load(Load load_symbol) {
        *this = {};
        return load_symbol("aicpu_begin_native_invocation", reinterpret_cast<void **>(&begin)) &&
               load_symbol("aicpu_reset_execution_entry", reinterpret_cast<void **>(&reset)) &&
               load_symbol("aicpu_finish_native_invocation", reinterpret_cast<void **>(&finish)) && ready();
    }
    NativeInvocationEntry bind(int32_t callbacks, uint64_t base, uint64_t epoch) const {
        return {callbacks, base, epoch, begin, reset, finish};
    }
};
