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

#include "common/module_thread_value.h"
#include "host/diagnostic_credit_signal.h"

namespace {
struct Value {
    uintptr_t first;
    uintptr_t second;
};
simpler::ModuleThreadValue<Value> value({17, 29});
}  // namespace

extern "C" __attribute__((visibility("default"))) bool test_module_value(uintptr_t expected, uintptr_t next) {
    const auto before = value.get();
    if (before.first != expected || before.second != expected + 12) return false;
    if (value.try_set({next, next + 12}) != 0) return false;
    using Suppression = simpler::dfx::credit::DiagnosticCreditSignal::SuppressSelfNotify;
    if (Suppression::active()) return false;
    {
        Suppression outer;
        if (!Suppression::active()) return false;
        {
            Suppression inner;
            if (!Suppression::active()) return false;
        }
        if (!Suppression::active()) return false;
    }
    return !Suppression::active();
}
