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

#include <pto/pto-inst.hpp>

#include <stdexcept>

extern "C" void signal_event(int event) {
    __builtin_cce_ffts_cross_core_sync(PIPE_MTE3, pto::getFFTSMsg(FFTS_MODE_VAL, event));
}

extern "C" void wait_event(int event) { __builtin_cce_wait_flag_dev(event); }

extern "C" void legacy_signal_event(int event) {
    ffts_cross_core_sync(PIPE_MTE3, pto::getFFTSMsg(FFTS_MODE_VAL, event));
}

extern "C" void legacy_wait_event(int event) { wait_flag_dev(event); }

extern "C" bool rejects_mode(int mode) {
    try {
        __builtin_cce_ffts_cross_core_sync(PIPE_MTE3, pto::getFFTSMsg(mode, 0));
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

extern "C" bool rejects_count(int count) {
    try {
        __builtin_cce_ffts_cross_core_sync(PIPE_MTE3, pto::getFFTSMsg(FFTS_MODE_VAL, 0, count));
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}
