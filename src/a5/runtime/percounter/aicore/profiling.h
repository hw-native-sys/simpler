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
#include "scheduler/profiling_memory.h"

namespace simpler::percounter {
namespace {

class ProfileScope {
public:
    inline __aicore__ ProfileScope(PercounterScalarTrace *trace, ProfilePhase phase) :
        trace_(trace),
        phase_(phase) {
        if constexpr (profile_compiled) {
            if (trace_) {
                depth_ = ++trace_->depth;
                begin_ = percounter_scalar_clock();
            }
        }
    }
    inline __aicore__ void close() {
        if constexpr (profile_compiled) {
            if (trace_) {
                const auto end = percounter_scalar_clock();
                --trace_->depth;
                trace_->append_function(phase_, begin_, end, -1, depth_);
                trace_ = nullptr;
            }
        }
    }
    inline __aicore__ ~ProfileScope() { close(); }
    ProfileScope(const ProfileScope &) = delete;
    ProfileScope &operator=(const ProfileScope &) = delete;

private:
    PercounterScalarTrace *trace_;
    ProfilePhase phase_;
    uint64_t begin_{0};
    uint32_t depth_{0};
};
}  // namespace
}  // namespace simpler::percounter
