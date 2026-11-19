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

#include "inner_kernel.h"
#include "runtime_types.h"
#include "scheduler/profiling_record.h"

namespace simpler::percounter {
namespace {
template <>
struct ProfileCompilation<> {
    static constexpr bool enabled = true;
};
}  // namespace
}  // namespace simpler::percounter

#include "scheduler/async_completion.h"
#include "scheduler/ssbuf.h"

#include <gtest/gtest.h>
#include <array>
#include <vector>

void exercise_unprofiled_memory_helpers();

namespace {
using namespace simpler::percounter;

struct Observation {
    std::array<uint64_t, 64> words{};
    PercounterProfileBuffer header{};
    PercounterScalarTrace trace{};
    PercounterScalarBinding binding;

    Observation() {
        trace.records = words.data();
        trace.buffer = &header;
        trace.window_start = percounter_scalar_clock();
        binding.attach(&trace);
    }

    std::vector<ProfilePhase> phases() const {
        std::vector<ProfilePhase> result;
        uint32_t cursor = 0;
        while (cursor < trace.storage_words) {
            SchedulerScalarRecord record{};
            EXPECT_TRUE(
                percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, trace.window_start, record)
            );
            result.push_back(static_cast<ProfilePhase>(record.kind & 0xffffU));
        }
        return result;
    }
};

TEST(PercounterScalarMemory, ProfiledCounterReadsKeepTheirTranslationUnitPolicy) {
    Observation observation;
    exercise_unprofiled_memory_helpers();
    EXPECT_TRUE(observation.phases().empty());
    uint32_t value = 7;
    uint32_t (*volatile read)(volatile uint32_t &, int) = &percounter_gm_read;
    EXPECT_EQ(read(value, __ATOMIC_ACQUIRE), 7);
    const CompletionToken token{
        reinterpret_cast<uint64_t>(&value), 7, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
    };
    CompletionPollResult (*volatile poll)(const CompletionToken &) = &percounter_poll_valid_completion;
    EXPECT_EQ(poll(token).state, CompletionPollState::READY);
    EXPECT_EQ(observation.phases(), (std::vector{ProfilePhase::BypassLoad, ProfilePhase::BypassLoad}));
}

TEST(PercounterScalarMemory, SimulatedSynchronizationTokensRetainEveryAccess) {
    Observation observation;
    uint32_t token = 0;
    percounter_ssbuf_store_release(&token, 11);
    EXPECT_EQ(percounter_ssbuf_load_acquire(&token), 11);
    EXPECT_EQ(observation.phases(), (std::vector{ProfilePhase::SsbufStore, ProfilePhase::SsbufLoad}));
}

TEST(PercounterScalarMemory, SimulatedCcuReadAndClearRetainEveryAccess) {
    Observation observation;
    uint64_t event = 9;
    uint64_t (*volatile read)(uint64_t *) = &percounter_ccu_event_read;
    void (*volatile clear)(uint64_t *) = &percounter_ccu_event_clear;
    EXPECT_EQ(read(&event), 9);
    clear(&event);
    EXPECT_EQ(read(&event), 0);
    EXPECT_EQ(
        observation.phases(),
        (std::vector{ProfilePhase::BypassLoad, ProfilePhase::BypassStoreIssue, ProfilePhase::BypassLoad})
    );
}
}  // namespace
