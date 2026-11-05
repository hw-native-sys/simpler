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

#include "scheduler_async_directory.h"
#include "scheduler_async_lease.h"

enum class SchedulerAsyncWorkKind : uint8_t { CHECK, FINISH };
enum class SchedulerAsyncDiscoverResult : uint8_t { NONE, CLAIMED, INVALID_STATE };
enum class SchedulerAsyncAdmission : uint8_t { BEFORE_ADMISSION, ADMISSION_DENIED, PERMIT_GRANTED };

struct SchedulerAsyncSliceBudget {
    uint32_t directory_words{4};
    uint32_t lease_probes{2};
    uint32_t claim_attempts{2};
    uint32_t work_units{8};
};

struct SchedulerAsyncDirectoryCursor {
    uint64_t word{0};
    uint32_t bit{0};
};

// A core-local view of an immutable run allocation. These pointers are never
// serialized or handed from one worker to another.
struct SchedulerAsyncDirectoryView {
    __gm__ SchedulerAsyncDirectoryWord *directory{nullptr};
    __gm__ SchedulerAsyncLease *leases{nullptr};
    uint64_t record_count{0};
};

struct SchedulerAsyncDiscoveryState {
    SchedulerAsyncDirectoryCursor check{};
    SchedulerAsyncDirectoryCursor finish{};
    SchedulerAsyncWorkKind next_kind{SchedulerAsyncWorkKind::CHECK};
};

struct SchedulerAsyncClaim {
    SchedulerAsyncWorkKind kind{SchedulerAsyncWorkKind::CHECK};
    uint64_t index{0};
    uint64_t owned_word{0};
};

// Every normal dispatch-loop iteration resets this gate. Both admission paths
// use the same spent bit, and a granted cohort permit always goes to its kernel.
struct SchedulerAsyncServiceGate {
    uint32_t business_returns{0};
    bool spent{false};

    inline __aicore__ void begin_iteration() { spent = false; }
    inline __aicore__ void kernel_returned() {
        if (business_returns < 4) ++business_returns;
    }
    inline __aicore__ bool take(bool enabled, SchedulerAsyncAdmission admission, bool runnable_business) {
        if (!enabled || spent || admission == SchedulerAsyncAdmission::PERMIT_GRANTED) return false;
        if (admission == SchedulerAsyncAdmission::BEFORE_ADMISSION && runnable_business && business_returns < 4)
            return false;
        spent = true;
        business_returns = 0;
        return true;
    }
};

inline __aicore__ uint64_t scheduler_async_directory_word_count(uint64_t records) {
    return records == 0 ? 0 : (records - 1) / 64 + 1;
}

inline __aicore__ SchedulerAsyncDirectoryCursor scheduler_async_directory_start(uint64_t records, uint64_t worker_id) {
    SchedulerAsyncDirectoryCursor cursor{};
    const uint64_t words = scheduler_async_directory_word_count(records);
    if (words != 0) {
        cursor.word = worker_id % words;
        const uint64_t remaining = records - cursor.word * 64;
        const uint64_t bits = remaining < 64 ? remaining : 64;
        cursor.bit = static_cast<uint32_t>((worker_id % bits) * 17 % bits);
    }
    return cursor;
}

inline __aicore__ void scheduler_async_directory_advance(SchedulerAsyncDirectoryCursor *cursor, uint64_t words) {
    cursor->word = cursor->word + 1 == words ? 0 : cursor->word + 1;
    cursor->bit = 0;
}

// The policy only abstracts the runtime-owned atomics for deterministic race
// tests; backend completion addresses never enter this discovery path.
struct SchedulerAsyncDiscoveryAtomics {
    inline __aicore__ static uint64_t directory(__gm__ SchedulerAsyncDirectoryWord &word) {
        return scheduler_async_directory_load(word);
    }
    inline __aicore__ static uint64_t lease(__gm__ SchedulerAsyncLease &value) {
        return scheduler_async_lease_load(value);
    }
    inline __aicore__ static SchedulerAsyncLeaseOutcome claim(__gm__ SchedulerAsyncLease &value, uint64_t observed) {
        return scheduler_async_lease_try_claim(value, observed);
    }
};

template <typename Atomics = SchedulerAsyncDiscoveryAtomics>
inline __aicore__ SchedulerAsyncDiscoverResult scheduler_async_discover_directory(
    const SchedulerAsyncDirectoryView &view, SchedulerAsyncDirectoryCursor *cursor, SchedulerAsyncSliceBudget *budget,
    SchedulerAsyncClaim *claim
) {
    if (view.record_count == 0) return SchedulerAsyncDiscoverResult::NONE;
    const uint64_t words = scheduler_async_directory_word_count(view.record_count);
    if (view.directory == nullptr || view.leases == nullptr || cursor->word >= words || cursor->bit >= 64)
        return SchedulerAsyncDiscoverResult::INVALID_STATE;
    // At most one directory traversal per call, including a final word suffix.
    // A later slice revisits bits below the initial cursor; no empty snapshot
    // persists beyond this invocation.
    uint64_t visited = 0;
    while (budget->directory_words != 0 && budget->lease_probes != 0 && budget->claim_attempts != 0 &&
           visited < words) {
        --budget->directory_words;
        ++visited;
        const uint64_t word_index = cursor->word;
        uint64_t bits = Atomics::directory(view.directory[word_index]);
        const uint64_t valid_count = view.record_count - word_index * 64;
        if (valid_count < 64 && (bits >> valid_count) != 0) return SchedulerAsyncDiscoverResult::INVALID_STATE;
        bits &= UINT64_MAX << cursor->bit;
        while (bits != 0 && budget->lease_probes != 0 && budget->claim_attempts != 0) {
            const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(bits));
            bits &= bits - 1;
            const uint64_t index = word_index * 64 + bit;
            cursor->bit = bit + 1;
            if (cursor->bit == 64 || cursor->bit == valid_count) scheduler_async_directory_advance(cursor, words);
            --budget->lease_probes;
            const uint64_t observed = Atomics::lease(view.leases[index]);
            if (!scheduler_async_lease_valid(observed)) return SchedulerAsyncDiscoverResult::INVALID_STATE;
            if (scheduler_async_lease_phase(observed) == SchedulerAsyncPhase::DORMANT)
                return SchedulerAsyncDiscoverResult::INVALID_STATE;
            if (scheduler_async_lease_phase(observed) != SchedulerAsyncPhase::RUNNABLE) continue;
            --budget->claim_attempts;
            const auto result = Atomics::claim(view.leases[index], observed);
            if (result.result == SchedulerAsyncLeaseResult::SUCCESS) {
                // AICore atomic helpers do not synthesize acquire barriers.
                // The new owner may read mutable progress only after this fence.
                scheduler_cache_barrier();
                claim->index = index;
                claim->owned_word = result.word;
                return SchedulerAsyncDiscoverResult::CLAIMED;
            }
            if (result.result != SchedulerAsyncLeaseResult::CONTENDED &&
                result.result != SchedulerAsyncLeaseResult::UNAVAILABLE)
                return SchedulerAsyncDiscoverResult::INVALID_STATE;
        }
        if (bits == 0 && cursor->word == word_index) scheduler_async_directory_advance(cursor, words);
    }
    return SchedulerAsyncDiscoverResult::NONE;
}

template <typename Atomics = SchedulerAsyncDiscoveryAtomics>
inline __aicore__ SchedulerAsyncDiscoverResult scheduler_async_discover(
    const SchedulerAsyncDirectoryView &checks, const SchedulerAsyncDirectoryView &finishes,
    SchedulerAsyncDiscoveryState *state, SchedulerAsyncSliceBudget *budget, SchedulerAsyncClaim *claim
) {
    const auto first = state->next_kind;
    state->next_kind =
        first == SchedulerAsyncWorkKind::CHECK ? SchedulerAsyncWorkKind::FINISH : SchedulerAsyncWorkKind::CHECK;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        claim->kind = pass == 0 ? first : state->next_kind;
        const bool check = claim->kind == SchedulerAsyncWorkKind::CHECK;
        const auto result = scheduler_async_discover_directory<Atomics>(
            check ? checks : finishes, check ? &state->check : &state->finish, budget, claim
        );
        if (result != SchedulerAsyncDiscoverResult::NONE) return result;
    }
    return SchedulerAsyncDiscoverResult::NONE;
}
