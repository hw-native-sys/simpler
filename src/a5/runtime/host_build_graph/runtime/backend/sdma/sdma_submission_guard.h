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

#include "async_completion_writer.h"
#include "async_sdma_guard.h"
#include "scheduler/scheduler_memory.h"

namespace pto2::detail {
enum class SdmaGuardResult : uint8_t { ACQUIRED, BUSY, FAILED };

inline __aicore__ SdmaGuardResult sdma_guard_try_acquire(__gm__ AsyncSdmaSubmissionGuard &guard) {
    const uint64_t previous = scheduler_gm_compare_exchange(guard.word, UINT64_C(0), UINT64_C(1));
    if (previous == 0) {
        scheduler_cache_barrier();
        return SdmaGuardResult::ACQUIRED;
    }
    return previous == 1 ? SdmaGuardResult::BUSY : SdmaGuardResult::FAILED;
}

inline __aicore__ bool sdma_guard_release(__gm__ AsyncSdmaSubmissionGuard &guard, bool success) {
    scheduler_cache_barrier();
    const uint64_t previous =
        scheduler_gm_compare_exchange(guard.word, UINT64_C(1), success ? UINT64_C(0) : UINT64_C(2));
    scheduler_cache_barrier();
    return previous == 1;
}

struct SdmaSubmissionClock {
    inline __aicore__ static uint64_t now() {
#if defined(__CCE_AICORE__)
        return get_sys_cnt();
#elif defined(__CPU_SIM)
        return get_sys_cnt_aicore();
#else
        return 0;
#endif
    }
};

struct SdmaSubmissionMetadata {
    inline __aicore__ void publish(__gm__ uint8_t *workspace) const {
#if defined(__CCE_AICORE__)
        // SDK session reconstruction must not observe a predecessor's dirty SQ
        // tail or overwrite a newer tail with that predecessor's cached copy.
        pipe_barrier(PIPE_ALL);
        dcci(workspace, ENTIRE_DATA_CACHE, CACHELINE_OUT);
        dsb(DSB_DDR);
#else
        (void)workspace;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    }
};

struct SdmaMetadataAlreadyPublished {
    inline __aicore__ void publish(__gm__ uint8_t *) const {}
};

// The lock covers session creation, submission and publication of SDK metadata,
// not the lifetime of the asynchronous transfer. CHECK observes completion after
// this scope has released the submitting core's channel group.
class SdmaSubmissionScope {
public:
    SdmaSubmissionScope() = default;
    SdmaSubmissionScope(const SdmaSubmissionScope &) = delete;
    SdmaSubmissionScope &operator=(const SdmaSubmissionScope &) = delete;

    template <typename Clock = SdmaSubmissionClock>
    inline __aicore__ bool acquire(AsyncCtx &ctx, uint32_t first_queue, uint32_t queue_count = 1) {
        if (ctx.sdma_guard_context == 0) return true;  // Existing legacy/AICPU ownership contract.
        const auto *context = reinterpret_cast<__gm__ const AsyncSdmaGuardContext *>(ctx.sdma_guard_context);
        if ((ctx.task_token.is_valid() && !async_completion_writer_valid(ctx)) ||
            !async_sdma_guard_context_valid(context) || queue_count == 0 || first_queue >= context->guard_count ||
            queue_count > context->guard_count - first_queue)
            return reject(ctx);
        guard_ = reinterpret_cast<__gm__ AsyncSdmaSubmissionGuard *>(ctx.sdma_guard_context + sizeof(*context)) +
                 first_queue;
        const uint64_t started = Clock::now();
        constexpr uint64_t timeout = UINT64_C(5) * PLATFORM_PROF_SYS_CNT_FREQ;
        // All callers acquire physical queues in ascending order. A partial
        // acquisition has no SDK side effects and is released on failure.
        while (owned_ < queue_count) {
            const auto result = sdma_guard_try_acquire(guard_[owned_]);
            if (result == SdmaGuardResult::ACQUIRED) {
                ++owned_;
                continue;
            }
            if (result == SdmaGuardResult::FAILED || Clock::now() - started >= timeout) {
                unlock(true);
                guard_ = nullptr;
                return reject(ctx);
            }
        }
        return true;
    }

    template <typename Metadata = SdmaSubmissionMetadata>
    inline __aicore__ bool release(bool success, __gm__ uint8_t *workspace, Metadata metadata = {}) {
        if (!owned_) return guard_ == nullptr;
        if (success) metadata.publish(workspace);
        return unlock(success);
    }

    inline __aicore__ ~SdmaSubmissionScope() {
        if (owned_) (void)unlock(false);
    }

private:
    __gm__ AsyncSdmaSubmissionGuard *guard_{nullptr};
    uint32_t owned_{0};
    inline __aicore__ bool unlock(bool success) {
        bool valid = true;
        while (owned_ != 0) {
            --owned_;
            valid = sdma_guard_release(guard_[owned_], success && valid) && valid;
        }
        return valid;
    }
    inline __aicore__ static bool reject(AsyncCtx &ctx) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
};
}  // namespace pto2::detail
