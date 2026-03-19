#include "aicpu/platform_aicpu_affinity.h"

#include <atomic>
#include <cstdint>

#include "common/unified_log.h"

static std::atomic<int32_t> s_thread_counter{0};
static std::atomic<int32_t> s_cleanup_counter{0};

bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched) {
    if (logical_count >= total_launched) {
        return true;
    }

    int32_t idx = s_thread_counter.fetch_add(1, std::memory_order_acq_rel);
    bool survive = (idx < logical_count);

    if (!survive) {
        LOG_INFO("AICPU affinity gate (sim): thread idx=%d DROPPED (logical=%d, launched=%d)",
                 idx, logical_count, total_launched);
    }

    // Last thread resets state for next invocation
    int32_t cleanup_idx = s_cleanup_counter.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_thread_counter.store(0, std::memory_order_release);
        s_cleanup_counter.store(0, std::memory_order_release);
    }

    return survive;
}
