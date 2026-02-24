#include "aicpu/device_time.h"

#include <chrono>

#include "common/platform_config.h"

uint64_t get_sys_cnt_aicpu() {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    constexpr uint64_t kNsPerSec = std::nano::den;
    uint64_t seconds = elapsed_ns / kNsPerSec;
    uint64_t remaining_ns = elapsed_ns % kNsPerSec;
    uint64_t ticks = seconds * PLATFORM_PROF_SYS_CNT_FREQ +
                     (remaining_ns * PLATFORM_PROF_SYS_CNT_FREQ) / kNsPerSec;
    return ticks;
}
