#pragma once
#include <cstdint>

// Returns true if this thread should call aicpu_execute().
// Returns false if this thread should exit (dropped).
// logical_count: desired active threads (from runtime.sche_cpu_num)
// total_launched: actual threads launched (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)
bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched);
