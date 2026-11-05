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
// Actual registration and lookup state in the runtime's own translation unit.
#include "aicpu_loader/finalize_args.h"
#include "aicpu_executor.cpp"  // NOLINT(bugprone-suspicious-include)

extern "C" void unified_log_info(const char *, const char *, ...) {}
extern "C" void unified_log_error(const char *, const char *, ...) {}
extern "C" void unified_log_timing(const char *, const char *, ...) {}
extern "C" void unified_log_debug(const char *, const char *, ...) {}
int bind_orchestration_host_log_state(void *, const char **) { return 0; }
int get_orch_device_id() { return 0; }
extern "C" __attribute__((visibility("default"))) int probe_selected_config(int slot) {
    if (slot < 0 || slot >= MAX_REGISTERED_CALLABLE_IDS) return -1;
    const auto &entry = g_aicpu_executor.orch_so_table_[slot];
    if (entry.config_func == nullptr) return -1;
    ChipTaskArgs args{};
    return entry.config_func(args).expected_arg_count;
}
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *) { return 0; }
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *) { return 0; }

namespace {
int fail_close_at = -1;
int close_calls = 0;
int CloseCallable(void *handle) {
    if (close_calls++ == fail_close_at) return -1;
    return dlclose(handle);
}
}  // namespace
extern "C" __attribute__((visibility("default"))) void probe_fail_close_at(int attempt) {
    fail_close_at = attempt;
    close_calls = 0;
}
extern "C" __attribute__((visibility("default"))) int probe_live_handles() {
    int count = 0;
    for (const auto &entry : g_aicpu_executor.orch_so_table_)
        if (entry.handle != nullptr) ++count;
    return count;
}
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_finalize(void *arg) {
    auto *args = static_cast<AicpuFinalizeArgs *>(arg);
    if (args == nullptr || args->result_addr == 0) return -1;
    auto *result = reinterpret_cast<AicpuFinalizeResult *>(args->result_addr);
    result->status = g_aicpu_executor.release_callables(CloseCallable);
    result->completed = 1;
    return 0;
}
