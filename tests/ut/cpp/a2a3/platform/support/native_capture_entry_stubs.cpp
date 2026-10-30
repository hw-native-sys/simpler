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
// Hardware/configuration boundaries only; kernel.cpp and result publication stay real.
#include <cstdint>
#include <sched.h>

#include "aicpu/aicpu_device_config.h"
#include "aicpu/args_dump_aicpu.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_log.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"

extern bool native_capture_test_affinity;
int platform_aicpu_affinity_thread_idx() { return 0; }
bool platform_aicpu_affinity_gate_filter(const int *, int, int) { return native_capture_test_affinity; }
void init_log_switch() {}
extern "C" int32_t aicpu_release_callables() { return 0; }
extern "C" void set_log_level(int) {}
extern "C" void set_orch_device_id(int) {}
extern "C" void set_scheduler_timeout_ms(int) {}
extern "C" void set_dma_workspace_addr(int, unsigned long long) {}
extern "C" void set_dev_l2_cache_offset(unsigned long long) {}
extern "C" void set_platform_phase_base(uint64_t) {}
extern "C" uint64_t get_platform_phase_base() { return 0; }
extern "C" void set_platform_regs(uint64_t) {}
extern "C" void set_platform_pmu_reg_addrs(uint64_t) {}
extern "C" void set_platform_dump_base(uint64_t) {}
extern "C" void set_platform_dump_args_run_terminal(uint64_t) {}
extern "C" void set_platform_chip_swimlane_base(uint64_t) {}
extern "C" void set_platform_chip_swimlane_run_terminal_bank(uint64_t) {}
extern "C" void set_platform_chip_swimlane_aicore_rotation_table(uint64_t) {}
extern "C" void set_platform_pmu_base(uint64_t) {}
extern "C" void set_platform_pmu_run_terminal(uint64_t) {}
extern "C" void set_platform_dep_gen_base(uint64_t) {}
extern "C" void set_platform_dep_gen_run_terminal(uint64_t) {}
extern "C" void set_platform_scope_stats_base(uint64_t) {}
extern "C" void set_platform_scope_stats_run_terminal(uint64_t) {}
extern "C" void set_dump_args_enabled(bool) {}
extern "C" void set_chip_swimlane_enabled(bool) {}
extern "C" void set_pmu_enabled(bool) {}
extern "C" void set_dep_gen_enabled(bool) {}
extern "C" void set_scope_stats_enabled(bool) {}

#ifdef _WIN32
extern "C" int sched_getscheduler(pid_t) { return SCHED_OTHER; }
extern "C" int sched_setscheduler(pid_t, int, const struct sched_param *) { return 0; }
#endif
