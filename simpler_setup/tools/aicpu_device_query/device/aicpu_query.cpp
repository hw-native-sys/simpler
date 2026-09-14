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
//
// aicpu_query.cpp — device-side AICPU SO that runs HAL queries plus the full
// affinity preflight (serial enum helper, atomic-flag handshake orch election,
// COND die scoring for non-orchestrator threads).

#include <cstdint>
#include <cstring>
#include <sched.h>

#include <driver/ascend_hal_base.h>

#include "common/platform_config.h"
#include "../shared/rtt_probe_types.h"

namespace {

constexpr uint32_t kHalSuccess = 0;

struct DeviceArgs {
    uint64_t reserved_pre[12];
    uint64_t q_input_addr;
    uint64_t q_input_count;
    uint64_t q_output_addr;
};

struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

#pragma pack(push, 4)
struct QueryRequest {
    int32_t module_type;
    int32_t info_type;
};
struct QueryResult {
    int32_t rc;
    int32_t _pad;
    int64_t value;
};
#pragma pack(pop)
static_assert(sizeof(QueryRequest) == 8, "QueryRequest size drift");
static_assert(sizeof(QueryResult) == 16, "QueryResult size drift");

extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);

constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelError = 3;

void DiagLog(const char *msg) { DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[aicpu-query] %s", msg); }

inline uint64_t SysCntAicpu() {
    uint64_t value;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(value));
    return value;
}

inline uint64_t SysCntFrequencyAicpu() {
    uint64_t value;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
}

void TouchDie(const uint64_t *aicore_regs, uint32_t first_core, uint32_t cores_per_die, uint32_t samples_per_core) {
    volatile uint32_t sink = 0;
    for (uint32_t core = first_core; core < first_core + cores_per_die; ++core) {
        auto *cond = reinterpret_cast<volatile uint32_t *>(aicore_regs[core] + REG_SPR_COND_OFFSET);
        for (uint32_t sample = 0; sample < samples_per_core; ++sample) {
            sink = *cond;
        }
    }
    (void)sink;
}

uint64_t MeasureDieTotalTicks(
    const uint64_t *aicore_regs, uint32_t first_core, uint32_t cores_per_die, uint32_t samples_per_core
) {
    const uint64_t begin = SysCntAicpu();
    TouchDie(aicore_regs, first_core, cores_per_die, samples_per_core);
    return SysCntAicpu() - begin;
}

void SetProbeError(AffinityPreflightOutput *output, AffinityProbeError error) {
    uint32_t expected = static_cast<uint32_t>(AffinityProbeError::kNone);
    (void)__atomic_compare_exchange_n(
        &output->error_code, &expected, static_cast<uint32_t>(error), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
    );
}

bool WaitForProbeValue(AffinityPreflightOutput *output, uint32_t *value, uint32_t target) {
    const uint64_t begin = SysCntAicpu();
    const uint64_t timeout_ticks = SysCntFrequencyAicpu() * kAffinityBarrierTimeoutSeconds;
    while (__atomic_load_n(value, __ATOMIC_ACQUIRE) < target) {
        if (__atomic_load_n(&output->error_code, __ATOMIC_ACQUIRE) !=
            static_cast<uint32_t>(AffinityProbeError::kNone)) {
            return false;
        }
        if (SysCntAicpu() - begin >= timeout_ticks) {
            SetProbeError(output, AffinityProbeError::kBarrierTimeout);
            return false;
        }
        __asm__ volatile("yield");
    }
    return true;
}

bool WaitEq(AffinityPreflightOutput *output, volatile uint32_t *cell, uint32_t expected) {
    const uint64_t begin = SysCntAicpu();
    const uint64_t timeout_ticks = SysCntFrequencyAicpu() * kAffinityBarrierTimeoutSeconds;
    while (__atomic_load_n(cell, __ATOMIC_ACQUIRE) != expected) {
        if (__atomic_load_n(&output->error_code, __ATOMIC_ACQUIRE) !=
            static_cast<uint32_t>(AffinityProbeError::kNone)) {
            return false;
        }
        if (SysCntAicpu() - begin >= timeout_ticks) {
            SetProbeError(output, AffinityProbeError::kBarrierTimeout);
            return false;
        }
        __asm__ volatile("yield");
    }
    return true;
}

uint64_t HandshakePair(
    AffinityPreflightOutput *output, uint32_t initiator, uint32_t responder, uint32_t my_idx, uint32_t iters
) {
    volatile uint32_t *req = &output->handshake_req[initiator][responder];
    volatile uint32_t *ack = &output->handshake_ack[responder][initiator];
    const uint64_t begin = SysCntAicpu();
    for (uint32_t iter = 1; iter <= iters; ++iter) {
        if (my_idx == initiator) {
            __atomic_store_n(req, iter, __ATOMIC_RELEASE);
            if (!WaitEq(output, ack, iter)) return 0;
        } else if (my_idx == responder) {
            if (!WaitEq(output, req, iter)) return 0;
            __atomic_store_n(ack, iter, __ATOMIC_RELEASE);
        }
    }
    return SysCntAicpu() - begin;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_query(void *args) {
    if (args == nullptr) {
        DiagLog("simpler_aicpu_query: args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<DeviceArgs *>(k->device_args);
    if (d == nullptr) {
        DiagLog("simpler_aicpu_query: device_args==nullptr");
        return 1;
    }
    if (d->q_input_addr == 0 || d->q_output_addr == 0 || d->q_input_count == 0) {
        DiagLog("simpler_aicpu_query: empty I/O buffers");
        return 1;
    }

    auto *requests = reinterpret_cast<const QueryRequest *>(d->q_input_addr);
    auto *results = reinterpret_cast<QueryResult *>(d->q_output_addr);
    const uint64_t n = d->q_input_count;
    const uint32_t self_did = 0;

    for (uint64_t i = 0; i < n; ++i) {
        int64_t value = 0;
        drvError_t rc = halGetDeviceInfo(self_did, requests[i].module_type, requests[i].info_type, &value);
        results[i].rc = static_cast<int32_t>(rc);
        results[i]._pad = 0;
        results[i].value = (rc == kHalSuccess) ? value : 0;
    }

    return 0;
}

// Phase 1 helper: launched with aicpu_num=1; records sched_getcpu().
__attribute__((visibility("default"))) int simpler_aicpu_affinity_enum(void *args) {
    if (args == nullptr) {
        DiagLog("simpler_aicpu_affinity_enum: args==nullptr");
        return 1;
    }
    auto *kernel_args = reinterpret_cast<KernelArgs *>(args);
    auto *device_args = reinterpret_cast<AffinityEnumDeviceArgs *>(kernel_args->device_args);
    if (device_args == nullptr || device_args->output_addr == 0) {
        DiagLog("simpler_aicpu_affinity_enum: invalid buffers");
        return 1;
    }
    auto *output = reinterpret_cast<AffinityEnumOutput *>(device_args->output_addr);
    output->cpu_id = sched_getcpu();
    __atomic_store_n(&output->ready, 1u, __ATOMIC_RELEASE);
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_affinity_preflight(void *args) {
    if (args == nullptr) {
        DiagLog("simpler_aicpu_affinity_preflight: args==nullptr");
        return 1;
    }
    auto *kernel_args = reinterpret_cast<KernelArgs *>(args);
    auto *device_args = reinterpret_cast<AffinityPreflightDeviceArgs *>(kernel_args->device_args);
    if (device_args == nullptr || device_args->output_addr == 0 || device_args->aicore_regs_addr == 0 ||
        device_args->user_pool_addr == 0) {
        DiagLog("simpler_aicpu_affinity_preflight: invalid buffers");
        return 1;
    }
    const uint32_t pool_count = device_args->pool_count;
    if (pool_count < 2 || pool_count > kAffinityMaxPool || device_args->aicore_count == 0 ||
        device_args->aicore_count % PLATFORM_NUM_DIES != 0 || device_args->handshake_iters == 0 ||
        device_args->samples_per_core == 0) {
        DiagLog("simpler_aicpu_affinity_preflight: invalid topology");
        return 1;
    }

    const int cpu_id = sched_getcpu();
    const auto *user_pool = reinterpret_cast<const int32_t *>(device_args->user_pool_addr);
    int32_t pool_idx = -1;
    for (uint32_t idx = 0; idx < pool_count; ++idx) {
        if (user_pool[idx] == cpu_id) {
            pool_idx = static_cast<int32_t>(idx);
            break;
        }
    }
    if (pool_idx < 0) return 0;

    auto *output = reinterpret_cast<AffinityPreflightOutput *>(device_args->output_addr);
    const auto *aicore_regs = reinterpret_cast<const uint64_t *>(device_args->aicore_regs_addr);
    const uint32_t cores_per_die = device_args->aicore_count / PLATFORM_NUM_DIES;
    const uint32_t iters = device_args->handshake_iters;

    const uint32_t bit = 1u << static_cast<uint32_t>(pool_idx);
    const uint32_t previous = __atomic_fetch_or(&output->claimed_mask, bit, __ATOMIC_ACQ_REL);
    if ((previous & bit) != 0) {
        SetProbeError(output, AffinityProbeError::kDuplicatePool);
        return 1;
    }

    AffinityPoolSlot &slot = output->slots[pool_idx];
    slot.pool_idx = pool_idx;
    slot.cpu_id = cpu_id;

    __atomic_add_fetch(&output->ready_count, 1u, __ATOMIC_ACQ_REL);
    if (!WaitForProbeValue(output, &output->ready_count, pool_count)) return 1;
    if (pool_idx == 0) {
        __atomic_store_n(&output->pool_count, pool_count, __ATOMIC_RELEASE);
    }

    // Phase 2: pairwise atomic-flag handshake for every unordered pair.
    uint64_t handshake_sum = 0;
    uint32_t handshake_pairs = 0;
    for (uint32_t i = 0; i < pool_count; ++i) {
        for (uint32_t j = i + 1; j < pool_count; ++j) {
            const uint64_t ticks = HandshakePair(output, i, j, static_cast<uint32_t>(pool_idx), iters);
            if (__atomic_load_n(&output->error_code, __ATOMIC_ACQUIRE) !=
                static_cast<uint32_t>(AffinityProbeError::kNone)) {
                return 1;
            }
            if (static_cast<uint32_t>(pool_idx) == i || static_cast<uint32_t>(pool_idx) == j) {
                const uint64_t avg = ticks / static_cast<uint64_t>(iters);
                handshake_sum += avg;
                ++handshake_pairs;
                if (static_cast<uint32_t>(pool_idx) == i) {
                    // Store avg+1 so a true zero average still publishes.
                    const uint64_t published_avg = avg + 1;
                    __atomic_store_n(
                        &output->handshake_pair_ticks[i * kAffinityMaxPool + j], published_avg, __ATOMIC_RELEASE
                    );
                    __atomic_store_n(
                        &output->handshake_pair_ticks[j * kAffinityMaxPool + i], published_avg, __ATOMIC_RELEASE
                    );
                }
            }
            // All threads wait until the pair's matrix entry is published.
            volatile uint64_t *published = &output->handshake_pair_ticks[i * kAffinityMaxPool + j];
            const uint64_t begin = SysCntAicpu();
            const uint64_t timeout_ticks = SysCntFrequencyAicpu() * kAffinityBarrierTimeoutSeconds;
            while (__atomic_load_n(published, __ATOMIC_ACQUIRE) == 0) {
                if (__atomic_load_n(&output->error_code, __ATOMIC_ACQUIRE) !=
                    static_cast<uint32_t>(AffinityProbeError::kNone)) {
                    return 1;
                }
                if (SysCntAicpu() - begin >= timeout_ticks) {
                    SetProbeError(output, AffinityProbeError::kBarrierTimeout);
                    return 1;
                }
                __asm__ volatile("yield");
            }
        }
    }

    slot.avg_handshake_ticks =
        handshake_pairs > 0 ? handshake_sum / static_cast<uint64_t>(handshake_pairs) : UINT64_MAX;
    __atomic_store_n(&slot.handshake_valid, 1u, __ATOMIC_RELEASE);

    for (uint32_t idx = 0; idx < pool_count; ++idx) {
        if (!WaitForProbeValue(output, &output->slots[idx].handshake_valid, 1u)) return 1;
    }

    // Elect orch: minimum average handshake latency; tie-break on smaller pool_idx.
    if (pool_idx == 0) {
        uint32_t best = 0;
        uint64_t best_avg = output->slots[0].avg_handshake_ticks;
        for (uint32_t idx = 1; idx < pool_count; ++idx) {
            const uint64_t avg = output->slots[idx].avg_handshake_ticks;
            if (avg < best_avg || (avg == best_avg && idx < best)) {
                best = idx;
                best_avg = avg;
            }
        }
        output->slots[best].is_orch = 1;
        __atomic_store_n(&output->orch_pool_idx, best, __ATOMIC_RELEASE);
        __atomic_store_n(&output->handshake_done, 1u, __ATOMIC_RELEASE);
    }
    if (!WaitForProbeValue(output, &output->handshake_done, 1u)) return 1;
    const uint32_t orch_idx = __atomic_load_n(&output->orch_pool_idx, __ATOMIC_ACQUIRE);

    // Phase 3: serial COND die scoring for non-orch threads.
    for (uint32_t turn = 0; turn < pool_count; ++turn) {
        if (turn == orch_idx) {
            if (static_cast<uint32_t>(pool_idx) == turn) {
                slot.die0_sum_ticks = 0;
                slot.die1_sum_ticks = 0;
                __atomic_store_n(&slot.die_valid, 1u, __ATOMIC_RELEASE);
            }
        } else if (static_cast<uint32_t>(pool_idx) == turn) {
            TouchDie(aicore_regs, 0, cores_per_die, kAffinityWarmupSamplesPerCore);
            TouchDie(aicore_regs, cores_per_die, cores_per_die, kAffinityWarmupSamplesPerCore);
            slot.die0_sum_ticks = MeasureDieTotalTicks(aicore_regs, 0, cores_per_die, device_args->samples_per_core);
            slot.die1_sum_ticks =
                MeasureDieTotalTicks(aicore_regs, cores_per_die, cores_per_die, device_args->samples_per_core);
            __atomic_store_n(&slot.die_valid, 1u, __ATOMIC_RELEASE);
        }
        if (!WaitForProbeValue(output, &output->slots[turn].die_valid, 1u)) return 1;
    }

    if (pool_idx == 0) {
        __atomic_store_n(&output->die_done, 1u, __ATOMIC_RELEASE);
    }
    if (!WaitForProbeValue(output, &output->die_done, 1u)) return 1;
    return 0;
}

// Back-compat alias used by older host descriptors.
__attribute__((visibility("default"))) int simpler_aicpu_rtt_probe(void *args) {
    return simpler_aicpu_affinity_preflight(args);
}

}  // extern "C"
