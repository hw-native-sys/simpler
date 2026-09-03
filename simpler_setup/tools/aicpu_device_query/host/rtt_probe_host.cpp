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

#include "rtt_probe_host.h"

#include "aicpu_topology_probe.h"
#include "common/acl_hal_device.h"
#include "common/platform_config.h"
#include "host/host_regs.h"
#include "../shared/rtt_probe_types.h"

#include <driver/ascend_hal.h>
#include <runtime/runtime/rts/rts_kernel.h>

#include <algorithm>
#include <cstdio>
#include <dlfcn.h>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace aicpu_device_query {
namespace {

thread_local bool g_affinity_probe_timed_out = false;

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() {
        if (ptr_ != nullptr) aclrtFree(ptr_);
    }

    bool Allocate(size_t bytes, const char *description) {
        const aclError rc = aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc == ACL_SUCCESS) return true;
        ptr_ = nullptr;
        std::fprintf(stderr, "aclrtMalloc failed: %d (%s)\n", static_cast<int>(rc), description);
        return false;
    }

    void *get() const { return ptr_; }

private:
    void *ptr_{nullptr};
};

bool CheckAcl(aclError rc, const char *call, const char *description) {
    if (rc == ACL_SUCCESS) return true;
    g_affinity_probe_timed_out = rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT;
    std::fprintf(stderr, "%s failed: %d (%s)\n", call, static_cast<int>(rc), description);
    return false;
}

bool MapAicoreRegisters(int device_id, std::vector<uint64_t> *registers) {
    using HalResMapFn = int (*)(uint32_t, struct res_map_info *, uint64_t *, uint32_t *);
    auto hal_res_map = reinterpret_cast<HalResMapFn>(dlsym(nullptr, "halResMap"));
    if (hal_res_map == nullptr) {
        std::fprintf(stderr, "halResMap not found: %s\n", dlerror());
        return false;
    }

    const auto physical_device_id = static_cast<uint32_t>(pto::acl_to_hal_device_id(device_id));
    struct res_map_info map_info = {};
    map_info.target_proc_type = PROCESS_CP1;
    map_info.res_type = RES_AICORE;
    registers->assign(DAV_3510::PLATFORM_MAX_PHYSICAL_CORES, 0);
    for (uint32_t core = 0; core < DAV_3510::PLATFORM_MAX_PHYSICAL_CORES; ++core) {
        map_info.res_id = core;
        uint32_t length = REG_AICORE_MAP_SIZE;
        const int rc = hal_res_map(physical_device_id, &map_info, &(*registers)[core], &length);
        if (rc != 0) {
            std::fprintf(stderr, "halResMap failed for AICore %u: %d\n", core, rc);
            return false;
        }
    }
    return true;
}

bool ResolveNamed(void *binary_handle, uint64_t fingerprint, const char *base_name, rtFuncHandle *handle) {
    char op_type[128];
    std::snprintf(op_type, sizeof(op_type), "%s_%016lx", base_name, fingerprint);
    const rtError_t rc = rtsFuncGetByName(binary_handle, op_type, handle);
    if (rc == RT_ERROR_NONE) return true;
    std::fprintf(stderr, "rtsFuncGetByName %s failed: %d\n", base_name, static_cast<int>(rc));
    return false;
}

bool LaunchCpu(rtFuncHandle handle, uint32_t block_dim, void *device_args, aclrtStream stream) {
    struct LaunchArgs {
        uint64_t pad[5] = {0};
        uint64_t device_args_ptr = 0;
        uint64_t reserved[20] = {0};
    } launch_args = {};
    launch_args.device_args_ptr = reinterpret_cast<uint64_t>(device_args);

    rtCpuKernelArgs_t cpu_args = {};
    cpu_args.baseArgs.args = &launch_args;
    cpu_args.baseArgs.argsSize = sizeof(launch_args);
    rtLaunchKernelAttr_t attr = {};
    rtKernelLaunchCfg_t cfg = {&attr, 0};
    const rtError_t rc = rtsLaunchCpuKernel(handle, block_dim, stream, &cfg, &cpu_args);
    if (rc == RT_ERROR_NONE) return true;
    std::fprintf(stderr, "rtsLaunchCpuKernel failed: %d\n", static_cast<int>(rc));
    return false;
}

std::string MakeDescriptorEntry(
    uint64_t fingerprint, const std::string &so_basename, const char *op_type_base, const char *function_name
) {
    char op_type[128];
    std::snprintf(op_type, sizeof(op_type), "%s_%016lx", op_type_base, fingerprint);
    std::string descriptor = "  \"";
    descriptor += op_type;
    descriptor += "\": {\n    \"opInfo\": {\n";
    descriptor += "      \"functionName\": \"";
    descriptor += function_name;
    descriptor += "\",\n      \"kernelSo\": \"";
    descriptor += so_basename;
    descriptor += "\",\n      \"opKernelLib\": \"AICPUKernel\",\n";
    descriptor += "      \"computeCost\": \"100\",\n      \"engine\": \"DNN_VM_AICPU\",\n";
    descriptor += "      \"flagAsync\": \"False\",\n      \"flagPartial\": \"False\",\n";
    descriptor += "      \"userDefined\": \"False\"\n    }\n  }";
    return descriptor;
}

std::vector<int32_t> EnumerateUserPool(
    rtFuncHandle enum_handle, void *device_args, size_t device_args_bytes, aclrtStream stream,
    const pto::a5::AicpuTopology &topology
) {
    std::vector<int32_t> occupy_cpus;
    for (const auto &cpu : topology.os_schedulable_cpus) {
        occupy_cpus.push_back(cpu.cpu_id);
    }
    std::sort(occupy_cpus.begin(), occupy_cpus.end());
    occupy_cpus.erase(std::unique(occupy_cpus.begin(), occupy_cpus.end()), occupy_cpus.end());
    if (occupy_cpus.size() > static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)) {
        std::fprintf(
            stderr, "affinity preflight OCCUPY population %zu exceeds supported launch maximum %d\n",
            occupy_cpus.size(), PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH
        );
        return {};
    }

    // Driver and verified-JSON topology already provide the complete OCCUPY
    // intersection. Use it directly instead of hoping repeated one-thread
    // launches eventually visit every CPU. An incomplete topology falls back
    // to the serial discovery path below and must still prove completeness.
    if (topology.source != pto::a5::AicpuTopologySource::kOccupyFallback) {
        std::vector<int32_t> topology_cpus;
        if (pto::a5::build_complete_topology_user_pool(topology, topology_cpus)) return topology_cpus;
        std::fprintf(stderr, "affinity preflight topology does not completely cover OCCUPY; trying serial discovery\n");
    }

    DeviceBuffer enum_out;
    if (!enum_out.Allocate(sizeof(AffinityEnumOutput), "affinity enum output")) return {};

    std::set<int32_t> seen;
    const int attempts = static_cast<int>(occupy_cpus.size()) * 3 + 4;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        AffinityEnumOutput host_out = {};
        if (!CheckAcl(
                aclrtMemcpy(enum_out.get(), sizeof(host_out), &host_out, sizeof(host_out), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy", "zero enum output"
            )) {
            return {};
        }
        std::vector<uint8_t> host_args(device_args_bytes, 0);
        auto *args = reinterpret_cast<AffinityEnumDeviceArgs *>(host_args.data());
        args->output_addr = reinterpret_cast<uint64_t>(enum_out.get());
        if (!CheckAcl(
                aclrtMemcpy(
                    device_args, device_args_bytes, host_args.data(), device_args_bytes, ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy", "enum device args"
            )) {
            return {};
        }
        if (!LaunchCpu(enum_handle, 1u, device_args, stream)) return {};
        if (!CheckAcl(
                aclrtSynchronizeStreamWithTimeout(stream, kAffinityHostSyncTimeoutMs),
                "aclrtSynchronizeStreamWithTimeout", "enum sync"
            )) {
            return {};
        }
        if (!CheckAcl(
                aclrtMemcpy(&host_out, sizeof(host_out), enum_out.get(), sizeof(host_out), ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy", "enum D2H"
            )) {
            return {};
        }
        if (host_out.ready == 0 || host_out.cpu_id < 0) continue;
        seen.insert(host_out.cpu_id);
        if (!occupy_cpus.empty() && seen.size() >= occupy_cpus.size()) break;
    }

    const std::vector<int32_t> pool(seen.begin(), seen.end());
    if (pool != occupy_cpus) {
        std::fprintf(
            stderr, "affinity preflight serial discovery incomplete: saw %zu of %zu OCCUPY CPUs\n", pool.size(),
            occupy_cpus.size()
        );
        return {};
    }
    return pool;
}

std::string FormatAffinityJson(
    int device_id, const pto::a5::AicpuTopology &topology, const std::vector<int32_t> &user_pool,
    const AffinityPreflightOutput &output
) {
    std::ostringstream json;
    json << "{\n  \"schema_version\": 3,\n"
         << "  \"measurement_method\": \"atomic-flag-orch+cond-die-v1\",\n"
         << "  \"soc_name\": \"" << topology.soc_name << "\",\n"
         << "  \"device_id\": " << device_id << ",\n"
         << "  \"handshake_iters\": " << kAffinityHandshakeIters << ",\n"
         << "  \"samples_per_core\": " << kAffinitySamplesPerCore << ",\n"
         << "  \"user_pool_cpus\": [";
    for (size_t i = 0; i < user_pool.size(); ++i) {
        if (i) json << ", ";
        json << user_pool[i];
    }
    json << "],\n  \"orch_pool_idx\": " << output.orch_pool_idx << ",\n  \"pool\": [\n";
    for (uint32_t idx = 0; idx < output.pool_count; ++idx) {
        const AffinityPoolSlot &slot = output.slots[idx];
        if (idx) json << ",\n";
        json << "    {\"pool_idx\": " << slot.pool_idx << ", \"cpu_id\": " << slot.cpu_id
             << ", \"avg_handshake_ticks\": " << slot.avg_handshake_ticks
             << ", \"die0_sum_ticks\": " << slot.die0_sum_ticks << ", \"die1_sum_ticks\": " << slot.die1_sum_ticks
             << ", \"is_orch\": " << slot.is_orch << "}";
    }
    json << "\n  ]\n}\n";
    return json.str();
}

}  // namespace

std::string MakeRttDescriptorEntry(uint64_t fingerprint, const std::string &so_basename) {
    return MakeDescriptorEntry(fingerprint, so_basename, "simpler_aicpu_rtt_probe", "simpler_aicpu_rtt_probe");
}

std::string MakeAffinityEnumDescriptorEntry(uint64_t fingerprint, const std::string &so_basename) {
    return MakeDescriptorEntry(fingerprint, so_basename, "simpler_aicpu_affinity_enum", "simpler_aicpu_affinity_enum");
}

std::string MakeAffinityPreflightDescriptorEntry(uint64_t fingerprint, const std::string &so_basename) {
    return MakeDescriptorEntry(
        fingerprint, so_basename, "simpler_aicpu_affinity_preflight", "simpler_aicpu_affinity_preflight"
    );
}

bool RunAffinityPreflight(
    int device_id, void *binary_handle, uint64_t fingerprint, void *device_args, size_t device_args_bytes,
    aclrtStream stream, const pto::a5::AicpuTopology &topology
) {
    g_affinity_probe_timed_out = false;
    rtFuncHandle enum_handle = nullptr;
    rtFuncHandle preflight_handle = nullptr;
    if (!ResolveNamed(binary_handle, fingerprint, "simpler_aicpu_affinity_enum", &enum_handle)) return false;
    if (!ResolveNamed(binary_handle, fingerprint, "simpler_aicpu_affinity_preflight", &preflight_handle)) {
        // Fall back to legacy symbol name if present.
        if (!ResolveNamed(binary_handle, fingerprint, "simpler_aicpu_rtt_probe", &preflight_handle)) return false;
    }

    const std::vector<int32_t> user_pool =
        EnumerateUserPool(enum_handle, device_args, device_args_bytes, stream, topology);
    if (user_pool.size() < 2) {
        std::fprintf(stderr, "affinity preflight needs at least 2 user CPUs; got %zu\n", user_pool.size());
        return false;
    }
    if (user_pool.size() > static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH) ||
        user_pool.size() > kAffinityMaxPool) {
        std::fprintf(
            stderr, "affinity preflight pool %zu exceeds supported launch maximum %d\n", user_pool.size(),
            PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH
        );
        return false;
    }
    if (user_pool.size() < 5) {
        // Emit a shrink-only probe result so Python can skip full packing.
        std::ostringstream json;
        json << "{\n  \"schema_version\": 3,\n"
             << "  \"measurement_method\": \"atomic-flag-orch+cond-die-v1\",\n"
             << "  \"soc_name\": \"" << topology.soc_name << "\",\n"
             << "  \"device_id\": " << device_id << ",\n"
             << "  \"pool_too_small\": true,\n"
             << "  \"user_pool_cpus\": [";
        for (size_t i = 0; i < user_pool.size(); ++i) {
            if (i) json << ", ";
            json << user_pool[i];
        }
        json << "]\n}\n";
        std::fputs(json.str().c_str(), stdout);
        return true;
    }

    std::vector<uint64_t> aicore_registers;
    if (!MapAicoreRegisters(device_id, &aicore_registers)) return false;

    DeviceBuffer pool_buf, regs_buf, out_buf;
    if (!pool_buf.Allocate(user_pool.size() * sizeof(int32_t), "user pool") ||
        !regs_buf.Allocate(aicore_registers.size() * sizeof(uint64_t), "aicore regs") ||
        !out_buf.Allocate(sizeof(AffinityPreflightOutput), "preflight output")) {
        return false;
    }
    AffinityPreflightOutput zero_out = {};
    if (!CheckAcl(
            aclrtMemcpy(
                pool_buf.get(), user_pool.size() * sizeof(int32_t), user_pool.data(),
                user_pool.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy", "H2D pool"
        ) ||
        !CheckAcl(
            aclrtMemcpy(
                regs_buf.get(), aicore_registers.size() * sizeof(uint64_t), aicore_registers.data(),
                aicore_registers.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy", "H2D regs"
        ) ||
        !CheckAcl(
            aclrtMemcpy(out_buf.get(), sizeof(zero_out), &zero_out, sizeof(zero_out), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy", "H2D output"
        )) {
        return false;
    }

    std::vector<uint8_t> host_args(device_args_bytes, 0);
    auto *args = reinterpret_cast<AffinityPreflightDeviceArgs *>(host_args.data());
    args->output_addr = reinterpret_cast<uint64_t>(out_buf.get());
    args->aicore_regs_addr = reinterpret_cast<uint64_t>(regs_buf.get());
    args->user_pool_addr = reinterpret_cast<uint64_t>(pool_buf.get());
    args->pool_count = static_cast<uint32_t>(user_pool.size());
    args->aicore_count = DAV_3510::PLATFORM_MAX_PHYSICAL_CORES;
    args->handshake_iters = kAffinityHandshakeIters;
    args->samples_per_core = kAffinitySamplesPerCore;
    if (!CheckAcl(
            aclrtMemcpy(device_args, device_args_bytes, host_args.data(), device_args_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy", "preflight device args"
        )) {
        return false;
    }

    const uint32_t launch_count = static_cast<uint32_t>(
        topology.device_occupancy.occupy_valid ? __builtin_popcountll(topology.device_occupancy.occupy) :
                                                 user_pool.size()
    );
    if (!LaunchCpu(preflight_handle, launch_count, device_args, stream)) return false;
    if (!CheckAcl(
            aclrtSynchronizeStreamWithTimeout(stream, kAffinityHostSyncTimeoutMs), "aclrtSynchronizeStreamWithTimeout",
            "preflight sync"
        )) {
        return false;
    }

    AffinityPreflightOutput output = {};
    if (!CheckAcl(
            aclrtMemcpy(&output, sizeof(output), out_buf.get(), sizeof(output), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy", "preflight D2H"
        )) {
        return false;
    }
    if (output.error_code != static_cast<uint32_t>(AffinityProbeError::kNone)) {
        std::fprintf(stderr, "affinity preflight device error_code=%u\n", output.error_code);
        return false;
    }
    if (output.pool_count != user_pool.size() || output.handshake_done == 0 || output.die_done == 0) {
        std::fprintf(stderr, "affinity preflight incomplete output\n");
        return false;
    }
    // Normalize pair ticks (device stored avg+1).
    for (uint32_t i = 0; i < output.pool_count; ++i) {
        for (uint32_t j = 0; j < output.pool_count; ++j) {
            uint64_t &cell = output.handshake_pair_ticks[i * kAffinityMaxPool + j];
            if (cell > 0) --cell;
        }
    }

    std::fputs(FormatAffinityJson(device_id, topology, user_pool, output).c_str(), stdout);
    return true;
}

bool LastAffinityProbeTimedOut() { return g_affinity_probe_timed_out; }

bool RunRttProbe(
    int device_id, void *binary_handle, uint64_t fingerprint, void *device_args, size_t device_args_bytes,
    aclrtStream stream, const pto::a5::AicpuTopology &topology, const pto::a5::AicpuLaunchPlan &
) {
    return RunAffinityPreflight(
        device_id, binary_handle, fingerprint, device_args, device_args_bytes, stream, topology
    );
}

}  // namespace aicpu_device_query
