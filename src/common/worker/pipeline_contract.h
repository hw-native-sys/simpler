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

/**
 * Admission check for the PipelineContract a host runtime declares.
 *
 * Lives beside the contract rather than inside ChipWorker so the rules a build
 * will honor are stated in one place and can be exercised on their own.
 */

#pragma once

#include <cstdint>

#include "runtime_c_api.h"
#include "execution_mode.h"

/**
 * Whether `contract` has a valid structure and mode-specific byte counts.
 * Serviceability and runtime-specific resource sets are checked separately.
 *
 * Rejects a contract whose ABI version is not the one compiled in, that
 * declares more resources than fit, that declares no pipeline depth at all, or
 * whose resources carry an unspecified or out-of-range kind, an out-of-range
 * class, or a byte count incompatible with the mode and resource class.
 *
 * There is no fixed ABI ceiling on `pipeline_depth`: it is what a runtime says
 * it can support. The Worker chooses the context's grant, and owners size their
 * per-run tables and native run storage to that grant during initialization.
 * This count bounds admitted run resources; it is not a byte budget for device
 * workspace, which has its own allocator and optional accounting limit.
 *
 * The unspecified-kind rule is what catches a `resource_count` larger than the
 * entries a runtime actually filled in: the trailing entries are still zeroed,
 * and zero is not a resource. Repeated kinds stay legal, so a runtime may
 * declare several resources of one kind.
 */
// Accept the raw mode value so unknown values can be rejected before an enum cast.
inline bool is_valid_pipeline_contract(const PipelineContract *contract, uint32_t mode) {
    if ((mode != SIMPLER_MODE_PROGRAM && mode != SIMPLER_MODE_KERNEL) || contract == nullptr ||
        contract->abi_version != PTO_PIPELINE_CONTRACT_ABI_VERSION ||
        contract->resource_count > PTO_PIPELINE_MAX_RESOURCES || contract->pipeline_depth == 0) {
        return false;
    }
    for (uint32_t i = 0; i < contract->resource_count; ++i) {
        const PipelineResource &resource = contract->resources[i];
        if (resource.kind == PTO_PIPELINE_KIND_UNSPECIFIED || resource.kind > PTO_PIPELINE_AICORE_STREAM ||
            resource.resource_class > PTO_PIPELINE_REPORTED) {
            return false;
        }
        // Per-run images may be sized by host orchestration at bind time.
        // Shared scratch has a config-derived size before kernel init; handles
        // never carry bytes, and a reported region's size is chosen when its
        // pool acquires it. Program declarations remain topology-only.
        if (mode == SIMPLER_MODE_PROGRAM || resource.resource_class == PTO_PIPELINE_EXEC_HANDLE ||
            resource.resource_class == PTO_PIPELINE_REPORTED) {
            if (resource.bytes_per_copy != 0) return false;
        } else if (resource.resource_class == PTO_PIPELINE_SCRATCH && resource.bytes_per_copy == 0) {
            return false;
        }
    }
    return true;
}

// Program-mode compatibility overload.
inline bool is_valid_pipeline_contract(const PipelineContract *contract) {
    return is_valid_pipeline_contract(contract, SIMPLER_MODE_PROGRAM);
}

/** Return the number of concrete copies required for one resource. */
inline uint32_t pipeline_resource_copy_count(const PipelineContract &contract, const PipelineResource &resource) {
    return resource.resource_class == PTO_PIPELINE_DELIVERED ? contract.pipeline_depth : 1u;
}

/** Select the concrete copy of `resource` owned by `lease`. */
inline uint32_t pipeline_resource_slot(
    const PipelineContract &contract, const PipelineResource &resource, const PipelineSlotLease &lease
) {
    return pipeline_resource_copy_count(contract, resource) == 1 ? 0u : lease.slot_id;
}

/** Find a declared resource kind, or return nullptr when the runtime does not use it. */
// The caller must validate the contract before looking up a resource.
inline const PipelineResource *find_pipeline_resource(const PipelineContract &contract, uint32_t kind) {
    for (uint32_t i = 0; i < contract.resource_count; ++i) {
        if (contract.resources[i].kind == kind) return &contract.resources[i];
    }
    return nullptr;
}

/** Select the bank of each pooled region owned by the lease at `slot_id`. */
inline RegionBanks pipeline_resource_banks_for_slot(const PipelineContract &contract, uint32_t slot_id) {
    const PipelineSlotLease lease{slot_id, 0, 0};
    RegionBanks banks{0, 0, 0};
    if (const PipelineResource *heap = find_pipeline_resource(contract, PTO_PIPELINE_GM_HEAP))
        banks.heap = pipeline_resource_slot(contract, *heap, lease);
    if (const PipelineResource *sm = find_pipeline_resource(contract, PTO_PIPELINE_GM_SM))
        banks.sm = pipeline_resource_slot(contract, *sm, lease);
    if (const PipelineResource *image = find_pipeline_resource(contract, PTO_PIPELINE_RUNTIME_IMAGE))
        banks.image = pipeline_resource_slot(contract, *image, lease);
    return banks;
}

/**
 * Whether each pooled device region has a distinct, selectable declaration.
 *
 * GM heap, GM shared memory and the runtime image are selected independently:
 * one may be shared scratch while another is delivered per run. The only
 * structural requirements are that a kind is declared at most once — a second
 * entry of the same kind would never be read — and that an arena region is
 * either delivered or scratch; a stream or a reported region cannot be one.
 * The caller must validate the contract before checking this topology.
 */
inline bool has_serviceable_arena_topology(const PipelineContract &contract) {
    constexpr uint32_t ARENA_KINDS[] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_GM_SM, PTO_PIPELINE_RUNTIME_IMAGE};
    for (uint32_t kind : ARENA_KINDS) {
        uint32_t declared = 0;
        const PipelineResource *resource = nullptr;
        for (uint32_t i = 0; i < contract.resource_count; ++i) {
            if (contract.resources[i].kind != kind) continue;
            ++declared;
            resource = &contract.resources[i];
        }
        if (declared > 1) return false;
        if (resource == nullptr) continue;
        if (resource->resource_class != PTO_PIPELINE_DELIVERED && resource->resource_class != PTO_PIPELINE_SCRATCH) {
            return false;
        }
    }
    return true;
}

// Each required execution role is supplied exactly once; an absent AICPU has no handle.
inline bool has_serviceable_stream_topology(const PipelineContract &contract, bool requires_aicpu = true) {
    if (contract.resource_count > PTO_PIPELINE_MAX_RESOURCES) return false;
    constexpr uint32_t STREAM_KINDS[] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_AICORE_STREAM};
    for (uint32_t kind : STREAM_KINDS) {
        uint32_t count = 0;
        for (uint32_t i = 0; i < contract.resource_count; ++i) {
            const auto &resource = contract.resources[i];
            if (resource.kind != kind) continue;
            if (resource.resource_class != PTO_PIPELINE_EXEC_HANDLE) return false;
            ++count;
        }
        const uint32_t required_count = kind == PTO_PIPELINE_AICPU_STREAM && !requires_aicpu ? 0 : 1;
        if (count != required_count) return false;
    }
    return true;
}

// Complete TMR kernel admission, including structural checks before any lookup.
inline bool is_valid_tmr_kernel_pipeline_contract(const PipelineContract *contract) {
    // Exact cardinality plus required-kind checks excludes extra TASK_ARGS
    // entries, which neither the arena nor stream topology check covers.
    if (!is_valid_pipeline_contract(contract, SIMPLER_MODE_KERNEL) || contract->resource_count != 6 ||
        !has_serviceable_arena_topology(*contract) || !has_serviceable_stream_topology(*contract)) {
        return false;
    }
    constexpr PipelineResource REQUIRED_STORAGE[] = {
        {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_SCRATCH, 0},
        {PTO_PIPELINE_GM_SM, PTO_PIPELINE_SCRATCH, 0},
        {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_SCRATCH, 0},
        {PTO_PIPELINE_TASK_ARGS, PTO_PIPELINE_DELIVERED, 0},
    };
    for (const auto &required : REQUIRED_STORAGE) {
        const auto *resource = find_pipeline_resource(*contract, required.kind);
        if (resource == nullptr || resource->resource_class != required.resource_class || resource->bytes_per_copy == 0)
            return false;
    }
    return true;
}
