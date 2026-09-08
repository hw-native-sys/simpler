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
 * declares more resources than fit, that asks for a pipeline depth outside
 * the supported range, or whose resources carry an unspecified or out-of-range kind, an
 * out-of-range class, or a byte count incompatible with the mode and kind.
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
        contract->resource_count > PTO_PIPELINE_MAX_RESOURCES || contract->pipeline_depth == 0 ||
        contract->pipeline_depth > PTO_PIPELINE_MAX_DEPTH) {
        return false;
    }
    for (uint32_t i = 0; i < contract->resource_count; ++i) {
        const PipelineResource &resource = contract->resources[i];
        if (resource.kind == PTO_PIPELINE_KIND_UNSPECIFIED || resource.kind > PTO_PIPELINE_AICORE_STREAM ||
            resource.resource_class > PTO_PIPELINE_EXEC_HANDLE) {
            return false;
        }
        const bool sized_arena = mode == SIMPLER_MODE_KERNEL && resource.kind <= PTO_PIPELINE_RUNTIME_IMAGE;
        if (sized_arena ? resource.bytes_per_copy == 0 : resource.bytes_per_copy != 0) return false;
    }
    return true;
}

inline bool is_valid_pipeline_contract(const PipelineContract *contract) {
    return is_valid_pipeline_contract(contract, SIMPLER_MODE_PROGRAM);
}

/** Return the number of concrete copies required for one resource. */
inline uint32_t pipeline_resource_copy_count(const PipelineContract &contract, const PipelineResource &resource) {
    return resource.resource_class == PTO_PIPELINE_DEVICE_SCRATCH ? 1u : contract.pipeline_depth;
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

/**
 * Whether the three pooled device regions can be served by one bank selector.
 *
 * GM heap, GM shared memory, and the runtime image are committed together by
 * one `setup_static_arena` call into one arena bank, so they cannot be given
 * different copy counts. A contract that classifies them inconsistently — or
 * declares one of them twice, where only the first entry would ever be read —
 * describes a layout the executor cannot produce, and is rejected at load
 * rather than at the first launch that would need the second bank.
 * The caller must validate the contract before checking this topology.
 */
inline bool has_serviceable_arena_topology(const PipelineContract &contract) {
    constexpr uint32_t ARENA_KINDS[] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_GM_SM, PTO_PIPELINE_RUNTIME_IMAGE};
    bool seen = false;
    uint32_t shared_copies = 0;
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
        uint32_t copies = pipeline_resource_copy_count(contract, *resource);
        if (seen && copies != shared_copies) return false;
        shared_copies = copies;
        seen = true;
    }
    return true;
}

// Each execution role is supplied exactly once. Copy counts do not prescribe
// the number of physical streams created by the platform.
inline bool has_serviceable_stream_topology(const PipelineContract &contract) {
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
        if (count != 1) return false;
    }
    return true;
}

// Complete TMR kernel admission, including structural checks before any lookup.
inline bool is_valid_tmr_kernel_pipeline_contract(const PipelineContract *contract) {
    if (!is_valid_pipeline_contract(contract, SIMPLER_MODE_KERNEL) || contract->pipeline_depth != 1 ||
        contract->resource_count != 6 || !has_serviceable_arena_topology(*contract) ||
        !has_serviceable_stream_topology(*contract)) {
        return false;
    }
    for (uint32_t kind = PTO_PIPELINE_GM_HEAP; kind <= PTO_PIPELINE_TASK_ARGS; ++kind) {
        const auto *resource = find_pipeline_resource(*contract, kind);
        const auto expected_class =
            kind == PTO_PIPELINE_TASK_ARGS ? PTO_PIPELINE_HOST_PER_RUN : PTO_PIPELINE_DEVICE_SCRATCH;
        if (resource == nullptr || resource->resource_class != expected_class) return false;
    }
    return true;
}
