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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <type_traits>
extern "C" {
#endif

#define SIMPLER_CCU_MODULE_ABI_VERSION 2U
#define SIMPLER_CCU_REGISTRATION_BINDING_LAYOUT_VERSION 2U
#define SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION 2U
#define SIMPLER_CCU_WORKSPACE_LAYOUT_VERSION 2U
#define SIMPLER_CCU_MODULE_MANIFEST_SYMBOL "simpler_ccu_module_manifest_v2"
#define SIMPLER_CCU_BUILD_ID_SIZE 32U
#define SIMPLER_CCU_EVENT_MASK 1ULL
#define SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT 1U
#define SIMPLER_CCU_TASK_ARGUMENT_COUNT 4U

#define SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS (1U << 0)
#define SIMPLER_CCU_KERNEL_CAP_MASK SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS

typedef struct SimplerCcuModuleManifestV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t manifest_size;
    uint32_t kernel_record_size;
    uint32_t kernel_count;
    uint32_t kernel_records_offset;
    uint32_t string_table_offset;
    uint32_t string_table_size;
    uint32_t capability_flags;
    uint32_t instance_record_size;
    uint8_t sdk_build_id[SIMPLER_CCU_BUILD_ID_SIZE];
    uint8_t pto_isa_build_id[SIMPLER_CCU_BUILD_ID_SIZE];
    uint32_t instance_count;
    uint32_t instance_records_offset;
    uint64_t reserved[3];
} SimplerCcuModuleManifestV2;

typedef struct SimplerCcuKernelRecordV2 {
    uint32_t struct_size;
    uint32_t kernel_id;
    uint32_t reserved0;
    uint32_t task_argument_layout_version;
    uint64_t parameter_size;
    uint32_t parameter_alignment;
    uint32_t capability_flags;
    uint32_t entry_symbol_offset;
    uint32_t entry_symbol_size;
    uint32_t registration_binding_layout_version;
    uint32_t registration_binding_size;
    uint32_t registration_argument_count;
    uint32_t task_argument_count;
    uint32_t reserved[2];
} SimplerCcuKernelRecordV2;

/* Table position is the module-local logical index, independent of SDK registration order. */
typedef struct SimplerCcuMissionInstanceV2 {
    uint32_t struct_size;
    uint32_t mission_index;
    uint32_t kernel_id;
    uint32_t flags;
    uint64_t reserved[2];
} SimplerCcuMissionInstanceV2;

typedef struct SimplerCcuRegistrationBindingV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layout_version;
    uint32_t kernel_id;
    uint32_t mission_index;
    uint32_t rank_id;
    uint32_t rank_count;
    uint32_t flags;
    uint64_t channel_handle;
    uint64_t ready_event_handle;
    uint64_t trigger_event_handle;
    uint64_t done_event_handle;
    uint64_t reserved[4];
} SimplerCcuRegistrationBindingV2;

typedef struct SimplerCcuLaunchArgumentsV2 {
    uint64_t layout_version;
    uint64_t parameter_addr;
    uint64_t parameter_token;
    uint64_t reserved;
} SimplerCcuLaunchArgumentsV2;

typedef struct SimplerCcuEventDescriptorV2 {
    uint64_t address;
    uint64_t mask;
} SimplerCcuEventDescriptorV2;

typedef struct SimplerCcuMissionDescriptorV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t run_generation;
    uint32_t mission_index;
    uint32_t kernel_id;
    uint32_t parameter_alignment;
    uint32_t flags;
    uint64_t parameter_addr;
    uint64_t parameter_size;
    uint64_t parameter_token;
    SimplerCcuEventDescriptorV2 ready;
    SimplerCcuEventDescriptorV2 trigger;
    SimplerCcuEventDescriptorV2 done;
    uint32_t module_index;
    uint32_t reserved0;
} SimplerCcuMissionDescriptorV2;

/* Published through the domain runtime context; missions_addr points to a dense descriptor array. */
typedef struct SimplerCcuWorkspaceV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layout_version;
    uint32_t mission_count;
    uint64_t run_generation;
    uint64_t missions_addr;
    uint32_t mission_descriptor_size;
    uint32_t flags;
    uint64_t domain_id;
    uint64_t reserved[2];
} SimplerCcuWorkspaceV2;

typedef const SimplerCcuModuleManifestV2 *(*SimplerCcuModuleManifestEntryV2)(void);
typedef int32_t (*SimplerCcuNativeKernelEntryV2)(const SimplerCcuRegistrationBindingV2 *binding);

#ifdef __cplusplus
}

#define SIMPLER_CCU_ASSERT_POD(type) \
    static_assert(std::is_trivially_copyable_v<type> && std::is_standard_layout_v<type>, #type " must be C/POD")

SIMPLER_CCU_ASSERT_POD(SimplerCcuModuleManifestV2);
SIMPLER_CCU_ASSERT_POD(SimplerCcuKernelRecordV2);
SIMPLER_CCU_ASSERT_POD(SimplerCcuRegistrationBindingV2);
SIMPLER_CCU_ASSERT_POD(SimplerCcuLaunchArgumentsV2);
SIMPLER_CCU_ASSERT_POD(SimplerCcuMissionInstanceV2);
SIMPLER_CCU_ASSERT_POD(SimplerCcuEventDescriptorV2);
SIMPLER_CCU_ASSERT_POD(SimplerCcuMissionDescriptorV2);
SIMPLER_CCU_ASSERT_POD(SimplerCcuWorkspaceV2);

static_assert(sizeof(SimplerCcuModuleManifestV2) == 136, "CCU manifest ABI size changed");
static_assert(sizeof(SimplerCcuKernelRecordV2) == 64, "CCU kernel record ABI size changed");
static_assert(sizeof(SimplerCcuRegistrationBindingV2) == 96, "CCU registration binding ABI size changed");
static_assert(sizeof(SimplerCcuLaunchArgumentsV2) == 32, "CCU launch arguments ABI size changed");
static_assert(sizeof(SimplerCcuMissionInstanceV2) == 32, "CCU instance ABI size changed");
static_assert(sizeof(SimplerCcuEventDescriptorV2) == 16, "CCU Event descriptor ABI size changed");
static_assert(sizeof(SimplerCcuMissionDescriptorV2) == 112, "CCU mission descriptor ABI size changed");
static_assert(sizeof(SimplerCcuWorkspaceV2) == 64, "CCU workspace ABI size changed");

#undef SIMPLER_CCU_ASSERT_POD
#endif
