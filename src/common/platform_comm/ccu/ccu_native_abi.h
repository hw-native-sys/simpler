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

#define SIMPLER_CCU_MODULE_MANIFEST_SYMBOL "simpler_ccu_module_manifest"
#define SIMPLER_CCU_BUILD_ID_SIZE 32U
#define SIMPLER_CCU_ABI_VERSION 2U
#define SIMPLER_CCU_EVENT_MASK 1ULL
#define SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT 1U
#define SIMPLER_CCU_TASK_ARGUMENT_COUNT 2U
#define SIMPLER_CCU_MAX_RANKS 64U

#define SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS (1U << 0)
#define SIMPLER_CCU_KERNEL_CAP_MASK SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS

typedef struct SimplerCcuModuleManifest {
    uint32_t struct_size;
    uint32_t manifest_size;
    uint32_t kernel_record_size;
    uint32_t kernel_count;
    uint32_t kernel_records_offset;
    uint32_t string_table_offset;
    uint32_t string_table_size;
    uint32_t capability_flags;
    uint8_t sdk_build_id[SIMPLER_CCU_BUILD_ID_SIZE];
    uint8_t pto_isa_build_id[SIMPLER_CCU_BUILD_ID_SIZE];
    uint32_t abi_version;
    uint32_t reserved;
} SimplerCcuModuleManifest;

typedef struct SimplerCcuKernelRecord {
    uint32_t struct_size;
    uint32_t kernel_id;
    uint64_t parameter_size;
    uint32_t parameter_alignment;
    uint32_t capability_flags;
    uint32_t entry_symbol_offset;
    uint32_t entry_symbol_size;
    uint32_t registration_binding_size;
    uint32_t registration_argument_count;
    uint32_t task_argument_count;
    uint32_t required_rank_count;  // Zero accepts any supported domain size.
    uint32_t reserved;
} SimplerCcuKernelRecord;

typedef struct SimplerCcuRegistrationBinding {
    uint32_t struct_size;
    uint32_t ccu_id;
    uint32_t rank_id;
    uint32_t rank_count;
    uint64_t channels[SIMPLER_CCU_MAX_RANKS];
    uint64_t ready_event_handle;
    uint64_t trigger_event_handle;
    uint64_t done_event_handle;
} SimplerCcuRegistrationBinding;

typedef struct SimplerCcuLaunchArguments {
    uint64_t parameter_addr;
    uint64_t parameter_token;
} SimplerCcuLaunchArguments;

typedef struct SimplerCcuEventDescriptor {
    uint64_t address;
    uint64_t mask;
} SimplerCcuEventDescriptor;

typedef struct SimplerCcuControl {
    uint32_t struct_size;
    uint32_t ccu_id;
    uint64_t run_generation;
    uint32_t parameter_alignment;
    uint32_t flags;
    uint64_t parameter_addr;
    uint64_t parameter_size;
    uint64_t parameter_token;
    SimplerCcuEventDescriptor ready;
    SimplerCcuEventDescriptor trigger;
    SimplerCcuEventDescriptor done;
} SimplerCcuControl;

/* Published through the domain runtime context; controls_addr points to ccu_id-indexed controls. */
typedef struct SimplerCcuWorkspace {
    uint32_t struct_size;
    uint32_t ccu_count;
    uint64_t run_generation;
    uint64_t controls_addr;
    uint32_t control_size;
    uint32_t flags;
    uint64_t windows_addr;
    uint32_t window_count;
    uint32_t window_size;
} SimplerCcuWorkspace;

/* In C, (void) specifies zero arguments; () leaves arguments unspecified. */
// NOLINTNEXTLINE(modernize-redundant-void-arg)
typedef const SimplerCcuModuleManifest *(*SimplerCcuModuleManifestEntry)(void);
typedef int32_t (*SimplerCcuNativeKernelEntry)(const SimplerCcuRegistrationBinding *binding);

#ifdef __cplusplus
}

#define SIMPLER_CCU_ASSERT_POD(type) \
    static_assert(std::is_trivially_copyable_v<type> && std::is_standard_layout_v<type>, #type " must be C/POD")

SIMPLER_CCU_ASSERT_POD(SimplerCcuModuleManifest);
SIMPLER_CCU_ASSERT_POD(SimplerCcuKernelRecord);
SIMPLER_CCU_ASSERT_POD(SimplerCcuRegistrationBinding);
SIMPLER_CCU_ASSERT_POD(SimplerCcuLaunchArguments);
SIMPLER_CCU_ASSERT_POD(SimplerCcuEventDescriptor);
SIMPLER_CCU_ASSERT_POD(SimplerCcuControl);
SIMPLER_CCU_ASSERT_POD(SimplerCcuWorkspace);

static_assert(sizeof(SimplerCcuModuleManifest) == 104, "CCU manifest ABI size changed");
static_assert(sizeof(SimplerCcuKernelRecord) == 56, "CCU kernel record ABI size changed");
static_assert(sizeof(SimplerCcuRegistrationBinding) == 552, "CCU registration binding ABI size changed");
static_assert(sizeof(SimplerCcuLaunchArguments) == 16, "CCU launch arguments ABI size changed");
static_assert(sizeof(SimplerCcuEventDescriptor) == 16, "CCU Event descriptor ABI size changed");
static_assert(sizeof(SimplerCcuControl) == 96, "CCU control ABI size changed");
static_assert(sizeof(SimplerCcuWorkspace) == 48, "CCU workspace ABI size changed");

#undef SIMPLER_CCU_ASSERT_POD
#endif
