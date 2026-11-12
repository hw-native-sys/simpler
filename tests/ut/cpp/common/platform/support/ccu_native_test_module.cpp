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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "platform_comm/ccu/ccu_native_abi.h"

extern "C" int simpler_ccu_test_manifest_variant;

namespace {

struct TestManifest {
    SimplerCcuModuleManifestV2 manifest;
    SimplerCcuKernelRecordV2 records[2];
    SimplerCcuMissionInstanceV2 instances[3];
    char strings[50];
};

TestManifest make_manifest() {
    TestManifest result{};
    result.manifest.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
    result.manifest.struct_size = sizeof(SimplerCcuModuleManifestV2);
    result.manifest.manifest_size = sizeof(TestManifest);
    result.manifest.kernel_record_size = sizeof(SimplerCcuKernelRecordV2);
    result.manifest.kernel_count = 2;
    result.manifest.kernel_records_offset = offsetof(TestManifest, records);
    result.manifest.instance_record_size = sizeof(SimplerCcuMissionInstanceV2);
    result.manifest.instance_count = 3;
    result.manifest.instance_records_offset = offsetof(TestManifest, instances);
    result.instances[0] = {sizeof(SimplerCcuMissionInstanceV2), 0, 3, 0, {0, 0}};
    result.instances[1] = {sizeof(SimplerCcuMissionInstanceV2), 1, 3, 0, {0, 0}};
    result.instances[2] = {sizeof(SimplerCcuMissionInstanceV2), 2, 4, 0, {0, 0}};
    result.manifest.string_table_offset = offsetof(TestManifest, strings);
    result.manifest.string_table_size = sizeof(result.strings);
    std::memset(result.manifest.sdk_build_id, 0x11, SIMPLER_CCU_BUILD_ID_SIZE);
    std::memset(result.manifest.pto_isa_build_id, 0x22, SIMPLER_CCU_BUILD_ID_SIZE);

    static constexpr char first[] = "simpler_ccu_test_kernel_a";
    static constexpr char second[] = "simpler_ccu_test_kernel_b";
    static_assert(sizeof(first) - 1 + sizeof(second) - 1 == sizeof(result.strings));
    std::memcpy(result.strings, first, sizeof(first) - 1);
    std::memcpy(result.strings + sizeof(first) - 1, second, sizeof(second) - 1);
    result.records[0] = {
        sizeof(SimplerCcuKernelRecordV2),
        3,
        0,
        SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION,
        128,
        64,
        SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS,
        static_cast<uint32_t>(offsetof(TestManifest, strings)),
        sizeof(first) - 1,
        SIMPLER_CCU_REGISTRATION_BINDING_LAYOUT_VERSION,
        sizeof(SimplerCcuRegistrationBindingV2),
        SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
        SIMPLER_CCU_TASK_ARGUMENT_COUNT,
        {0, 0}
    };
    result.records[1] = {
        sizeof(SimplerCcuKernelRecordV2),
        4,
        0,
        SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION,
        256,
        128,
        SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS,
        static_cast<uint32_t>(offsetof(TestManifest, strings) + sizeof(first) - 1),
        sizeof(second) - 1,
        SIMPLER_CCU_REGISTRATION_BINDING_LAYOUT_VERSION,
        sizeof(SimplerCcuRegistrationBindingV2),
        SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
        SIMPLER_CCU_TASK_ARGUMENT_COUNT,
        {0, 0}
    };
    return result;
}

}  // namespace

#ifndef CCU_TEST_ENTRY_RESULT
#define CCU_TEST_ENTRY_RESULT 3
#endif
extern "C" int simpler_ccu_test_kernel_a(const SimplerCcuRegistrationBindingV2 *) noexcept {
    return CCU_TEST_ENTRY_RESULT;
}
extern "C" int simpler_ccu_test_kernel_b(const SimplerCcuRegistrationBindingV2 *) noexcept { return 4; }

extern "C" const SimplerCcuModuleManifestV2 *simpler_ccu_module_manifest_v2() noexcept {
    static TestManifest manifest;
    manifest = make_manifest();
    switch (simpler_ccu_test_manifest_variant) {
    case 1:
        ++manifest.manifest.abi_version;
        break;
    case 2:
        manifest.manifest.reserved[0] = 1;
        break;
    case 3:
        manifest.records[1].kernel_id = manifest.records[0].kernel_id;
        break;
    case 4:
        manifest.records[0].capability_flags = UINT32_MAX;
        break;
    case 5:
        manifest.records[0].parameter_alignment = 3;
        break;
    case 6:
        manifest.strings[0] = '9';
        break;
    case 7:
        manifest.records[0].entry_symbol_offset = manifest.manifest.manifest_size;
        break;
    case 8:
        manifest.records[0].reserved0 = 1;
        break;
    case 9:
        manifest.manifest.string_table_offset = manifest.manifest.kernel_records_offset;
        break;
    case 10:
        ++manifest.records[0].task_argument_count;
        break;
    case 11:
        manifest.manifest.abi_version = 1;
        break;
    case 12:
        manifest.instances[1].mission_index = 0;
        break;
    case 13:
        manifest.instances[1].mission_index = 3;
        break;
    case 14:
        manifest.instances[1].kernel_id = 99;
        break;
    case 15:
        manifest.instances[2].kernel_id = 3;
        break;
    case 16:
        manifest.manifest.instance_records_offset = manifest.manifest.kernel_records_offset;
        break;
    case 17:
        manifest.manifest.instance_records_offset = manifest.manifest.string_table_offset;
        break;
    case 18:
        manifest.manifest.instance_count = 0;
        break;
    case 19:
        manifest.manifest.instance_records_offset = UINT32_MAX;
        break;
    case 20:
        ++manifest.manifest.instance_record_size;
        break;
    case 21:
        manifest.instances[1].flags = 1;
        break;
    case 22:
        manifest.instances[1].reserved[0] = 1;
        break;
    case 23:
        ++manifest.instances[1].struct_size;
        break;
    case 24:
        ++manifest.manifest.instance_records_offset;
        break;
    default:
        break;
    }
    return &manifest.manifest;
}
