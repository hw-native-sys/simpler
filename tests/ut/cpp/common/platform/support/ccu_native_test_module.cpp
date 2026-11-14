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
    SimplerCcuModuleManifest manifest;
    SimplerCcuKernelRecord records[2];
    char strings[50];
};

TestManifest make_manifest() {
    TestManifest result{};
    result.manifest.struct_size = sizeof(SimplerCcuModuleManifest);
    result.manifest.abi_version = SIMPLER_CCU_ABI_VERSION;
    result.manifest.manifest_size = sizeof(TestManifest);
    result.manifest.kernel_record_size = sizeof(SimplerCcuKernelRecord);
    result.manifest.kernel_count = 2;
    result.manifest.kernel_records_offset = offsetof(TestManifest, records);
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
        sizeof(SimplerCcuKernelRecord),
        3,
        128,
        64,
        SIMPLER_CCU_KERNEL_CAP_MASK,
        static_cast<uint32_t>(offsetof(TestManifest, strings)),
        sizeof(first) - 1,
        sizeof(SimplerCcuRegistrationBinding),
        SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
        SIMPLER_CCU_TASK_ARGUMENT_COUNT
    };
    result.records[1] = {
        sizeof(SimplerCcuKernelRecord),
        4,
        256,
        128,
        SIMPLER_CCU_KERNEL_CAP_MASK,
        static_cast<uint32_t>(offsetof(TestManifest, strings) + sizeof(first) - 1),
        sizeof(second) - 1,
        sizeof(SimplerCcuRegistrationBinding),
        SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
        SIMPLER_CCU_TASK_ARGUMENT_COUNT
    };
    return result;
}

}  // namespace

#ifndef CCU_TEST_ENTRY_RESULT
#define CCU_TEST_ENTRY_RESULT 3
#endif
extern "C" int simpler_ccu_test_kernel_a(const SimplerCcuRegistrationBinding *) noexcept {
    return CCU_TEST_ENTRY_RESULT;
}
extern "C" int simpler_ccu_test_kernel_b(const SimplerCcuRegistrationBinding *) noexcept { return 4; }

extern "C" const SimplerCcuModuleManifest *simpler_ccu_module_manifest() noexcept {
    static TestManifest manifest;
    manifest = make_manifest();
    switch (simpler_ccu_test_manifest_variant) {
    case 1:
        ++manifest.manifest.struct_size;
        break;
    case 2:
        manifest.manifest.capability_flags = 1;
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
        manifest.records[0].parameter_alignment = 8192;
        break;
    case 9:
        manifest.manifest.string_table_offset = manifest.manifest.kernel_records_offset;
        break;
    case 10:
        ++manifest.records[0].task_argument_count;
        break;
    case 11:
        manifest.manifest.kernel_record_size = 1;
        break;
    case 12:
        manifest.manifest.manifest_size = 32;
        break;
    case 13:
        manifest.manifest.kernel_count = 0;
        break;
    case 14:
        manifest.manifest.kernel_records_offset = 1;
        break;
    case 15:
        manifest.records[0].registration_binding_size = 1;
        break;
    case 16:
        manifest.manifest.abi_version = 1;
        break;
    case 17:
        manifest.manifest.reserved = 1;
        break;
    case 18:
        manifest.records[0].required_rank_count = SIMPLER_CCU_MAX_RANKS + 1;
        break;
    default:
        break;
    }
    return &manifest.manifest;
}
