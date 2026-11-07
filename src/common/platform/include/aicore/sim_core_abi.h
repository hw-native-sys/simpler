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

#include <cstdint>
#include "common/core_type.h"

class Runtime;

// Same-build internal ABI shared by the loader, wrappers and test fixtures.
using SimIdentitySetFn = int (*)(uint32_t);
using SimIdentityHelpersInstallFn = void (*)(SimIdentitySetFn, SimIdentitySetFn);
using SimAdmissionReportFn = bool (*)(void *, int);
using SimA2a3CoreEntryFn =
    int (*)(Runtime *, int, CoreType, uint32_t, uint64_t, uint32_t, uint64_t, uint64_t, SimAdmissionReportFn, void *);
using SimA5CoreEntryFn = int (*)(
    Runtime *, int, CoreType, uint32_t, uint64_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t,
    SimAdmissionReportFn, void *
);
extern "C" void sim_install_core_identity_setters(SimIdentitySetFn, SimIdentitySetFn);
