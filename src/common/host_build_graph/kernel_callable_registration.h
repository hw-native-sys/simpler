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
#include <type_traits>

namespace simpler::kernel {

// Device execution binding for an existing mainline callable registration.
// It allocates no IDs and owns no code images. The owner orders publication
// before use and excludes revocation while readers remain. Launch packets
// cannot install or replace this record.
struct HbgCallableBinding {
    uint64_t runtime_address;
    uint64_t context_generation;
    uint64_t callable_generation;
    int32_t callable_id;
    int32_t tensor_count;
    int32_t scalar_count;
    uint32_t reserved;
};
static_assert(sizeof(HbgCallableBinding) == 40);
static_assert(std::is_trivially_copyable_v<HbgCallableBinding> && std::is_standard_layout_v<HbgCallableBinding>);

}  // namespace simpler::kernel

// Cold device control ABI; args names a readable sizeof(HbgCallableBinding)
// record. Null, invalid counts/ID/reserved/generations or a missing/mismatched
// context return -1. Exact duplicate registration succeeds; replacing a live
// binding requires prior revoke. No ID or code allocation occurs here.
// The Host owner serializes these calls outside capture and excludes readers.
extern "C" int simpler_aicpu_l1_hbg_register_callable(void *args);
// Revoke requires the same complete binding identity and context. An exact
// repeated revoke succeeds; malformed, absent or conflicting records return -1.
// Success invalidates the generation; freeing code/Runtime is the Host's job.
extern "C" int simpler_aicpu_l1_hbg_revoke_callable(void *args);
