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
 * @file acl_hal_device.h
 * @brief ACL-logical to driver-visible device-id translation for direct HAL calls.
 */
#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "common/unified_log.h"

namespace pto {

/**
 * Translate an ACL/rt *logical* device id into the *driver-visible* id that the
 * raw driver HAL indexes.
 *
 * Contract — every device id crossing into the driver must pick the right space:
 *   - Direct `hal*` / `dlsym`'d driver calls (`halMemCtl`, `halGetDeviceInfo*`,
 *     ...) index the driver-visible space and MUST translate through this
 *     function: `hal_fn(acl_to_hal_device_id(device_id))`.
 *   - ACL/rt entry points (`aclrtSetDevice`, `rtMalloc`, `rtMemcpy`, stream
 *     APIs, ...) take the logical id and MUST NOT.
 *   - One ACL exception, measured on CANN 9.0.0 / a2a3:
 *     `aclrtMemAccessDesc::location.id` reaches the driver untranslated and MUST
 *     translate, even though `aclrtMemSetAccess` is an ACL entry point. Its
 *     sibling `aclrtPhysicalMemProp::location.id` MUST NOT: it is read in the
 *     logical space, and must in fact name the *bound* logical device.
 *     The A5 ACL runtime validates both fields in the ACL-logical space, so neither
 *     field may be translated there.
 *
 *     Measured under `ASCEND_RT_VISIBLE_DEVICES=2,3` bound to logical 0 (card 2).
 *     The grant matters: the translated id of the bound device (2) has to fall
 *     outside the logical range `[0,1]`, or the same value is legal in both
 *     spaces and no outcome can tell them apart.
 *
 *       aclrtMallocPhysical  prop.location.id=0 (bound logical) -> 0        2 GiB appears on card 2
 *                            prop.location.id=2 (translated)    -> 107001   ACL_ERROR_RT_INVALID_DEVICEID
 *                            prop.location.id=1 (other logical) -> 507899
 *       aclrtMemSetAccess    desc.location.id=0 (logical)        -> 507899
 *                            desc.location.id=2 (translated)     -> 0        OK
 *
 *     The 107001 row is what settles the field's space: under a driver-visible
 *     reading, 2 *is* the bound card and the call would have succeeded.
 *     `aclrtMemGetAllocationGranularity` rejects it identically, so the check is
 *     not specific to the allocation call. A symmetric "fix" of the memory
 *     property therefore fails loudly rather than silently allocating elsewhere —
 *     107001 for the translated id, 507899 for any other logical id.
 *
 *     The asymmetry is per-field, not per-API, so neither half generalises to the
 *     other.
 *
 * `ASCEND_RT_VISIBLE_DEVICES` renumbers the logical space to `0..N-1` while the
 * HAL keeps indexing the driver-visible space; a direct HAL call made with the
 * logical id then targets the wrong device (`rc=42`). This returns the identity
 * when the variable is unset (legacy / no isolation).
 *
 * The whole list is validated before any entry is trusted: one malformed token
 * means the environment is misconfigured, so we fall back to the unremapped
 * logical id rather than act on a half-parsed guess. Every fallback is logged so
 * a silent mis-target becomes a two-second diagnosis.
 *
 * Note: the returned value is the *driver-visible* id, which need not equal the
 * absolute physical (`npu-smi`) id — a container-level `ASCEND_VISIBLE_DEVICES`
 * remaps the driver's own numbering, and `ASCEND_RT_VISIBLE_DEVICES` indexes
 * into that already-remapped set. Indexing the driver-visible set is exactly
 * what the HAL needs.
 */
inline int64_t acl_to_hal_device_id(int64_t logical_id) {
    const char *vis = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (vis == nullptr || vis[0] == '\0' || logical_id < 0) {
        return logical_id;
    }
    const std::string_view list(vis);
    int64_t idx = 0;
    int64_t driver_id = -1;  // driver-visible id at logical_id, once seen
    std::size_t pos = 0;
    while (pos < list.size()) {
        while (pos < list.size() && (list[pos] == ' ' || list[pos] == ',')) {
            ++pos;
        }
        if (pos >= list.size()) {
            break;
        }
        int value = 0;
        const auto [ptr, ec] = std::from_chars(list.data() + pos, list.data() + list.size(), value);
        // from_chars reports the token end via ptr and overflow via ec; a token
        // must also be terminated by a list delimiter or end-of-string ("6x" is
        // malformed), and negative ids are invalid.
        const bool terminated = ptr == list.data() + list.size() || *ptr == ' ' || *ptr == ',';
        if (ptr == list.data() + pos || ec != std::errc() || value < 0 || !terminated) {
            LOG_WARN(
                "ASCEND_RT_VISIBLE_DEVICES=\"%s\" is malformed; using logical device id %lld unremapped", vis,
                static_cast<long long>(logical_id)
            );
            return logical_id;
        }
        if (idx == logical_id) {
            driver_id = value;
        }
        ++idx;
        pos = static_cast<std::size_t>(ptr - list.data());
    }
    if (driver_id < 0) {
        // Distinct from the unset case above: the variable is set but too short,
        // which is a definite misconfiguration rather than a no-isolation default.
        LOG_WARN(
            "logical device id %lld is out of range of ASCEND_RT_VISIBLE_DEVICES=\"%s\" (%lld visible); "
            "using it unremapped",
            static_cast<long long>(logical_id), vis, static_cast<long long>(idx)
        );
        return logical_id;
    }
    LOG_DEBUG("device id: acl %lld -> hal %lld", static_cast<long long>(logical_id), static_cast<long long>(driver_id));
    return driver_id;
}

}  // namespace pto
