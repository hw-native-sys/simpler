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
 * ACL / RT Error-Code Annotation (host failure line)
 *
 * Emits the human-readable half of a CANN failure: what the return code means, what to do
 * next, and CANN's own last error message.
 *
 * A macro rather than a function on purpose. LOG_ERROR bakes in __FUNCTION__, __FILENAME__
 * and __LINE__ at its expansion point, so a function would stamp all of these lines with this
 * header instead of the call that failed. There are a dozen expansion sites (rtSetDevice,
 * rtMalloc, rtStreamCreate, the stream syncs, the kernel launches, ...) and *which* one failed
 * is the first thing a reader needs, so the annotation has to inherit the caller's provenance.
 *
 * The annotation goes on its own lines, after the call site's existing "<call> failed: <rc>"
 * line. That line -- and the "<op> failed with code <rc>" exception text it eventually turns
 * into -- is parsed by conftest to detect a poisoned device, so it is left untouched.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_ACL_ERROR_LOG_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_ACL_ERROR_LOG_H_

#include <acl/acl.h>

#include <cstdint>

#include "acl_error_names.h"
#include "common/unified_log.h"

// Annotates a non-zero CANN return code. An uncharacterised code is left alone rather than
// labelled with a guess; CANN's own message is still printed in that case, since it is the
// driver's account of what went wrong and beats any static table we could write.
#define ACL_LOG_ERROR_DETAIL(rc)                                                                                \
    do {                                                                                                        \
        const int32_t acl_rc_ = static_cast<int32_t>(rc);                                                       \
        if (acl_rc_ != 0) {                                                                                     \
            const char *acl_name_ = acl_error_name(acl_rc_);                                                    \
            if (acl_name_ != nullptr) {                                                                         \
                LOG_ERROR("ACL error detail: %d %s - %s", acl_rc_, acl_name_, acl_error_desc(acl_rc_));         \
                const char *acl_hint_ = acl_error_hint(acl_rc_);                                                \
                if (acl_hint_ != nullptr) {                                                                     \
                    LOG_ERROR("ACL error hint: %s. See docs/troubleshooting/device-error-codes.md", acl_hint_); \
                }                                                                                               \
            }                                                                                                   \
            const char *acl_recent_ = aclGetRecentErrMsg();                                                     \
            if (acl_recent_ != nullptr && acl_recent_[0] != '\0') {                                             \
                LOG_ERROR("ACL recent error message: %s", acl_recent_);                                         \
            }                                                                                                   \
        }                                                                                                       \
    } while (0)

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_ACL_ERROR_LOG_H_
