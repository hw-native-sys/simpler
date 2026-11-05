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

enum class AsyncUseDeclarationKind : uint32_t {
    UNKNOWN,
    COUNTER_PAYLOAD,
    CONSUMER_CLEAN_READ,
    SDMA_TRANSFER,
    COUNTER_ONLY
};

// Native author/compiler contract, not a cache probe: publishers complete
// writeback and a publication barrier before signalling; every observer's
// touched cache lines have no unpublished dirty alias when acquisition runs.
// It covers whole cache lines, including bytes neighbouring the logical view.
// Native code violating this contract is outside the declared interface.
// A consumer may write after entry (INOUT); its writes retain the ordinary
// kernel output-publication obligation. Clean is required before maintenance,
// not forever after it.
enum class AsyncUseCacheContract : uint32_t { UNQUALIFIED, PUBLISHED_CLEAN_LINES };

// Host orchestration declaration, not a Tensor/Definition wire field. Argument
// positions resolve against each invocation's actual task payload. No address,
// captured owner or allocation generation is stored in a reusable declaration.
struct AsyncUseDeclaration {
    AsyncUseDeclarationKind kind{AsyncUseDeclarationKind::UNKNOWN};
    uint32_t condition_arg{UINT32_MAX};
    uint32_t payload_arg{UINT32_MAX};
    uint32_t source_arg{UINT32_MAX};
    uint32_t context_scalar{UINT32_MAX};
    uint32_t record_stride{0};
    AsyncUseCacheContract cache_contract{AsyncUseCacheContract::UNQUALIFIED};

    bool operator==(const AsyncUseDeclaration &other) const {
        return kind == other.kind && condition_arg == other.condition_arg && payload_arg == other.payload_arg &&
               source_arg == other.source_arg && context_scalar == other.context_scalar &&
               record_stride == other.record_stride && cache_contract == other.cache_contract;
    }
};

inline bool async_use_declaration_valid(const AsyncUseDeclaration &use, uint32_t tensor_count, uint32_t scalar_count) {
    if (use.cache_contract != AsyncUseCacheContract::UNQUALIFIED &&
        use.cache_contract != AsyncUseCacheContract::PUBLISHED_CLEAN_LINES)
        return false;
    switch (use.kind) {
    case AsyncUseDeclarationKind::COUNTER_PAYLOAD:
    case AsyncUseDeclarationKind::COUNTER_ONLY:
        if (use.kind == AsyncUseDeclarationKind::COUNTER_ONLY ? use.payload_arg != UINT32_MAX :
                                                                use.payload_arg >= tensor_count)
            return false;
        return use.condition_arg < tensor_count && use.source_arg == UINT32_MAX && use.context_scalar == UINT32_MAX &&
               use.record_stride >= sizeof(int32_t) && use.record_stride % alignof(int32_t) == 0;
    case AsyncUseDeclarationKind::CONSUMER_CLEAN_READ:
        return use.payload_arg < tensor_count && use.condition_arg == UINT32_MAX && use.source_arg == UINT32_MAX &&
               use.context_scalar == UINT32_MAX && use.record_stride == 0;
    case AsyncUseDeclarationKind::SDMA_TRANSFER:
        return use.payload_arg < tensor_count && use.condition_arg == UINT32_MAX && use.source_arg < tensor_count &&
               use.context_scalar < scalar_count && use.record_stride == 0;
    default:
        return false;
    }
}

inline bool async_use_is_counter(AsyncUseDeclarationKind kind) {
    return kind == AsyncUseDeclarationKind::COUNTER_PAYLOAD || kind == AsyncUseDeclarationKind::COUNTER_ONLY;
}

// Counter declarations select a signal argument's record layout. Several
// payloads may be associated with that layout; no-payload and a payload claim
// for the same selector disagree, as do two different record strides. SDMA
// selectors include their context argument, so peers sharing source/destination
// argument positions remain distinct uses.
inline bool async_use_declarations_conflict(const AsyncUseDeclaration &a, const AsyncUseDeclaration &b) {
    if (async_use_is_counter(a.kind) && async_use_is_counter(b.kind) && a.condition_arg == b.condition_arg)
        return a.kind != b.kind || a.record_stride != b.record_stride || a.cache_contract != b.cache_contract;
    if (a.kind == b.kind && a.payload_arg == b.payload_arg && a.source_arg == b.source_arg &&
        a.context_scalar == b.context_scalar)
        return a.cache_contract != b.cache_contract;
    return false;
}
