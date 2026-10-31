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

#include <cstddef>
#include <cstdint>

// One machine word: an aligned pointer plus a value tag in its low bits.
// Value-initialization produces null; default-initialized output storage is written before use.
template <typename T, typename Tag, int kTagBits>
class TaggedPtr {
    static_assert(kTagBits > 0 && kTagBits < sizeof(uintptr_t) * 8);
    static_assert(Tag::kBitWidth <= kTagBits);
    static constexpr uintptr_t kTagMask = (uintptr_t{1} << kTagBits) - 1;

public:
    TaggedPtr() = default;

    TaggedPtr(T *ptr, Tag tag) :
        word_(reinterpret_cast<uintptr_t>(ptr) | tag.bits()) {
        static_assert(alignof(T) >= (size_t{1} << kTagBits));
    }

    T *ptr() const { return reinterpret_cast<T *>(word_ & ~kTagMask); }
    Tag tag() const { return Tag::from_low_bits(word_ & kTagMask); }
    bool is_null() const { return ptr() == nullptr; }

private:
    uintptr_t word_;
};
