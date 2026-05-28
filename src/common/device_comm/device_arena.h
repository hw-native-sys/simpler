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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

/**
 * Single-shot bump arena over one backing allocation.
 *
 * Workflow:
 *   1. reserve(size, align)  — declare every sub-region up-front, returning
 *      its offset (also serving as region id);
 *   2. commit(base_align)    — perform exactly one underlying allocation,
 *      sized to fit all reservations plus forward-alignment padding;
 *   3. region_ptr(offset)    — read back the device pointer of any region.
 *
 * Backing alloc/free are injected as function pointers + ctx, so this header
 * is usable from both host code (forwarding to MemoryAllocator) and AICPU
 * code (forwarding to libc). The class deliberately avoids STL containers
 * (std::vector / std::function) since it is compiled into the AICPU binary.
 *
 * RAII: the destructor calls release(). Copy is disallowed.
 */
class DeviceArena {
public:
    using AllocFn = void *(*)(void *ctx, size_t size);
    using FreeFn = void (*)(void *ctx, void *ptr);

    // Upper bound on the number of distinct sub-regions a single arena can
    // hold. Current AICPU init reserves ~30 regions; 128 leaves comfortable
    // headroom for per-thread / per-ring expansions.
    static constexpr size_t kMaxRegions = 128;

    // Default base alignment for the underlying allocation. 1024 bytes is a
    // hardware requirement (HBM DMA granularity / PTO2_PACKED_OUTPUT_ALIGN),
    // not a software-side optimization — do NOT lower it to the 64-byte cache
    // line that satisfies every sub-region alignment requirement. The cost is
    // a one-time forward-alignment pad of up to 1023 bytes per arena, which is
    // negligible compared to typical region payloads (GM heap MBs, runtime KBs).
    static constexpr size_t kDefaultBaseAlign = 1024;

    // Default libc backend used when the caller does not supply alloc/free.
    static void *default_alloc(void * /*ctx*/, size_t size) noexcept { return std::malloc(size); }
    static void default_free(void * /*ctx*/, void *ptr) noexcept { std::free(ptr); }

    // Default ctor: libc-backed (most AICPU / UT use sites). For callers that
    // need a custom backend (e.g. host-side MemoryAllocator), use the 3-arg ctor.
    DeviceArena() noexcept :
        alloc_(&default_alloc),
        free_(&default_free),
        ctx_(nullptr) {}

    DeviceArena(AllocFn alloc, FreeFn free_fn, void *ctx) noexcept :
        alloc_(alloc),
        free_(free_fn),
        ctx_(ctx) {}

    ~DeviceArena() noexcept { release(); }

    // Non-copyable, and implicitly non-movable: user-declared copy operations
    // suppress the implicit move ctor/assignment. The arena owns a heap-allocated
    // buffer, so moves would need to transfer raw_base_ + region table — keep it
    // pinned in place.
    DeviceArena(const DeviceArena &) = delete;
    DeviceArena &operator=(const DeviceArena &) = delete;

    // Phase 1: declare a sub-region. Returns its offset from base() (also
    // serves as region id). Asserts if called after commit() or if the region
    // count would exceed kMaxRegions.
    size_t reserve(size_t size, size_t align) noexcept;

    // Phase 2: perform the backing allocation, forward-align the base, and
    // mark the arena committed. Idempotent: repeated calls return the same
    // base without re-allocating. Returns nullptr if the backing alloc fails.
    //
    // NOT noexcept: the injected alloc_ function pointer may run code that
    // throws (e.g. host-side MemoryAllocator goes through std::mutex /
    // std::unordered_set which can throw on OOM or lock failure). Letting
    // exceptions propagate lets the extern "C" boundary's catch-all turn
    // them into status codes; a noexcept here would std::terminate before
    // that boundary gets a chance. (release() stays noexcept because it
    // runs from ~DeviceArena(), where throwing would terminate anyway —
    // the trampoline's free path must therefore be nothrow.)
    void *commit(size_t base_align = kDefaultBaseAlign);

    // Phase 2 alternative: attach to an externally-owned buffer instead of
    // allocating one. Caller guarantees:
    //   (a) `external_base` is already `base_align`-aligned — attach does
    //       NOT forward-align, since the prebuilt image was constructed for
    //       the address the caller advertised;
    //   (b) the buffer is at least `total_size()` bytes (the sum of sizes
    //       passed to reserve()), since attach uses no forward-alignment
    //       slack of its own;
    //   (c) all region offsets the caller plans to read back via
    //       `region_ptr(off)` are held by the caller — attach does NOT
    //       repopulate the internal region table, and reserve() cannot run
    //       after attach (it asserts !committed_). `region_size()` likewise
    //       returns 0 for attached arenas; treat the arena post-attach as
    //       a base-pointer wrapper.
    //
    // Re-attach (release + attach the same or another buffer) is permitted
    // so the AICPU boot path can rebind the same pooled image each run.
    //
    // The external buffer is NOT freed by release()/~DeviceArena(); ownership
    // stays with the caller. Used for the prebuilt-arena fast path where
    // a host-built image is rtMemcpy'd into a device buffer that DeviceRunner
    // owns across runs.
    void attach(void *external_base, size_t base_align = kDefaultBaseAlign) noexcept;

    // Phase 3: pointer to the sub-region at `offset`. Asserts if called
    // before commit().
    void *region_ptr(size_t offset) const noexcept;

    // Size of the sub-region whose offset matches `offset`. Linear scan;
    // intended for debug / assertions, not hot path. Returns 0 for an
    // attached arena (attach() does not repopulate the region table) —
    // callers in the prebuilt-image path keep sizes alongside their offsets
    // instead.
    size_t region_size(size_t offset) const noexcept;

    // Free the backing buffer (if any) and reset to the pre-commit state so
    // a fresh reserve+commit cycle can run.
    void release() noexcept;

    bool is_committed() const noexcept { return committed_; }
    void *base() const noexcept { return base_; }

    // Total bytes reserved across all regions (excludes base_align padding).
    size_t total_size() const noexcept { return cursor_; }

    // Diagnostics for UT — counts of underlying alloc/free invocations.
    size_t alloc_count() const noexcept { return alloc_count_; }
    size_t free_count() const noexcept { return free_count_; }

private:
    struct Region {
        size_t offset;
        size_t size;
    };

    AllocFn alloc_;
    FreeFn free_;
    void *ctx_;

    Region regions_[kMaxRegions]{};
    size_t region_count_{0};

    size_t cursor_{0};
    void *raw_base_{nullptr};
    size_t raw_size_{0};
    void *base_{nullptr};
    bool committed_{false};
    // True when committed via attach(): the backing buffer is externally
    // owned, so release() must not call free_().
    bool attached_{false};

    size_t alloc_count_{0};
    size_t free_count_{0};
};

inline size_t DeviceArena::reserve(size_t size, size_t align) noexcept {
    assert(!committed_ && "DeviceArena::reserve() called after commit()");
    assert(region_count_ < kMaxRegions && "DeviceArena: exceeded kMaxRegions");
    assert(align > 0 && (align & (align - 1)) == 0 && "DeviceArena: align must be a power of two");
    cursor_ = (cursor_ + align - 1) & ~(align - 1);
    const size_t off = cursor_;
    regions_[region_count_++] = Region{off, size};
    cursor_ += size;
    return off;
}

inline void *DeviceArena::commit(size_t base_align) {
    if (committed_) return base_;
    assert(base_align > 0 && (base_align & (base_align - 1)) == 0 && "DeviceArena: base_align must be a power of two");
    // Allocate enough to fit the layout starting from any pointer returned by
    // the backend, then forward-align the visible base.
    raw_size_ = cursor_ + base_align - 1;
    raw_base_ = alloc_(ctx_, raw_size_);
    if (raw_base_ == nullptr) return nullptr;
    ++alloc_count_;
    const auto raw = reinterpret_cast<uintptr_t>(raw_base_);
    base_ = reinterpret_cast<void *>((raw + base_align - 1) & ~(static_cast<uintptr_t>(base_align) - 1));
    committed_ = true;
    return base_;
}

inline void DeviceArena::attach(void *external_base, size_t base_align) noexcept {
    // Re-attach (e.g. AICPU boot path attaches each run) is fine: only an
    // attached state can be "re-attached" — release() it first to keep
    // semantics tight. A real commit() (alloc-backed) must not be silently
    // dropped, so still trap on that.
    if (committed_) {
        assert(attached_ && "DeviceArena::attach() called after commit (only re-attach is allowed)");
        release();
    }
    // The external buffer must already be base_align-aligned by the caller —
    // forward-align in-place would shift the visible base off the address the
    // caller advertised (and that the prebuilt image was constructed for).
    // The checks below are promoted to unconditional aborts (rather than
    // plain assert()) because a misaligned attach silently produces a buffer
    // whose visible base disagrees with every offset the prebuilt image was
    // laid out against — release builds, which strip assert(), would still
    // run on a corrupted arena. Aborting at the breakage point is far cheaper
    // to triage than the downstream wild-pointer accesses.
    const auto raw = reinterpret_cast<uintptr_t>(external_base);
    const bool ok = (external_base != nullptr) && (base_align > 0) && ((base_align & (base_align - 1)) == 0) &&
                    ((raw & (static_cast<uintptr_t>(base_align) - 1)) == 0);
    if (!ok) {
        assert(false && "DeviceArena::attach(): null base, non-power-of-two align, or pre-alignment violated");
        std::abort();
    }
    base_ = external_base;
    raw_base_ = nullptr;
    raw_size_ = 0;
    committed_ = true;
    attached_ = true;
}

inline void *DeviceArena::region_ptr(size_t offset) const noexcept {
    assert(committed_ && "DeviceArena::region_ptr() called before commit()");
    return reinterpret_cast<char *>(base_) + offset;
}

inline size_t DeviceArena::region_size(size_t offset) const noexcept {
    for (size_t i = 0; i < region_count_; ++i) {
        if (regions_[i].offset == offset) return regions_[i].size;
    }
    return 0;
}

inline void DeviceArena::release() noexcept {
    // attached arenas wrap externally-owned memory — never free.
    if (raw_base_ != nullptr && !attached_) {
        free_(ctx_, raw_base_);
        ++free_count_;
    }
    raw_base_ = nullptr;
    base_ = nullptr;
    raw_size_ = 0;
    cursor_ = 0;
    region_count_ = 0;
    committed_ = false;
    attached_ = false;
}
