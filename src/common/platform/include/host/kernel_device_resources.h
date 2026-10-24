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
#include <vector>

#include "worker/pipeline_contract.h"
#include "host/memory_allocator.h"

// Available only to resource prepare/close, never to launch binding.
struct KernelResourceOps {
    void *context{nullptr};
    void *(*allocate)(void *, size_t){nullptr};
    int (*release)(void *, void *){nullptr};
    bool valid() const { return allocate != nullptr && release != nullptr; }

    static KernelResourceOps from_allocator(MemoryAllocator &allocator) {
        return {
            &allocator,
            [](void *ctx, size_t bytes) {
                return static_cast<MemoryAllocator *>(ctx)->alloc(bytes);
            },
            [](void *ctx, void *ptr) {
                return static_cast<MemoryAllocator *>(ctx)->free(ptr);
            }
        };
    }
};

// Runtime-defined region ordering is fixed for a context. Offsets describe
// destinations inside the contract's arenas, not addresses in a launch packet.
// Repeated preparation preserves arena kinds and offsets. An empty region has
// zero offset and binds to the view {0, 0}.
struct KernelResourceRegion {
    uint32_t arena_kind;
    uint64_t offset;
    uint64_t bytes;
};
struct KernelResourceLayout {
    uint64_t schema{0};
    PipelineContract contract{};
    std::vector<KernelResourceRegion> regions;
};
struct KernelResourceView {
    uint64_t address{0};
    uint64_t capacity{0};
};
// Owned snapshot of the frozen views. bind() copies the views, so this value
// remains readable after KernelDeviceResources or KernelExecutionState is
// closed or prepared again. The addresses it holds still name resources that
// are only valid while those resources are live.
struct KernelResourceBinding {
    std::vector<KernelResourceView> regions;
};

// Serialized by KernelExecutionState. The destructor performs no device calls;
// explicit close follows external quiescence and graph destruction.
class KernelDeviceResources {
public:
    KernelDeviceResources() = default;
    KernelDeviceResources(const KernelDeviceResources &) = delete;
    KernelDeviceResources &operator=(const KernelDeviceResources &) = delete;

    // First call allocates every arena resource and builds the region views.
    // Repeated calls require the same schema, arena kinds and offsets, and may
    // only shrink or clear a region; the allocation capacity stays frozen.
    int prepare(const KernelResourceLayout &layout, const KernelResourceOps &ops);
    // Freezes the prepared capacity; later prepare() calls cannot grow it.
    int freeze();
    // Copies the frozen views into out.regions. The snapshot owns its storage;
    // use the addresses only while these resources remain prepared.
    int bind(uint64_t schema, const uint64_t *required, size_t count, KernelResourceBinding &out) const;
    // Releases allocations in reverse order. A failed release retains that
    // allocation and keeps the object closing for an explicit retry.
    int close();
    bool prepared() const { return prepared_; }
    bool frozen() const { return frozen_; }
    bool closing() const { return closing_; }
    bool has_live_resources() const;
    int cleanup_error() const { return cleanup_error_; }

private:
    struct Allocation {
        uint32_t kind;
        uint64_t bytes;
        void *allocation{nullptr};
        uint64_t base{0};
    };
    KernelResourceOps ops_{};
    KernelResourceLayout layout_{};
    std::vector<Allocation> allocations_;
    std::vector<KernelResourceView> views_;
    bool prepared_{false};
    bool frozen_{false};
    bool closing_{false};
    int cleanup_error_{0};
};
