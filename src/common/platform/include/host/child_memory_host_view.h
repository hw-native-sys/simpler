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
 * @file child_memory_host_view.h
 * @brief Host mappings of child-memory allocations, held for the allocation's lifetime.
 *
 * A `host_build_graph` orchestration runs on the host and may read or write a
 * child-memory tensor's bytes to shape the graph. Serving that needs a host
 * mapping of the device allocation, and establishing one is not cheap enough
 * to repeat per run: measured on a2a3 / CANN 9.0.0, a
 * `halHostRegister` + `halHostUnregister` pair costs ~5.2 µs before any bytes
 * are mapped and ~7.0 ms/GiB beyond that, so a two-tensor bind pays ~24 µs
 * every run — see docs/investigations/2026-09-hbg-per-run-host-view-rebuild.md.
 *
 * So a mapping is established once and kept. The unit is the **allocation**,
 * not the tensor: several tensors and views can sit inside one child buffer and
 * share its mapping, and the allocation is what a later free invalidates.
 *
 * This class is the table only — it never calls the platform. The runner that
 * owns `device_malloc` / `device_free` performs the register and unregister and
 * drives this table around them, which is what keeps a cached host VA from
 * outliving the pages behind it: a mapping is dropped by the same `free_tensor`
 * that releases its allocation, and by `finalize`.
 *
 * No lock. One runner owns one table, and every mutator runs on that runner's
 * allocation path.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

class ChildMemoryHostViewCache {
public:
    struct Entry {
        void *host_view;
        std::size_t bytes;
    };

    /**
     * The host address serving `dev_ptr`, given the base of the allocation that
     * contains it.
     *
     * @return nullptr when this allocation has no mapping yet. The returned
     *         address carries `dev_ptr`'s offset within the allocation, so the
     *         caller reads and writes it directly.
     */
    void *lookup(const void *alloc_base, const void *dev_ptr) const {
        auto it = entries_.find(const_cast<void *>(alloc_base));
        if (it == entries_.end()) return nullptr;
        const auto offset =
            static_cast<const unsigned char *>(dev_ptr) - static_cast<const unsigned char *>(alloc_base);
        return static_cast<unsigned char *>(it->second.host_view) + offset;
    }

    /** Record a mapping the caller just established over the whole allocation. */
    void insert(void *alloc_base, std::size_t bytes, void *host_view) {
        auto [it, inserted] = entries_.try_emplace(alloc_base, Entry{host_view, bytes});
        if (inserted) mapped_bytes_ += bytes;
    }

    /**
     * Drop this allocation's mapping.
     *
     * @return false when nothing was cached, which is the common case — most
     *         allocations are never touched by an orchestration. The caller
     *         unregisters by allocation base, so nothing is handed back.
     */
    bool take(void *alloc_base) {
        auto it = entries_.find(alloc_base);
        if (it == entries_.end()) return false;
        mapped_bytes_ -= it->second.bytes;
        entries_.erase(it);
        return true;
    }

    /** Drop every mapping, returning the allocation bases to unregister. */
    std::vector<void *> take_all() {
        std::vector<void *> bases;
        bases.reserve(entries_.size());
        for (auto &[alloc_base, entry] : entries_)
            bases.push_back(alloc_base);
        entries_.clear();
        mapped_bytes_ = 0;
        return bases;
    }

    std::size_t count() const noexcept { return entries_.size(); }

    /** Device bytes currently mapped into this process on behalf of orchestration. */
    std::uint64_t mapped_bytes() const noexcept { return mapped_bytes_; }

private:
    std::unordered_map<void *, Entry> entries_;
    std::uint64_t mapped_bytes_ = 0;
};
