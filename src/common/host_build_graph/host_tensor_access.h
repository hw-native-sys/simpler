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
 * @file host_tensor_access.h
 * @brief Tensor-byte access policy for the host orchestrator.
 *
 * In program mode, `simpler::hbg::Tensor::buffer.addr` is a device address.
 * host_build_graph runs the orchestrator on the host, so `get_tensor_data` /
 * `set_tensor_data` cannot assume the CPU executing them can load that address.
 * This is the seam where that platform capability is resolved.
 *
 * Kernel mode instead accepts only explicit caller-owned Host-copy arguments.
 * It registers their Host addresses directly, permits reads, and denies writes.
 * Device mappings and device-copy fallbacks are disabled, so Host build cannot
 * inspect device storage or introduce a stream synchronization.
 *
 * The current bind path registers one region per host-memory tensor, backed by
 * the caller's host tensor buffer, which the bind has just copied in H2D:
 *
 *   - A read observes that caller buffer.
 *   - A write mutates that caller buffer, then uses the device-copy hook so the
 *     device observes it too. This is visible even for an `IN` argument if its
 *     host orchestration calls `set_tensor_data`.
 *
 * A child-memory tensor has no such buffer: it arrives already on the device
 * and the bind stages nothing for it. Its region is therefore added
 * **unresolved** (`add_child_memory`) and costs nothing until an access lands
 * inside it, at which point one of two means is chosen for the whole
 * allocation:
 *
 *   - `acquire_child_memory_host_view` returns a host mapping the platform owns
 *     and keeps for the allocation's lifetime — reads and writes go straight
 *     through it, coherent by construction.
 *   - It returns null (a5 onboard has no host-map path; issue #1531 refuses
 *     ordinary-page small allocations on 64 KiB-page hosts), and every access
 *     is served by a device copy instead. That path holds no state, so a read
 *     cannot observe stale bytes and a write lands on the device immediately.
 *
 * Resolving on access rather than at bind means the set that gets a mapping is
 * exactly the set the orchestration touched — a tensor it never reads costs a
 * vector entry and nothing else.
 *
 * `add` also retains a null-fallback platform path: it asks the platform for a
 * host-readable mapping whose address may equal or differ from `dev_base`, and
 * always accesses the returned address. The current runtime-maker path cannot
 * reach it: host-memory tensors always have the caller buffer, while pure outputs
 * are deliberately left unregistered. The path remains as an explicit
 * platform-capability escape hatch in `add` and is covered directly by unit
 * tests; no current production caller reaches it.
 *
 * An address no region covers is a failure, never a raw dereference. Pure
 * outputs and GM-heap tensors the orchestrator created have no region, so both
 * reads and writes resolve to nothing.
 *
 * Regions and any optional mappings are owned by one orchestration run — the
 * window between copy-in and the first dispatched task. A caller-buffer view
 * holds the copied-in bytes, and nothing has executed yet to make it stale; once
 * tasks run, that view would be indistinguishable from live device memory.
 * `HostTensorAccessor` bounds the window and releases its mappings on every
 * exit path. A child-memory mapping is the exception it does not own: the
 * platform holds that one for the allocation's lifetime, so `close` leaves it
 * alone.
 *
 * `host/host_tensor_access.cpp` holds the only definitions of the read/write
 * pair, and libhost_runtime.so links them. Nothing in the AICPU build reaches
 * either: every caller is host-side.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

struct HostApi;  // common/host_api.h — fwd-declared so this header stays out of platform includes

enum class HostTensorAccessMode : uint8_t {
    // Program mode may stage, map or copy device storage as part of its
    // self-managed synchronous run.
    Program,
    // Kernel-mode Host build may read only explicit caller-owned Host copies.
    // Device mappings, D2H fallback and Host writes are unrepresentable.
    KernelHostCopiesOnly,
};

/**
 * The registered regions of one orchestration run, and any optional mappings
 * that run installed to serve them.
 *
 * One accessor per run, mutated only by the thread running that run's
 * orchestration. The region and mapping tables are plain vectors with no lock,
 * so concurrent `add` / `close` on one accessor is a data race; concurrent runs
 * are isolated by each owning a separate accessor, which is what makes two runs
 * unable to see or drop each other's regions.
 *
 * `add` is the only producer of mappings and the only caller of
 * `register_device_memory_to_host`; `close` unregisters exactly the mappings
 * this accessor installed and nothing else. Both are reached on every return
 * path — `close` is idempotent and the destructor calls it — so a mapping
 * cannot outlive the run that made it.
 *
 * A null `api` makes every `add` fail, so a registered region always implies a
 * usable `api`; `write`'s mirror push-back relies on that and does not re-check.
 *
 * The state lives behind `Impl` so this header pulls in no standard containers;
 * they stay in `host/host_tensor_access.cpp`.
 */
class HostTensorAccessor {
public:
    explicit HostTensorAccessor(const HostApi *api, HostTensorAccessMode mode = HostTensorAccessMode::Program);
    ~HostTensorAccessor();

    HostTensorAccessor(const HostTensorAccessor &) = delete;
    HostTensorAccessor &operator=(const HostTensorAccessor &) = delete;

    /**
     * Register `[dev_base, dev_base + size)`, using `fallback_host_view` (the
     * caller's host tensor buffer) when available and asking the platform for a
     * host mapping otherwise. The current runtime-maker always supplies the
     * fallback for host-memory tensors and skips pure outputs, so its bind path does
     * not install mappings.
     *
     * @return false for an empty region, a null `api`, or when neither a
     *         mapping nor a fallback view is available.
     */
    bool add(uint64_t dev_base, uint64_t size, void *fallback_host_view);

    /**
     * Register `[dev_base, dev_base + size)` as a child-memory region, with no
     * means of access yet.
     *
     * A plain push: the platform is not consulted and nothing is mapped. The
     * first read or write landing inside the region resolves it, so an
     * orchestration that never touches this tensor pays nothing for it.
     *
     * @return false for an empty region or a null `api`.
     */
    bool add_child_memory(uint64_t dev_base, uint64_t size);

    /**
     * Register an explicit host-only duplicate for HBG kernel Host build.
     * `logical_base` is the address carried by that HOST ChipTensor and
     * `host_view` is the same caller-owned storage. No platform mapping or
     * device copy is attempted, and writes through the accessor remain denied.
     */
    bool add_host_copy(uint64_t logical_base, uint64_t size, const void *host_view);

    bool read(uint64_t dev_addr, void *dst, uint64_t bytes);
    bool write(uint64_t dev_addr, const void *src, uint64_t bytes);

    /** Drop every region and unregister every mapping this accessor installed. */
    void close() noexcept;

    /** Mappings installed by `add` and not yet dropped by `close`. */
    size_t mapping_count() const noexcept;

    /** Total bytes covered by those mappings; excludes caller-buffer views. */
    uint64_t mapped_bytes() const noexcept;

    /**
     * Accesses served by a device copy because no host mapping was available.
     *
     * One PCIe round trip each, so this is the number to look at when a host
     * that cannot map (a5, or a 64 KiB-page host per issue #1531) orchestrates
     * more slowly than one that can.
     */
    uint64_t device_copy_count() const noexcept;

private:
    struct Impl;
    Impl *impl_;
};

/**
 * Read `bytes` at the accessor's logical address `dev_addr` into `dst`.
 *
 * @return false when no registered region covers the whole span; `dst` is
 *         untouched.
 */
bool host_tensor_read(HostTensorAccessor *accessor, uint64_t dev_addr, void *dst, uint64_t bytes);

/**
 * Write `bytes` from `src` to device address `dev_addr`, leaving the bytes
 * visible to the device.
 *
 * @return false when no registered region covers the whole span, or when the
 *         push-back to the device fails.
 */
bool host_tensor_write(HostTensorAccessor *accessor, uint64_t dev_addr, const void *src, uint64_t bytes);
