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
 * @brief simpler::hbg::Tensor-byte access for the host orchestrator, over device buffers.
 *
 * `simpler::hbg::Tensor::buffer.addr` is a device address. host_build_graph runs the
 * orchestrator on the host, so `get_tensor_data` / `set_tensor_data` cannot
 * assume the CPU executing them can load that address — whether it can is a
 * platform capability, not a property of the runtime. This is the seam where
 * that capability is resolved, so the orchestrator core never dereferences a
 * device address itself.
 *
 * The bind path registers staged input tensors and explicit resident host views,
 * backed by the caller's host tensor buffer:
 *
 *   - A read observes that caller buffer.
 *   - A write mutates that caller buffer, then uses the device-copy hook so the
 *     device observes it too. This is visible even for an `IN` argument if its
 *     host orchestration calls `set_tensor_data`.
 *
 * `add` also retains a null-fallback platform path: it asks the platform for a
 * host-readable mapping whose address may equal or differ from `dev_base`, and
 * always accesses the returned address. The current runtime-maker path cannot
 * reach it: staged and resident views supply a host buffer, while pure outputs
 * are deliberately left unregistered. The path remains as an explicit
 * platform-capability escape hatch in `add` and is covered directly by unit
 * tests; no current production caller reaches it.
 *
 * An address no registered region covers is a failure, never a raw
 * dereference. Pure outputs, GM-heap tensors and child buffers without an
 * explicit host view remain inaccessible.
 *
 * Regions belong to one host orchestration run, before tasks are dispatched.
 * Staged views contain the bytes just uploaded. Resident IN views remain
 * coherent across rounds under the caller's no-device-writes contract; host
 * writes update both copies. Resident INOUT views are refreshed from the
 * device at every bind. These windows do not expose task-produced values or
 * provide coherence while device tasks execute.
 *
 * `host/host_tensor_access.cpp` holds the only definitions of the read/write
 * pair, and libhost_runtime.so links them. Nothing in the AICPU build reaches
 * either: every caller is host-side.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "task_interface/arg_direction.h"
#include "task_interface/tensor.h"

struct HostApi;  // common/host_api.h — fwd-declared so this header stays out of platform includes

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
    explicit HostTensorAccessor(const HostApi *api);
    ~HostTensorAccessor();

    HostTensorAccessor(const HostTensorAccessor &) = delete;
    HostTensorAccessor &operator=(const HostTensorAccessor &) = delete;

    /**
     * Register `[dev_base, dev_base + size)`, using `fallback_host_view` (the
     * caller's host tensor buffer) when available and asking the platform for a
     * host mapping otherwise. The current runtime-maker always supplies the
     * fallback for staged and explicit resident views, and skips pure outputs; it
     * not install mappings.
     *
     * @return false for an empty region, a null `api`, or when neither a
     *         mapping nor a fallback view is available.
     */
    bool add(uint64_t dev_base, uint64_t size, void *fallback_host_view);
    // An IN view stays coherent because kernels do not mutate it and host writes
    // push back. An INOUT view is refreshed from the device before orchestration.
    bool add_resident(const ChipTensor &tensor, ArgDirection direction, uint64_t host_addr, uint64_t host_bytes);
    bool read(uint64_t dev_addr, void *dst, uint64_t bytes) const;
    bool write(uint64_t dev_addr, const void *src, uint64_t bytes) const;

    /** Drop every region and unregister every mapping this accessor installed. */
    void close() noexcept;

    /** Mappings installed by `add` and not yet dropped by `close`. */
    size_t mapping_count() const noexcept;

    /** Total bytes covered by those mappings; excludes fallback staging views. */
    uint64_t mapped_bytes() const noexcept;

private:
    struct Impl;
    Impl *impl_;
};

/**
 * Read `bytes` at device address `dev_addr` into `dst`.
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

// Byte span from buffer.addr through the last reachable element, including
// start_offset and stride gaps. Reject malformed or out-of-bounds geometry.
bool host_tensor_span(const ChipTensor &tensor, size_t *bytes);
