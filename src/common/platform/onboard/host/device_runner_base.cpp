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
 * `DeviceRunnerBase` — onboard host lifecycle shared by a2a3 and a5.
 *
 * Constructor wires the three arenas to call back into `mem_alloc_` via
 * the static trampolines declared in the header. Per-region commit is
 * still driven by the subclass's `setup_static_arena`.
 *
 * Shared lifecycle methods own runner-level resources; architecture-specific
 * launch, completion, and reset behavior remains in each DeviceRunner.
 */

#include "device_runner_base.h"

#include <runtime/rt.h>
#include <acl/acl.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

#include "callable.h"
#include "callable_protocol.h"
#include "call_config.h"
#include "chip_callable_layout.h"
#include "common/core_type.h"
#include "common/host_api.h"
#include "common/platform_config.h"
#include "common/sdma_warmup_layout.h"
#include "common/unified_log.h"
#include "host/acl_error_log.h"
#include "host/arena_replacement_transaction.h"
#include "kernel_platform_ops.h"
#include "host/host_phase_records_artifact.h"
#include "host/raii_scope_guard.h"
#include "host/run_boundary.h"
#include "host_log.h"
#include "platform_comm/comm.h"
#include "runtime_c_api.h"
#include "task_args_wire.h"
#include "utils/elf_build_id.h"
// `runtime.h` (pulled in via `device_runner_helpers.h` in the base header)
// supplies the per-arch `Handshake` + `Runtime` types used by
// `print_handshake_results` / `bind_callable_to_runtime` /
// `prepare_orch_so`.

// Implemented by each runtime's host part (runtime_maker.cpp). Reports the
// AICPU entry symbols this runtime exports beyond the base {exec, init} set, so
// the common AICPU loader carries no runtime-specific symbol knowledge. TMARB
// returns simpler_aicpu_register_callable; host_build_graph returns none.
extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count);

namespace {

HostRuntimeTimeoutConfig resolve_onboard_timeout_config() {
    RuntimeTimeoutConfig order_defaults{
        PLATFORM_OP_EXECUTE_TIMEOUT_US, PLATFORM_STREAM_SYNC_TIMEOUT_MS, PLATFORM_SCHEDULER_TIMEOUT_MS
    };
    RuntimeTimeoutParseStatus parse_status;
    RuntimeTimeoutConfig cfg = resolve_runtime_timeout_config(order_defaults, &parse_status);

    if (parse_status.op_execute_env_set && !parse_status.op_execute_valid) {
        const char *op_env = std::getenv(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV);
        LOG_WARN(
            "%s=%s invalid, using default %llu", SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, op_env,
            (unsigned long long)order_defaults.op_execute_timeout_us
        );
    }

    if (parse_status.stream_sync_env_set && !parse_status.stream_sync_valid) {
        const char *sync_env = std::getenv(SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV);
        LOG_WARN(
            "%s=%s invalid, using default %d", SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, sync_env,
            order_defaults.stream_sync_timeout_ms
        );
    }

    if (parse_status.scheduler_env_set && !parse_status.scheduler_valid) {
        const char *sched_env = std::getenv(SIMPLER_SCHEDULER_TIMEOUT_MS_ENV);
        LOG_WARN(
            "%s=%s invalid, using default %d", SIMPLER_SCHEDULER_TIMEOUT_MS_ENV, sched_env,
            order_defaults.scheduler_timeout_ms
        );
    }

    bool host_timeout_env_set =
        parse_status.op_execute_env_set || parse_status.stream_sync_env_set || parse_status.scheduler_env_set;
    RuntimeTimeoutOrderStatus order_status = validate_runtime_timeout_order(cfg);
    // The scheduler override is forwarded to the device (via InitArgs at init)
    // only when explicitly set, valid, and consistent with the op/stream
    // ordering. 0 means "no override" — the AICPU scheduler then keeps its
    // compile-time default. op/stream remain host-side acl knobs.
    int32_t scheduler_override = (parse_status.scheduler_env_set && parse_status.scheduler_valid &&
                                  order_status == RuntimeTimeoutOrderStatus::OK) ?
                                     cfg.scheduler_timeout_ms :
                                     0;
    if (host_timeout_env_set && order_status != RuntimeTimeoutOrderStatus::OK) {
        LOG_WARN(
            "Ignoring timeout env overrides: %s (scheduler=%d ms, op_execute=%llu us, stream_sync=%d ms)",
            runtime_timeout_order_status_name(order_status), cfg.scheduler_timeout_ms,
            (unsigned long long)cfg.op_execute_timeout_us, cfg.stream_sync_timeout_ms
        );
        return HostRuntimeTimeoutConfig{
            order_defaults.op_execute_timeout_us, order_defaults.stream_sync_timeout_ms, scheduler_override
        };
    }
    return HostRuntimeTimeoutConfig{cfg.op_execute_timeout_us, cfg.stream_sync_timeout_ms, scheduler_override};
}

/**
 * The ACL event operations a `RunCompletionFence` is built on.
 *
 * `aclrtCreateEventExWithFlag` rather than the plain form because its events
 * re-record without an `aclrtResetEvent`, which is what lets one slot's pair
 * serve every run it hosts. `ACL_EVENT_SYNC` is the completion-only flag: these
 * boundaries are never read for a timestamp, so they carry none of the timeline
 * cost a readable event would.
 *
 * `aclrtQueryEventStatus` reports not-ready as a status rather than an error
 * code, so a pending boundary is never confused with a failed query.
 */
RunCompletionFence::DeviceEventOps make_acl_event_ops() {
    RunCompletionFence::DeviceEventOps ops;
    ops.create = [](void **out_event) -> int {
        aclrtEvent event = nullptr;
        aclError rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
        if (rc != ACL_SUCCESS) {
            LOG_ERROR("aclrtCreateEventExWithFlag (run completion boundary) failed: %d", static_cast<int>(rc));
            ACL_LOG_ERROR_DETAIL(rc);
            return static_cast<int>(rc);
        }
        *out_event = event;
        return 0;
    };
    ops.record = [](void *event, void *stream) -> int {
        aclError rc = aclrtRecordEvent(static_cast<aclrtEvent>(event), static_cast<aclrtStream>(stream));
        if (rc != ACL_SUCCESS) {
            LOG_ERROR("aclrtRecordEvent (run completion boundary) failed: %d", static_cast<int>(rc));
            ACL_LOG_ERROR_DETAIL(rc);
            return static_cast<int>(rc);
        }
        return 0;
    };
    ops.query = [](void *event, bool *complete) -> int {
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        aclError rc = aclrtQueryEventStatus(static_cast<aclrtEvent>(event), &status);
        if (rc != ACL_SUCCESS) {
            LOG_ERROR("aclrtQueryEventStatus (run completion boundary) failed: %d", static_cast<int>(rc));
            ACL_LOG_ERROR_DETAIL(rc);
            return static_cast<int>(rc);
        }
        *complete = status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
        return 0;
    };
    ops.wait = [](void *event, int timeout_ms) -> int {
        aclError rc = aclrtSynchronizeEventWithTimeout(static_cast<aclrtEvent>(event), timeout_ms);
        return rc == ACL_SUCCESS ? 0 : static_cast<int>(rc);
    };
    ops.destroy = [](void *event) -> int {
        aclError rc = aclrtDestroyEvent(static_cast<aclrtEvent>(event));
        if (rc != ACL_SUCCESS) {
            LOG_ERROR("aclrtDestroyEvent (run completion boundary) failed: %d", static_cast<int>(rc));
            ACL_LOG_ERROR_DETAIL(rc);
            return static_cast<int>(rc);
        }
        return 0;
    };
    return ops;
}

/**
 * The ACL event operations the passive boundary markers are built on.
 *
 * `ACL_EVENT_TIME_LINE` (0x8), not the completion-only `ACL_EVENT_SYNC` (0x1) the fences use:
 * only the timeline flag carries a timestamp, which `aclrtEventGetTimestamp` documents as "get
 * syscnt when event recorded" — the device's own system counter at the instant the stream reached
 * the record, not a host clock. These markers re-record one event per run with no intervening
 * `aclrtResetEvent`: `aclrtRecordEvent` overwrites the prior timestamp in place.
 *
 * Retrieval is `aclrtSynchronizeEventWithTimeout` on the marker itself, then
 * `aclrtEventGetTimestamp`: the device having passed the record does not by itself make the
 * timestamp retrievable. The header documents the first as blocking the host, and the timeout form
 * rather than the plain one bounds that block by the same configured budget the completion fence
 * waits under. Neither call queues anything on a stream.
 *
 * Separate ops rather than the fences', so borrowing a timestamp cannot become a reason to change
 * what the completion events promise. Which reading belongs to which run is `RunBoundaryMarks`'
 * business; see that header.
 */
RunBoundaryMarks::DeviceEventOps make_acl_timing_event_ops() {
    RunBoundaryMarks::DeviceEventOps ops;
    ops.create = [](void **out_event) -> int {
        aclrtEvent event = nullptr;
        aclError rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_TIME_LINE);
        if (rc != ACL_SUCCESS) {
            LOG_WARN(
                "aclrtCreateEventExWithFlag (run boundary marker) failed: %d; device boundary times are "
                "unavailable for this runner",
                static_cast<int>(rc)
            );
            return static_cast<int>(rc);
        }
        *out_event = event;
        return 0;
    };
    ops.record = [](void *event, void *stream) -> int {
        aclError rc = aclrtRecordEvent(static_cast<aclrtEvent>(event), static_cast<aclrtStream>(stream));
        return rc == ACL_SUCCESS ? 0 : static_cast<int>(rc);
    };
    ops.synchronize = [](void *event, int timeout_ms) -> int {
        aclError rc = aclrtSynchronizeEventWithTimeout(static_cast<aclrtEvent>(event), timeout_ms);
        return rc == ACL_SUCCESS ? 0 : static_cast<int>(rc);
    };
    ops.read_timestamp = [](void *event, uint64_t *out_timestamp) -> int {
        aclError rc = aclrtEventGetTimestamp(static_cast<aclrtEvent>(event), out_timestamp);
        return rc == ACL_SUCCESS ? 0 : static_cast<int>(rc);
    };
    ops.destroy = [](void *event) -> int {
        aclError rc = aclrtDestroyEvent(static_cast<aclrtEvent>(event));
        if (rc != ACL_SUCCESS) {
            LOG_ERROR("aclrtDestroyEvent (run boundary marker) failed: %d", static_cast<int>(rc));
            ACL_LOG_ERROR_DETAIL(rc);
            return static_cast<int>(rc);
        }
        return 0;
    };
    return ops;
}

}  // namespace

DeviceRunnerBase::DeviceRunnerBase() {
    for (auto &bank : arena_banks_) {
        bank = std::make_unique<ArenaBank>(&arena_alloc_trampoline, &arena_free_trampoline, this);
    }
    for (auto &fence : run_fences_) {
        fence = std::make_unique<RunCompletionFence>(make_acl_event_ops());
    }
    queued_waits_ = std::make_unique<QueuedStreamWaits>(make_acl_event_ops());
    boundary_marks_ = std::make_unique<RunBoundaryMarks>(make_acl_timing_event_ops());
}

uint64_t DeviceRunnerBase::arena_bank_gm_heap_base(uint32_t bank_id) const {
    if (bank_id >= arena_banks_.size()) return 0;
    const ArenaBank &bank = *arena_banks_[bank_id];
    return bank.gm_heap.is_committed() ? reinterpret_cast<uint64_t>(bank.gm_heap.base()) : 0;
}

uint64_t DeviceRunnerBase::retained_temp_addr(uint32_t slot_id) const {
    if (slot_id >= retained_temp_addrs_.size()) return 0;
    return reinterpret_cast<uint64_t>(retained_temp_addrs_[slot_id]);
}

void *DeviceRunnerBase::allocate_tensor(std::size_t bytes) { return mem_alloc_.alloc(bytes); }

void DeviceRunnerBase::free_tensor(void *dev_ptr) {
    if (dev_ptr != nullptr) {
        // Before the pages go: a mapping outliving them would hand a live host
        // VA to whatever the driver puts there next.
        if (child_memory_host_views_.take(dev_ptr)) {
            unregister_device_memory_from_host(dev_ptr);
        }
        mem_alloc_.free(dev_ptr);
    }
}

void *DeviceRunnerBase::acquire_child_memory_host_view(void *dev_ptr, std::size_t bytes) {
    if (dev_ptr == nullptr || bytes == 0) return nullptr;

    void *alloc_base = nullptr;
    std::size_t alloc_size = 0;
    if (!mem_alloc_.owning_allocation(dev_ptr, &alloc_base, &alloc_size)) {
        LOG_ERROR("acquire_child_memory_host_view: %p is not inside a tracked device allocation", dev_ptr);
        return nullptr;
    }
    const auto *end = static_cast<const unsigned char *>(dev_ptr) + bytes;
    if (end > static_cast<const unsigned char *>(alloc_base) + alloc_size) {
        LOG_ERROR(
            "acquire_child_memory_host_view: [%p, +%zu) overruns its allocation [%p, +%zu)", dev_ptr, bytes, alloc_base,
            alloc_size
        );
        return nullptr;
    }

    if (void *cached = child_memory_host_views_.lookup(alloc_base, dev_ptr); cached != nullptr) {
        return cached;
    }

    void *host_view = register_device_memory_to_host(alloc_base, alloc_size);
    if (host_view == nullptr) {
        return nullptr;
    }
    child_memory_host_views_.insert(alloc_base, alloc_size, host_view);
    LOG_INFO(
        "host-orch: mapped child-memory allocation %p (%zu bytes); %zu mapping(s), %llu bytes held", alloc_base,
        alloc_size, child_memory_host_views_.count(),
        static_cast<unsigned long long>(child_memory_host_views_.mapped_bytes())
    );
    return child_memory_host_views_.lookup(alloc_base, dev_ptr);
}

void DeviceRunnerBase::release_child_memory_host_views() {
    for (void *alloc_base : child_memory_host_views_.take_all()) {
        // A block whose last consumer could not be proven finished keeps its
        // mapping: unregistering returns the host range that covers the whole
        // allocation, and the bytes behind it may still be written. The record
        // is dropped either way, so nothing reaches this allocation again.
        if (workspace_.must_keep(alloc_base)) {
            std::size_t mapped = workspace_.block_bytes(alloc_base);
            workspace_.note_mapping_retained(alloc_base, mapped);
            continue;
        }
        if (unregister_device_memory_from_host(alloc_base) == 0) continue;
        // Still mapped, and this registry no longer names it. A managed block
        // is quarantined so that nothing — the sweep below, a later growth, or
        // a second close — releases storage a live host address still covers.
        // An unmanaged allocation keeps the behaviour it always had: the
        // allocator's finalize frees it regardless.
        if (workspace_.note_mapping_unregister_failed(alloc_base)) {
            LOG_ERROR(
                "release_child_memory_host_views: %p could not be unmapped; its workspace block is quarantined",
                alloc_base
            );
        }
    }
}

int DeviceRunnerBase::drop_child_memory_host_view(void *alloc_base) {
    if (alloc_base == nullptr) return 0;
    if (child_memory_host_views_.lookup(alloc_base, alloc_base) == nullptr) return 0;
    const int rc = unregister_device_memory_from_host(alloc_base);
    if (rc == 0) child_memory_host_views_.take(alloc_base);
    return rc;
}

namespace {
// The run identity this thread's workspace requests belong to. Thread-scoped so
// a prepared successor built on another thread cannot be charged to this one.
struct WorkspacePlanIdentity {
    std::uint64_t epoch{0};
    WorkspaceManager::RegionKey region{};
};
thread_local WorkspacePlanIdentity g_workspace_plan;
}  // namespace

void DeviceRunnerBase::begin_workspace_plan(uint32_t pipeline_slot, std::uint64_t run_epoch) noexcept {
    g_workspace_plan.epoch = run_epoch;
    g_workspace_plan.region = WorkspaceManager::staging_region(pipeline_slot);
}

void DeviceRunnerBase::end_workspace_plan() noexcept { g_workspace_plan = WorkspacePlanIdentity{}; }

void DeviceRunnerBase::set_workspace_plan_region(const WorkspaceManager::RegionKey &region) noexcept {
    g_workspace_plan.region = region;
}

WorkspaceManager::RegionKey DeviceRunnerBase::workspace_plan_region() noexcept { return g_workspace_plan.region; }

int DeviceRunnerBase::reference_bank_arenas(
    uint32_t arena_bank, const ArenaRegionRequest *requests, std::size_t count
) {
    for (std::size_t i = 0; i < count; ++i) {
        const DeviceArena *arena = requests[i].arena;
        // A region asked to hold nothing has no backing and no consumer. Its
        // previous block was reported detached by the transaction, so nothing
        // here has to speak for it.
        if (arena == nullptr || !arena->is_committed()) continue;
        const WorkspaceManager::RegionKey region =
            WorkspaceManager::arena_region(arena_bank, static_cast<WorkspaceManager::ArenaRegion>(i));
        // The ledger is keyed by what its own allocation callback handed over,
        // which is the raw block — `base()` is the forward-aligned address
        // inside it and is not the same value when the platform returns an
        // under-aligned pointer.
        void *const owned = arena->raw_backing();
        if (workspace_.reference(owned, g_workspace_plan.epoch)) {
            // The transaction published every region before this ran, so this
            // address is what the region is using now: any earlier generation
            // of it becomes obsolete and its bytes become reclaimable once its
            // own consumers retire.
            workspace_.note_published(region, owned);
            continue;
        }
        LOG_ERROR(
            "setup_static_arena: bank %u region %s could not register this run as a consumer of %p", arena_bank,
            requests[i].name, owned
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return 0;
}

void *DeviceRunnerBase::acquire_arena_backing(std::size_t size) {
    if (!workspace_.enabled()) return mem_alloc_.alloc(size);
    // Which region is asking is not in the callback's arguments, so the setup
    // that drives the arenas names it around each commit. Without that every
    // region of every bank would share one pool, and a growing GM heap could be
    // handed the block a still-attached shared-memory region is published at.
    return workspace_.acquire(g_workspace_plan.region, g_workspace_plan.epoch, size);
}

void DeviceRunnerBase::note_arena_region_disposition(uint32_t arena_bank, ArenaRegionDisposition what, void *base) {
    if (!workspace_.enabled() || base == nullptr) return;
    if (!workspace_.note_unpublished(base)) return;
    const char *reason =
        what == ArenaRegionDisposition::StageAborted ? "its staging was aborted" : "its region now holds nothing";
    LOG_INFO(
        "setup_static_arena: bank %u gave up workspace block %p (%s); its bytes are reclaimable once its last "
        "consumer retires",
        arena_bank, base, reason
    );
}

void DeviceRunnerBase::release_arena_backing(void *p) {
    // A managed block is owned by the ledger, not by the arena that was using
    // it: the arena dropping its base is not permission to free, and the bytes
    // stay charged until the last run referencing them retires. An unmanaged
    // context frees exactly as it always did.
    if (workspace_.owns(p)) return;
    mem_alloc_.free(p);
}

int DeviceRunnerBase::set_workspace_budget(std::uint64_t limit_bytes) {
    WorkspaceManager::Backend backend{};
    backend.ctx = this;
    backend.acquire = [](void *ctx, std::size_t bytes) -> void * {
        // Both ownership records exist before the device call: the ledger
        // reserved its node, and this reservation holds the allocator's lock
        // and its tracking node until the commit that follows.
        auto *self = static_cast<DeviceRunnerBase *>(ctx);
        MemoryAllocator::Reservation res = self->mem_alloc_.begin_reservation();
        return res.commit_alloc(bytes);
    };
    backend.release = [](void *ctx, void *base) -> int {
        auto *self = static_cast<DeviceRunnerBase *>(ctx);
        // Unmap first, and only free what is proven unmapped. A host mapping
        // covers the whole allocation, so freeing bytes still behind one would
        // give the next allocation a range this process holds a live host
        // address over. A failed unmap is reported as a failed release, which
        // keeps the block owned and charged here instead of leaving a mapping
        // with nothing naming it.
        const int unmap_rc = self->drop_child_memory_host_view(base);
        if (unmap_rc != 0) return unmap_rc;
        return self->mem_alloc_.free(base);
    };
    if (!workspace_.configure(limit_bytes, backend)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    return 0;
}

bool DeviceRunnerBase::workspace_report(SimplerWorkspaceReport *out) const { return workspace_.report(out); }

int DeviceRunnerBase::acquire_retained_temp(
    uint32_t pipeline_slot, std::size_t bytes, void **addr_out, std::size_t *size_out
) {
    if (addr_out == nullptr || size_out == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (pipeline_slot >= retained_temp_addrs_.size()) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    *addr_out = retained_temp_addrs_[pipeline_slot];
    *size_out = retained_temp_sizes_[pipeline_slot];
    if (bytes == 0 || bytes <= retained_temp_sizes_[pipeline_slot]) {
        // A request the retained block already covers allocates nothing, but
        // this run is about to write and read it, so it registers as a consumer
        // anyway — otherwise a later growth would find no reference and treat
        // those bytes as free to overwrite.
        if (bytes != 0 && workspace_.enabled() && *addr_out != nullptr &&
            !workspace_.reference(*addr_out, g_workspace_plan.epoch)) {
            LOG_ERROR(
                "acquire_retained_temp: slot %u could not register this run as a consumer of %p", pipeline_slot,
                *addr_out
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        return 0;
    }

    if (!workspace_.enabled()) {
        // Unmanaged: the sequence RetainedTempBump used to run itself — release
        // the old block, take a bigger one, and stop naming the old one either
        // way, so a later run cannot free it twice.
        void *previous = retained_temp_addrs_[pipeline_slot];
        if (previous != nullptr) mem_alloc_.free(previous);
        void *grown = mem_alloc_.alloc(bytes);
        set_retained_temp_buffer(pipeline_slot, grown, grown == nullptr ? 0 : bytes);
        if (grown == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        *addr_out = grown;
        *size_out = bytes;
        return 0;
    }

    // Managed: the previous generation keeps its address and its contents until
    // its last consumer retires, so this request takes a block of its own
    // inside the budget. A refusal leaves the slot naming the old block.
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(pipeline_slot);
    void *grown = workspace_.acquire(region, g_workspace_plan.epoch, bytes);
    if (grown == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    // The previous block keeps its address and its contents: the slot stops
    // naming it, and its earlier consumers release it when they retire.
    set_retained_temp_buffer(pipeline_slot, grown, bytes);
    // The slot names the new block from here on, which makes the one it named
    // before an obsolete generation.
    workspace_.note_published(region, grown);
    *addr_out = grown;
    *size_out = bytes;
    return 0;
}

int DeviceRunnerBase::copy_to_device(void *dev_ptr, const void *host_ptr, std::size_t bytes) {
    return rtMemcpy(dev_ptr, bytes, host_ptr, bytes, RT_MEMCPY_HOST_TO_DEVICE);
}

int DeviceRunnerBase::copy_from_device(void *host_ptr, const void *dev_ptr, std::size_t bytes) {
    return rtMemcpy(host_ptr, bytes, dev_ptr, bytes, RT_MEMCPY_DEVICE_TO_HOST);
}

int DeviceRunnerBase::device_memset(void *dev_ptr, int value, std::size_t bytes) {
    return aclrtMemset(dev_ptr, bytes, value, bytes);
}

void DeviceRunnerBase::get_retained_temp_buffer(uint32_t pipeline_slot, void **addr, size_t *size) {
    if (pipeline_slot >= retained_temp_addrs_.size()) {
        if (addr != nullptr) *addr = nullptr;
        if (size != nullptr) *size = 0;
        return;
    }
    if (addr != nullptr) *addr = retained_temp_addrs_[pipeline_slot];
    if (size != nullptr) *size = retained_temp_sizes_[pipeline_slot];
}

void DeviceRunnerBase::set_retained_temp_buffer(uint32_t pipeline_slot, void *addr, size_t size) {
    if (pipeline_slot >= retained_temp_addrs_.size()) return;
    retained_temp_addrs_[pipeline_slot] = addr;
    retained_temp_sizes_[pipeline_slot] = size;
}

int DeviceRunnerBase::acquire_graph_definition_block(
    uint32_t pipeline_slot, size_t bytes, size_t alignment, void **device_out, void **staging_out
) {
    if (device_out == nullptr || staging_out == nullptr) return -1;
    *device_out = nullptr;
    *staging_out = nullptr;
    if (pipeline_slot >= graph_definition_blocks_.size() || bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0 || bytes > SIZE_MAX - (alignment - 1)) {
        return -1;
    }
    RetainedGraphBlock &block = graph_definition_blocks_[pipeline_slot];
    if (block.aligned_addr == nullptr || block.capacity < bytes ||
        reinterpret_cast<uintptr_t>(block.aligned_addr) % alignment != 0) {
        const size_t allocation_bytes = bytes + alignment - 1;
        void *allocation = mem_alloc_.alloc(allocation_bytes);
        if (allocation == nullptr) return -1;
        const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation);
        if (raw > UINTPTR_MAX - (alignment - 1)) {
            mem_alloc_.free(allocation);
            return -1;
        }
        void *aligned_addr = reinterpret_cast<void *>((raw + alignment - 1) & ~(alignment - 1));
        if (device_memset(aligned_addr, 0, bytes) != 0) {
            mem_alloc_.free(allocation);
            return -1;
        }
        if (block.allocation != nullptr && mem_alloc_.free(block.allocation) != 0) {
            mem_alloc_.free(allocation);
            return -1;
        }
        block.allocation = allocation;
        block.aligned_addr = aligned_addr;
        block.capacity = bytes;
    }
    // Grow-only and never shrunk, so a steady-state bind assembles its objects
    // in host memory it neither acquires nor returns.
    if (block.staging.size() < bytes) block.staging.resize(bytes);
    *device_out = block.aligned_addr;
    *staging_out = block.staging.data();
    return 0;
}

void DeviceRunnerBase::get_graph_definition_staging(uint32_t pipeline_slot, void **addr, size_t *size) {
    if (addr != nullptr) *addr = nullptr;
    if (size != nullptr) *size = 0;
    if (pipeline_slot >= graph_definition_blocks_.size()) return;
    RetainedGraphBlock &block = graph_definition_blocks_[pipeline_slot];
    if (block.staging.empty()) return;
    if (addr != nullptr) *addr = block.staging.data();
    if (size != nullptr) *size = block.staging.size();
}

int DeviceRunnerBase::acquire_sm_mirror(uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out) {
    if (addr_out == nullptr) return -1;
    *addr_out = nullptr;
    if (pipeline_slot >= sm_mirrors_.size() || bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        bytes > SIZE_MAX - (alignment - 1)) {
        return -1;
    }
    RetainedSmMirror &mirror = sm_mirrors_[pipeline_slot];
    // Grow-only and never shrunk: the task capacity is fixed for a given run
    // configuration, so past the first bind the image is written into host pages
    // that are already mapped, and each page of it faults once per process rather
    // than once per bind.
    const size_t needed = bytes + alignment - 1;
    if (mirror.capacity < needed) {
        // `new[]` on a trivially-typed array default-initializes, so the block
        // costs no page until a bind writes one; make_unique would zero it. The
        // outgoing block's bytes are not carried over, because nothing reads a byte
        // this bind did not write.
        std::unique_ptr<std::byte[]> storage(new (std::nothrow) std::byte[needed]);
        if (storage == nullptr) return -1;
        mirror.storage = std::move(storage);
        mirror.capacity = needed;
    }
    const uintptr_t raw = reinterpret_cast<uintptr_t>(mirror.storage.get());
    *addr_out = reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    return 0;
}

int DeviceRunnerBase::acquire_run_image_staging(
    uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out
) {
    if (addr_out == nullptr) return -1;
    *addr_out = nullptr;
    if (pipeline_slot >= run_image_stagings_.size() || bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0 || bytes > SIZE_MAX - (alignment - 1)) {
        return -1;
    }
    RetainedSmMirror &staging = run_image_stagings_[pipeline_slot];
    // Grow-only, like the mirror: an image's size follows the graph a run builds,
    // so a repeated workload writes host pages that are already mapped.
    const size_t needed = bytes + alignment - 1;
    if (staging.capacity < needed) {
        // `new[]` default-initializes a trivially-typed array, so the block costs
        // no page until the bind writes one. The outgoing block's bytes are not
        // carried over: a publication ships what its own bind assembled.
        std::unique_ptr<std::byte[]> storage(new (std::nothrow) std::byte[needed]);
        if (storage == nullptr) return -1;
        staging.storage = std::move(storage);
        staging.capacity = needed;
    }
    const uintptr_t raw = reinterpret_cast<uintptr_t>(staging.storage.get());
    *addr_out = reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    return 0;
}

void DeviceRunnerBase::release_run_image_stagings() {
    for (RetainedSmMirror &staging : run_image_stagings_) {
        staging.storage.reset();
        staging.capacity = 0;
    }
}

void DeviceRunnerBase::release_sm_mirrors() {
    for (RetainedSmMirror &mirror : sm_mirrors_) {
        mirror.storage.reset();
        mirror.capacity = 0;
    }
}

void DeviceRunnerBase::release_graph_definition_blocks() {
    for (RetainedGraphBlock &block : graph_definition_blocks_) {
        if (block.allocation != nullptr) mem_alloc_.free(block.allocation);
        block = RetainedGraphBlock{};
    }
}

void DeviceRunnerBase::abandon_graph_definition_blocks() {
    for (RetainedGraphBlock &block : graph_definition_blocks_) {
        block = RetainedGraphBlock{};
    }
}

void DeviceRunnerBase::clear_temporary_buffer() {
    for (size_t slot = 0; slot < retained_temp_addrs_.size(); ++slot) {
        if (retained_temp_addrs_[slot] == nullptr) continue;
        // A managed block is released by the ledger that owns it, once no
        // consumer references it; the slot only stops naming it here.
        if (!workspace_.owns(retained_temp_addrs_[slot])) {
            mem_alloc_.free(retained_temp_addrs_[slot]);
        }
        retained_temp_addrs_[slot] = nullptr;
        retained_temp_sizes_[slot] = 0;
    }
}

void *DeviceRunnerBase::acquire_pooled_gm_heap(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).gm_heap;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

void *DeviceRunnerBase::acquire_pooled_gm_sm(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).gm_sm;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

void *DeviceRunnerBase::acquire_pooled_runtime_arena(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).runtime_pool;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

bool DeviceRunnerBase::lookup_prebuilt_runtime_arena_cache(
    uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
    void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
) const {
    // The cache holds one entry and its bases point into bank 0, so any other
    // bank must rebuild rather than be handed a region it does not own.
    if (arena_bank != 0) return false;
    return prebuilt_runtime_arena_cache_.lookup(
        hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off, image_data, image_size
    );
}

void DeviceRunnerBase::mark_prebuilt_runtime_arena_cached(
    uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
    void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
) {
    // Single-entry cache owned by bank 0; see lookup_prebuilt_runtime_arena_cache.
    if (arena_bank != 0) return;
    prebuilt_runtime_arena_cache_.store(
        hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off, image_data, image_size
    );
}

int DeviceRunnerBase::setup_static_arena(
    uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size
) {
    if (arena_bank >= arena_banks_.size()) {
        LOG_ERROR("arena bank %u is outside [0, %zu)", arena_bank, arena_banks_.size());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // Three independent device_malloc'd buffers: GM heap, shared memory,
    // prebuilt runtime arena. Split out from a single large allocation because
    // the combined size can exceed the device allocator's largest contiguous
    // block. Each arena commits exactly one region, so its base() is the
    // pooled pointer the caller wants.
    //
    // A request an existing region already covers costs nothing: the region is
    // kept, so a repeated workload allocates nothing here. A region that must
    // grow is replaced through the staged transaction below, which is what
    // makes an allocation failure leave every region's address and capacity as
    // this call found them — at the price of holding old and new backing
    // together until publication.
    ArenaBank &bank = this->arena_bank(arena_bank);

    // Captured graphs can retain committed base addresses, so kernel mode
    // forbids growing or releasing a region that is already committed. The
    // check covers all three regions and completes before the transaction
    // below, so a refusal allocates nothing and frees nothing.
    if (execution_mode_latch().is_kernel()) {
        const struct {
            const DeviceArena &arena;
            size_t cached_size;
            size_t requested_size;
            const char *name;
        } regions[] = {
            {bank.gm_heap, bank.cached_gm_heap_size, gm_heap_size, "gm_heap"},
            {bank.gm_sm, bank.cached_gm_sm_size, gm_sm_size, "gm_sm"},
            {bank.runtime_pool, bank.cached_runtime_arena_size, runtime_arena_size, "runtime_pool"},
        };
        for (const auto &region : regions) {
            if (!kernel_arena_change_is_forbidden(
                    region.arena.is_committed(), region.cached_size, region.requested_size
                )) {
                continue;
            }
            LOG_ERROR(
                "setup_static_arena: kernel mode forbids %s committed region %s (cached %zu, requested %zu)",
                region.requested_size == 0 ? "releasing" : "growing", region.name, region.cached_size,
                region.requested_size
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    }

    // The ledger keys a block by the one region that owns it, and the arena
    // allocation callback sees only a byte count, so each region names itself
    // just before its own backing is staged.
    ArenaRegionAnnounce announce{this, arena_bank};
    auto name_region = [](void *ctx, std::size_t region_index) {
        auto *a = static_cast<ArenaRegionAnnounce *>(ctx);
        a->runner->set_workspace_plan_region(
            WorkspaceManager::arena_region(a->bank, static_cast<WorkspaceManager::ArenaRegion>(region_index))
        );
    };
    // The two boundaries at which a region stops publishing a block without a
    // successor taking over. Reported to the ledger, which owns the block the
    // arena is only using.
    auto region_gave_up = [](void *ctx, std::size_t region_index, ArenaRegionDisposition what, void *base) {
        auto *a = static_cast<ArenaRegionAnnounce *>(ctx);
        (void)region_index;
        a->runner->note_arena_region_disposition(a->bank, what, base);
    };
    ArenaRegionRequest requests[] = {
        {&bank.gm_heap, &bank.cached_gm_heap_size, gm_heap_size, "gm_heap", name_region, &announce, region_gave_up,
         &announce},
        {&bank.gm_sm, &bank.cached_gm_sm_size, gm_sm_size, "gm_sm", name_region, &announce, region_gave_up, &announce},
        {&bank.runtime_pool, &bank.cached_runtime_arena_size, runtime_arena_size, "runtime_pool", name_region,
         &announce, region_gave_up, &announce},
    };
    constexpr size_t kRegionCount = sizeof(requests) / sizeof(requests[0]);
    // One region at a time, each named to the ledger, because the allocation
    // callbacks carry no region of their own and a capacity hit calls none at
    // all. The transaction itself is unchanged: it still stages every region
    // and publishes only when all of them succeeded.
    const BankArenaSetupOutcome outcome = run_bank_arena_setup(
        requests, kRegionCount, /*owns_prebuilt_cache=*/arena_bank == 0, &prebuilt_runtime_arena_cache_,
        DeviceArena::kDefaultBaseAlign
    );
    if (outcome.rc == 0 && workspace_.enabled()) {
        // Every region this bank now publishes is used by this plan, including
        // the ones whose existing capacity was enough and therefore allocated
        // nothing. A plan that read and wrote a region without registering as
        // its consumer would let a later growth treat those bytes as free.
        const int ref_rc = reference_bank_arenas(arena_bank, requests, kRegionCount);
        if (ref_rc != 0) return ref_rc;
    }
    if (outcome.rc != 0) {
        const int failed = outcome.transaction.failed_region;
        const char *name = (failed >= 0 && static_cast<size_t>(failed) < kRegionCount) ? requests[failed].name : "?";
        const size_t requested =
            (failed >= 0 && static_cast<size_t>(failed) < kRegionCount) ? requests[failed].requested_size : 0;
        LOG_ERROR(
            "setup_static_arena: staging %s (%zu bytes) failed on bank %u; this bank keeps its committed addresses "
            "and capacities",
            name, requested, arena_bank
        );
    }
    return outcome.rc;
}

std::thread DeviceRunnerBase::create_thread(std::function<void()> fn) {
    // A freshly spawned thread carries no CANN device context of its own, so
    // this bind creates one rather than taking anything from the caller — it
    // is the one rtSetDevice a borrowed-device context still owns, and it is
    // scoped to a thread this runner created.
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn)]() {
        rtSetDevice(dev_id);
        fn();
    });
}

int DeviceRunnerBase::bind_current_thread(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("Invalid device_id: %d", device_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (device_id_ != -1 && device_id_ != device_id) {
        LOG_ERROR(
            "DeviceRunner already initialized on device %d; reset/finalize before switching to device %d", device_id_,
            device_id
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // CANN device context is per-thread, so every caller must attach explicitly.
    int rc = rtSetDevice(device_id);
    if (rc != 0) {
        LOG_ERROR("rtSetDevice(%d) failed: %d", device_id, rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    return 0;
}

int DeviceRunnerBase::attach_current_thread(int device_id) {
    // rtSetDevice and the op-execute watchdog below are acts of device
    // ownership, so this entry belongs to a program context. A kernel context
    // reaches its device through adopt_borrowed_device instead; the one caller
    // here that runs under both identities is DeviceRunner::finalize(), which
    // skips this call on a kernel latch.
    if (execution_mode_latch().is_kernel()) {
        LOG_ERROR("attach_current_thread: refused — a kernel-mode context does not own the caller's device");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }

    int rc = bind_current_thread(device_id);
    if (rc != 0) return rc;

    // Both writers of device_id_ — this one and adopt_borrowed_device — guard
    // on the still-unset value, and both run before any prepare, execution or
    // collector thread attaches. Prepared-run admission and execution
    // subsequently attach different host threads, so repeated same-value
    // writes here would still be a C++ data race.
    if (device_id_ == -1) {
        timeout_config_ = resolve_onboard_timeout_config();
        // aclrtSetOpExecuteTimeOutV2 is device-global: it changes the
        // op-execute timeout of every operator any process runs on this
        // device, including the host framework's own. Only a context that
        // owns the device may set it.
        configure_aicore_op_timeout();
        device_id_ = device_id;
    }
    return 0;
}

int DeviceRunnerBase::adopt_borrowed_device(int device_id) {
    // The caller already holds this device current on its own threads, so the
    // only thing a kernel context takes from it is the identity: no
    // rtSetDevice, and no configure_aicore_op_timeout, which would rewrite the
    // op-execute watchdog for every other user of that card. Resolving the
    // timeout config is pure environment parsing and stays, because the stream
    // and scheduler timeouts derived from it are read on both identities.
    if (!execution_mode_latch().is_kernel()) {
        LOG_ERROR("adopt_borrowed_device: refused — the context has not latched kernel mode");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    if (device_id < 0) {
        LOG_ERROR("Invalid device_id: %d", device_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (device_id_ != -1 && device_id_ != device_id) {
        LOG_ERROR("DeviceRunner already on device %d; close before adopting device %d", device_id_, device_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (device_id_ == -1) {
        timeout_config_ = resolve_onboard_timeout_config();
        device_id_ = device_id;
    }
    return 0;
}

void DeviceRunnerBase::configure_aicore_op_timeout() {
    uint64_t actual_timeout = 0;
    int rc = aclrtSetOpExecuteTimeOutV2(timeout_config_.op_execute_timeout_us, &actual_timeout);
    if (rc != 0) {
        LOG_ERROR(
            "aclrtSetOpExecuteTimeOutV2(%llu us) failed: %d", (unsigned long long)timeout_config_.op_execute_timeout_us,
            rc
        );
    } else {
        LOG_INFO(
            "aclrtSetOpExecuteTimeOutV2: requested=%llu us, actual=%llu us",
            (unsigned long long)timeout_config_.op_execute_timeout_us, (unsigned long long)actual_timeout
        );
    }
}

int DeviceRunnerBase::ensure_device_initialized() {
    // Attach the current thread to the device (device_id_ was set in
    // attach_current_thread() during simpler_init) and create the persistent
    // AICPU/AICore streams. Streams live for the DeviceRunner's lifetime and
    // are destroyed in finalize().
    int rc = attach_current_thread(device_id_);
    if (rc != 0) {
        return rc;
    }

    // The point this runner is bound to a device is the point it wants to hear
    // about that device's faults. A failed install is not fatal: the callback
    // is a reporting channel today, and nothing decides a run from it.
    (void)acquire_device_fault_monitor();

    bool aicpu_created_here = false;
    bool aicore_created_here = false;
    if (stream_aicpu_ == nullptr) {
        rc = rtStreamCreate(&stream_aicpu_, 0);
        if (rc != 0) {
            LOG_ERROR("rtStreamCreate (AICPU) failed: %d", rc);
            ACL_LOG_ERROR_DETAIL(rc);
            return rc;
        }
        aicpu_created_here = true;
    }
    if (stream_aicore_ == nullptr) {
        rc = rtStreamCreate(&stream_aicore_, 0);
        if (rc != 0) {
            LOG_ERROR("rtStreamCreate (AICore) failed: %d", rc);
            ACL_LOG_ERROR_DETAIL(rc);
            // Roll back only the AICPU stream we just created, not a
            // pre-existing persistent one.
            if (aicpu_created_here) {
                rtStreamDestroy(stream_aicpu_);
                stream_aicpu_ = nullptr;
            }
            return rc;
        }
        aicore_created_here = true;
    }
    if (aicpu_created_here || aicore_created_here) {
        LOG_INFO("DeviceRunner: device=%d set, streams created", device_id_);
    }

    // Latch the AICore stream's block_dim ceiling. resolve_block_dim() is then
    // pure arithmetic and can run before any per-run stream work.
    if (max_block_dim_ == 0) {
        max_block_dim_ = query_max_block_dim(stream_aicore_, &max_cube_cores_, &max_vector_cores_);
        LOG_INFO(
            "DeviceRunner: device=%d max_block_dim=%d (cube=%u, vector=%u)", device_id_, max_block_dim_,
            max_cube_cores_, max_vector_cores_
        );
    }

    rc = ensure_binaries_loaded(stream_aicpu_);
    if (rc != 0) return rc;

    // Before the AICPU init launch: that launch is what publishes the workspace
    // addresses, and it happens once.
    rc = ensure_dma_workspace_provisioned();
    if (rc != 0) return rc;

    rc = ensure_aicpu_init_launched(stream_aicpu_);
    if (rc != 0) return rc;

    return ensure_dma_workspace_warmed();
}

int DeviceRunnerBase::init_kernel_context(int device_id) {
    int rc = adopt_borrowed_device(device_id);
    if (rc != 0) return rc;

    rc = kernel_exec_state_.initialize(device_id_, make_onboard_kernel_context_ops());
    if (rc != 0) {
        LOG_ERROR("init_kernel_context: context stream/event creation failed: %d", rc);
        return rc;
    }

    rtStream_t control_stream = static_cast<rtStream_t>(kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu));
    rtStream_t aicore_stream = static_cast<rtStream_t>(kernel_exec_state_.hidden_stream(KernelStreamKind::Aicore));

    // Same latch as the program path: resolve_block_dim() is pure arithmetic
    // once this holds.
    if (max_block_dim_ == 0) {
        max_block_dim_ = query_max_block_dim(aicore_stream, &max_cube_cores_, &max_vector_cores_);
        LOG_INFO(
            "DeviceRunner: kernel context device=%d max_block_dim=%d (cube=%u, vector=%u)", device_id_, max_block_dim_,
            max_cube_cores_, max_vector_cores_
        );
    }

    rc = ensure_binaries_loaded(control_stream);
    if (rc != 0) return rc;

    // The async-DMA workspace is program mode's SDMA channel; kernel mode
    // provisions none, so this launch publishes the all-zero addresses that
    // mean "that engine is unavailable".
    return ensure_aicpu_init_launched(control_stream);
}

PersistentArgsOps DeviceRunnerBase::persistent_args_ops() {
    PersistentArgsOps ops{};
    ops.context = this;
    ops.alloc = [](void *context, size_t bytes) -> void * {
        return static_cast<DeviceRunnerBase *>(context)->mem_alloc_.alloc(bytes);
    };
    ops.free_ = [](void *context, void *ptr) -> int {
        return static_cast<DeviceRunnerBase *>(context)->mem_alloc_.free(ptr);
    };
    ops.copy_h2d = [](void *, void *dst, size_t dst_bytes, const void *src, size_t src_bytes) -> int {
        return static_cast<int>(rtMemcpy(dst, dst_bytes, src, src_bytes, RT_MEMCPY_HOST_TO_DEVICE));
    };
    ops.fill_arch_fields = [](void *context, KernelArgs *args, uint64_t device_id) -> int {
        return static_cast<DeviceRunnerBase *>(context)->fill_persistent_arch_fields(args, device_id);
    };
    return ops;
}

int DeviceRunnerBase::prepare_kernel_callable(int32_t callable_id) {
    rtStream_t control_stream = static_cast<rtStream_t>(kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu));
    if (control_stream == nullptr) {
        LOG_ERROR("prepare_kernel_callable: no live kernel context");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }

    int rc = register_callable_on_device(callable_id, control_stream);
    if (rc != 0) return rc;

    // Idempotent: only the first prepared callable allocates. The uploaded
    // Runtime keeps its per-callable and per-invocation fields at the sentinels
    // Runtime() sets, because no callable is bound into this image: a launch
    // resolves its own target, and writing one here would make every launch
    // run whichever callable was prepared first.
    //
    // One set of execution resources, and this context reads no pipeline depth.
    // The kernel contract `build_kernel_pipeline_contract_impl` produces and
    // the init entry validates declares depth 2, which nothing consumes — the
    // number is the contract's, not this owner's, and one set is what a context
    // admitting a single invocation at a time needs.
    rc = persistent_args_.prepare_once(kernel_runtime_, persistent_args_ops(), static_cast<uint64_t>(device_id_));
    if (rc != 0) return rc;

    return kernel_exec_state_.mark_ready_enqueued();
}

/**
 * Distance from a GM address to its nocache alias on `device_id`, or 0.
 *
 * The device maps each page twice, once cached and once not, and the driver owns
 * that layout — so the distance is a per-device value only it can report, never a
 * constant a caller may bake in. A kernel adds it to a base address to reach the
 * uncached mapping.
 *
 * Every failure yields 0, which is the value that leaves loads cached, because
 * the kernel's `addr + 0` is the ordinary address. A device without the alias
 * answers ACL_ERROR_RT_FEATURE_NOT_SUPPORT, and that is not a defect: the alias
 * is an L2 optimization, so its absence costs bandwidth, not correctness.
 */
int DeviceRunnerBase::ensure_aicpu_init_launched(rtStream_t control_stream) {
    if (aicpu_init_launched_) {
        return 0;
    }

    InitArgs init_args{};
    init_args.device_id = static_cast<uint32_t>(device_id_);
    // The device threshold is set here and never again: this entry launches once
    // per Worker, so a later host-side set_level does not reach the AICPU. That is
    // the intended contract, not a missing refresh — recreate the Worker to change
    // it. docs/logging.md records why.
    init_args.log_level = static_cast<uint32_t>(HostLogger::get_instance().level());
    // Per-device scheduler watchdog override, resolved once at attach into
    // timeout_config_. 0 -> the AICPU scheduler keeps its compile-time default.
    init_args.scheduler_timeout_ms = timeout_config_.scheduler_timeout_ms;
    // Publish the provisioned async-DMA workspace addresses (all-zero unless the
    // Worker opted into SDMA). ensure_dma_workspace_provisioned() runs first, so
    // this single launch carries them; the AICPU SO stays resident, and the
    // values survive every subsequent per-task launch.
    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
        init_args.dma_workspace_addr[kind] = dma_workspace_addr_[kind];
    }
    fill_init_arch_fields(init_args);

    LOG_INFO("=== launch_aicpu_payload %s ===", host::KernelNames::InitName);
    int rc = launch_aicpu_payload(
        control_stream, &init_args, sizeof(init_args), host::KernelNames::InitName, /*aicpu_num=*/1
    );
    if (rc != 0) {
        LOG_ERROR("ensure_aicpu_init_launched: launch_aicpu_payload failed: %d", rc);
        return rc;
    }

    rc = aclrtSynchronizeStreamWithTimeout(control_stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc != 0) {
        LOG_ERROR("ensure_aicpu_init_launched: stream sync failed: %d (device_id=%d)", rc, device_id_);
        return rc;
    }
    aicpu_init_launched_ = true;
    return 0;
}

int DeviceRunnerBase::ensure_binaries_loaded(rtStream_t control_stream) {
    // Check if already loaded (binaries are owned by the runner via
    // set_executors and live for the runner's lifetime).
    if (binaries_loaded_) {
        return 0;
    }

    // The control stream is what the bootstrap launch rides; a context that
    // has not created one yet has no device to bootstrap on.
    if (control_stream == nullptr) {
        LOG_ERROR("Device not set before loading binaries");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    if (dispatcher_so_binary_.empty()) {
        LOG_ERROR(
            "DeviceRunner: dispatcher SO bytes not provided; pass dispatcher_path through ChipWorker.init "
            "(RuntimeBinaries.dispatcher_path)"
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // One-shot bootstrap: libaicpu_extend_kernels invokes our dispatcher,
    // which writes the runtime AICPU SO bytes to
    // simpler_inner_<fp>_<device_id>.so in the device-side preinstall path.
    // The dispatcher SO itself is never persisted to disk — only the
    // transient libaicpu_extend_kernels dlopen. Subsequent per-task AICPU
    // launches resolve symbols via rtsBinaryLoadFromFile + rtsFuncGetByName +
    // rtsLaunchCpuKernel directly against the preinstall file.
    int rc = load_aicpu_op_.BootstrapDispatcher(
        dispatcher_so_binary_.data(), dispatcher_so_binary_.size(), aicpu_so_binary_.data(), aicpu_so_binary_.size(),
        control_stream, device_id_
    );
    if (rc != 0) {
        LOG_ERROR("LoadAicpuOp::BootstrapDispatcher failed: %d", rc);
        return rc;
    }
    LOG_INFO("DeviceRunner: inner SO uploaded to preinstall via dispatcher bootstrap");

    // JSON-register the inner SO and resolve its runtime entry handles. The
    // runtime reports any AICPU entries it exports beyond the base set so the
    // loader stays runtime-agnostic.
    std::vector<std::string> extra_symbols;
    size_t extra_count = 0;
    const char *const *extra = runtime_extra_aicpu_symbols(&extra_count);
    for (size_t i = 0; i < extra_count && extra != nullptr; ++i) {
        if (extra[i] != nullptr) extra_symbols.emplace_back(extra[i]);
    }
    rc = load_aicpu_op_.Init(extra_symbols);
    if (rc != 0) {
        LOG_ERROR("LoadAicpuOp::Init failed: %d", rc);
        return rc;
    }
    LOG_INFO("DeviceRunner: inner SO registered (runtime entry handles ready)");

    // Release host bytes — bootstrap is done. Per-task launches go through
    // the cached rtFuncHandle owned by LoadAicpuOp; dispatcher SO bytes are
    // never referenced again; the aicpu kernel SO's host buffer is no longer
    // needed either (we used to H2D it through AicpuSoInfo as a CANN-internal
    // bookkeeping workaround; that's gone).
    dispatcher_so_binary_.clear();
    dispatcher_so_binary_.shrink_to_fit();
    aicpu_so_binary_.clear();
    aicpu_so_binary_.shrink_to_fit();

    binaries_loaded_ = true;
    LOG_INFO("DeviceRunner: binaries loaded");
    return 0;
}

int DeviceRunnerBase::query_max_block_dim(rtStream_t stream, uint32_t *out_cube, uint32_t *out_vector) {
    uint32_t cube_limit = 0, vector_limit = 0;
    bool got_limits = (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, &cube_limit) == ACL_ERROR_NONE) &&
                      (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector_limit) == ACL_ERROR_NONE) &&
                      cube_limit > 0 && vector_limit > 0;
    if (out_cube != nullptr) *out_cube = got_limits ? cube_limit : 0;
    if (out_vector != nullptr) *out_vector = got_limits ? vector_limit : 0;
    if (got_limits) {
        // Cap by PLATFORM_MAX_BLOCKDIM as well: runtime handshake/scheduler
        // arrays are statically sized to RUNTIME_MAX_WORKER (= PLATFORM_MAX_BLOCKDIM
        // * PLATFORM_CORES_PER_BLOCKDIM), so even if ACL reports more cores
        // than the platform cap we must not exceed it.
        int from_stream = static_cast<int>(
            std::min(cube_limit / PLATFORM_AIC_CORES_PER_BLOCKDIM, vector_limit / PLATFORM_AIV_CORES_PER_BLOCKDIM)
        );
        return std::min(from_stream, PLATFORM_MAX_BLOCKDIM);
    }
    return PLATFORM_MAX_BLOCKDIM;
}

void DeviceRunnerBase::print_handshake_results(const KernelArgsHelper &kernel_args) {
    // Every consumer of this copy is a DEBUG record below, so the threshold
    // decides whether the D2H happens at all, not just whether it is printed.
    if (!HostLogger::get_instance().is_enabled(simpler::log::LogLevel::DEBUG)) {
        return;
    }
    if (stream_aicpu_ == nullptr || worker_count_ == 0 || kernel_args.args.runtime_args == nullptr) {
        return;
    }

    // Allocate temporary buffer to read handshake data from device
    std::vector<Handshake> workers(worker_count_);
    size_t total_size = sizeof(Handshake) * worker_count_;
    int rc = rtMemcpy(
        workers.data(), total_size, kernel_args.args.runtime_args->get_workers(), total_size, RT_MEMCPY_DEVICE_TO_HOST
    );
    if (rc != 0) {
        // The buffer holds no device content on this path, so it is not
        // printed. A diagnostic read carries no run verdict.
        LOG_WARN("rtMemcpy(handshake results) D2H failed: %d", rc);
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG(
            "  Core %d: aicore_done=%d aicpu_ready=%d task=0x%lx", i, workers[i].aicore_done, workers[i].aicpu_ready,
            static_cast<uint64_t>(workers[i].task)
        );
    }
}

// =============================================================================
// Group D — chip-callable upload + per-callable_id registration
// =============================================================================

// Whether this runtime's device scheduler dispatches from resolved kernel-entry
// addresses rather than resolving each entry out of the CoreCallable object it
// is handed. Defined by every runtime's runtime_maker.cpp, so adding a runtime
// cannot leave the answer implicit, and true only where a device consumer reads
// the entry view.
extern "C" bool runtime_uses_callable_entry_table_impl();

uint64_t DeviceRunnerBase::upload_chip_callable_buffer(const ChipCallable *callable) {
    if (callable == nullptr) {
        return 0;
    }
    // The upload allocates and copies; it needs a bound device and nothing
    // else. A kernel-mode context has no stream_aicpu_ — its AICPU stream
    // belongs to KernelExecutionState — so the stream is not the precondition
    // to test here.
    if (device_id_ < 0) {
        LOG_ERROR("No device bound before upload_chip_callable_buffer()");
        return 0;
    }

    const ChipCallableLayout layout = compute_chip_callable_layout(callable);

    // Content-hash dedup: identical bytes → return cached chip_dev.
    auto it = chip_callable_buffers_.find(layout.content_hash);
    if (it != chip_callable_buffers_.end() && it->second.release_pending) {
        // Present only as a retained owner of a block whose release failed, so
        // it names nothing a caller may use. Refuse rather than hand out an
        // address whose contents are gone or never arrived; the retry that
        // frees it runs at close.
        LOG_ERROR("Chip callable hash=0x%lx names a block awaiting release retry; upload refused", layout.content_hash);
        return 0;
    }
    if (it != chip_callable_buffers_.end()) {
        it->second.refcount++;
        LOG_DEBUG(
            "Chip callable dedup hit: chip_dev=0x%lx, size=%zu, hash=0x%lx, refcount=%d", it->second.chip_dev,
            it->second.total_size, layout.content_hash, it->second.refcount
        );
        return it->second.chip_dev;
    }

    // Sizing, before anything is allocated, so an unaddressable func_id refuses
    // the registration rather than leaving a block behind. Each table starts on
    // a CALLABLE_ALIGN boundary past the code, which is also the alignment the
    // block's own base carries, so a table entry's address is the base plus a
    // known constant on every callable.
    //
    // The arithmetic is 64-bit over sources the ChipCallable wire ABI caps at
    // 32 bits: `layout.total_size` is the header plus a `storage_used` bounded
    // by one `uint32_t` binary size plus one `uint32_t` child offset plus a
    // CoreCallable header, the tables add at most `RUNTIME_MAX_FUNC_ID`
    // eight-byte entries each, and each alignment step adds less than one line.
    // The static_assert states that bound, so no checked addition is needed
    // here.
    static_assert(sizeof(size_t) >= 8, "the chip-callable tail sizing below assumes a 64-bit size_t");
    static_assert(
        static_cast<uint64_t>(offsetof(ChipCallable, storage_)) + UINT32_MAX + UINT32_MAX +
                CoreCallable::binary_data_offset() + 2 * CALLABLE_ALIGN +
                2 * static_cast<uint64_t>(RUNTIME_MAX_FUNC_ID) * sizeof(uint64_t) <
            static_cast<uint64_t>(SIZE_MAX),
        "a ChipCallable plus both function tables must not be able to overflow size_t"
    );
    uint32_t table_len = 0;
    int32_t bad_func_id = 0;
    if (!chip_callable_table_length(callable, RUNTIME_MAX_FUNC_ID, &table_len, &bad_func_id)) {
        LOG_ERROR("Chip callable declares func_id=%d outside [0, %d)", bad_func_id, RUNTIME_MAX_FUNC_ID);
        return 0;
    }
    const bool want_entry_table = table_len != 0 && runtime_uses_callable_entry_table_impl();
    const size_t table_bytes = static_cast<size_t>(table_len) * sizeof(uint64_t);
    const auto align_up = [](size_t bytes) {
        return (bytes + CALLABLE_ALIGN - 1) & ~(static_cast<size_t>(CALLABLE_ALIGN) - 1);
    };
    const size_t object_table_off = align_up(layout.total_size);
    const size_t entry_table_off = align_up(object_table_off + table_bytes);
    const size_t alloc_size = want_entry_table ? entry_table_off + table_bytes : object_table_off + table_bytes;

    // Every host allocation this upload needs is taken before the device one,
    // so no allocation that can throw sits between the device block's creation
    // and the publication that gives it an owner. What runs in between — the
    // scratch fill, the patch, the table fill and the copy — allocates nothing.
    std::vector<uint8_t> scratch(alloc_size);
    std::vector<uint64_t> object_table(table_len, 0);
    std::vector<uint64_t> entry_table;
    if (want_entry_table) entry_table.assign(table_len, 0);

    void *gm_addr = mem_alloc_.alloc(alloc_size);
    if (gm_addr == nullptr) {
        LOG_ERROR("Failed to allocate device GM for ChipCallable buffer (size=%zu)", alloc_size);
        return 0;
    }
    const uint64_t chip_dev = reinterpret_cast<uint64_t>(gm_addr);
    assert((chip_dev & (CALLABLE_ALIGN - 1)) == 0 && "device alloc must be CALLABLE_ALIGN-byte aligned");

    // Ownership of the device block until the map entry has it. Only the
    // publication below can still fail once the copy has succeeded, and it
    // allocates a map node; this releases the block if it does. The
    // copy-failure path keeps its own rollback, because that is the one that
    // can retain a reportable owner — an allocation this guard must not make,
    // since it also runs while an exception is propagating.
    bool published = false;
    auto block_guard = RAIIScopeGuard([&]() {
        if (published) return;
        if (mem_alloc_.free(gm_addr) != 0) {
            LOG_ERROR("Upload rollback: free of chip_dev=0x%lx failed — block leaks until device reset", chip_dev);
        }
    });

    // Fill the host scratch with each child's resolved_addr_ fixed up to the
    // device-side address of that child's binary code (so the AICPU dispatch
    // path's `reinterpret_cast<CoreCallable*>(addr)->resolved_addr()` lands
    // on the right device offset).
    std::memcpy(scratch.data(), callable, layout.total_size);
    patch_chip_callable_scratch_for_device(callable, layout, chip_dev, scratch.data());

    // The function tables go in the alignment-padded tail of that same scratch,
    // so the one copy below delivers them alongside the code their entries
    // name.
    chip_callable_fill_tables(
        callable, layout, scratch.data(), chip_dev, table_len, object_table.data(),
        want_entry_table ? entry_table.data() : nullptr
    );
    if (table_len != 0) {
        std::memcpy(scratch.data() + object_table_off, object_table.data(), table_bytes);
        if (want_entry_table) std::memcpy(scratch.data() + entry_table_off, entry_table.data(), table_bytes);
    }

    int rc = rtMemcpy(gm_addr, alloc_size, scratch.data(), alloc_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy chip callable H2D failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        // Nothing owns this block yet — the map entry is added below — so a
        // failed release on the retaining path has to record one, or the
        // allocation is reachable only through `MemoryAllocator`, whose
        // finalize() clears its tracking map even when rtFree fails. The entry
        // goes in with refcount zero: it names a block to retry releasing, not
        // a callable anyone can use.
        if (mem_alloc_.free(gm_addr) != 0) {
            if (retains_failed_callable_release()) {
                LOG_ERROR(
                    "Upload rollback: free of chip_dev=0x%lx failed — retaining ownership for retry at close", chip_dev
                );
                chip_callable_buffers_.emplace(
                    layout.content_hash, ChipCallableBuffer{chip_dev, alloc_size, 0, /*release_pending=*/true}
                );
            } else {
                LOG_ERROR("Upload rollback: free of chip_dev=0x%lx failed — block leaks until device reset", chip_dev);
            }
        }
        block_guard.dismiss();
        return 0;
    }
    mark_run_streams_stale();

    // One publication point. The entry is inserted with no tables and the
    // vectors moved in afterwards through `std::vector`'s noexcept move
    // assignment, so the insertion is the last thing that can fail and no
    // caller can reach a published entry whose tables are missing.
    ChipCallableBuffer retained{chip_dev, alloc_size, 1};
    retained.table_len = table_len;
    if (table_len != 0) {
        retained.object_table_dev = chip_dev + object_table_off;
        if (want_entry_table) retained.entry_table_dev = chip_dev + entry_table_off;
    }
    auto inserted = chip_callable_buffers_.emplace(layout.content_hash, std::move(retained));
    inserted.first->second.object_table = std::move(object_table);
    published = true;
    LOG_DEBUG(
        "Uploaded chip callable: chip_dev=0x%lx, size=%zu, child_count=%d, table_len=%u, hash=0x%lx", chip_dev,
        alloc_size, callable->child_count(), table_len, layout.content_hash
    );
    return chip_dev;
}

int DeviceRunnerBase::release_chip_callable_buffer(uint64_t hash) {
    if (hash == 0) {
        return 0;
    }
    auto it = chip_callable_buffers_.find(hash);
    if (it == chip_callable_buffers_.end()) {
        LOG_WARN("release_chip_callable_buffer: hash=0x%lx not found", hash);
        return 0;
    }
    if (it->second.release_pending || --it->second.refcount <= 0) {
        it->second.refcount = 0;
        it->second.release_pending = retains_failed_callable_release();
        // On the retaining path the map entry is the only reportable owner of
        // this allocation: `MemoryAllocator::finalize()` clears its tracking map
        // even when rtFree fails, so erasing on a failed free would leave the
        // block with no owner able to retry it, and the next close would report
        // success over a live allocation. The refcount stays at zero, so a later
        // release or finalize retries the free rather than double-counting it.
        const int free_rc = mem_alloc_.free(reinterpret_cast<void *>(it->second.chip_dev));
        if (free_rc != 0 && retains_failed_callable_release()) {
            LOG_ERROR(
                "release_chip_callable_buffer: free of chip_dev=0x%lx failed: %d — retaining ownership for retry",
                it->second.chip_dev, free_rc
            );
            return free_rc;
        }
        if (free_rc != 0) {
            LOG_ERROR(
                "release_chip_callable_buffer: free of chip_dev=0x%lx failed: %d — block leaks until device reset",
                it->second.chip_dev, free_rc
            );
        } else {
            LOG_DEBUG(
                "Freed chip callable buffer: chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev,
                it->second.total_size, hash
            );
        }
        chip_callable_buffers_.erase(it);
        return free_rc;
    }
    return 0;
}

int DeviceRunnerBase::stamp_orch_so(Runtime &runtime, int32_t cid) {
    // Registered-callable flow only: the orch SO was already H2D'd and
    // dlopen'd device-side at record_device_orch_callable / launch_device_register
    // time. All that remains for a run is to tell the AICPU which orch_so_table_
    // slot to dispatch — the active callable_id.
    if (cid < 0) {
        LOG_ERROR("stamp_orch_so: invalid callable_id=%d", cid);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("stamp_orch_so: callable_id=%d not registered", cid);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    runtime.set_active_callable_id(cid);
    return 0;
}

int DeviceRunnerBase::prepare_orch_so(Runtime &runtime) {
    const int32_t cid = runtime.get_active_callable_id();
    if (cid < 0) {
        LOG_ERROR("prepare_orch_so: no active callable_id; registered-callable flow required");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return stamp_orch_so(runtime, cid);
}

int DeviceRunnerBase::commit_device_register(int32_t cid) {
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("commit_device_register: callable_id=%d not registered", cid);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const auto &state = it->second;
    if (state.host_dlopen_handle != nullptr) {
        return 0;
    }
    const bool inserted = aicpu_seen_callable_ids_.insert(cid).second;
    if (inserted) {
        ++aicpu_dlopen_total_;
        LOG_INFO("AICPU callable load committed cid=%d (count=%zu)", cid, aicpu_dlopen_total_);
    }
    return 0;
}

int DeviceRunnerBase::launch_device_register(int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("launch_device_register: callable_id=%d not registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (it->second.host_dlopen_handle != nullptr) {
        return 0;
    }

    const int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("launch_device_register: ensure_device_initialized failed: %d", rc);
        return rc;
    }
    return register_callable_on_device(callable_id, stream_aicpu_);
}

int DeviceRunnerBase::register_callable_on_device(int32_t callable_id, rtStream_t control_stream) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("register_callable_on_device: callable_id=%d not registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (it->second.host_dlopen_handle != nullptr) {
        return 0;
    }
    int rc = 0;

    // Build the orch-SO descriptor straight from CallableState — no full
    // Runtime H2D as the old prewarm path did. Registration always (re)dlopens
    // the SO device-side, so there is no per-callable "new?" bit to carry.
    const CallableState &state = it->second;
    RegisterCallableArgs reg_args{};
    reg_args.active_callable_id = callable_id;
    reg_args.dev_orch_so_addr = state.dev_orch_so_addr;
    reg_args.dev_orch_so_size = state.dev_orch_so_size;
    snprintf(reg_args.device_orch_func_name, sizeof(reg_args.device_orch_func_name), "%s", state.func_name.c_str());
    snprintf(
        reg_args.device_orch_config_name, sizeof(reg_args.device_orch_config_name), "%s", state.config_name.c_str()
    );

    LOG_INFO("=== launch_aicpu_payload %s ===", host::KernelNames::RegisterCallableName);
    rc = launch_aicpu_payload(
        control_stream, &reg_args, sizeof(reg_args), host::KernelNames::RegisterCallableName, /*aicpu_num=*/1
    );
    if (rc != 0) {
        // Submission itself failed, so no AICPU task exists and nothing on the
        // device can be reading the uploaded orchestration SO. The caller may
        // release the registration.
        LOG_ERROR("register_callable_on_device: launch_aicpu_payload failed: %d", rc);
        return rc;
    }

    // Past this point the registration task is enqueued and the AICPU reads the
    // uploaded orchestration SO out of the callable's device buffer. A failed or
    // timed-out wait does not establish that it stopped, so admission is
    // poisoned rather than the buffer released: `accepts_dispatch()` turning
    // false is how the caller learns it must not unregister this callable.
    rc = aclrtSynchronizeStreamWithTimeout(control_stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "register_callable_on_device: stream sync timeout timeout_ms=%d device_id=%d — retaining callable "
            "ownership, the AICPU may still be reading its image",
            PLATFORM_STREAM_SYNC_TIMEOUT_MS, device_id_
        );
        kernel_exec_state_.poison(rc);
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR(
            "register_callable_on_device: aclrtSynchronizeStreamWithTimeout failed: %d — retaining callable "
            "ownership, completion is not established",
            rc
        );
        ACL_LOG_ERROR_DETAIL(rc);
        kernel_exec_state_.poison(rc);
        return rc;
    }

    return commit_device_register(callable_id);
}

int DeviceRunnerBase::record_device_orch_callable(
    int32_t callable_id, uint64_t chip_buffer_hash, uint64_t aicore_image_hash, uint64_t chip_dev,
    const void *orch_so_data, size_t orch_so_size, const char *func_name, const char *config_name,
    std::vector<ArgDirection> signature
) {
    // The AICPU executor reserves `orch_so_table_[MAX_REGISTERED_CALLABLE_IDS]`
    // (declared in src/common/task_interface/callable_protocol.h) and indexes
    // it by callable_id; rejecting an out-of-range id here keeps the host and
    // AICPU sides in sync and avoids an OOB access at run time.
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "record_device_orch_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (orch_so_data == nullptr || orch_so_size == 0) {
        LOG_ERROR("record_device_orch_callable: empty orch SO for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (chip_buffer_hash == 0 || chip_dev == 0) {
        LOG_ERROR("record_device_orch_callable: missing chip buffer for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_device_orch_callable: callable_id=%d already registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    const uint64_t hash = simpler::common::utils::elf_build_id_64(orch_so_data, orch_so_size);

    CallableState state;
    state.hash = hash;
    state.chip_buffer_hash = chip_buffer_hash;
    state.aicore_image_hash = aicore_image_hash;
    state.dev_orch_so_addr = chip_dev + offsetof(ChipCallable, storage_);
    state.dev_orch_so_size = orch_so_size;
    state.func_name = (func_name != nullptr) ? func_name : "";
    state.config_name = (config_name != nullptr) ? config_name : "";
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    LOG_INFO(
        "record_device_orch_callable: cid=%d orch_hash=0x%lx chip_hash=0x%lx %zu bytes", callable_id, hash,
        chip_buffer_hash, orch_so_size
    );
    return 0;
}

int DeviceRunnerBase::record_host_orch_callable(
    int32_t callable_id, uint64_t chip_buffer_hash, uint64_t aicore_image_hash, void *host_dlopen_handle,
    void *host_orch_func_ptr, std::vector<ArgDirection> signature
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "record_host_orch_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (host_dlopen_handle == nullptr || host_orch_func_ptr == nullptr) {
        LOG_ERROR("record_host_orch_callable: null handle/fn for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (chip_buffer_hash == 0) {
        LOG_ERROR("record_host_orch_callable: missing chip buffer for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_host_orch_callable: callable_id=%d already registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    CallableState state;
    state.chip_buffer_hash = chip_buffer_hash;
    state.aicore_image_hash = aicore_image_hash;
    state.host_dlopen_handle = host_dlopen_handle;
    state.host_orch_func_ptr = host_orch_func_ptr;
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    ++host_dlopen_total_;
    LOG_INFO("record_host_orch_callable: cid=%d (host dlopen #%zu)", callable_id, host_dlopen_total_);
    return 0;
}

int DeviceRunnerBase::unregister_callable(int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        return 0;
    }
    CallableState state = std::move(it->second);
    callables_.erase(it);
    aicpu_seen_callable_ids_.erase(callable_id);
    release_chip_callable_buffer(state.chip_buffer_hash);

    if (state.host_dlopen_handle != nullptr) {
        // hbg path: no device-side orch SO handle, just dlclose the host handle.
        dlclose(state.host_dlopen_handle);
        return 0;
    }
    return 0;
}

bool DeviceRunnerBase::has_callable(int32_t callable_id) const { return callables_.count(callable_id) != 0; }

int DeviceRunnerBase::ensure_dma_workspace_provisioned() {
    if (dma_workspace_handle_ != nullptr) {
        return 0;
    }
    const uint32_t supported = dma_workspace_supported_mask();
    constexpr uint32_t kSdmaBit = uint32_t{1} << DMA_WORKSPACE_SDMA;
    // Opting in on a device that cannot provide SDMA is a caller error, not a
    // silent no-op: a Worker built for TPREFETCH_ASYNC must not reach its first
    // run reading a zero workspace address.
    if (sdma_requested_ && (supported & kSdmaBit) == 0) {
        LOG_ERROR("dma workspace: SDMA requested where unsupported (supported=0x%x)", supported);
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    // Everything this device supports, minus what the caller declined. SDMA is
    // the only declinable engine: its workspace cannot be obtained without also
    // creating 48 CP-process STARS streams, which halves this Worker's
    // post-fault reset budget, so a Worker that did not ask for it must not end
    // up holding them. Every other supported engine carries no such cost and is
    // provisioned unconditionally.
    const uint32_t required_mask = sdma_requested_ ? supported : (supported & ~kSdmaBit);
    if (required_mask == 0) {
        return 0;
    }
    // Dormant while one engine is supported, since required_mask is a subset of
    // supported. It arms itself on the day dma_workspace_supported_mask() widens,
    // which is the day the single-handle contract breaks — dma_workspace_release()
    // casts the opaque handle back to the one provider type it can be, so a second
    // engine would be released as the type of the first. A rejection here beats
    // that silent type confusion.
    if ((required_mask & (required_mask - 1)) != 0) {
        LOG_ERROR(
            "dma workspace: mask=0x%x names %d engines; one handle owns one provider", required_mask,
            __builtin_popcount(required_mask)
        );
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }

    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
        dma_workspace_addr_[kind] = 0;

    // The provisioned addresses are stable for the Worker's life.
    int rc =
        dma_workspace_provision(required_mask, dma_workspace_addr_, DMA_WORKSPACE_KIND_COUNT, &dma_workspace_handle_);
    if (rc != 0) {
        LOG_ERROR("dma workspace: mask=0x%x failed: %d", required_mask, rc);
        for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
            dma_workspace_addr_[kind] = 0;
        dma_workspace_handle_ = nullptr;
        return rc;
    }
    return 0;
}

int DeviceRunnerBase::ensure_dma_workspace_warmed() {
    if (dma_workspace_handle_ == nullptr || sdma_warmed_) {
        return 0;
    }
    // An unavailable warmup leaves init successful, because the only cost is
    // first-call latency. A device error does not: the card the warmup just
    // faulted on would otherwise reach the first run. No dma_workspace_release()
    // on that path — launch_sdma_warmup_kernel() has marked the runner unusable,
    // and per-resource release on a faulted card is exactly what finalize()'s
    // fatal path exists to avoid. The workspace handle stays set so that path
    // still sees an SDMA generation and applies its handoff delay.
    const int rc = launch_sdma_warmup_kernel(sdma_warmup_binary_.data(), sdma_warmup_binary_.size());
    if (rc != 0) {
        LOG_ERROR("dma workspace: sdma warmup left the device unusable: %d", rc);
        return rc;
    }
    sdma_warmed_ = true;
    sdma_warmup_binary_.clear();
    sdma_warmup_binary_.shrink_to_fit();
    return 0;
}

int DeviceRunnerBase::launch_sdma_warmup_kernel(const void *binary, size_t size) {
    // Reaching here means the workspace was provisioned, so this platform does
    // support SDMA and a missing ELF is a build/staging regression rather than
    // the expected state — worth a warning even though it is not fatal.
    if (binary == nullptr || size == 0) {
        LOG_WARN("sdma warmup: no warmup ELF supplied; the first TPREFETCH_ASYNC will pay the cold control path");
        return 0;
    }
    const uint32_t channel_count = dma_workspace_channel_count();
    const uint64_t workspace = dma_workspace_addr_[DMA_WORKSPACE_SDMA];
    if (channel_count == 0 || workspace == 0) {
        LOG_INFO("sdma warmup: no channels or no workspace address; skipping");
        return 0;
    }

    if (sdma_warmup_bin_handle_ == nullptr) {
        rtDevBinary_t warmup_binary;
        std::memset(&warmup_binary, 0, sizeof(warmup_binary));
        // AIVEC, not the executor's ELF magic: this binary has no cube half.
        warmup_binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
        warmup_binary.version = 0;
        warmup_binary.data = binary;
        warmup_binary.length = size;
        int reg_rc = rtRegisterAllKernel(&warmup_binary, &sdma_warmup_bin_handle_);
        if (reg_rc != 0 || sdma_warmup_bin_handle_ == nullptr) {
            LOG_WARN("sdma warmup: rtRegisterAllKernel failed: %d; skipping warmup", reg_rc);
            sdma_warmup_bin_handle_ = nullptr;
            return 0;
        }
    }

    // One cache line per channel, see sdma_warmup_layout.h. Transient: the launch
    // is synchronized here, so nothing outlives this call.
    const size_t status_bytes = static_cast<size_t>(channel_count) * kSdmaWarmupStatusStrideBytes;
    void *status_dev = allocate_tensor(status_bytes);
    if (status_dev == nullptr) {
        LOG_WARN("sdma warmup: could not allocate %zu status bytes; skipping warmup", status_bytes);
        return 0;
    }
    // Zero means "no core reached this channel", so the buffer must start clean
    // for the readback below to be meaningful.
    int rc = device_memset(status_dev, 0, status_bytes);
    if (rc != 0) {
        LOG_WARN("sdma warmup: status zero-fill failed: %d; skipping warmup", rc);
        free_tensor(status_dev);
        return 0;
    }

    struct Args {
        uint64_t workspace;
        uint64_t status;
        uint64_t channel_count;
    };
    Args args = {workspace, reinterpret_cast<uint64_t>(status_dev), channel_count};
    rtArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    // block_dim is kSdmaWarmupBlockDim, NOT channel_count: the cold cost is
    // per-channel and serializes in the engine regardless of how many cores push
    // on it, so extra blocks buy nothing (measured) and tying block_dim to the
    // channel count would break on any chip with fewer AIVs than channels. The
    // kernel walks channels grid-stride to cover all of them from 8 blocks.
    const auto started = std::chrono::steady_clock::now();
    rc = rtKernelLaunchWithHandleV2(
        sdma_warmup_bin_handle_, 0, kSdmaWarmupBlockDim, &rt_args, nullptr, stream_aicore_, &cfg
    );
    if (rc == 0) {
        rc = rtStreamSynchronize(stream_aicore_);
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (rc != 0) {
        LOG_ERROR("sdma warmup: launch/sync failed: %d; the card is left poisoned", rc);
        // No free_tensor(status_dev): rtFree on a card that just failed an AICore
        // operation can block in DEV_RUNNING_DOWN. The allocation stays tracked by
        // mem_alloc_ and is forgotten wholesale by the fatal teardown the mark
        // below routes finalize() into.
        recover_device_or_mark_unusable(rc);
        return rc;
    }

    report_sdma_warmup_status(status_dev, channel_count, elapsed_ms);
    return 0;
}

void DeviceRunnerBase::report_sdma_warmup_status(void *status_dev, uint32_t channel_count, double elapsed_ms) {
    const size_t status_bytes = static_cast<size_t>(channel_count) * kSdmaWarmupStatusStrideBytes;
    std::vector<uint8_t> status_host(status_bytes, 0);
    const int rc = copy_from_device(status_host.data(), status_dev, status_bytes);
    free_tensor(status_dev);
    if (rc != 0) {
        LOG_WARN("sdma warmup: status D2H failed: %d; warmup ran but is unverified", rc);
        return;
    }

    uint32_t warmed = 0;
    uint32_t declined = 0;
    for (uint32_t channel = 0; channel < channel_count; ++channel) {
        uint32_t slot = 0;
        std::memcpy(
            &slot, status_host.data() + static_cast<size_t>(channel) * kSdmaWarmupStatusStrideBytes, sizeof(slot)
        );
        if (slot == kSdmaWarmupStatusOk) {
            ++warmed;
        } else if (slot == kSdmaWarmupStatusFailed) {
            ++declined;
        }
    }
    // TIMING, not INFO: this is a one-off multi-millisecond init cost paid to
    // remove the same cost from the first run, so it belongs with the other
    // performance markers that stay visible at the default threshold.
    if (warmed == channel_count) {
        LOG_TIMING("sdma warmup: %u/%u channels warmed in %.2f ms", warmed, channel_count, elapsed_ms);
    } else {
        // The two shortfalls have different causes: a declined channel was reached
        // but failed the warmup's preconditions (unpopulated SQ, non-empty queue),
        // while an unreached one means the walk itself did not cover the channel.
        LOG_WARN(
            "sdma warmup: only %u/%u channels warmed in %.2f ms (%u declined, %u unreached); "
            "the rest keep their cold-start cost",
            warmed, channel_count, elapsed_ms, declined, channel_count - warmed - declined
        );
    }
}

uint64_t DeviceRunnerBase::callable_hash(int32_t callable_id) const {
    auto it = callables_.find(callable_id);
    return it == callables_.end() ? 0 : it->second.hash;
}

// Per-run binding half, defined in each runtime's runtime_maker.cpp and linked
// into this same host_runtime.so. Declared here (rather than only in
// c_api_shared.cpp) so bind_callable_to_runtime can call it directly, keeping
// the CallableState-derived host_orch_func_ptr / signature internal to the
// runner instead of returning them across the c_api boundary.
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);

int DeviceRunnerBase::bind_callable_to_runtime(
    Runtime &runtime, int32_t callable_id, const HostApi *api, const void *orch_args, const uint64_t *ring_task_window,
    const uint64_t *ring_heap, const uint64_t *ring_dep_pool
) {
    // Clear before anything else, including before the registry lookup: an id
    // that is gone is exactly the case where the reference still standing names
    // a block `unregister_callable` already freed, and the scheduler
    // dereferences these addresses for the AICore to call what it finds there.
    // So no bind — refused, failed, or successful — may leave a predecessor's.
    runtime.clear_callable_tables();
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d not registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const auto &state = it->second;

    auto block = chip_callable_buffers_.find(state.chip_buffer_hash);
    if (block == chip_callable_buffers_.end() || block->second.release_pending) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d has no retained registration block", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // The tables were built and copied by the registration that retained this
    // block, and the content hash keying it covers the child func_ids and
    // offsets they are derived from, so every callable_id sharing the block
    // binds the same complete view. A callable with no children publishes no
    // table and every lookup against it reads 0.
    const ChipCallableBuffer &tables = block->second;
    runtime.set_callable_tables(
        tables.object_table.empty() ? nullptr : tables.object_table.data(), tables.object_table_dev,
        tables.entry_table_dev, tables.table_len
    );
    // Tell the AICPU which orch_so_table_ slot this run dispatches. The orch SO
    // descriptor itself was delivered at register time via RegisterCallableArgs.
    runtime.set_active_callable_id(callable_id);

    // Per-run binding (tensor args, GM heap, SM alloc). host_orch_func_ptr is
    // non-null only on the hbg path; signature is the cached ChipCallable
    // signature_[], plumbed end-to-end for per-tensor H2D/D2H direction
    // decisions in runtime_maker (trb consumes it, hbg ignores it). Both stay
    // internal to the runner now — they are no longer returned to the c_api.
    return bind_callable_to_runtime_impl(
        &runtime, api, reinterpret_cast<const ChipStorageTaskArgs *>(orch_args), state.host_orch_func_ptr,
        state.signature.empty() ? nullptr : state.signature.data(), static_cast<int>(state.signature.size()),
        ring_task_window, ring_heap, ring_dep_pool
    );
}

// Eager prebuilt-arena warm-up. A runtime that has a prebuilt runtime arena
// (tensormap_and_ringbuffer) provides a strong prewarm_config_impl in its
// runtime_maker.cpp that overrides this weak no-op default. Runtimes without one
// (host_build_graph, or an arch that has not implemented it yet) link this weak
// default and treat prewarm as a no-op. simpler_init calls it directly for the
// fork-constant ring sizing once the device is up.
extern "C" __attribute__((weak)) int prewarm_config_impl(
    const HostApi * /*api*/, const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/,
    const uint64_t * /*ring_dep_pool*/
) {
    return 0;
}

void DeviceRunnerBase::apply_call_config(const CallConfig &config) {
    set_chip_swimlane_enabled(config.enable_chip_swimlane);
    set_output_prefix(config.output_prefix);
}

void DeviceRunnerBase::begin_host_phase_run(uint32_t pipeline_slot, const DfxRunConfig &dfx) {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    host_phase_runs_[pipeline_slot].begin(dfx);
}

HostPhaseRecordPool *
DeviceRunnerBase::host_phase_pool_arm(uint32_t pipeline_slot, bool producer_wants_records) noexcept {
    if (pipeline_slot >= host_phase_runs_.size()) return nullptr;
    HostPhaseRunState &run = host_phase_runs_[pipeline_slot];
    // Only a host-orchestrating bind reaches this, so arriving here is what
    // makes the run's orchestrator phases host-produced.
    run.host_orchestrated = run.chip_swimlane_level == ChipSwimlaneLevel::ORCH_PHASES;

    // arm() allocates the pool's buffers, so it can throw; this path is noexcept,
    // where an escaping exception is std::terminate. A pass that cannot get its
    // storage collects no records and says so by handing back nullptr.
    HostPhaseRecordPool *pool = nullptr;
    try {
        pool = run.records.arm(run.wants_records(producer_wants_records));
    } catch (...) {
        LOG_WARN("Host phase pool could not be armed; this pass collects no per-event records");
    }
    if (!run.host_orchestrated) return pool;

    return pool;
}

void DeviceRunnerBase::publish_host_phase_run_to_collector(uint32_t pipeline_slot) noexcept {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    const HostPhaseRunState &run = host_phase_runs_[pipeline_slot];
    // Read by the collector's initialize() when it sizes the orch phase pool, so
    // this has to precede it — both now run from the launch arming.
    chip_swimlane_collector_.set_host_orchestrated(run.host_orchestrated);
}

void DeviceRunnerBase::publish_host_phase_records_to_swimlane(uint32_t pipeline_slot) {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    const simpler::dfx::HostPhaseRecordStore &records = host_phase_runs_[pipeline_slot].records;
    if (!records.finished()) return;
    chip_swimlane_collector_.set_host_phase_records(
        records.submit_records(), records.device_upload_records(), records.submitted_tasks(), records.total_records(),
        records.dropped_records()
    );
}

// =============================================================================
// Group E (minimal) — shared AICPU launch helper
// =============================================================================

int DeviceRunnerBase::launch_aicpu_payload(
    rtStream_t stream, void *args, size_t args_size, const char *kernel_name, int aicpu_num
) {
    // For the run entry, kernel_name is host::KernelNames::RunName — the runtime
    // SO's actual exported symbol (simpler_aicpu_exec). LaunchBuiltInOp
    // dispatches via rtsLaunchCpuKernel on the cached rtFuncHandle resolved by
    // LoadAicpuOp::Init at first-time bootstrap.
    return load_aicpu_op_.LaunchBuiltInOp(stream, args, args_size, aicpu_num, kernel_name);
}

int DeviceRunnerBase::finalize_common() { return finalize_common_impl(false); }

int DeviceRunnerBase::abandon_common_after_device_failure() { return finalize_common_impl(true); }

int DeviceRunnerBase::finalize_common_impl(bool abandon_device_resources) {
    int rc = 0;
    auto capture = [&rc](int err) {
        if (err != 0 && rc == 0) rc = err;
    };

    // Teardown invariant: finalize_common() is the single place that releases
    // every RTS/device-owning resource, and the subclass runs it BEFORE its
    // device reset / aclFinalize. Several base-class members have destructors
    // that themselves call an RTS API -- LoadAicpuOp::~ -> rtsBinaryUnload,
    // MemoryAllocator::~ -> finalize -> rtFree, DeviceArena::~ -> release ->
    // rtFree. A member destructor runs (per C++ rules) only AFTER finalize()
    // returns, i.e. AFTER aclFinalize has torn down the RTS context, and
    // touching an RTS interface on a dead context segfaults on a5 (a2a3 happens
    // to tolerate it). So each such member is released explicitly here while RTS
    // is live; every release is idempotent (guarded on a handle / committed_ /
    // raw_base_ flag) so the eventual destructor no-ops. Any new member owning
    // an RTS/device resource must be released here, with an idempotent
    // destructor as the backstop. See issue #1197.
    // Streams are persistent for the DeviceRunner's lifetime; destroy them here.
    // Intentionally no pre-destroy sync: when a run hits the AICore op-timeout
    // chain (PR #718), the AICPU stream surfaces ACL_ERROR_RT_AICPU_EXCEPTION
    // (507018) at run-path sync; calling aclrtSynchronizeStream* again on the
    // error-state stream at finalize wedges subsequent tests (observed: 507018
    // / 507899 / 507901 cascade across the whole st-onboard-a2a3 suite).
    // rtStreamDestroy on an error-state stream is the supported teardown path.
    if (abandon_device_resources) {
        LOG_WARN("Fatal teardown: force reset/quarantine finished; skipping per-resource RTS destroy/free calls");
    }
    // Anything this device reported and nobody has read yet is reported now:
    // after this the runner stops looking, and a notification that arrived
    // during teardown is the one most worth having in the log. Observation only
    // — the admission this could refuse is already over, and the consumer itself
    // starts no drain, reset or recovery, which this path could not survive.
    (void)consume_device_fault_notices();
    release_device_fault_monitor();
    // Completion-boundary events are released ahead of the streams they were
    // recorded on: no run is left to wait on them here, and a destroyed stream
    // cannot be the thing that proves a surviving event safe to drop.
    for (auto &fence : run_fences_) {
        if (abandon_device_resources) {
            fence->abandon();
        } else {
            capture(fence->release());
        }
    }
    // After the fences, in both directions. An abandoned fence is what lets its
    // tokens be dropped at all; a released one refused while any was live, so
    // reaching here with an undischarged wait means the release above already
    // reported it and the proof events stay named rather than destroyed.
    if (abandon_device_resources) {
        capture(queued_waits_->abandon());
    } else {
        capture(queued_waits_->release_events());
    }
    // Last of the event owners, and on a condition the others do not have: nothing waits on a
    // boundary marker, but a *run* stream that kept its handle because its own destroy failed may
    // still hold a queued record naming one. Destroying the event then is exactly what the
    // markers must not do, so they are retained instead and the retention is reported. The
    // bootstrap pair below records no markers, so its own destruction is not this condition.
    if (abandon_device_resources) {
        boundary_marks_->abandon();
    } else if (marker_recording_streams_retired()) {
        capture(boundary_marks_->release());
    } else {
        boundary_marks_->retain();
        LOG_ERROR(
            "finalize: a run stream survived its own destroy, so the %zu boundary-marker event(s) it may still "
            "name are kept for this process rather than destroyed",
            boundary_marks_->live_event_count()
        );
        capture(PTO_RUNTIME_ERR_INVALID_STATE);
    }
    if (stream_aicpu_ != nullptr) {
        if (!abandon_device_resources) {
            capture(rtStreamDestroy(stream_aicpu_));
        }
        stream_aicpu_ = nullptr;
    }
    if (stream_aicore_ != nullptr) {
        if (!abandon_device_resources) {
            capture(rtStreamDestroy(stream_aicore_));
        }
        stream_aicore_ = nullptr;
    }

    // Release the async-DMA provider (SDMA STARS streams + workspace) only on
    // healthy teardown. A fatal reset invalidates its device resources as a
    // group, so running its per-stream destructor afterwards is unsafe.
    if (dma_workspace_handle_ != nullptr) {
        if (!abandon_device_resources) {
            dma_workspace_release(dma_workspace_handle_);
        }
        dma_workspace_handle_ = nullptr;
    }
    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
        dma_workspace_addr_[kind] = 0;

    bool kernel_cleanup_failed = false;

    // LoadAicpuOp holds a binary_handle_ from rtsBinaryLoadFromFile; unload it
    // here while RTS is live so ~LoadAicpuOp's idempotent Finalize() no-ops
    // instead of unloading after aclFinalize (see the invariant above).
    if (abandon_device_resources) {
        load_aicpu_op_.ForgetWithoutUnload();
        // A force reset invalidates every device allocation at once. If the
        // reset failed, the device is quarantined and per-allocation rtFree is
        // still unsafe. Forget allocator ownership before the shared host-side
        // cleanup below, so arena/free backstops become local no-ops. Per-run
        // kernel arguments live on PreparedExecution and are abandoned by
        // cleanup_execution() before finalize is reached.
        mem_alloc_.abandon_after_device_failure();
    } else {
        // A failed unload keeps the loader's handle, so a kernel close must
        // report the failure: its device is borrowed and never reset, which
        // makes this close the only thing that can retire the binary.
        const int loader_rc = load_aicpu_op_.Finalize();
        capture(loader_rc);
        kernel_cleanup_failed = execution_mode_latch().is_kernel() && loader_rc != 0;
    }

    // aicore_bin_handle_ was registered once via rtRegisterAllKernel; CANN
    // releases its device-side state when the device context tears down. Same for
    // the SDMA warmup ELF's separate handle.
    aicore_bin_handle_ = nullptr;
    sdma_warmup_bin_handle_ = nullptr;
    sdma_warmed_ = false;
    binaries_loaded_ = false;
    // The inner AICPU SO is unloaded with the binaries above, so its latched
    // globals are gone too — clear the one-shot guard so a reused runner
    // re-launches simpler_aicpu_init after the next ensure_binaries_loaded().
    aicpu_init_launched_ = false;

    // Release any chip callable buffers callers forgot to unregister. On a
    // kernel context a failed free keeps its entry so an explicit close can
    // retry it; dropping it here would make MemoryAllocator::finalize() clear
    // the last ownership record and turn the retry into a false success. A
    // A program context erases regardless: it resets its device below, and an
    // entry that outlived that reset would answer the next generation's dedup
    // lookup with an address from the one that just ended.
    if (!abandon_device_resources) {
        for (auto it = chip_callable_buffers_.begin(); it != chip_callable_buffers_.end();) {
            const int free_rc = mem_alloc_.free(reinterpret_cast<void *>(it->second.chip_dev));
            if (free_rc != 0) {
                capture(free_rc);
                if (retains_failed_callable_release()) {
                    kernel_cleanup_failed = true;
                    ++it;
                    continue;
                }
                LOG_ERROR(
                    "finalize: free of chip_dev=0x%lx failed: %d — clearing callable bookkeeping before device reset",
                    it->second.chip_dev, free_rc
                );
                it = chip_callable_buffers_.erase(it);
                continue;
            }
            LOG_DEBUG(
                "Freed chip callable buffer: chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev,
                it->second.total_size, it->first
            );
            it = chip_callable_buffers_.erase(it);
        }
    } else {
        chip_callable_buffers_.clear();
    }

    // hbg path: dlclose any host orch handles callers forgot to unregister.
    // finalize() is the last chance; Worker.close() does not auto-unregister
    // each callable_id, so without this loop the host process leaks one
    // dlopen handle per (re)created Worker — observable in long-running
    // pytest sessions.
    for (auto &kv : callables_) {
        if (kv.second.host_dlopen_handle != nullptr) {
            dlclose(kv.second.host_dlopen_handle);
        }
    }
    callables_.clear();
    aicpu_seen_callable_ids_.clear();
    aicpu_dlopen_total_ = 0;

    // Release the three per-Worker pooled arenas (GM heap, shared memory, optional
    // trb prebuilt runtime arena — each its own device_malloc). Must precede
    // mem_alloc_.finalize() so the arenas free through the still-live
    // allocator, not after it.
    for (auto &bank : arena_banks_) {
        if (abandon_device_resources) {
            bank->gm_heap.abandon_after_device_failure();
            bank->gm_sm.abandon_after_device_failure();
            bank->runtime_pool.abandon_after_device_failure();
        } else {
            bank->gm_heap.release();
            bank->gm_sm.release();
            bank->runtime_pool.release();
        }
    }
    prebuilt_runtime_arena_cache_.invalidate();

    if (abandon_device_resources) {
        abandon_graph_definition_blocks();
        retained_temp_addrs_.fill(nullptr);
        retained_temp_sizes_.fill(0);
        // Forget the mappings without unregistering: the reset invalidated them
        // and the call would be a further device operation.
        (void)child_memory_host_views_.take_all();
    } else {
        release_graph_definition_blocks();
        clear_temporary_buffer();
    }
    // Pure host memory, so both are returned on either path — a force reset
    // invalidated device allocations, not these pages.
    release_sm_mirrors();
    release_run_image_stagings();

    // Free each slot's device-phase/task-timing buffer (allocated lazily in
    // run()) while mem_alloc_ and the device context are still live.
    // free_tensor() routes through mem_alloc_.free(), so it must run before
    // mem_alloc_.finalize() and before the subclass's `rtDeviceReset()` tears
    // down the device runtime.
    for (void *&slot_ptr : device_wall_dev_ptrs_) {
        if (slot_ptr == nullptr) continue;
        if (!abandon_device_resources) {
            free_tensor(slot_ptr);
        }
        slot_ptr = nullptr;
    }
    device_timing_armed_.fill(false);

    // Same ordering constraint as the timing buffers above: free while
    // mem_alloc_ and the device context are still live.
    for (void *&slot_ptr : device_run_result_dev_ptrs_) {
        if (slot_ptr == nullptr) continue;
        if (!abandon_device_resources) {
            free_tensor(slot_ptr);
        }
        slot_ptr = nullptr;
    }
    device_run_results_.fill(DeviceRunResultRegion{});
    device_run_result_initialized_.fill(false);
    device_run_result_reads_.reset();

    // The AICore register-address tables are device constants committed once per
    // device context, so this is where they are returned — same window and same
    // ordering constraint as the device-wall buffers above. Release keys on the
    // address, not the committed flag: a table whose host-to-device copy failed
    // is still owned and still has to be freed. Both fields are cleared on both
    // paths, so a re-provisioned runner commits again for its new device
    // generation, and on the fatal path the allocator has already forgotten the
    // block. An address whose free fails is retained for a later retry, matching
    // the slot blocks below.
    auto release_reg_table = [&](uint64_t &table, bool &committed) {
        if (table == 0) {
            committed = false;
            return;
        }
        if (abandon_device_resources) {
            table = 0;
        } else if (mem_alloc_.free(reinterpret_cast<void *>(table)) == 0) {
            table = 0;
        } else {
            capture(PTO_RUNTIME_ERR_INTERNAL);
        }
        committed = false;
    };
    release_reg_table(aicore_ctrl_reg_table_dev_, aicore_ctrl_reg_table_committed_);
    release_reg_table(aicore_pmu_reg_table_dev_, aicore_pmu_reg_table_committed_);

    // Each slot's KernelArgs / runtime blocks outlive the runs that
    // use them, so this is where they are returned — same reason and same
    // ordering constraint as the device-wall buffer above. A failing free keeps
    // its block recorded, so reporting the error is what lets a caller retry
    // reach it.
    for (SlotPersistentArgs &slot : slot_persistent_args_) {
        if (abandon_device_resources) {
            abandon_slot_persistent_args(slot);
        } else {
            const int slot_rc = release_slot_persistent_args(slot, mem_alloc_);
            capture(slot_rc);
            kernel_cleanup_failed = kernel_cleanup_failed || (execution_mode_latch().is_kernel() && slot_rc != 0);
        }
    }

    // Kernel-mode context resources. The argument blocks route through
    // mem_alloc_, so they are released before its finalize below; the streams
    // and events do not, and go after them so a caller that inspects the
    // teardown sees arguments released while their owning context still
    // exists. Both are no-ops on a program-mode context.
    if (abandon_device_resources) {
        persistent_args_.abandon();
    } else {
        const int args_rc = persistent_args_.finalize_once();
        if (args_rc != 0 && rc == 0) rc = args_rc;
        const int close_rc = kernel_exec_state_.close();
        if (close_rc != 0 && rc == 0) rc = close_rc;
        kernel_cleanup_failed =
            kernel_cleanup_failed || (execution_mode_latch().is_kernel() && (args_rc != 0 || close_rc != 0));
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    if (!abandon_device_resources) {
        // A failed persistent release keeps its address for explicit close
        // retry. MemoryAllocator::finalize() deliberately clears its tracking
        // map even when rtFree fails, which would turn that retry into a false
        // success. Preserve the allocator and all remaining runner state until
        // the kernel cleanup has completed.
        if (kernel_cleanup_failed) return rc;
        // The mappings name the allocations mem_alloc_ is about to free, so
        // they cannot be released after it. A force reset already invalidated
        // both, and the unregister would be a further device call.
        if (!workspace_.enabled()) {
            // Unmanaged: the original terminal sequence, unchanged.
            release_child_memory_host_views();
            capture(mem_alloc_.finalize());
        } else {
            // Mappings first, exactly as the unmanaged path does: the releases
            // below hand allocations back to the platform, and a host mapping
            // over one of them would outlive its pages. The ledger keeps the
            // block of any mapping that could not be dropped, and the managed
            // release path unmaps each block it frees, so no ordering here can
            // leave a freed range mapped.
            release_child_memory_host_views();
            // Every block no consumer references goes back the ordinary way
            // first, so the sweep below is left with what could not be proven
            // unused.
            capture(workspace_.release_unreferenced());
            // The sweep holds the ledger's lock for its whole duration, taken
            // before the allocator's — the order every other path uses. Its
            // callbacks work through a view that assumes that lock is held, so
            // they allocate nothing, cannot throw, and cannot leave the
            // allocator's tracking map half-cleared.
            WorkspaceManager::TerminalSweep sweep = workspace_.begin_terminal_sweep();
            capture(mem_alloc_.finalize_except(
                [](void *base, std::size_t /*bytes*/, void *ctx) {
                    return static_cast<WorkspaceManager::TerminalSweep *>(ctx)->must_keep(base) ?
                               MemoryAllocator::SweepAction::KeepIt :
                               MemoryAllocator::SweepAction::FreeIt;
                },
                [](void *base, int rc, MemoryAllocator::SweepAction acted, void *ctx) {
                    static_cast<WorkspaceManager::TerminalSweep *>(ctx)->note_result(
                        base, rc, acted == MemoryAllocator::SweepAction::KeepIt
                    );
                },
                &sweep
            ));
        }
    }

    block_dim_ = 0;
    worker_count_ = 0;
    // Tied to stream_aicore_, destroyed above: a re-provisioned runner
    // re-queries rather than trusting the previous stream's limits.
    max_block_dim_ = 0;
    max_cube_cores_ = 0;
    max_vector_cores_ = 0;
    aicore_kernel_binary_.clear();
    for (auto &bank : arena_banks_) {
        bank->cached_gm_heap_size = 0;
        bank->cached_gm_sm_size = 0;
        bank->cached_runtime_arena_size = 0;
    }
    if (abandon_device_resources) {
        LOG_WARN("Fatal teardown: host-side ownership cleared without further device calls");
    }
    return rc;
}

int DeviceRunnerBase::launch_aicore_kernel(rtStream_t stream, const KernelArgs &k_args) {
    // Lazy-register the AICore binary on first call; reuse cached handle
    // thereafter. CANN has no public rtUnregisterAllKernel, so re-registering
    // every run would pin another device-side copy of the ELF and quickly
    // exhaust HBM — surfaced in CI as 207001 at rtKernelLaunchWithHandleV2
    // with a 507899 cascade at rtStreamCreate.
    if (aicore_bin_handle_ == nullptr) {
        if (aicore_kernel_binary_.empty()) {
            LOG_ERROR("AICore kernel binary is empty");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        rtDevBinary_t binary;
        std::memset(&binary, 0, sizeof(binary));
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.version = 0;
        binary.data = aicore_kernel_binary_.data();
        binary.length = aicore_kernel_binary_.size();
        int rc = rtRegisterAllKernel(&binary, &aicore_bin_handle_);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtRegisterAllKernel failed: %d", rc);
            ACL_LOG_ERROR_DETAIL(rc);
            aicore_bin_handle_ = nullptr;
            return rc;
        }
    }

    // `AicoreLaunchArgs` is the host-side image of the entry's parameter list
    // and is defined per-arch beside the entry's ABI, in common/kernel_args.h.
    // The driver copies `argsSize` bytes during the launch call, so a stack
    // local satisfies its host-buffer lifetime requirement.
    AicoreLaunchArgs args{};
    fill_shared_launch_args(args, k_args);
    fill_arch_launch_args(args, k_args);
    rtArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    int rc = rtKernelLaunchWithHandleV2(aicore_bin_handle_, 0, block_dim_, &rt_args, nullptr, stream, &cfg);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtKernelLaunchWithHandleV2 failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    return rc;
}

// =============================================================================
// run() sub-sequence helpers — head + tail chunks shared by both arches
// =============================================================================

int DeviceRunnerBase::validate_launch_aicpu_num(int launch_aicpu_num) {
    if (launch_aicpu_num == 1 || launch_aicpu_num < 0 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "launch_aicpu_num (%d) must be 0 (auto) or in range [2, %d]", launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return 0;
}

int DeviceRunnerBase::resolve_aicpu_thread_num(int requested, int usable, int arch_default) {
    if (usable < 2) {
        LOG_ERROR("AICPU usable count %d < 2 (need >=1 orchestrator + >=1 scheduler)", usable);
        return -1;
    }
    int desired = (requested > 0) ? requested : arch_default;
    int total = std::min(desired, usable);
    if (total < desired) {
        LOG_WARN(
            "AICPU: requested %d active threads, only %d usable on this die — running 1 orch + %d sched", desired,
            usable, total - 1
        );
    }
    return total;
}

const DeviceRunnerBase::DeviceRunTiming &DeviceRunnerBase::device_run_timing(uint32_t pipeline_slot) const {
    static const DeviceRunTiming kEmpty{};
    if (pipeline_slot >= device_run_timing_.size()) return kEmpty;
    return device_run_timing_[pipeline_slot];
}

void DeviceRunnerBase::release_device_run_timing(uint32_t pipeline_slot) {
    if (pipeline_slot >= device_timing_armed_.size()) return;
    device_timing_armed_[pipeline_slot] = false;
}

const uint8_t *
DeviceRunnerBase::device_run_result(uint32_t pipeline_slot, uint64_t run_epoch, size_t *bytes_out) const {
    if (bytes_out != nullptr) *bytes_out = 0;
    if (pipeline_slot >= device_run_results_.size()) return nullptr;
    const DeviceRunResultRegion &region = device_run_results_[pipeline_slot];
    if (!device_run_result_published(region, run_epoch)) return nullptr;
    if (bytes_out != nullptr) *bytes_out = region.payload_bytes;
    return region.payload;
}

int DeviceRunnerBase::ensure_device_run_result_region(
    uint32_t pipeline_slot, uint64_t run_epoch, KernelArgsHelper &kernel_args
) {
    kernel_args.args.run_result_data_base = 0;
    kernel_args.args.run_result_epoch = 0;
    if (pipeline_slot >= device_run_result_dev_ptrs_.size()) {
        LOG_ERROR("run-result region: pipeline slot %u is out of range", pipeline_slot);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // Epoch 0 would be indistinguishable from never-written device memory, so a
    // run without one cannot be given a region it could later mis-read.
    if (run_epoch == 0) {
        LOG_ERROR("run-result region: run epoch 0 cannot be published");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    void *&slot_ptr = device_run_result_dev_ptrs_[pipeline_slot];
    if (slot_ptr == nullptr) {
        slot_ptr = allocate_tensor(device_run_result_bytes());
        device_run_result_initialized_[pipeline_slot] = false;
    }
    if (slot_ptr == nullptr) {
        // Failing prepare is the point. Launching anyway would run a device side
        // with nowhere to put its result, and leave the host to read a region
        // whose contents belong to nobody.
        LOG_ERROR("run-result region: allocation failed for slot %u", pipeline_slot);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // A fresh allocation has to be zeroed once. `allocate_tensor` is an
    // `rtMalloc`: the bytes it returns are whatever the device left there, so
    // `published` cannot be assumed to differ from the epoch this run is about
    // to look for. Only the first use of an allocation pays this — steady-state
    // reuse is distinguished by epoch and needs no per-run H2D.
    if (!device_run_result_initialized_[pipeline_slot]) {
        const uint64_t unpublished = 0;
        if (copy_to_device(slot_ptr, &unpublished, sizeof(unpublished)) != 0) {
            // Publish no base: a region whose `published` is still unknown could
            // read back as this run's own epoch. The allocation stays so the
            // next prepare on this slot retries the initialization.
            LOG_ERROR("run-result region: initial clear failed for slot %u", pipeline_slot);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        device_run_result_initialized_[pipeline_slot] = true;
    }
    kernel_args.args.run_result_data_base = reinterpret_cast<uint64_t>(slot_ptr);
    kernel_args.args.run_result_epoch = run_epoch;
    return 0;
}

// The retained bank is indexed by the run's pipeline slot directly, not by a
// slot-derived modulus: the slot space and the bank array are the same size, so
// a slot outside the array is a contract break to report rather than to fold.
static_assert(
    PLATFORM_RUN_TERMINAL_BANKS == PTO_PIPELINE_MAX_DEPTH,
    "swimlane terminal banks must cover exactly the pipeline's retained runs"
);

uint64_t DeviceRunnerBase::arm_chip_swimlane_run_terminal_bank(uint32_t pipeline_slot, uint64_t run_epoch) {
    // Zero means "publish no snapshot". Every path that cannot resolve a bank —
    // swimlane off, collector not initialized, slot out of range, no run identity
    // — returns it rather than letting the device derive an address.
    return reinterpret_cast<uint64_t>(chip_swimlane_collector_.arm_run_terminal_bank(pipeline_slot, run_epoch));
}

int DeviceRunnerBase::read_device_run_result(uint32_t pipeline_slot, uint64_t run_epoch) {
    if (pipeline_slot >= device_run_results_.size()) return 0;
    // The region is this slot's, and the slot is not handed to another run until
    // the run holding it finalizes, so this read races nothing. What makes the
    // record this run's rather than a successor's is that its device side wrote
    // and published it before its kernel returned.
    return device_run_result_reads_.read_with_status(
        pipeline_slot, run_epoch, device_run_results_[pipeline_slot], device_run_result_dev_ptrs_[pipeline_slot],
        [](void *dst, const void *src) {
            // The copy's own status is the caller's: this is an SDK call on the
            // run's drain path, and a code it reports is an error this thread
            // has observed. Returning it rather than a bool is what keeps a
            // later zero from standing in for it.
            int rc = rtMemcpy(
                dst, sizeof(DeviceRunResultRegion), src, sizeof(DeviceRunResultRegion), RT_MEMCPY_DEVICE_TO_HOST
            );
            if (rc != 0) {
                LOG_WARN("rtMemcpy(run_result) D2H failed: %d", rc);
                ACL_LOG_ERROR_DETAIL(rc);
            }
            return rc;
        }
    );
}

RunRecordRead DeviceRunnerBase::device_run_result_read_status(uint32_t pipeline_slot, uint64_t run_epoch) const {
    if (pipeline_slot >= device_run_results_.size()) return RunRecordRead::NotAttempted;
    return device_run_result_reads_.state(pipeline_slot, run_epoch);
}

RunCompletionFence::Completion DeviceRunnerBase::observed_run_boundaries(const NativeRunIdentity &identity) const {
    return run_boundaries_observed_.observed(identity);
}

DeviceRunTerminal DeviceRunnerBase::device_run_terminal(uint32_t pipeline_slot, uint64_t run_epoch) const {
    DeviceRunTerminal undecided;
    if (run_epoch == 0) {
        undecided.reason = "run has no epoch";
        return undecided;
    }
    if (pipeline_slot >= device_run_results_.size()) {
        undecided.reason = "pipeline slot out of range";
        return undecided;
    }
    switch (device_run_result_reads_.state(pipeline_slot, run_epoch)) {
    case RunRecordRead::NotAttempted:
        undecided.reason = "no read taken for this run";
        return undecided;
    case RunRecordRead::Failed:
        undecided.reason = "result read-back failed";
        return undecided;
    case RunRecordRead::Ok:
        break;
    }
    return device_run_result_terminal(device_run_results_[pipeline_slot], run_epoch);
}

void DeviceRunnerBase::ensure_device_wall_buffer(uint32_t pipeline_slot, KernelArgsHelper &kernel_args) {
    if (!device_phase_capture_enabled() || pipeline_slot >= device_wall_dev_ptrs_.size()) {
        // A null base makes the AICPU stamping helpers no-op.
        kernel_args.args.device_wall_data_base = 0;
        return;
    }
    // Fixed header followed by per-thread AICPU phase records (thread-major:
    // AicpuPhaseRecord[NUM_AICPU_PHASES] per launched AICPU thread). Slot
    // AicpuPhase::RunWall keeps the original whole-run wall; the rest subdivide
    // the on-NPU portion. Each surviving AICPU thread writes its own records
    // (plain stores, no atomics); read_device_wall_ns() reduces RunWall as
    // max(end) - min(start) and surfaces the other phases as trace markers.
    // Each pipeline slot owns its own buffer, allocated lazily and reset every
    // run, so the device's writes for one run cannot land in storage whose
    // result another run has not read yet.
    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    using BufferImage = DevicePhaseBufferStorage<kThreads>;
    constexpr size_t kBytes = device_phase_buffer_bytes(kThreads);
    static_assert(sizeof(BufferImage) == kBytes, "device-phase buffer layout drift");
    void *&slot_ptr = device_wall_dev_ptrs_[pipeline_slot];
    if (slot_ptr == nullptr) {
        slot_ptr = allocate_tensor(kBytes);
    }
    if (slot_ptr != nullptr) {
        kernel_args.args.device_wall_data_base = reinterpret_cast<uint64_t>(slot_ptr);
    }
}

int DeviceRunnerBase::arm_device_wall_buffer(uint32_t pipeline_slot, KernelArgsHelper &kernel_args) {
    if (pipeline_slot >= device_wall_dev_ptrs_.size()) return 0;
    void *slot_ptr = device_wall_dev_ptrs_[pipeline_slot];
    if (slot_ptr == nullptr || kernel_args.args.device_wall_data_base == 0) return 0;
    if (device_timing_armed_[pipeline_slot]) {
        // The slot's previous result was never consumed, so its owner's finalize
        // did not run. Arming anyway would overwrite an unread result; report it
        // and leave this run without capture rather than corrupt both.
        LOG_WARN(
            "device_phase slot %u still holds an unconsumed result; disabling phase capture this run", pipeline_slot
        );
        kernel_args.args.device_wall_data_base = 0;
        return 0;
    }
    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    using BufferImage = DevicePhaseBufferStorage<kThreads>;
    static const BufferImage init = [] {
        BufferImage image{};
        reset_device_phase_buffer(&image, kThreads);
        return image;
    }();
    if (copy_to_device(slot_ptr, &init, sizeof(init)) != 0) {
        // Reset failed — disable capture for this run so stale slot data
        // can't leak into the reduction. Keep the slot's allocation alive:
        // the next run on this slot retries the reset.
        LOG_WARN("device_phase reset H2D failed; disabling phase capture this run");
        kernel_args.args.device_wall_data_base = 0;
        return 0;
    }
    device_timing_armed_[pipeline_slot] = true;
    return 0;
}

int DeviceRunnerBase::resolve_block_dim() {
    if (max_block_dim_ < 1) {
        LOG_ERROR(
            "block_dim ceiling not resolved (cube=%u, vector=%u); ensure_device_initialized must run first",
            max_cube_cores_, max_vector_cores_
        );
        return -1;
    }
    LOG_INFO("block_dim resolved to %d (cube=%u, vector=%u)", max_block_dim_, max_cube_cores_, max_vector_cores_);
    return max_block_dim_;
}

int DeviceRunnerBase::prepare_launch_shape(Runtime &runtime, const CallConfig &config) {
    if (validate_launch_aicpu_num(config.aicpu_thread_num) != 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    int block_dim = resolve_block_dim();
    if (block_dim < 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    int num_aicore = block_dim * cores_per_blockdim_;
    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("block_dim (%d) exceeds RUNTIME_MAX_WORKER (%d)", block_dim, RUNTIME_MAX_WORKER);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    runtime.set_worker_count(num_aicore);
    runtime.set_aicpu_thread_num(config.aicpu_thread_num);

    // First `block_dim` cores are AIC; remaining ~2/3 are AIV. The rule is host
    // state: every consumer of it runs on the host and needs it before any core
    // has reported. `dev.workers[]` is not written here — it sits outside the
    // per-run uploaded prefix, the device owns every word of it once the block
    // has been initialized, and a host store would only be published by
    // accident on a block's first publication.
    runtime.set_core_type_rule(num_aicore, block_dim);
    return 0;
}

void DeviceRunnerBase::activate_launch_shape(const Runtime &runtime) {
    worker_count_ = runtime.get_worker_count();
    block_dim_ = worker_count_ / cores_per_blockdim_;
}

int DeviceRunnerBase::sync_stream_pair(rtStream_t aicpu_stream, rtStream_t aicore_stream) {
    LOG_INFO("=== aclrtSynchronizeStreamWithTimeout AICPU stream ===");
    int rc = aclrtSynchronizeStreamWithTimeout(aicpu_stream, timeout_config_.stream_sync_timeout_ms);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICPU timeout_ms=%d device_id=%d block_dim=%d",
            timeout_config_.stream_sync_timeout_ms, device_id_, block_dim_
        );
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICPU) failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    LOG_INFO("=== aclrtSynchronizeStreamWithTimeout AICore stream ===");
    rc = aclrtSynchronizeStreamWithTimeout(aicore_stream, timeout_config_.stream_sync_timeout_ms);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICore timeout_ms=%d device_id=%d block_dim=%d",
            timeout_config_.stream_sync_timeout_ms, device_id_, block_dim_
        );
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICore) failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    return 0;
}

namespace {

const char *stream_role_name(RunCompletionFence::StreamRole role) {
    return role == RunCompletionFence::StreamRole::Aicore ? "AICore" : "AICPU";
}

}  // namespace

int DeviceRunnerBase::arm_run_fence(const PreparedExecution &prepared) {
    int rc = run_fence(prepared.pipeline_slot).arm(prepared.identity);
    if (rc != 0) {
        LOG_ERROR("arm_run_fence: slot %u could not take a completion fence: %d", prepared.pipeline_slot, rc);
    }
    return rc;
}

int DeviceRunnerBase::record_run_boundary(
    const PreparedExecution &prepared, RunCompletionFence::StreamRole role, rtStream_t stream
) {
    RunCompletionFence &fence = run_fence(prepared.pipeline_slot);
    // The one point where this stream is certainly live and certainly carrying a
    // run, so it is where its driver id is captured for the fault channel's
    // filter. A number survives both the stream's replacement and the device
    // reset that would invalidate the handle; asking the handle later does not.
    {
        int32_t stream_id = -1;
        if (aclrtStreamGetId(static_cast<aclrtStream>(stream), &stream_id) == ACL_SUCCESS) {
            run_stream_ids_.note(stream_id);
        } else {
            // A stream this run submits on whose id is unknown. Skipping it
            // silently would leave the history claiming to be whole while
            // missing exactly the entry a later notice on this stream would
            // carry, and that notice would then read as another runner's.
            LOG_WARN(
                "aclrtStreamGetId failed for the %s stream of slot %u; the fault channel's stream history is "
                "incomplete from here, so an unmatched notice reads as undecided rather than as another runner's",
                stream_role_name(role), prepared.pipeline_slot
            );
            run_stream_ids_.note_unidentified_stream();
        }
    }
    // The submission is a fact the instant the device queue accepted it, and it
    // has to be recorded before anything that can still fail — otherwise a
    // failing record below would leave the run looking unsubmitted.
    fence.note_kernel_submitted(prepared.identity, role);
    int rc = fence.record(prepared.identity, role, stream);
    if (rc != 0) {
        LOG_ERROR(
            "record_run_boundary: %s boundary of slot %u was not recorded: %d; this run holds submitted work no "
            "fence covers",
            stream_role_name(role), prepared.pipeline_slot, rc
        );
    }
    return rc;
}

int DeviceRunnerBase::poll_run_fence(
    const PreparedExecution &prepared, rtStream_t aicpu_stream, rtStream_t aicore_stream
) {
    const RunCompletionFence::Completion completion =
        poll_and_retain_run_boundaries(run_fence(prepared.pipeline_slot), run_boundaries_observed_, prepared.identity);
    switch (completion) {
    case RunCompletionFence::Completion::Complete:
        // Boundaries prove the kernels exited; the streams carry the device's
        // verdict on them, which a completed run still has to be asked for.
        return query_stream_pair_error(aicpu_stream, aicore_stream) == 0 ? SIMPLER_NATIVE_RUN_POLL_COMPLETE :
                                                                           SIMPLER_NATIVE_RUN_POLL_ERROR;
    case RunCompletionFence::Completion::Pending:
        return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
    case RunCompletionFence::Completion::Unfenced:
        return query_stream_pair_nonblocking(aicpu_stream, aicore_stream);
    case RunCompletionFence::Completion::Error:
        break;
    }
    return SIMPLER_NATIVE_RUN_POLL_ERROR;
}

int DeviceRunnerBase::wait_run_fence(
    const PreparedExecution &prepared, rtStream_t aicpu_stream, rtStream_t aicore_stream
) {
    const RunBoundaryWait observed = wait_and_retain_run_boundaries(
        run_fence(prepared.pipeline_slot), run_boundaries_observed_, prepared.identity,
        timeout_config_.stream_sync_timeout_ms
    );
    if (observed.completion == RunCompletionFence::Completion::Unfenced) {
        LOG_WARN(
            "wait_run_fence: slot %u holds submitted work no boundary covers; falling back to the bounded "
            "whole-stream wait",
            prepared.pipeline_slot
        );
        return sync_stream_pair(aicpu_stream, aicore_stream);
    }

    LOG_INFO("=== aclrtSynchronizeEventWithTimeout run completion boundaries ===");
    int rc = observed.rc;
    if (rc == ACL_ERROR_RT_EVENT_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Run fence wait timeout: timeout_ms=%d device_id=%d block_dim=%d slot=%u",
            timeout_config_.stream_sync_timeout_ms, device_id_, block_dim_, prepared.pipeline_slot
        );
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeEventWithTimeout (run completion boundary) failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    // Completion is settled above. What decides the run is this run's own
    // evidence: the boundaries just observed plus the record its device side
    // published before its kernel returned. The read is taken here, ahead of
    // any verdict, and it is the run's one read — finalize's later call for
    // the same epoch reuses these bytes.
    const int transfer_rc = read_device_run_result(prepared.pipeline_slot, prepared.identity.run_epoch);
    RunOutcomeEvidence evidence;
    evidence.boundaries = observed.completion;
    evidence.record_read = device_run_result_read_status(prepared.pipeline_slot, prepared.identity.run_epoch);
    evidence.terminal = device_run_terminal(prepared.pipeline_slot, prepared.identity.run_epoch);

    switch (decide_run_drain(transfer_rc, evidence)) {
    case RunDrainAction::ReportTransferError: {
        // An SDK error this thread has already observed. The synchronize still
        // runs so the pair converges and the health path sees a code, but the
        // observed error is what the run returns: a later zero does not annul
        // it, and a later non-zero does not replace it.
        const int sync_rc = sync_stream_pair(aicpu_stream, aicore_stream);
        LOG_ERROR(
            "Run result transfer failed: %d (device_id=%d block_dim=%d slot=%u); the converging stream "
            "synchronize returned %d and does not replace it",
            transfer_rc, device_id_, block_dim_, prepared.pipeline_slot, sync_rc
        );
        return transfer_rc;
    }
    case RunDrainAction::AcceptRecordedSuccess:
        // Both this run's boundaries completed, its transfer reported nothing,
        // and the record it published for this exact identity says Ok. That is
        // what this branch asserts — not that the device raised no exception
        // for anything else on the stream. A fault no participant recorded can
        // still reach the caller only through a later API call or the health
        // channel, which is the accepted cost of not waiting here for runs
        // queued behind this one. See docs/design/run-completion-fence.md.
        return 0;
    case RunDrainAction::Synchronize:
        break;
    }

    // Every other shape keeps the device's verdict, and on this SDK a stream
    // synchronize is the only call measured to produce one. Measured on a2a3
    // for a run whose AICPU kernel returned a fatal status: both boundaries
    // complete (so the kernels did exit), rtStreamQuery reports both streams
    // drained and error-free, aclrtPeekAtLastError reports nothing, and only
    // aclrtSynchronizeStreamWithTimeout surfaces the 507018 — after which peek
    // reports it too. A zero timeout is rejected outright (107000), so there is
    // no non-blocking form of the same check. See
    // docs/design/run-completion-fence.md.
    rc = sync_stream_pair(aicpu_stream, aicore_stream);
    if (rc != 0) {
        LOG_ERROR(
            "Run completed its boundaries but the device reports an error executing it: %d (device_id=%d "
            "block_dim=%d slot=%u)",
            rc, device_id_, block_dim_, prepared.pipeline_slot
        );
    }
    return rc;
}

void DeviceRunnerBase::retire_run_fence(const PreparedExecution &prepared) noexcept {
    if (run_fence(prepared.pipeline_slot).retire(prepared.identity) != 0) {
        LOG_ERROR(
            "retire_run_fence: slot %u keeps its fence — a queued cross-run wait still names a boundary of it",
            prepared.pipeline_slot
        );
    }
}

size_t DeviceRunnerBase::queued_boundary_wait_count() const { return queued_waits_->live_count(); }

int DeviceRunnerBase::predecessor_boundary_unfired(const NativeRunJoin &join, bool *unfired) const {
    if (unfired == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    *unfired = false;
    const uint32_t slot = join.predecessor_identity.pipeline_slot;
    if (join.predecessor_owner == nullptr || slot >= PTO_PIPELINE_MAX_DEPTH) return PTO_RUNTIME_ERR_INTERNAL;
    bool complete = false;
    const int rc =
        run_fences_[slot]->query_boundary(join.predecessor_identity, RunCompletionFence::StreamRole::Aicpu, &complete);
    if (rc != 0) return rc;
    *unfired = !complete;
    return 0;
}

int DeviceRunnerBase::mark_run_boundary(
    RunBoundaryMarks::Position position, const NativeRunIdentity &identity, void *stream
) {
    return boundary_marks_->record(position, identity, stream);
}

RunBoundaryMarks::Mark
DeviceRunnerBase::run_boundary_mark(RunBoundaryMarks::Position position, const NativeRunIdentity &identity) {
    return boundary_marks_->read(position, identity, timeout_config_.stream_sync_timeout_ms);
}

DeviceRunnerBase::JoinedLaunchRecord
DeviceRunnerBase::note_joined_launch(const NativeRunJoin &join, const NativeRunIdentity &successor) {
    JoinedLaunchRecord record;
    record.successor = successor;
    record.predecessor = join.predecessor_identity;
    bool unfired = false;
    record.query_rc = predecessor_boundary_unfired(join, &unfired);
    record.observed = record.query_rc == 0;
    record.predecessor_unfired = record.observed && unfired;
    // A query that could not answer is an anomaly an operator wants to see
    // without parsing the trace: the successor is correctly ordered either way,
    // but nothing is known about when. The favourable case is not a warning and
    // is carried by the span the caller emits.
    if (!record.observed) {
        LOG_WARN(
            "joined launch: the identified predecessor's whole-operator boundary could not be read: %d "
            "(successor_epoch=%llu predecessor_epoch=%llu)",
            record.query_rc, static_cast<unsigned long long>(successor.run_epoch),
            static_cast<unsigned long long>(join.predecessor_identity.run_epoch)
        );
    }
    return record;
}

int DeviceRunnerBase::open_own_boundary_wait(const PreparedExecution &prepared, void **core_done_out) {
    int rc = queued_waits_->open(
        QueuedStreamWaits::Shape::Own, prepared.identity, prepared.identity, run_fence(prepared.pipeline_slot),
        RunCompletionFence::StreamRole::Aicore, RunCompletionFence::StreamRole::Aicpu, core_done_out
    );
    if (rc != 0) {
        LOG_ERROR(
            "open_own_boundary_wait: slot %u cannot reference its own AICore boundary: %d", prepared.pipeline_slot, rc
        );
    }
    return rc;
}

int DeviceRunnerBase::open_cross_run_wait(const PreparedExecution &prepared, void **predecessor_boundary_out) {
    const NativeRunJoin &join = prepared.join;
    if (join.predecessor_owner == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    const uint32_t predecessor_slot = join.predecessor_identity.pipeline_slot;
    if (predecessor_slot >= PTO_PIPELINE_MAX_DEPTH || predecessor_slot == prepared.pipeline_slot) {
        LOG_ERROR(
            "open_cross_run_wait: slot %u was joined to slot %u, which is not another live pipeline slot",
            prepared.pipeline_slot, predecessor_slot
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (!has_whole_operator_boundary(join.predecessor_identity)) {
        LOG_ERROR(
            "open_cross_run_wait: slot %u was joined to slot %u, whose boundary does not cover its whole operator",
            prepared.pipeline_slot, predecessor_slot
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    int rc = queued_waits_->open(
        QueuedStreamWaits::Shape::CrossRun, prepared.identity, join.predecessor_identity, run_fence(predecessor_slot),
        RunCompletionFence::StreamRole::Aicpu, RunCompletionFence::StreamRole::Aicore, predecessor_boundary_out
    );
    if (rc != 0) {
        LOG_ERROR(
            "open_cross_run_wait: slot %u cannot reference the whole-operator boundary of slot %u: %d",
            prepared.pipeline_slot, predecessor_slot, rc
        );
    }
    return rc;
}

void DeviceRunnerBase::note_whole_operator_boundary(const PreparedExecution &prepared) {
    std::lock_guard<std::mutex> lk(whole_operator_mu_);
    whole_operator_boundaries_[prepared.pipeline_slot] = prepared.identity;
}

void DeviceRunnerBase::clear_whole_operator_boundary(const PreparedExecution &prepared) noexcept {
    std::lock_guard<std::mutex> lk(whole_operator_mu_);
    NativeRunIdentity &published = whole_operator_boundaries_[prepared.pipeline_slot];
    if (published == prepared.identity) published = NativeRunIdentity{};
}

bool DeviceRunnerBase::has_whole_operator_boundary(const NativeRunIdentity &identity) const {
    if (identity.pipeline_slot >= PTO_PIPELINE_MAX_DEPTH || identity.run_epoch == 0) return false;
    std::lock_guard<std::mutex> lk(whole_operator_mu_);
    return whole_operator_boundaries_[identity.pipeline_slot] == identity;
}

int DeviceRunnerBase::commit_boundary_wait(QueuedStreamWaits::Shape shape, const PreparedExecution &prepared) {
    int rc = queued_waits_->commit(shape, prepared.identity);
    if (rc != 0) {
        LOG_ERROR("commit_boundary_wait: slot %u could not commit its queued wait: %d", prepared.pipeline_slot, rc);
    }
    return rc;
}

int DeviceRunnerBase::revoke_boundary_wait(QueuedStreamWaits::Shape shape, const PreparedExecution &prepared) {
    int rc = queued_waits_->revoke(shape, prepared.identity);
    if (rc != 0) {
        LOG_ERROR("revoke_boundary_wait: slot %u could not drop its reservation: %d", prepared.pipeline_slot, rc);
    }
    return rc;
}

int DeviceRunnerBase::record_cross_run_proof(const PreparedExecution &prepared, rtStream_t waiter_stream) {
    int rc = queued_waits_->record_proof(prepared.identity, static_cast<void *>(waiter_stream));
    if (rc != 0) {
        LOG_ERROR(
            "record_cross_run_proof: slot %u queued its cross-run wait but could not record the event that proves "
            "the wait consumed: %d; the reference now needs a quiescence proof to retire",
            prepared.pipeline_slot, rc
        );
    }
    return rc;
}

int DeviceRunnerBase::discharge_boundary_waits(
    const PreparedExecution &prepared, bool boundaries_complete, rtStream_t aicpu_stream, rtStream_t aicore_stream
) {
    if (!queued_waits_->holds_reference_to(prepared.identity)) return 0;

    int first_rc =
        queued_waits_->discharge(prepared.identity, boundaries_complete, timeout_config_.stream_sync_timeout_ms);
    if (!queued_waits_->holds_reference_to(prepared.identity)) return first_rc;

    // Nothing cheaper is left. A pair synchronize covers the boundary and
    // everything queued behind it, which is exactly what an unproven queued
    // wait needs — and it is a failure-path cost only, because a successful
    // drain has already retired both rungs above.
    LOG_WARN(
        "discharge_boundary_waits: slot %u still holds a queued wait on its own boundary; falling back to the "
        "bounded whole-stream synchronize",
        prepared.pipeline_slot
    );
    const int sync_rc = sync_stream_pair(aicpu_stream, aicore_stream);
    if (sync_rc == 0) {
        const int quiesce_rc = queued_waits_->discharge_on_quiescence(prepared.identity);
        if (first_rc == 0) first_rc = quiesce_rc;
        if (!queued_waits_->holds_reference_to(prepared.identity)) return first_rc;
    } else if (first_rc == 0) {
        first_rc = sync_rc;
    }

    // No proof was obtained, so a queued wait may still name an event of this
    // run. Nothing may be released against it; the device generation itself has
    // to end, which is what invalidates the reference.
    LOG_ERROR(
        "discharge_boundary_waits: slot %u could not prove its queued waits consumed (synchronize returned %d); "
        "the device is recovered or marked unusable rather than releasing an event the device may still name",
        prepared.pipeline_slot, sync_rc
    );
    if (first_rc == 0) first_rc = PTO_RUNTIME_ERR_INTERNAL;
    recover_device_or_mark_unusable(first_rc);
    return first_rc;
}

void DeviceRunnerBase::discharge_boundary_waits_noexcept(
    const PreparedExecution &prepared, bool boundaries_complete, rtStream_t aicpu_stream, rtStream_t aicore_stream
) noexcept {
    try {
        (void)discharge_boundary_waits(prepared, boundaries_complete, aicpu_stream, aicore_stream);
    } catch (...) {
        LOG_ERROR("discharge_boundary_waits threw for slot %u", prepared.pipeline_slot);
        recover_device_or_mark_unusable(PTO_RUNTIME_ERR_INTERNAL);
    }
}

DeviceFaultMonitor *DeviceRunnerBase::fault_monitor_if_held() noexcept {
    if (!fault_monitor_held_) return nullptr;
    if (fault_monitor_pid_ != static_cast<long>(getpid())) {
        // Inherited across a fork. The reference belongs to the parent, and the
        // monitor has already reset itself for this process, so both the hold
        // and the read position are meaningless here: dropping them is what
        // stops this runner from releasing a reference it never took, and from
        // reading a stream that now starts behind its cursor.
        fault_monitor_held_ = false;
        fault_monitor_pid_ = -1;
        fault_notices_ = DeviceFaultNoticeCursor{};
        return nullptr;
    }
    return device_fault_monitor();
}

int DeviceRunnerBase::acquire_device_fault_monitor() {
    if (fault_monitor_if_held() != nullptr) return 0;
    DeviceFaultMonitor *monitor = device_fault_monitor();
    if (monitor == nullptr) {
        // Nobody bound the process's monitor into this module — a host runtime
        // opened directly rather than through a loader. There is nothing to
        // listen on, and nothing depends on this channel.
        return 0;
    }
    const int rc = monitor->acquire();
    if (rc != 0) {
        LOG_WARN("device fault monitor: could not install the process callback: %d", rc);
        return rc;
    }
    fault_monitor_held_ = true;
    fault_monitor_pid_ = static_cast<long>(getpid());
    // Whatever this process reported before this runner existed is not this
    // runner's to report, so start from where the stream already stands.
    fault_notices_.skip_to_current(*monitor);
    return 0;
}

void DeviceRunnerBase::release_device_fault_monitor() noexcept {
    DeviceFaultMonitor *monitor = fault_monitor_if_held();
    if (monitor == nullptr) return;
    fault_monitor_held_ = false;
    fault_monitor_pid_ = -1;
    monitor->release();
}

int DeviceRunnerBase::retire_device_generation_after_confirmed_reset() noexcept {
    // `fault_monitor_if_held()` answering null is not a reason to skip the local
    // retirement — see `retire_after_confirmed_device_reset`, which keeps that
    // half unconditional.
    const DeviceGenerationRetirement retirement =
        retire_after_confirmed_device_reset(device_health_, run_stream_ids_, fault_monitor_if_held(), fault_notices_);
    if (retirement.monitor_reinstalled && retirement.monitor_reinstall_rc != 0) {
        LOG_WARN("device fault monitor: re-install after device reset failed: %d", retirement.monitor_reinstall_rc);
    }
    if (retirement.cleared_suspicion) {
        LOG_WARN(
            "device %d: confirmed reset retired the suspect generation; fault notices reported before it are no "
            "longer this device's (generation is now %llu)",
            device_id_, static_cast<unsigned long long>(device_health_.generation())
        );
    }
    return retirement.monitor_reinstall_rc;
}

uint64_t DeviceRunnerBase::consume_device_fault_notices() noexcept {
    DeviceFaultMonitor *monitor = fault_monitor_if_held();
    if (monitor == nullptr) return 0;
    const uint32_t own_device = static_cast<uint32_t>(device_id_);

    uint64_t own = 0;
    const DeviceFaultNoticeCursor::Progress progress =
        fault_notices_.consume(*monitor, [&](const DeviceFaultNotice &notice) {
            // The notice's device id is logical, the same space this runner names
            // its own device in — measured with card 5 bound as logical 0 through
            // ASCEND_RT_VISIBLE_DEVICES, which reported device_id=0. So this
            // compares directly and must not translate through
            // acl_to_hal_device_id.
            //
            // Its stream id is matched against the ids this device's runs were
            // recorded on at launch, not against the handles live right now: this
            // runs at teardown, where a force reset may already have invalidated
            // those handles, and where a stream a run used may since have been
            // replaced.
            const bool my_device = notice.device_id == own_device;
            const RunStreamIdentities::Attribution attribution =
                my_device ? run_stream_ids_.attribute(notice.stream_id) : RunStreamIdentities::Attribution::NotMine;
            const char *scope = !my_device ? "; names another device in this process" :
                                attribution == RunStreamIdentities::Attribution::Mine ?
                                             "" :
                                attribution == RunStreamIdentities::Attribution::Undecided ?
                                             "; names a stream this runner cannot place — its identity history "
                                             "is incomplete, so attribution is undecided" :
                                             "; names a stream no run of this runner submitted on";
            LOG_ERROR(
                "device fault reported: device_id=%u stream_id=%u task_id=%u error_code=%u thread_id=%u "
                "(device-level; not attributed to any run%s)",
                notice.device_id, notice.stream_id, notice.task_id, notice.error_code, notice.thread_id, scope
            );
            if (attribution != RunStreamIdentities::Attribution::Mine) {
                device_health_.note_unattributed_fault();
                return;
            }
            ++own;
            device_health_.note_own_device_fault(notice.error_code);
        });
    // Counted process-wide: several devices can report into one ring, and it
    // reserves no share per device. An undelivered notice may have named a run
    // stream of this runner's and the channel cannot say, so it counts as
    // unattributable rather than as this device's — the same treatment a notice
    // naming another stream gets, for the same reason: acting on it refuses
    // healthy work on evidence that names nothing.
    if (progress.lost != 0) {
        LOG_ERROR(
            "device fault notices lost before the host read them: %llu overwritten in this process (ring holds %llu)",
            static_cast<unsigned long long>(progress.lost),
            static_cast<unsigned long long>(DeviceFaultMonitor::retained_notices())
        );
    }
    if (progress.newly_dropped != 0) {
        LOG_ERROR(
            "device fault notices dropped by the driver-thread reporter: %llu in this process",
            static_cast<unsigned long long>(progress.newly_dropped)
        );
    }
    device_health_.note_undelivered_notices(progress.lost, progress.newly_dropped);
    if (own == 0) return own;

    // A matched notice refuses future admission, and nothing more. `accepts_new_run`
    // reads the suspicion this recorded; no drain, reset or recovery starts here, and
    // the run being finalized keeps the outcome its own channels gave it — a notice
    // carries no run identity and can arrive late (16 s is the longest lag measured,
    // not a bound), so it can name a fault from an earlier run than this one.
    //
    // "Matched" is membership, not provenance: the stream is one this device's runs
    // were recorded on, which on a5 also carries binary load, AICPU init and callable
    // registration. So a refusal says a fault landed on a stream this runner uses, not
    // that a run caused it and not that a run was impaired. That is the availability
    // trade this policy takes deliberately — a late or recycled identity can refuse
    // work the device would have served.
    //
    // Result, health and resource retirement stay three decisions with three inputs:
    // a run can fail for its own reasons on a healthy card, and can succeed on a card
    // that faulted underneath it, both measured here.
    LOG_ERROR(
        "device %d: fault channel matched %llu notice(s) to its run streams (first code=%u, generation=%llu). "
        "Admission is refused until a confirmed reset retires this generation; no run's outcome is changed by it.",
        device_id_, static_cast<unsigned long long>(own), device_health_.first_error_code(),
        static_cast<unsigned long long>(device_health_.generation())
    );
    return own;
}

void DeviceRunnerBase::read_device_wall_ns(uint32_t pipeline_slot) {
    // Pull the per-thread AICPU phase records back from the device buffer that
    // AICPU writes through via KernelArgs::device_wall_data_base. (We can't use
    // the device_k_args_ shadow here — CANN's rtAicpuKernelLaunchExWithArgs
    // copies KernelArgs into AICPU-private memory at launch, so AICPU's writes
    // to its local copy don't propagate to device_k_args_.) Failure path is a
    // soft warn — the slot's record stays zeroed.
    if (pipeline_slot >= device_run_timing_.size()) return;
    DeviceRunTiming &out = device_run_timing_[pipeline_slot];
    out = DeviceRunTiming{};
    // The device this run was stamped on, and the unit its ticks are in, travel
    // with the result: the emit path must not read them off the runner, whose
    // state may already describe a later run.
    out.device_id = device_id_;
    out.sys_cnt_hz = device_sys_cnt_frequency_hz();
    if (!device_phase_capture_enabled()) return;
    // Gate on this slot's arming, not just on the buffer existing. A failed
    // reset leaves the allocation in place but publishes a null device base, so
    // AICPU never stamped for this run — reading anyway would republish the
    // previous run on this slot (or, on a slot's first run, whatever
    // `allocate_tensor` handed back) as this run's result.
    if (!device_timing_armed_[pipeline_slot]) return;
    void *slot_ptr = device_wall_dev_ptrs_[pipeline_slot];
    if (slot_ptr == nullptr) return;

    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    using BufferPrefix = DevicePhaseBufferPrefixStorage<kThreads>;
    static_assert(sizeof(BufferPrefix) == task_timing_tail_offset(kThreads), "device-phase prefix layout drift");
    BufferPrefix buf{};
    int wall_rc = rtMemcpy(&buf, sizeof(buf), slot_ptr, sizeof(buf), RT_MEMCPY_DEVICE_TO_HOST);
    if (wall_rc != 0) {
        LOG_WARN("rtMemcpy(device_phase) D2H failed: %d", wall_rc);
        return;
    }

    // Reduce across threads: per phase, min(start) + span = max(end) - min(start)
    // in cycles. RunWall (slot 0) is published as wall_ns for backward
    // compatibility; its duration is the whole-run wall.
    uint64_t start_cycles[NUM_AICPU_PHASES];
    uint64_t span_cycles[NUM_AICPU_PHASES];
    reduce_aicpu_phase_windows(buf.phases, kThreads, start_cycles, span_cycles);

    // Origin = earliest sub-phase start (Preamble..SchedWindow share the device
    // clock; RunWall is the bracket at offset 0). Sub-phase start offsets from
    // this origin give a common device-clock timeline so the orchestrator and
    // scheduler windows are comparable (their union is the "Effective" window).
    uint64_t origin = kPhaseUnset;
    for (int p = static_cast<int>(AicpuPhase::Preamble); p < NUM_AICPU_PHASES; ++p) {
        if (start_cycles[p] != kPhaseUnset && start_cycles[p] < origin) origin = start_cycles[p];
    }

    for (int p = 0; p < NUM_AICPU_PHASES; ++p) {
        out.phase_ns[p] = span_cycles[p] > 0 ? static_cast<uint64_t>(cycles_to_us(span_cycles[p]) * 1000.0) : 0;
        if (p != static_cast<int>(AicpuPhase::RunWall) && start_cycles[p] != kPhaseUnset && origin != kPhaseUnset &&
            start_cycles[p] >= origin) {
            out.phase_start_ns[p] = static_cast<uint64_t>(cycles_to_us(start_cycles[p] - origin) * 1000.0);
        }
    }
    out.wall_ns = out.phase_ns[static_cast<int>(AicpuPhase::RunWall)];

    // The offsets above are rebased on this run's origin, so they cannot express
    // the interval between two runs. Keep RunWall's bounds as raw sys-counter
    // ticks: that counter is a monotone rescaling of CNTVCT_EL0 and is not reset
    // per run, so consecutive runs' ticks are directly comparable, and a
    // consumer differences the ticks before converting rather than converting
    // each bound first.
    const uint64_t run_wall_start = start_cycles[static_cast<int>(AicpuPhase::RunWall)];
    if (run_wall_start != kPhaseUnset) {
        out.run_wall_start_cycles = run_wall_start;
        out.run_wall_end_cycles = run_wall_start + span_cycles[static_cast<int>(AicpuPhase::RunWall)];
    }

    // A nonzero header means the last AICPU thread found at least one
    // dispatched timing slot. The conditional callback D2Hs the optional tail
    // and resolves it on the phase `origin` timeline.
    constexpr int kTailRecords = task_timing_buffer_slots(kThreads);
    int tail_rc = read_task_timing_tail_if_used(buf.header, [&]() {
        TaskTimingRecord tail[kTailRecords] = {};
        const void *tail_src = reinterpret_cast<const uint8_t *>(slot_ptr) + task_timing_tail_offset(kThreads);
        int rc = rtMemcpy(tail, sizeof(tail), tail_src, sizeof(tail), RT_MEMCPY_DEVICE_TO_HOST);
        if (rc != 0) return rc;
        resolve_task_timing_slots_ns(
            tail, kThreads, origin,
            [](uint64_t cyc) {
                return static_cast<uint64_t>(cycles_to_us(cyc) * 1000.0);
            },
            out.task_slot_dispatch_ns, out.task_slot_finish_ns
        );
        return 0;
    });
    if (tail_rc != 0) {
        LOG_WARN("rtMemcpy(task_timing) D2H failed: %d", tail_rc);
    }
}

int DeviceRunnerBase::init_runtime_args_with_metadata(
    Runtime &runtime, KernelArgsHelper &kernel_args, SlotPersistentArgs &slot
) {
    int rc = kernel_args.prepare_runtime_args(runtime, mem_alloc_, slot);
    if (rc != 0) {
        LOG_ERROR("prepare_runtime_args failed: %d", rc);
        return rc;
    }
    // A runtime whose entry values can travel as launch arguments publishes at
    // launch instead, because which route they take is only answerable once the
    // run holds the stream it will submit on. The snapshot taken above is what
    // that publication sends, so the values are still this run's either way.
    // Every other runtime publishes here, as it always has.
    if (!runtime_launch_entry_args_plan(runtime).supported) {
        rc = kernel_args.publish_runtime_args(/*launch_route_permitted=*/false);
        if (rc != 0) return rc;
    }
    // Log config and device ordinal are no longer published per-run on
    // KernelArgs — they were latched once into the AICPU SO globals by
    // simpler_aicpu_init (ensure_aicpu_init_launched) at device init.
    return 0;
}

int DeviceRunnerBase::start_shared_collectors_for_run(const DfxRunConfig &dfx, uint64_t run_epoch) {
    // Open each enabled collector's window and start its mgmt + poll threads
    // now, just before kernels launch. Both halves belong here: begin_run()
    // drops the previous run's records and republishes the device level, which
    // is only safe while this run holds the execution claim, and starting
    // earlier wastes CPU on empty queues and risks tripping ProfilerBase's
    // poll-loop idle-timeout if device-side init is slow.
    auto thread_factory = [this](std::function<void()> fn) {
        return create_thread(std::move(fn));
    };
    if (dfx.chip_swimlane_enabled()) {
        // Which of the two paths a run takes is decided by configuration, not
        // by what the collector answers: `run_begin` refusing means this run
        // cannot be retained, and falling back to `begin_run` would then reset
        // a store a predecessor is still publishing into. So a refusal fails
        // the run, here, before any kernel is submitted and before any
        // predecessor's records or slots are touched.
        if (chip_swimlane_collector_.retains_runs()) {
            // Reader shards before admission: admitting a run waits for every
            // shard to acknowledge the new run table, and a shard that has not
            // been spawned cannot acknowledge anything. From the second run on
            // they are already running, so this is what puts the first run on
            // the same path as the rest.
            chip_swimlane_collector_.start(thread_factory);
            if (!chip_swimlane_collector_.run_begin(run_epoch, dfx.output_prefix, dfx.chip_swimlane_level)) {
                LOG_ERROR(
                    "ChipSwimlane: run %llu was not admitted for retained collection",
                    static_cast<unsigned long long>(run_epoch)
                );
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        } else {
            chip_swimlane_collector_.begin_run(dfx.output_prefix, dfx.chip_swimlane_level);
            chip_swimlane_collector_.start(thread_factory);
        }
    }
    if (dfx.dump_args_enabled()) {
        dump_collector_.begin_run(dfx.output_prefix, dfx.dump_args_level);
        dump_collector_.start(thread_factory);
    }
    if (dfx.pmu_enabled) {
        pmu_collector_.begin_run(make_pmu_csv_path(dfx.output_prefix), dfx.pmu_event_type);
        pmu_collector_.start(thread_factory);
    }
    if (dfx.scope_stats_enabled) {
        scope_stats_collector_.begin_run();
        scope_stats_collector_.start(thread_factory);
    }
    return 0;
}

void DeviceRunnerBase::withdraw_unlaunched_collectors_for_run(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept {
    // Only for a launch that submitted nothing. `start_shared_collectors_for_run`
    // admits a run into a retained slot before any submission, so a transaction
    // that ends at `NotStarted` would otherwise leave that slot occupied and
    // targetless — invisible to the writer and to a flush, and two of them
    // exhaust the capacity the next admission waits on.
    if (!dfx.chip_swimlane_enabled()) return;
    if (!chip_swimlane_collector_.retains_runs()) return;
    // Nothing may escape a rollback path: the caller owes the layer above its
    // own transaction's rc and the ownership of this run's `prepared`. The
    // withdrawal publishes its outcome — a released slot, or a quarantine with
    // the fatal set and every waiter woken — before anything that can throw, so
    // this boundary cannot be what hides unreachable capacity; it exists so a
    // failure past that point cannot displace the launch failure either.
    try {
        (void)chip_swimlane_collector_.abandon_run(run_epoch);
    } catch (...) {}
}

int DeviceRunnerBase::flush_diagnostics(int timeout_ms, std::string *error) {
    if (!chip_swimlane_collector_.retains_runs()) return 0;
    return chip_swimlane_collector_.flush_retained_runs(timeout_ms, error) ? 0 : PTO_RUNTIME_ERR_INTERNAL;
}

void DeviceRunnerBase::finish_retained_runs() { chip_swimlane_collector_.finish_retained_runs(); }

void DeviceRunnerBase::write_host_phase_records_artifact(const std::string &output_prefix, uint32_t pipeline_slot) {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    simpler::dfx::HostPhaseRecordStore &records = host_phase_runs_[pipeline_slot].records;
    // Per-event view of the prepare path. Every phase it records is produced on
    // the host during bind, and the store is finished before launch, so this
    // touches no device state and is callable from any point after bind --
    // including a path that never launches. Keyed on the run's output_prefix
    // because it is non-empty exactly when this run produces diagnostic
    // artifacts, and on the store having finished a pass, which is what a
    // host-orchestrating runtime leaves behind. The store writes a pass at most
    // once, so every path that can end a run calls this unconditionally.
    if (!output_prefix.empty() && records.finished()) {
        (void)records.write_records_jsonl(make_host_phase_records_path(output_prefix));
    }
}

void DeviceRunnerBase::teardown_shared_collectors_after_run(
    const DfxRunConfig &dfx, uint32_t pipeline_slot, uint64_t run_epoch, bool device_execution_complete
) {
    // Tear down collectors. stop() joins mgmt then collector in the only safe
    // order (mgmt's final-drain pass into L2 has poll as its consumer).
    // Diagnostic exports use the per-task output prefix the user set on
    // CallConfig (CallConfig::validate() enforces non-empty upstream).
    if (dfx.chip_swimlane_enabled() && chip_swimlane_collector_.retains_runs()) {
        // A retained run keeps the run boundary's device-side reads — terminal
        // and live counters — and hands the rest to the writer. No quiesce: the
        // pipeline is shared with the successor and draining it here is what
        // the per-queue cut replaces.
        //
        // This run's host phase records go in first: the epoch's metadata
        // snapshot inside the close is what copies them, and the collector
        // holds one copy of them for every run it serves.
        simpler::dfx::runs::close_run_boundary(
            chip_swimlane_collector_, run_epoch, pipeline_slot, device_execution_complete, [this, pipeline_slot] {
                publish_host_phase_records_to_swimlane(pipeline_slot);
            }
        );
        write_host_phase_records_artifact(dfx.output_prefix, pipeline_slot);
        if (dfx.dump_args_enabled()) {
            dump_collector_.quiesce();
            dump_collector_.reconcile_counters();
            dump_collector_.export_dump_files();
        }
        if (dfx.pmu_enabled) {
            pmu_collector_.quiesce();
            pmu_collector_.reconcile_counters();
        }
        if (dfx.scope_stats_enabled) {
            scope_stats_collector_.quiesce();
            scope_stats_collector_.reconcile_counters();
            scope_stats_collector_.write_jsonl(dfx.output_prefix);
        }
        return;
    }
    if (dfx.chip_swimlane_enabled()) {
        chip_swimlane_collector_.quiesce();
        chip_swimlane_collector_.read_phase_header_metadata();
        chip_swimlane_collector_.reconcile_counters();
        // Only on the completion path. `device_execution_complete` is set by the
        // caller that observed the run's fence; the recovery path clears it, and
        // there a producer may never have reached its close — or may still be
        // running on a card the bounded drain did not prove clean. Reading then
        // would report a partial bank as if it were the run's accounting.
        if (device_execution_complete) {
            chip_swimlane_collector_.report_run_terminal_snapshot(pipeline_slot, run_epoch);
        }
        publish_host_phase_records_to_swimlane(pipeline_slot);
        chip_swimlane_collector_.export_swimlane_json();
    }

    write_host_phase_records_artifact(dfx.output_prefix, pipeline_slot);

    if (dfx.dump_args_enabled()) {
        dump_collector_.quiesce();
        dump_collector_.reconcile_counters();
        dump_collector_.export_dump_files();
    }

    if (dfx.pmu_enabled) {
        pmu_collector_.quiesce();
        pmu_collector_.reconcile_counters();
    }

    if (dfx.scope_stats_enabled) {
        scope_stats_collector_.quiesce();
        scope_stats_collector_.reconcile_counters();
        scope_stats_collector_.write_jsonl(dfx.output_prefix);
    }
}

bool DeviceRunnerBase::try_acquire_native_run(
    const void *owner, const NativeRunIdentity &identity, LaunchPermit *permit, const NativeRunJoin *join
) {
    if (owner == nullptr || permit == nullptr) return false;
    std::lock_guard<std::mutex> lk(native_run_mu_);
    bool reserved = false;
    for (const NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == owner) {
            reserved = true;
            break;
        }
    }
    if (!reserved) return false;
    if (native_run_claim_index(owner) < active_native_run_count_) return false;
    if (active_native_run_count_ >= active_native_runs_.size()) return false;
    if (active_native_run_count_ != 0) {
        // Only a run that presents the predecessor it was ordered behind may
        // join an occupied claim, and that predecessor must be the newest
        // holder: a join naming an older one would leave a run between the two
        // unordered with respect to this one.
        if (join == nullptr || join->predecessor_owner == nullptr) return false;
        const NativeRunClaim &newest = active_native_runs_[active_native_run_count_ - 1];
        if (newest.owner != join->predecessor_owner) return false;
        if (newest.identity != join->predecessor_identity) return false;
    }
    active_native_runs_[active_native_run_count_] = NativeRunClaim{owner, identity};
    ++active_native_run_count_;
    *permit = LaunchPermit(identity);
    return true;
}

void DeviceRunnerBase::release_native_run(const void *owner) {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    const size_t index = native_run_claim_index(owner);
    if (index >= active_native_run_count_) return;
    for (size_t next = index + 1; next < active_native_run_count_; ++next) {
        active_native_runs_[next - 1] = active_native_runs_[next];
    }
    --active_native_run_count_;
    active_native_runs_[active_native_run_count_] = NativeRunClaim{};
}

size_t DeviceRunnerBase::native_run_claim_index(const void *owner) const {
    if (owner == nullptr) return active_native_runs_.size();
    for (size_t i = 0; i < active_native_run_count_; ++i) {
        if (active_native_runs_[i].owner == owner) return i;
    }
    return active_native_runs_.size();
}

bool DeviceRunnerBase::native_run_active() const {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    return active_native_run_count_ != 0;
}

bool DeviceRunnerBase::native_run_owned_by(const void *owner) const {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    return native_run_claim_index(owner) < active_native_run_count_;
}

size_t DeviceRunnerBase::native_run_claim_count() const {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    return active_native_run_count_;
}

bool DeviceRunnerBase::try_reserve_native_run(
    const void *owner, uint32_t pipeline_slot, uint32_t arena_bank, bool allow_prepared_successor
) {
    if (owner == nullptr || pipeline_slot >= PTO_PIPELINE_MAX_DEPTH || arena_bank >= PTO_PIPELINE_MAX_DEPTH) {
        return false;
    }
    std::lock_guard<std::mutex> lk(native_run_mu_);

    size_t occupied = 0;
    const NativeRunReservation *existing = nullptr;
    for (const NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == nullptr) continue;
        if (reservation.owner == owner || reservation.pipeline_slot == pipeline_slot) {
            return false;
        }
        ++occupied;
        existing = &reservation;
    }
    if (occupied != 0) {
        // The one reservation already held must belong to a run that has taken
        // the claim: a successor may prepare alongside a *launched* run, not
        // alongside another merely prepared one.
        const bool existing_holds_claim =
            existing != nullptr && native_run_claim_index(existing->owner) < active_native_run_count_;
        if (!allow_prepared_successor || occupied != 1 || !existing->permits_prepared_successor ||
            !existing_holds_claim) {
            return false;
        }
    }

    for (NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == nullptr) {
            reservation = NativeRunReservation{owner, pipeline_slot, arena_bank, allow_prepared_successor};
            return true;
        }
    }
    return false;
}

bool DeviceRunnerBase::arena_bank_shared_with_other_run(const void *owner, uint32_t arena_bank) const {
    if (arena_bank >= PTO_PIPELINE_MAX_DEPTH) return false;
    std::lock_guard<std::mutex> lk(native_run_mu_);
    for (const NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == nullptr || reservation.owner == owner) continue;
        if (reservation.arena_bank == arena_bank) return true;
    }
    return false;
}

void DeviceRunnerBase::release_native_run_reservation(const void *owner) {
    if (owner == nullptr) return;
    std::lock_guard<std::mutex> lk(native_run_mu_);
    for (NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == owner) {
            reservation = NativeRunReservation{};
            return;
        }
    }
}

bool DeviceRunnerBase::native_runs_outstanding() const {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    for (const NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner != nullptr) return true;
    }
    return false;
}
