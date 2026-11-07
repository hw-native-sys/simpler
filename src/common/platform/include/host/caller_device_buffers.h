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

#include "host/device_buffer_owners.h"

/**
 * The device allocations a caller minted here, and which runs are still using them.
 *
 * "Caller" is a distinction inside this device context, not a layer boundary: it means whoever
 * allocated through `device_malloc_ctx` — a `Worker.malloc` / `alloc_child_tensor` from above — as
 * opposed to the regions this context mints for itself through `allocate_tensor`. Nothing here is
 * an L3-to-L2 protocol: no control command, no mailbox field, no change to the arguments' wire
 * form. Host-space tensors do not appear either, in any form: they are copied into a
 * `RetainedTempBump` slice the runner owns for the run's whole lifetime, which is a lifetime this
 * table has no question about, and the slice is not a caller mint so it could not resolve anyway.
 *
 * A caller device buffer is allocated by this process on the caller's request and freed on the
 * caller's request: the allocator's right to release is the caller's throughout, and nothing here
 * takes it. What this holds is a *borrow* — the statement that a run may still read or write those
 * bytes — and a release the borrow outlives is refused rather than performed.
 *
 * **Only caller mints are recorded.** The entry is made by the caller-facing device-malloc path, so
 * an address this process allocated for itself — a workspace region, a retained temporary, an arena
 * bank — is absent, and `resolve` therefore refuses it. A caller has no business naming one, and the
 * absence is what makes that refusal automatic rather than a list to maintain.
 *
 * **A span resolves to its containing allocation, not to a key.** A tensor may sit at an offset
 * inside a larger buffer, so the unit of proof is the allocation that covers the whole span — the
 * same rule `MemoryAllocator::owning_allocation` applies for host mappings, and for the same reason:
 * release is per allocation, so a borrow must be too.
 *
 * **A borrow belongs to a run identity, not to a count.** `borrow` is keyed on the run's own
 * identity, so a release names the run that took it and cannot discharge another run's reference.
 * Re-borrowing under one identity replaces that identity's set, which is what a re-prepared run
 * needs.
 *
 * **A borrow is about lifetime, and says nothing about readability.** Holding an allocation is not
 * producing it: two runs may legitimately take the same immutable input, and neither makes the
 * other's bytes unreadable. So the borrow answers only "may this be released", and a second,
 * separate fact answers "may this be read on the host now".
 *
 * **That second fact is a declared write, made by whoever knows the direction.** The storage a
 * run's arguments reach the lane in carries no direction, so the lane cannot supply it — but the
 * runtime's own bind receives the orchestration signature, so it declares which of its caller
 * spans it will write. `written_by_other_run` is then exactly the question a host reader has:
 * whether a *different* live run has said it produces these bytes. A shared read-only input is
 * declared by nobody and stays readable, which is what keeps every path that does not produce a
 * device tensor on the behaviour it already had.
 *
 * A declaration is keyed on the same identity as the borrow and released with it, so it cannot
 * outlive the run that made it — including when that run held no borrow, which is why the release
 * discharges the two facts independently. A span that resolves to no recorded allocation is
 * ignored: there is nothing to protect.
 *
 * **An unproven run's facts are permanent, and they belong to the allocation.** When a run ended
 * without its last consumer being proven finished, the device may still be reading those bytes and
 * may still be writing the ones it declared; no later event can prove otherwise. So the allocation
 * keeps both marks for the process's remaining life: `free` refuses from then on, and a host read
 * of those bytes is refused from then on. They sit on the allocation rather than on the run's
 * identity because the run is gone and its slot — and so its identity — is handed to the next run,
 * which must inherit neither its permissions nor its debts. In one sentence: every path that ends
 * a run unproven also poisons the lane, so what these marks actually guard is a `free` or a host
 * read arriving *after* that — the lane is gone, so the refusal has to live on the allocation.
 *
 * This table is not teardown's authority and makes no claim about it. What a retained reference
 * says is that *this* path must not release the allocation; what teardown then does with it — a
 * successful reset that ends the generation, a failed release it records, a quarantine — is
 * decided where those outcomes are known, not here.
 *
 * Its own mutex, and a leaf: nothing called while it is held takes another lock.
 */
class CallerDeviceBuffers final : private DeviceBufferOwners {
public:
    using DeviceBufferOwners::Allocation;
    using DeviceBufferOwners::allocation_count;
    using DeviceBufferOwners::borrow;
    using DeviceBufferOwners::borrow_count;
    using DeviceBufferOwners::borrowed;
    using DeviceBufferOwners::declare_writes;
    using DeviceBufferOwners::forget_if_unborrowed;
    using DeviceBufferOwners::record;
    using DeviceBufferOwners::recorded;
    using DeviceBufferOwners::release;
    using DeviceBufferOwners::resolve;
    using DeviceBufferOwners::retained_count;
    using DeviceBufferOwners::Span;
    using DeviceBufferOwners::write_declaration_count;
    using DeviceBufferOwners::written_by_other_run;
};
