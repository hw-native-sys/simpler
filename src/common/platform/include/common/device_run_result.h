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
 * @file device_run_result.h
 * @brief One run's device-side result, in storage that run owns.
 *
 * A runtime's device state — the shared-memory header, the rings, the heap — is
 * reused by whichever run occupies the arena next, and a successor resets it as
 * it starts. So a result left in that state is only readable until the next run
 * begins, and moving the host's read earlier does not fix it: the fence that
 * lets the host return is the same fence that releases the successor, so there
 * is no window for the host to claim.
 *
 * The ordering has to come from the device side. A run's device code copies what
 * the host needs into this region and publishes it **before its kernel
 * returns**, so the copy precedes the fence, which precedes any successor's
 * reset:
 *
 *     run N's error writes complete
 *       -> N copies them into N's region, then publishes
 *       -> N's kernel returns  -> fence N
 *       -> N+1 may reset the shared state
 *     host, later: reads N's region
 *
 * The platform owns one region per pipeline slot and never interprets it: the
 * payload is opaque bytes whose layout the runtime picks, because what a run
 * reports is a runtime concept (`tensormap_and_ringbuffer` publishes the error
 * tail of its `SharedMemoryHeader`; `host_build_graph` keeps no such shared
 * state and publishes nothing). A slot's region is not handed to another run
 * until the run holding it finalizes — the reservation is what guarantees that —
 * so the host's read of its own slot is unhurried.
 *
 * **`published` carries the run's epoch, not a constant.** Staleness is then
 * decided by value rather than by the host having cleared the region first, so
 * steady-state reuse costs no per-run H2D and a region the host failed to
 * prepare cannot be mistaken for this run's result — it still holds some
 * earlier run's epoch. A newly allocated region is the one case that must be
 * zeroed before use: `rtMalloc` returns whatever the device left there, which
 * cannot be assumed to differ from the epoch about to look for it.
 *
 * **What a match does and does not tell you.** A producer publishes only when it
 * has something to preserve — today, only a failing run does:
 *
 *   - `published == run_epoch` — this run published a result; the payload is
 *     that run's.
 *   - anything else — no result from this run. This does **not** distinguish a
 *     run that had nothing to report from one that never reached its publish
 *     point, such as a kernel the op-execute watchdog reaped.
 *
 * So the region is not a completion signal and must not be read as one. The
 * host decides whether a run failed from the execution error channel it already
 * has, and consults this region for the detail. A missing result never turns an
 * execution error into a success.
 *
 * **What this interface deliberately does not provide, and who owns it.** There
 * is no terminal per-run status here. Three consequences, all intended rather
 * than overlooked:
 *
 *   - a publication must carry a payload — `payload_bytes == 0` reads as absent,
 *     so there is no way to record "reached the end, nothing to report";
 *   - a failed device-to-host copy leaves the host's copy empty, which is
 *     therefore indistinguishable from a run that published nothing;
 *   - success is never recorded at all, so the region can corroborate a failure
 *     the host already knows about but can never be the thing that decides one.
 *
 * Carrying a run's terminal status is #2267's contract, not this one's, and it
 * is a wider change than relaxing the checks above. The shape it wants is a
 * small status published every run alongside the epoch, with the error tail
 * appended only on failure — not this failure-only payload with its emptiness
 * checks loosened, because a status channel has to settle three things this one
 * does not: when the host reads it, what an exception that never reached the
 * publish point reports, and who owns an error the region and the execution
 * channel disagree about. Until then, treat a run's success or failure as the
 * execution error channel's answer and this region as evidence attached to it.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

/**
 * Opaque payload capacity. Large enough for the biggest per-run result any
 * runtime publishes today with headroom; a producer static_asserts its own
 * payload size against this rather than assuming it fits.
 */
constexpr size_t DEVICE_RUN_RESULT_PAYLOAD_BYTES = 256;

/**
 * The region itself. Copied host<->device as bytes, so it stays POD and holds no
 * pointer into itself.
 */
struct DeviceRunResultRegion {
    uint64_t published;
    uint32_t payload_bytes;
    uint32_t reserved;
    uint8_t payload[DEVICE_RUN_RESULT_PAYLOAD_BYTES];
};

static_assert(
    std::is_trivially_copyable_v<DeviceRunResultRegion> && std::is_standard_layout_v<DeviceRunResultRegion>,
    "DeviceRunResultRegion crosses the host-device boundary as bytes"
);
static_assert(offsetof(DeviceRunResultRegion, payload) == 16, "run-result payload offset drift");
static_assert(sizeof(DeviceRunResultRegion) == DEVICE_RUN_RESULT_PAYLOAD_BYTES + 16, "run-result region layout drift");

/** Bytes the platform allocates per pipeline slot for one run's result. */
constexpr size_t device_run_result_bytes() { return sizeof(DeviceRunResultRegion); }

/**
 * Whether `region` holds the result published by the run whose epoch is
 * `run_epoch`. Epoch 0 is never accepted: it is the value a freshly cleared
 * region carries, so treating it as a match would make an unpublished region
 * look valid. A false answer means "no result from this run" and nothing more —
 * see the note on `published` above.
 */
inline bool device_run_result_published(const DeviceRunResultRegion &region, uint64_t run_epoch) {
    return run_epoch != 0 && region.published == run_epoch && region.payload_bytes != 0 &&
           region.payload_bytes <= DEVICE_RUN_RESULT_PAYLOAD_BYTES;
}
