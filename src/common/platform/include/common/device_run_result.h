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
 * @brief One run's device-side terminal result, in storage that run owns.
 *
 * A runtime's device state — the shared-memory header, the rings, the heap — is
 * reused by whichever run occupies the arena next, and a successor resets it as
 * it starts. So a result left in that state is only readable until the next run
 * begins, and moving the host's read earlier does not fix it: the fence that
 * lets the host return is the same fence that releases the successor, so there
 * is no window for the host to claim.
 *
 * The ordering has to come from the device side. A run's device code folds what
 * the host needs into this region and publishes it **before its kernel
 * returns**, so the copy precedes the fence, which precedes any successor's
 * reset:
 *
 *     run N's error writes complete
 *       -> N folds them into a terminal record in N's region, then publishes
 *       -> N's kernel returns  -> fence N
 *       -> N+1 may reset the shared state
 *     host, later: reads N's region
 *
 * The platform owns one region per pipeline slot. It interprets the header —
 * verdict, code and code source are a common vocabulary — but never the
 * payload, which is opaque diagnostic bytes whose layout the runtime picks
 * (`tensormap_and_ringbuffer` publishes the error tail of its
 * `SharedMemoryHeader`; `host_build_graph` keeps no such shared state and
 * publishes none). A slot's region is not handed to another run until the run
 * holding it finalizes — the reservation is what guarantees that — so the
 * host's read of its own slot is unhurried.
 *
 * **`published` carries the run's epoch, not a constant.** Staleness is then
 * decided by value rather than by the host having cleared the region first, so
 * steady-state reuse costs no per-run H2D and a region the host failed to
 * prepare cannot be mistaken for this run's result — it still holds some
 * earlier run's epoch. A newly allocated region is the one case that must be
 * zeroed before use: `rtMalloc` returns whatever the device left there, which
 * cannot be assumed to differ from the epoch about to look for it.
 *
 * **Every run publishes, success included.** A producer folds each
 * participant's error state and its runtime's shared error state into one
 * record and publishes it from a single cleanup owner after the last work that
 * could still change the run's outcome. So a matching epoch carries a verdict:
 *
 *   - `Ok` — the run reached the end of its audited normal path. Code 0, no
 *     source, no payload.
 *   - `Error` — the run failed, with the runtime's own signed code and the
 *     state that selected it. The payload, when present, is that runtime's
 *     diagnostic scene.
 *   - anything else, including a non-matching epoch — **undecided**. Absence of
 *     a record is not success and not failure: it is indistinguishable between
 *     a run that never reached its publish point (a kernel the op-execute
 *     watchdog reaped) and a read that failed.
 *
 * The code is the runtime's own signed status, not an ACL error: this channel
 * preserves what the runtime decided, and does not reproduce the `507018` a
 * stream synchronize reports for the same failure.
 *
 * **What a verdict does not settle.** A terminal record answers what this run's
 * execution did. It does not say the device is healthy, and it does not say the
 * run's resources are retirable — a device can publish a failure and keep
 * tearing down, so quiescence is proven by the run's completion boundaries and
 * its outstanding wait references, never by this record. The host keeps those
 * axes apart.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

/**
 * Opaque payload capacity. Large enough for the biggest per-run diagnostic any
 * runtime publishes today with headroom; a producer static_asserts its own
 * payload size against this rather than assuming it fits.
 */
constexpr size_t DEVICE_RUN_RESULT_PAYLOAD_BYTES = 256;

/** What the producer decided about this run. Wire values; do not renumber. */
enum class DeviceRunVerdict : uint32_t {
    None = 0,
    Ok = 1,
    Error = 2,
};

/** Which device-side state selected a failure's code. Wire values. */
enum class DeviceRunCodeSource : uint32_t {
    None = 0,
    Header = 1,      // the runtime's shared error state (orch / scheduler codes)
    ThreadRc = 2,    // a participant's own execution return
    ShutdownRc = 3,  // a participant's teardown return, with no earlier error
};

/**
 * Whether `source` is one a failure may carry. Both sides of the channel ask
 * this one question, so a producer cannot publish a source the host then reads
 * as undecided — an asymmetry that would turn a refusable producer defect into
 * a record that committed successfully and decides nothing.
 *
 * `None` is excluded: it is the value a success carries, and a failure whose
 * source is unrecorded names no state to look in.
 */
constexpr bool device_run_code_source_is_failure(DeviceRunCodeSource source) {
    return source == DeviceRunCodeSource::Header || source == DeviceRunCodeSource::ThreadRc ||
           source == DeviceRunCodeSource::ShutdownRc;
}

/**
 * The region itself. Copied host<->device as bytes, so it stays POD and holds no
 * pointer into itself.
 */
struct DeviceRunResultRegion {
    uint64_t published;       // this run's epoch; written and flushed last
    uint32_t payload_bytes;   // diagnostic length, 0 when there is none
    uint32_t verdict;         // DeviceRunVerdict
    int32_t completion_code;  // the runtime's signed status; 0 iff verdict is Ok
    uint32_t code_source;     // DeviceRunCodeSource
    uint8_t payload[DEVICE_RUN_RESULT_PAYLOAD_BYTES];
};

static_assert(
    std::is_trivially_copyable_v<DeviceRunResultRegion> && std::is_standard_layout_v<DeviceRunResultRegion>,
    "DeviceRunResultRegion crosses the host-device boundary as bytes"
);
static_assert(offsetof(DeviceRunResultRegion, payload) == 24, "run-result payload offset drift");
static_assert(sizeof(DeviceRunResultRegion) == DEVICE_RUN_RESULT_PAYLOAD_BYTES + 24, "run-result region layout drift");

/** Bytes the platform allocates per pipeline slot for one run's result. */
constexpr size_t device_run_result_bytes() { return sizeof(DeviceRunResultRegion); }

/** What the host concluded about a run from its region. */
enum class DeviceRunTerminalState : uint8_t {
    Undecided = 0,
    Succeeded = 1,
    Failed = 2,
};

/**
 * A validated terminal record. `reason` is set exactly when the state is
 * Undecided and names why, so a caller reports which of "never published",
 * "belongs to another run" and "self-inconsistent" it saw rather than
 * collapsing them into one silence.
 */
struct DeviceRunTerminal {
    DeviceRunTerminalState state{DeviceRunTerminalState::Undecided};
    int32_t code{0};
    DeviceRunCodeSource source{DeviceRunCodeSource::None};
    const char *reason{"not read"};
};

/**
 * Interpret `region` as the terminal record of the run whose epoch is
 * `run_epoch`, having already read it back successfully.
 *
 * Epoch 0 is never accepted: it is the value a freshly cleared region carries,
 * so treating it as a match would make an unpublished region look valid. A
 * record whose fields contradict each other is Undecided rather than trusted in
 * part — a success that carries a code, or a failure that carries none, is a
 * producer defect, and reading either as a verdict would hide it.
 */
inline DeviceRunTerminal device_run_result_terminal(const DeviceRunResultRegion &region, uint64_t run_epoch) {
    DeviceRunTerminal out;
    if (run_epoch == 0) {
        out.reason = "run has no epoch";
        return out;
    }
    if (region.published != run_epoch) {
        out.reason = "no record published under this run's epoch";
        return out;
    }
    const auto verdict = static_cast<DeviceRunVerdict>(region.verdict);
    if (verdict == DeviceRunVerdict::Ok) {
        if (region.completion_code != 0 || region.code_source != static_cast<uint32_t>(DeviceRunCodeSource::None) ||
            region.payload_bytes != 0) {
            out.reason = "success record carries a failure code, source or payload";
            return out;
        }
        out.state = DeviceRunTerminalState::Succeeded;
        out.reason = nullptr;
        return out;
    }
    if (verdict == DeviceRunVerdict::Error) {
        if (region.completion_code == 0) {
            out.reason = "failure record carries no code";
            return out;
        }
        const auto source = static_cast<DeviceRunCodeSource>(region.code_source);
        if (!device_run_code_source_is_failure(source)) {
            out.reason = "failure record carries no known code source";
            return out;
        }
        out.state = DeviceRunTerminalState::Failed;
        out.code = region.completion_code;
        out.source = source;
        out.reason = nullptr;
        return out;
    }
    out.reason =
        (verdict == DeviceRunVerdict::None) ? "record carries no verdict" : "record carries an unknown verdict";
    return out;
}

/**
 * Whether `region` carries a diagnostic payload published by the run whose
 * epoch is `run_epoch`. Diagnostic validity is deliberately separate from the
 * verdict's: a known failure stays a failure when its payload is absent or
 * overruns the capacity, and only the diagnostic view is refused.
 */
inline bool device_run_result_published(const DeviceRunResultRegion &region, uint64_t run_epoch) {
    return run_epoch != 0 && region.published == run_epoch && region.payload_bytes != 0 &&
           region.payload_bytes <= DEVICE_RUN_RESULT_PAYLOAD_BYTES;
}
