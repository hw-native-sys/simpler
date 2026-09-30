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
 * Where one run's entry arguments reach the AICPU, and the vocabulary both ends
 * of that choice speak.
 *
 * Two routes exist. The descriptor route publishes the values inside the run's
 * `Runtime` descriptor, which the host copies with one `rtMemcpy`. The launch
 * route appends them to the AICPU launch arguments, which RTS copies as part of
 * the launch it is already making. The same `EntryArgsSource` value is written
 * into both the descriptor and the launch header, so the device can reject a
 * pair that does not agree rather than decode whichever it happens to read.
 *
 * Shared by the host that selects the route, the platform AICPU entry that
 * forwards the launch header, and the runtime that decodes it.
 */

#pragma once

#include <cstddef>
#include <cstdint>

/** Which of the two routes carries this run's entry-argument values. */
enum class EntryArgsSource : uint32_t {
    Descriptor = 0,      // inside the published Runtime descriptor
    LaunchEnvelope = 1,  // appended to the AICPU launch arguments, copied by RTS
};

/**
 * Which carrier delivered this run's Graph Definition section.
 *
 * The section is the whole of the run's Definition bytes — every framed object,
 * as the bind packed them — and a task names its own object by an offset inside
 * it. `LaunchEnvelope` is the onboard carrier: the section is appended to the
 * AICPU launch arguments and RTS copies it, so each launched thread receives its
 * own copy and no address in it may be shared. `HostSnapshot` is simulation's,
 * where there is no launch package: the runner publishes a host snapshot it owns
 * until the run's last consumer ends. The two are alternatives, not fallbacks
 * for one another — a reader accepts exactly the one its platform uses, so a
 * simulated run cannot decode a stale onboard view or the reverse.
 */
enum class GraphSectionSource : uint32_t {
    None = 0,            // this run submitted no Graph task
    LaunchEnvelope = 1,  // appended to the AICPU launch arguments, copied by RTS
    HostSnapshot = 2,    // a run-owned host snapshot, simulation only
};

/**
 * Byte offset of the entry payload inside an AICPU launch package.
 *
 * One value for every architecture, so the payload's own offset is not a
 * per-arch quantity a decoder has to look up: it is a cache line multiple at or
 * above every arch's `sizeof(KernelArgs)` (each arch static_asserts its own
 * header fits). Cache-line aligned because the payload leads with tensor
 * descriptors whose type is 64-byte aligned — the copy target is what carries
 * that alignment, but keeping the source's offset aligned too means the two
 * agree whenever the launch base does.
 */
inline constexpr size_t LAUNCH_ENVELOPE_HEADER_BYTES = 192;

/**
 * Host bytes a persistent argument block keeps of the descriptor prefix its
 * launch route last published onto it.
 *
 * A cap, not a layout: a runtime that uses the launch route asserts its own
 * prefix fits, and one without a launch route never reads the storage. Fixed
 * and trivially copyable on purpose — recording a prefix after the copy that
 * put it on the device has to allocate nothing and throw nothing, or a
 * successful publication could be followed by a failure with no state to
 * report it in.
 */
inline constexpr size_t LAUNCH_ROUTE_PREFIX_CACHE_BYTES = 256;

/**
 * What the host needs to route one run's entry arguments, captured once per run
 * from the descriptor snapshot that publication will consume.
 *
 * Every field is a byte offset or count read at prepare time. The launch side
 * works from this and from the snapshot alone — never from the caller's
 * `Runtime`, which a successor's prepare may already have moved on.
 *
 * `supported == false` is the whole answer for a runtime that has no launch
 * route (`host_build_graph`): every other field stays zero and the descriptor
 * route is the only one its host code takes.
 *
 * `counts_valid == false` is a different answer and must not be folded into the
 * first one. It says this runtime does have a launch route but the counts it
 * read are outside capacity, which is a corrupted descriptor rather than a
 * routing decision — publishing those counts would hand a consumer that trusts
 * them a length it must not believe. A caller that sees it fails the run before
 * anything is copied. A runtime with no launch route examines no counts and
 * reports them valid.
 */
struct LaunchEntryArgsPlan {
    bool supported{false};
    bool counts_valid{true};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    // Offset of the three descriptor control words — tensor count, scalar count,
    // source — which publication patches in place. They sit inside every
    // publication length, including the shortest.
    size_t control_offset{0};
    // Where the values sit inside the descriptor snapshot. Two offsets because
    // the descriptor's storage keeps its arrays at fixed capacity, so the filled
    // parts are not adjacent there; the launch package packs them back to back.
    size_t tensor_offset{0};
    size_t scalar_offset{0};
    // Bytes the launch package's entry region occupies: the tensors this run
    // filled, then its scalars.
    size_t payload_bytes{0};
    // Descriptor bytes a publication sends when the launch route carries the
    // values: everything ahead of the entry storage, and nothing of it.
    size_t descriptor_bytes_when_launched{0};
};
