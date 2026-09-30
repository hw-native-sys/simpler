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

#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_entry_args.h"
#include "common/launch_entry_args.h"
#include "graph_image_view.h"
#include "runtime.h"

/**
 * This thread's view of the run's Graph Definition section.
 *
 * Built where a reader is about to use it and never stored: a launch package
 * belongs to the thread that entered with it, and the completion gate is a
 * last-one-out latch rather than a barrier, so a thread may return while its
 * peers are still materializing. A view kept in shared state would name a
 * package whose owner has gone.
 *
 * The carrier is taken from the run's descriptor and the reading is then done
 * against it, so the two have to agree: an onboard run reads its own launch
 * arguments and checks that the header they carry names the same section the
 * descriptor does, and a simulated run reads the run-owned snapshot. Neither
 * accepts the other's source, so a stale view is refused rather than decoded.
 *
 * An invalid view is the answer for a run with no Graph section at all, and its
 * caller reports the Definition as unreadable through the path it already has.
 */
inline GraphImageView graph_image_view_for_reader(const Runtime &runtime) {
    GraphImageView view{};
    const GraphSectionSource source = runtime.get_graph_section_source();
    const uint32_t bytes = runtime.get_graph_section_bytes();
    if (source == GraphSectionSource::None || bytes == 0) return view;

    if (source == GraphSectionSource::HostSnapshot) {
        const uint64_t base = runtime.get_graph_section_base();
        if (base == 0) return view;
        view.section = reinterpret_cast<const std::byte *>(base);
        view.bytes = bytes;
        return view;
    }

    if (source != GraphSectionSource::LaunchEnvelope) return view;
    // The descriptor published no address for this carrier, because there is no
    // one address: each launched thread has its own copy.
    if (runtime.get_graph_section_base() != 0) return view;
    const PlatformEntryArgs args = get_platform_entry_args(platform_aicpu_affinity_thread_idx());
    if (args.args_base == nullptr) return view;
    if (args.graph_source != static_cast<uint32_t>(GraphSectionSource::LaunchEnvelope)) return view;
    if (args.graph_offset != LAUNCH_ENVELOPE_HEADER_BYTES || args.graph_bytes != bytes) return view;
    view.section = static_cast<const std::byte *>(args.args_base) + args.graph_offset;
    view.bytes = bytes;
    return view;
}
