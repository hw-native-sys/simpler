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

#include <cstdint>
#include <exception>
#include <string>
#include <utility>

#include "host/chip_swimlane_collector.h"

namespace simpler::dfx::session {

/**
 * Name a boundary failure in the log, best effort.
 *
 * Every step of this allocates, and these paths run where an allocation may
 * have just failed. The caller has already set the state this describes, so a
 * report that cannot be built costs a log line and nothing else.
 */
inline void log_boundary_failure(const char *what_failed, const std::exception_ptr &failure) noexcept {
    try {
        std::string text("unknown exception");
        if (failure) {
            try {
                std::rethrow_exception(failure);
            } catch (const std::exception &e) {
                text = e.what();
            } catch (...) {}
        }
        LOG_ERROR("ChipSwimlane session: %s: %s", what_failed, text.c_str());
    } catch (...) {}
}

/**
 * Close one session run's boundary: publish what this run produced on the
 * host, then snapshot its epoch.
 *
 * `session_run_close` copies the collector's host-phase records — and, where a
 * runtime has one, its extension sections — into the epoch's own metadata. The
 * collector holds exactly one copy of that state, so the publication has to
 * reach it first; a snapshot taken ahead of the publication carries whatever
 * the previous run left behind.
 *
 * A publication that throws, at any point, marks this epoch incomplete before
 * the close reads it. That is what keeps an absent or half-written host state
 * from being sealed as a complete one: the artifact carries whatever did land
 * and settles at a partial verdict. The mark is set before anything is
 * formatted, and the close is attempted whether or not either succeeds — a
 * failure the log cannot describe still costs neither the verdict nor the
 * slot.
 *
 * What survives which failure:
 *
 * - **publication fails, close succeeds** — the epoch is sealed
 *   `metadata_complete: false` with a partial verdict, so the artifact itself
 *   carries the evidence, and the publication's exception is re-raised.
 * - **close fails** — the epoch may never be sealed at all, so *no* file need
 *   carry a verdict for it, and the slot it holds is not returned. The
 *   evidence is then the session's sticky fatal, which a flush and `close()`
 *   both read, plus the log lines above; the close's own exception propagates,
 *   and when the publication had failed too, that one is logged rather than
 *   raised, because only one can be.
 *
 * Both runner bases go through here, which is what keeps the two orders the
 * same.
 */
template <typename PublishHostState>
void close_session_run(
    ChipSwimlaneCollector &collector, uint64_t run_epoch, uint32_t pipeline_slot, bool device_execution_complete,
    PublishHostState &&publish_host_state
) {
    std::exception_ptr publication_failure;
    try {
        std::forward<PublishHostState>(publish_host_state)();
    } catch (...) {
        publication_failure = std::current_exception();
    }
    if (publication_failure) {
        // State first, and it allocates nothing. Describing the failure comes
        // after, where throwing costs only the description.
        collector.session_note_host_state_incomplete();
        log_boundary_failure("a run's host state did not fully publish", publication_failure);
    }
    try {
        collector.session_run_close(run_epoch, pipeline_slot, device_execution_complete);
    } catch (...) {
        std::exception_ptr close_failure;
        try {
            close_failure = std::current_exception();
        } catch (...) {}
        collector.session_note_boundary_close_failed();
        log_boundary_failure("a run boundary could not close its epoch", close_failure);
        if (publication_failure) {
            log_boundary_failure("the boundary that could not close had also failed to publish", publication_failure);
        }
        throw;
    }
    if (publication_failure) std::rethrow_exception(publication_failure);
}

}  // namespace simpler::dfx::session
