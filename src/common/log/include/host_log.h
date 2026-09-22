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
 * @file host_log.h
 * @brief Unified Host Logging System
 *
 * One threshold controls DEBUG/INFO/TIMING/WARN/ERROR/NUL. The integer values
 * match Python logging levels. CANN has no TIMING level, so onboard setup maps
 * both TIMING and WARN thresholds to CANN WARN.
 */

#pragma once

#include <atomic>
#include <cstdarg>
#include <cstdint>

#include <sys/types.h>

#include "common/host_log_state.h"
#include "common/log_level.h"

#if defined(__GNUC__) || defined(__clang__)
#define SIMPLER_HOST_LOG_LOCAL __attribute__((visibility("hidden")))
#else
#define SIMPLER_HOST_LOG_LOCAL
#endif

struct SimplerHostSpan;

class SIMPLER_HOST_LOG_LOCAL HostLogger {
public:
    static HostLogger &get_instance();

    // Bind this module-local logger implementation to process-owned state.
    // Must happen during module init, before the module starts worker threads.
    int bind_state(SimplerHostLogState *state);
    // Internal owner-side counterpart used when an embedding executable
    // supplies storage instead of this module's default state. Cross-DSO
    // loaders must use bind_state(), so a transient consumer cannot own the
    // process writer across dlclose().
    int adopt_state(SimplerHostLogState *state);
    SimplerHostLogState *state() const;

    void log(simpler::log::LogLevel level, const char *func, const char *fmt, ...);

    // va_list-taking primitive used by unified_log_* adapters. Caller is
    // responsible for `va_start` / `va_end`.
    void vlog(simpler::log::LogLevel level, const char *func, const char *fmt, va_list args);

    // Owner initialization starts the bounded writer by default. Hierarchical
    // workers defer it until their final local fork so no process forks with a
    // C++ thread already running.
    //
    // Takes effect immediately for every host module in the process, since each
    // reads the bound state's threshold per record — sim AICPU included, because
    // it is one of them. It does NOT reach onboard AICPU, which latches
    // InitArgs.log_level once in simpler_aicpu_init and keeps it for the
    // Worker's life; that is deliberate, not missing (see docs/logging.md,
    // "The threshold is live on the host and fixed on the device"). Recreate the
    // Worker to change the device threshold.
    void set_level(simpler::log::LogLevel level, bool defer_writer = false);
    bool start_writer();

    // A later hierarchical Worker may fork in the same process. Quiesce and
    // join an earlier writer before that fork; fail boundedly if its output is
    // pinned instead of forking while a C++ thread is alive.
    bool prepare_to_fork(uint32_t timeout_ms = 1000);

    // Drain accepted records through this process writer, including bound DSOs.
    // Write failures are tracked separately as drops. Producers must be
    // quiescent if the caller needs a strict shutdown boundary.
    bool flush(uint32_t timeout_ms = 1000);
    uint64_t dropped_records() const;
    // The same total, attributed. A caller sizing the queue or the claim budget
    // needs to know which of them the losses came from; the total alone cannot
    // distinguish "too small" from "too contended" from "destination broken".
    uint64_t dropped_records(SimplerHostLogDropReason reason) const;
    uint64_t pending_records() const;

    // Write this process's records to `path`/host.<pid>.log instead of stderr.
    // Python normally supplies a stable process-session spool; an embedding
    // caller may instead select an explicit persistent destination. The logger
    // never derives a path itself. The first non-empty path wins; a null or
    // empty one leaves the logger on
    // stderr. This is the logger's output, so it applies to every record: no
    // caller declares anything and no record kind is treated specially.
    void set_log_directory(const char *path);

    // The bound output directory, or nullptr while this logger writes to stderr.
    const char *log_directory() const;

    // Runtime modules read this from their bound process-owned state when
    // populating InitArgs.log_level at device init.
    int level() const;
    int cann_level() const;

    // Configure CANN before the device context is opened unless the user has
    // already selected ASCEND_GLOBAL_LOG_LEVEL. The callback shape matches
    // dlog_setlevel and keeps this policy independently testable.
    void configure_cann_log_level(int (*set_level)(int module_id, int level, int enable_event)) const;

    bool is_enabled(simpler::log::LogLevel level) const;

    // Fixed host-span adapter. The logger remains the sole STRACE grammar
    // owner even though its implementation is compiled into several DSOs.
    void log_host_span(const SimplerHostSpan *span);

private:
    HostLogger();
    ~HostLogger();

    HostLogger(const HostLogger &) = delete;
    HostLogger &operator=(const HostLogger &) = delete;
    HostLogger(HostLogger &&) = delete;
    HostLogger &operator=(HostLogger &&) = delete;

    const char *level_name(simpler::log::LogLevel level) const;
    bool emit(const char *level_tag, const char *func, const char *fmt, va_list args);
    bool emit_ungated(const char *level_tag, const char *func, const char *fmt, ...);
    // Write the loss breakdown into the log itself when it has grown since the
    // last report. The counters die with the process, so without this a reader
    // holding only the log file cannot tell that records are missing.
    void report_drops_if_grown();

    std::atomic<SimplerHostLogState *> state_;
    std::atomic<bool> state_owner_{true};
    // The queue implementation is private to host_log.cpp. Keeping only an
    // opaque pointer here prevents its C++ type and inline methods from becoming
    // preemptible symbols in every DSO that compiles the host logger.
    std::atomic<void *> sink_{nullptr};
};

#undef SIMPLER_HOST_LOG_LOCAL
