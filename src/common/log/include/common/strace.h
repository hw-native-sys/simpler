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
 * @file strace.h
 * @brief Host-side RAII trace markers ("simpler trace" — analogous to Android
 *        atrace/systrace) emitted to the unified log.
 *
 * A consumer (e.g. pypto-serving) reads host-side per-stage timing of each
 * simpler_run() purely from the log — no code change on its side, no API
 * contract. Markers go to the log (not a return value) because run() returns
 * nothing: an L3 parent and its L2 children are observed uniformly through the
 * one sink both processes share.
 *
 * Each span is one line, emitted on scope exit:
 *
 *   [STRACE] v=1 pid=<pid> tid=<tid> inv=<n> hid=<hash> depth=<d> \
 *            name=<dotted.name> ts=<ns> dur=<ns> [k=v ...]
 *
 *   v      format version (parser branches on it; lets device-side markers
 *          align later by reusing the same prefix + adding fields)
 *   pid    process id  (L3 parent vs each L2 child are distinct pids)
 *   tid    thread id   (multi-threaded orch stays attributable)
 *   inv    64-bit process-wide simpler_run() invocation id (atomic-allocated, so
 *          (pid, inv) is unique even across concurrent calls) — grouping key
 *          ONLY (gathers one call's spans together); not a token index. A
 *          lexical call sets it via StraceScope::next_inv(); a phased call
 *          allocates once and binds that id in each phase.
 *   hid    content-derived callable hash (ELF Build-ID 64); stable across slot
 *          reuse / processes / runs. Parser buckets by hid; the most-frequent
 *          bucket is decode, a once-seen bucket is prefill, etc.
 *   depth  thread-local nesting depth (++ on enter, -- on exit) — the parser
 *          rebuilds the call tree from depth, NOT from timestamp containment.
 *   name   dotted span name (self-locating even without the tree).
 *   ts,dur start + duration in ns on CLOCK_MONOTONIC (steady_clock). ts+dur
 *          maps 1:1 onto a Chrome-trace "X" event; same-host cross-process
 *          comparable. STRACE_A passes caller-supplied "k=v" attrs to the
 *          logger for delimiter encoding and bounded rendering.
 *
 * Gated on SIMPLER_HOST_STRACE (default on, see profiling_config.h — no env var)
 * and emitted at LOG_TIMING (the default-visible timing tier). In a
 * non-profiling build the macros compile to nothing.
 */

#pragma once

#include <cstdint>

#include "profiling_config.h"

#if SIMPLER_HOST_STRACE

#include <pthread.h>

#include <atomic>
#include <cstdlib>

#include "common/host_span.h"
#include "common/log_clock.h"

namespace simpler::strace {

// Per-thread trace state (active invocation id, nesting depth, callable hash).
// Held behind a pthread_key rather than C++ `thread_local`: the repo bans
// thread_local in SOs (ELF TLSDESC issues across dlopen — see
// docs/dynamic-linking.md), so all per-thread state uses POSIX TLS.
struct ThreadState {
    uint64_t inv = 0;
    int depth = 0;
    uint64_t hid = 0;
};

inline pthread_key_t &strace_key() {
    static pthread_key_t key;
    return key;
}

inline ThreadState *strace_state() {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, []() {
        pthread_key_create(&strace_key(), [](void *p) {
            std::free(p);
        });
    });
    auto *st = static_cast<ThreadState *>(pthread_getspecific(strace_key()));
    if (st == nullptr) {
        st = static_cast<ThreadState *>(std::calloc(1, sizeof(ThreadState)));
        pthread_setspecific(strace_key(), st);
    }
    return st;
}

inline void emit_record(
    const char *name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, int64_t timestamp_ns,
    int64_t duration_ns, const char *attributes
) {
    const SimplerHostSpan span{invocation_id, callable_hash, depth, 0, timestamp_ns, duration_ns, name, attributes};
    unified_log_host_span(&span);
}

class StraceScope {
public:
    explicit StraceScope(const char *name, const char *attrs = "") :
        name_(name),
        attrs_(attrs),
        t0_ns_(simpler::log::monotonic_now_ns()) {
        ++depth();
    }

    ~StraceScope() {
        const int64_t t1_ns = simpler::log::monotonic_now_ns();
        // depth printed is the scope's own level (post-decrement so the
        // outermost scope prints depth=0).
        const int d = --depth();
        emit_record(name_, inv(), hid(), d, t0_ns_, t1_ns - t0_ns_, attrs_);
    }

    StraceScope(const StraceScope &) = delete;
    StraceScope &operator=(const StraceScope &) = delete;

    /** Begin a lexical invocation: allocate a process-wide unique id and make
     *  it the active id for this thread. Call once at simpler_run entry.
     *
     *  The id generator is a process-wide atomic, not the per-thread counter, so
     *  `(pid, inv)` uniquely identifies one invocation even when several threads
     *  run simpler_run concurrently in the same process — otherwise each thread
     *  would start at 1 and the parser would merge their spans. The resolved id
     *  is stored in the per-thread slot (`inv()`) so nested scopes / emit_span_at
     *  on this thread read the right value. */
    static uint64_t allocate_inv() {
        static std::atomic<uint64_t> global_inv{0};
        return global_inv.fetch_add(1, std::memory_order_acq_rel) + 1;
    }
    static uint64_t next_inv() {
        const uint64_t id = allocate_inv();
        inv() = id;
        return id;
    }
    /** Set the callable hash for spans emitted on this thread. */
    static void set_hid(uint64_t h) { hid() = h; }
    /** Current invocation id / callable hash for this thread (for emit_span_at). */
    static uint64_t current_inv() { return inv(); }
    static uint64_t current_hid() { return hid(); }

private:
    static uint64_t &inv() { return strace_state()->inv; }
    static int &depth() { return strace_state()->depth; }
    static uint64_t &hid() { return strace_state()->hid; }

    const char *name_;
    const char *attrs_;
    int64_t t0_ns_;
};

/**
 * Temporarily bind one invocation to the current thread at a known parent
 * depth. The previous state is restored on exit, so phased callers do not
 * leave invocation identity or synthetic nesting active between API calls.
 */
class StraceContextScope {
public:
    StraceContextScope(uint64_t inv, uint64_t hid, int base_depth) :
        state_(strace_state()),
        saved_(*state_) {
        state_->inv = inv;
        state_->hid = hid;
        state_->depth = base_depth;
    }

    ~StraceContextScope() { *state_ = saved_; }

    StraceContextScope(const StraceContextScope &) = delete;
    StraceContextScope &operator=(const StraceContextScope &) = delete;

private:
    ThreadState *state_;
    ThreadState saved_;
};

/** Current steady-clock timestamp in the marker grammar's nanosecond unit. */
inline long long strace_now_ns() { return static_cast<long long>(simpler::log::monotonic_now_ns()); }

/**
 * Emit a marker for a span whose duration was measured elsewhere (e.g. a device
 * phase: AICPU cycles → ns). Shares the current thread's inv/hid grouping so the
 * parser nests it under the host call tree. `ts_ns` is a device-domain start (ns
 * on a common device-clock origin shared by all device spans of this
 * invocation), NOT the host clock — so the device spans are comparable to each
 * other (the orchestrator∪scheduler merged window, "Effective", is recoverable)
 * and the sub-phases nest/position correctly. `clk=dev` tags this; `depth` is
 * the explicit tree depth.
 */
inline void
emit_span_at(const char *name, long long ts_ns, long long dur_ns, int depth, const char *attrs = "clk=dev") {
    emit_record(name, StraceScope::current_inv(), StraceScope::current_hid(), depth, ts_ns, dur_ns, attrs);
}

/** Emit an explicitly timed host-domain span in the active invocation. */
inline void emit_host_span_at(const char *name, long long ts_ns, long long dur_ns, int depth, const char *attrs = "") {
    emit_span_at(name, ts_ns, dur_ns, depth, attrs);
}

}  // namespace simpler::strace

// Concatenation helpers so each scope gets a unique variable name per line.
#define STRACE_CAT_(a, b) a##b
#define STRACE_CAT(a, b) STRACE_CAT_(a, b)

/** Open a trace scope spanning the enclosing block. */
#define STRACE(name) ::simpler::strace::StraceScope STRACE_CAT(_strace_, __LINE__)(name)
/** Like STRACE but appends a caller-formatted "k=v ..." attribute string. */
#define STRACE_A(name, attrs) ::simpler::strace::StraceScope STRACE_CAT(_strace_, __LINE__)(name, attrs)
/** Begin a new invocation group (call once per simpler_run); returns inv id. */
#define STRACE_NEW_INV() ::simpler::strace::StraceScope::next_inv()
/** Allocate an invocation id without changing the current thread's context. */
#define STRACE_ALLOC_INV() ::simpler::strace::StraceScope::allocate_inv()
/** Set the callable hash for subsequent spans on this thread. */
#define STRACE_SET_HID(h) ::simpler::strace::StraceScope::set_hid(h)
/** Bind an existing invocation + its synthetic parent depth for this scope. */
#define STRACE_CONTEXT(inv, hid, depth) \
    ::simpler::strace::StraceContextScope STRACE_CAT(_strace_context_, __LINE__)((inv), (hid), (depth))
/** Read the current host monotonic clock in nanoseconds. */
#define STRACE_NOW_NS() ::simpler::strace::strace_now_ns()
/** Emit a host-domain span measured across disjoint API calls. */
#define STRACE_HOST_SPAN_AT(name, ts_ns, dur_ns, depth) \
    ::simpler::strace::emit_host_span_at((name), (ts_ns), (dur_ns), (depth))
/** Emit a disjoint host-domain span with caller-formatted attributes. */
#define STRACE_HOST_SPAN_AT_A(name, ts_ns, dur_ns, depth, attrs) \
    ::simpler::strace::emit_host_span_at((name), (ts_ns), (dur_ns), (depth), (attrs))
/** Emit a device-domain span (device-clock start `ts_ns` + measured `dur_ns`). */
#define STRACE_DEV_SPAN_AT(name, ts_ns, dur_ns, depth) \
    ::simpler::strace::emit_span_at((name), (ts_ns), (dur_ns), (depth))
/**
 * Like STRACE_DEV_SPAN_AT with caller-formatted attributes. `attrs` must carry
 * `clk=dev` itself — the parser keys the device clock domain off that token, and
 * supplying attrs replaces the default that would otherwise provide it.
 */
#define STRACE_DEV_SPAN_AT_A(name, ts_ns, dur_ns, depth, attrs) \
    ::simpler::strace::emit_span_at((name), (ts_ns), (dur_ns), (depth), (attrs))

#else  // !SIMPLER_HOST_STRACE

#define STRACE(name) ((void)0)
#define STRACE_A(name, attrs) ((void)0)
#define STRACE_NEW_INV() UINT64_C(0)
#define STRACE_ALLOC_INV() UINT64_C(0)
#define STRACE_SET_HID(h) ((void)0)
#define STRACE_CONTEXT(inv, hid, depth) ((void)0)
#define STRACE_NOW_NS() 0LL
#define STRACE_HOST_SPAN_AT(name, ts_ns, dur_ns, depth) ((void)0)
#define STRACE_HOST_SPAN_AT_A(name, ts_ns, dur_ns, depth, attrs) ((void)0)
#define STRACE_DEV_SPAN_AT(name, ts_ns, dur_ns, depth) ((void)0)

#endif  // SIMPLER_HOST_STRACE
