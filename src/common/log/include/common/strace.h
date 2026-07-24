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
 *   inv    process-wide simpler_run() invocation id (atomic-allocated, so
 *          (pid, inv) is unique even across concurrent calls) — grouping key
 *          ONLY (gathers one call's spans together); not a token index. Set
 *          once per call via StraceScope::next_inv().
 *   hid    content-derived callable hash (ELF Build-ID 64); stable across slot
 *          reuse / processes / runs. Parser buckets by hid; the most-frequent
 *          bucket is decode, a once-seen bucket is prefill, etc.
 *   depth  thread-local nesting depth (++ on enter, -- on exit) — the parser
 *          rebuilds the call tree from depth, NOT from timestamp containment.
 *   name   dotted span name (self-locating even without the tree).
 *   ts,dur start + duration in ns on CLOCK_MONOTONIC (steady_clock). ts+dur
 *          maps 1:1 onto a Chrome-trace "X" event; same-host cross-process
 *          comparable. STRACE_A appends caller-supplied "k=v" attrs verbatim.
 *
 * Gated on SIMPLER_HOST_STRACE (default on, see profiling_config.h — no env var)
 * and emitted at LOG_INFO_V9 (the must-see tier, default-visible). In a
 * non-profiling build the macros compile to nothing.
 */

#ifndef PLATFORM_STRACE_H_
#define PLATFORM_STRACE_H_

#include "profiling_config.h"

#if SIMPLER_HOST_STRACE

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

#include "common/unified_log.h"

namespace simpler::strace {

// Per-thread trace state (active invocation id, nesting depth, callable hash).
// Held behind a pthread_key rather than C++ `thread_local`: the repo bans
// thread_local in SOs (ELF TLSDESC issues across dlopen — see
// docs/dynamic-linking.md), so all per-thread state uses POSIX TLS.
struct ThreadState {
    unsigned inv = 0;
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

inline long strace_tid() {
#if defined(__linux__) && defined(SYS_gettid)
    return static_cast<long>(syscall(SYS_gettid));
#else
    // macOS and any platform without SYS_gettid: process id is a sufficient
    // lane key for the trace (per-process invocation grouping still holds).
    return static_cast<long>(getpid());
#endif
}

class StraceScope {
public:
    explicit StraceScope(const char *name, const char *attrs = "") :
        name_(name),
        attrs_(attrs),
        t0_(std::chrono::steady_clock::now()) {
        ++depth();
    }

    ~StraceScope() {
        const auto t1 = std::chrono::steady_clock::now();
        const long long ts = static_cast<long long>(t0_.time_since_epoch().count());
        const long long dur =
            static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count());
        // depth printed is the scope's own level (post-decrement so the
        // outermost scope prints depth=0).
        const int d = --depth();
        LOG_INFO_V9(
            "[STRACE] v=1 pid=%d tid=%ld inv=%u hid=%llx depth=%d name=%s ts=%lld dur=%lld %s",
            static_cast<int>(getpid()), strace_tid(), inv(), static_cast<unsigned long long>(hid()), d, name_, ts, dur,
            attrs_
        );
    }

    StraceScope(const StraceScope &) = delete;
    StraceScope &operator=(const StraceScope &) = delete;

    /** Begin a new invocation: allocate a process-wide unique id and make it the
     *  active id for this thread. Call once at simpler_run entry.
     *
     *  The id generator is a process-wide atomic, not the per-thread counter, so
     *  `(pid, inv)` uniquely identifies one invocation even when several threads
     *  run simpler_run concurrently in the same process — otherwise each thread
     *  would start at 1 and the parser would merge their spans. The resolved id
     *  is stored in the per-thread slot (`inv()`) so nested scopes / emit_span_at
     *  on this thread read the right value. */
    static unsigned next_inv() {
        static std::atomic<unsigned> global_inv{0};
        const unsigned id = global_inv.fetch_add(1, std::memory_order_acq_rel) + 1;
        inv() = id;
        return id;
    }
    /** Set the callable hash for spans emitted on this thread. */
    static void set_hid(uint64_t h) { hid() = h; }
    /** Propagate an invocation onto an auxiliary worker thread. */
    static void set_context(unsigned invocation, uint64_t h) {
        inv() = invocation;
        hid() = h;
        depth() = 0;
    }
    /** Current invocation id / callable hash for this thread (for emit_span_at). */
    static unsigned current_inv() { return inv(); }
    static uint64_t current_hid() { return hid(); }

private:
    static unsigned &inv() { return strace_state()->inv; }
    static int &depth() { return strace_state()->depth; }
    static uint64_t &hid() { return strace_state()->hid; }

    const char *name_;
    const char *attrs_;
    std::chrono::steady_clock::time_point t0_;
};

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
    LOG_INFO_V9(
        "[STRACE] v=1 pid=%d tid=%ld inv=%u hid=%llx depth=%d name=%s ts=%lld dur=%lld %s", static_cast<int>(getpid()),
        strace_tid(), StraceScope::current_inv(), static_cast<unsigned long long>(StraceScope::current_hid()), depth,
        name, ts_ns, dur_ns, attrs
    );
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
/** Set the callable hash for subsequent spans on this thread. */
#define STRACE_SET_HID(h) ::simpler::strace::StraceScope::set_hid(h)
/** Propagate the active invocation/hash onto an auxiliary thread. */
#define STRACE_SET_CONTEXT(inv, h) ::simpler::strace::StraceScope::set_context((inv), (h))
/** Emit a device-domain span (device-clock start `ts_ns` + measured `dur_ns`). */
#define STRACE_DEV_SPAN_AT(name, ts_ns, dur_ns, depth) \
    ::simpler::strace::emit_span_at((name), (ts_ns), (dur_ns), (depth))
/** Emit a host steady-clock span measured outside an RAII scope. */
#define STRACE_HOST_SPAN_AT(name, ts_ns, dur_ns, depth, attrs) \
    ::simpler::strace::emit_span_at((name), (ts_ns), (dur_ns), (depth), (attrs))

#else  // !SIMPLER_HOST_STRACE

#define STRACE(name) ((void)0)
#define STRACE_A(name, attrs) ((void)0)
#define STRACE_NEW_INV() ((void)0)
#define STRACE_SET_HID(h) ((void)0)
#define STRACE_SET_CONTEXT(inv, h) ((void)0)
#define STRACE_DEV_SPAN_AT(name, ts_ns, dur_ns, depth) ((void)0)
#define STRACE_HOST_SPAN_AT(name, ts_ns, dur_ns, depth, attrs) ((void)0)

#endif  // SIMPLER_HOST_STRACE

#endif  // PLATFORM_STRACE_H_
