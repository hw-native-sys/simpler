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

#include "host_build_graph/host/bind_phase.h"

#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <sys/resource.h>

#include "host_build_graph/host/graph_recorder_pool.h"
#include "host_build_graph/host/host_phase_trace.h"

// Host monotonic clock, shared with the record pool so spans and records can be
// read against each other.
static int64_t bind_now_ns() { return static_cast<int64_t>(host_phase_now_ns()); }

static uint64_t thread_cpu_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

static BindKernelCounters bind_kernel_counters() {
    BindKernelCounters counters{};
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        counters.minflt = static_cast<uint64_t>(usage.ru_minflt);
        counters.nivcsw = static_cast<uint64_t>(usage.ru_nivcsw);
        counters.nvcsw = static_cast<uint64_t>(usage.ru_nvcsw);
    }
#if defined(__linux__)
    // RUSAGE_THREAD is a Linux extension. Darwin has no per-thread rusage, so
    // thread_minflt stays zero there and the process total in minflt is the whole of
    // what is available.
    rusage thread_usage{};
    if (getrusage(RUSAGE_THREAD, &thread_usage) == 0) {
        counters.thread_minflt = static_cast<uint64_t>(thread_usage.ru_minflt);
    }
#endif
    counters.cpu_ns = thread_cpu_ns();
    counters.recorder_cpu_ns = graph_recorder_pool().worker_cpu_ns();
    return counters;
}

BindPhaseMark bind_phase_begin() {
    BindPhaseMark mark{};
    if (host_phase_breakdown_enabled()) {
        mark.counters = bind_kernel_counters();
    }
    mark.start_ns = bind_now_ns();
    return mark;
}

void record_bind_phase(HostPhaseKind kind, const BindPhaseMark &mark, const char *attrs, uint64_t payload) {
    const int64_t start_ns = mark.start_ns;
    if (!host_phase_breakdown_enabled()) {
        host_phase_record_bind(static_cast<uint32_t>(kind), static_cast<uint64_t>(start_ns), attrs, payload);
        return;
    }
    const BindKernelCounters now = bind_kernel_counters();
    // Closes both spans at the same instant. The counters have to be read before the
    // attribute string can be formatted, and the record path takes four early returns
    // before it would reach a clock of its own, so leaving the end timestamp to it
    // would let the duration cover work the counts do not. Both ends of the phase now
    // read counters and then the clock, one call apart.
    const int64_t end_ns = bind_now_ns();
    auto since = [](uint64_t current, uint64_t mark) {
        return current >= mark ? current - mark : 0;
    };
    // Counters first, caller attributes after. The buffer is the span attribute
    // field's own width, so an overlong segment loses its tail: a caller's
    // quantity is recoverable from the artifact's `detail` or the run's own
    // sizing, while the counters are this record's only copy and phase_time_split
    // has nothing to fall back on.
    char with_counters[kBindAttrsCapacity];
    const int formatted = snprintf(
        with_counters, sizeof(with_counters),
        "minflt=%" PRIu64 " tminflt=%" PRIu64 " nivcsw=%" PRIu64 " nvcsw=%" PRIu64 " cpu_ns=%" PRIu64
        " rec_cpu_ns=%" PRIu64 "%s%s",
        since(now.minflt, mark.counters.minflt), since(now.thread_minflt, mark.counters.thread_minflt),
        since(now.nivcsw, mark.counters.nivcsw), since(now.nvcsw, mark.counters.nvcsw),
        since(now.cpu_ns, mark.counters.cpu_ns), since(now.recorder_cpu_ns, mark.counters.recorder_cpu_ns),
        *attrs == '\0' ? "" : " ", attrs
    );
    // Mark the cut here, because nothing downstream can: this buffer is exactly
    // as wide as the span's attribute field, so the value the logger receives
    // already fits and its own `~` marker never fires. An unmarked truncation
    // reads as a complete attribute list that is one field short.
    if (formatted >= static_cast<int>(sizeof(with_counters))) {
        with_counters[sizeof(with_counters) - 2] = '~';
    }
    host_phase_record_bind(
        static_cast<uint32_t>(kind), static_cast<uint64_t>(start_ns), with_counters, payload,
        static_cast<uint64_t>(end_ns)
    );
}
