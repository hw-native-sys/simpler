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

#include <stdint.h>

#include "common/host_phase_kind.h"

// Minor faults the process has taken. First touch of a freshly mapped region traps
// once per page, and the bind maps its shared-memory mirror and arenas per call, so
// a phase's fault count is what separates work from page-table cost — a count, so it
// does not move with how loaded the box is.
//
// CPU time answers what a count cannot: whether a phase's wall time was spent running
// or waiting. It comes from per-thread CPU clocks and never from rusage —
// `ru_utime`/`ru_stime` are accounted per scheduler tick, 10 ms at CLK_TCK=100, so on
// a phase of a millisecond they quantise to either zero or a whole tick and no split
// survives. A per-thread clock reads the scheduler's running total in nanoseconds.
//
// Two CPU figures, because a phase runs on more than one thread. `cpu_ns` is the bind
// thread's own, so a phase's `dur - cpu_ns` is the time that thread spent off CPU;
// `recorder_cpu_ns` is every recording worker's summed, so its ratio to `dur` is how
// many threads' worth of work ran alongside. Only the first can be subtracted from the
// wall. `thread_minflt` is the same split applied to the fault count: it says how much
// of `minflt` the bind thread took rather than the recorders.
struct BindKernelCounters {
    uint64_t minflt;
    uint64_t thread_minflt;
    uint64_t nivcsw;  // involuntary: the scheduler took the CPU away
    uint64_t nvcsw;   // voluntary: the thread blocked
    uint64_t cpu_ns;
    uint64_t recorder_cpu_ns;
};

// One segment's two origins: the instant its duration measures from and the counter
// mark its faults measure from. Both live in the caller's frame, so a phase that
// opens while another is still open cannot disturb it — the two would share a single
// mark if one were held here, and the outer phase would then report counts measured
// from the inner phase's start while its duration still spanned its own.
//
// A phase's own counts are the delta since its own start, so they cover the same
// span its duration does. The marks do not partition the bind: the stretches
// between one phase's close and the next one's open belong to neither, in counts
// exactly as in time. The process-wide counters are process-wide on purpose, so a
// phase that runs while the Graph recorders are working is charged their faults too
// — it is the bind's total page-table cost that is being attributed, not one
// thread's.
struct BindPhaseMark {
    int64_t start_ns{0};
    BindKernelCounters counters{};
};

// Open one segment. Every phase start comes from here, so a phase's two spans cannot
// drift apart — a fault count taken over a wider span than the clock describes work
// the phase does not contain.
BindPhaseMark bind_phase_begin();

// Close one segment of the bind path, recording it and keeping its attributes for
// the line the breakdown prints at the end of the bind.
//
// The breakdown is LOG_TIMING lines rather than `[STRACE]` markers on purpose:
// the marker grammar is the platform's public per-run-stage contract (see
// runtime_c_api.h and docs/dfx/host-trace.md) whose consumers key off a fixed
// stage set, while everything below is host_build_graph's internal breakdown of
// one stage. LOG_TIMING sits at the default log threshold, so these are visible
// without a flag and at any --rounds.
void record_bind_phase(HostPhaseKind kind, const BindPhaseMark &mark, const char *attrs = "", uint64_t payload = 0);
