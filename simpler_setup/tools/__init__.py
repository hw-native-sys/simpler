# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-user profiling and debug CLIs shipped with the wheel.

Invoke via ``python -m simpler_setup.tools.<name>``:

- ``swimlane_converter``   : perf JSON -> Perfetto/Chrome trace
- ``sched_overhead_analysis``: scheduler overhead deep-dive
- ``critical_path``          : chip swimlane critical-path compute/stall analysis
- ``deps_viewer``           : deps.json -> text or pan/zoom HTML dependency graph
- ``wait_reduction_sim``   : deps.json -> bounded-bitmap WAIT reduction coverage vs full-DAG bound
- ``dump_viewer``           : inspect args dumps
- ``strace_timing``         : per-stage / per-round timing from [STRACE] log markers
- ``hbg.bind_phases``       : per-segment host_build_graph bind statistics from the chip.run.bind.* spans
- ``phase_time_split``      : the same segments split into on-CPU and off-CPU, from per-thread CPU clocks

``hbg/`` and ``tmr/`` are not CLIs but the two runtimes' private DFX vocabularies --
each holds its own TaskId layout, phase set, label style and the lanes only it draws.
A task id or a phase name carries no mark of which runtime produced it, so the common
tools above resolve the runtime a document names and ask ``_runtime_dispatch.get()``
for the implementation rather than importing either package directly. A tool that
belongs wholly to one runtime lives inside it (``hbg.bind_phases``,
``tmr.scope_stats_plot``).
"""
