# A5 Host Build Graph runtime

## Bind and execute

1. The worker registers the callable and loads its host orchestration library.
2. Bind runs orchestration, resolves task dependencies and validates recorded
   Graphs. The Host assigns the task blocks to physical lanes and packs
   calendar ranges, kernel bindings and dependency targets. The graph image
   preserves each top-level task's identity and keeps its argument pools.
3. Bind leases SoC SRAM slots and publishes the image and its bootstrap
   address. The AICPU validates the reported core topology, initializes the
   lease and releases execution.
4. The AICore indexes its own range list using its physical worker index. For
   a Graph range it waits for the range entry dependencies and the required
   prefix of child bindings. AICPU materializes those bindings in bounded
   slices while AICores run. Ordinary ranges carry prebound arguments.
5. Each AICore checks its row's conditions, performs any MIX or
   `sync_start` rendezvous, runs the kernel and retires the row. The output
   fence precedes DONE publication. A deferred row frees the physical lane
   before its DONE credit, which a polling consumer or range tail publishes.
6. AICPU observes the run's sink thresholds and deferred settlement, then
   requests exit. It waits for confirmed retirement before reusing the run's
   storage or collecting staged PMU records.

See the [calendar scheduler](../README.md) for range boundaries, SoC SRAM
capacity, completion and defer details.

## Storage

| Storage | Run data |
| ------- | -------- |
| GM | Graph definitions, compact task and argument image, per-lane calendar and binding tables, Graph prefix watermarks, deferred registrations, sampled timing and diagnostic buffers |
| SSBUF | Per-cluster rendezvous, current block context and the lane's defer marker |
| SoC SRAM | DONE, START and range counters, including sticky defer bits |
| SPR | Core launch and acknowledged exit |

The task window covers all top-level submissions. Each Graph uses one outer
slot, and its children get storage in the invocation's graph heap. Bind
validates the required storage and notification capacity before the lanes run.
Graph children execute with invocation-relative bindings and remain identifiable
by the outer task and the child's local ID.

## Online profile-guided planning

The worker caches planned calendars in memory. On a miss, the Host generates
a calendar from the submitted DAG and execution shapes and enables the online
samples for the chosen invocation. A completed run returns kernel start,
output-fence end, defer registration and defer completion timings to the Host.
The Host can use those timings to replace the cached calendar. Repeated Graphs
with the same key share a plan and retain separate bindings and counters.
Closing the worker releases its cache.

## Diagnostics and errors

Native launch retirement and Host status remain part of the run contract. A
timeout or lane error is reported through the run's status. Return from a
shutdown partition alone is insufficient to release its storage.

A5 HBG chip swimlane L1 shows actual kernel intervals, L2 shows calendar
transitions, L3 adds scheduler stages and L4 adds instrumented memory and
notification operations. Argument dumps use the selected kernel's before and
after boundaries. PMU samples are attached to the exact kernel execution and
collected after core retirement. See [Profiling levels](profiling_levels.md).
