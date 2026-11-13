# Run Pipeline and the KernelLaunch Stream

`Worker.run` and `ChipWorker.run` do not simply call `rtKernelLaunch` and
return. A run is prepared on the host, launched on the device, and finalized on
the host again; the depth-two run pipeline exists to overlap the first part of
one run with the device work of another.

That split is often mistaken for a duplicate of the CANN stream. This page
states what the stream provides, what it cannot provide for a run, and which
parts of the pipeline are current behavior versus target direction.

## What the stream provides

`rtKernelLaunch` submits a device operation to a stream. A stream is an ordered
queue, so operations submitted to one stream execute in submission order.
Submission itself is asynchronous: `launch_execution` enqueues the run's device
work and returns without waiting for the device to finish it. Completion is
observed later through the run's stream pair and the completion/finalize path.

## What the stream does not provide

### It orders operations, not the state they read

`rtKernelLaunch` copies its argument struct; it does not snapshot the memory
that struct points at. A run's device segment reads a `Runtime` image, graph
definitions, task arguments, staging and DFX regions that live in device
memory. The host writes some of those regions while preparing a run. If the
next run prepared into the same bytes, the device could still be reading the
previous run's content.

The current resource contract therefore classifies a region by whether the host
must give each in-flight run its own copy:

- `HOST_PER_RUN` regions are written by host preparation and need
  `pipeline_depth` copies, selected by the run's lease slot.
- `DEVICE_SCRATCH` regions are populated once and read by serialized device
  operations, so one copy is shared by every run.
- `EXEC_HANDLE` names the execution stream itself; one stream pair carries every
  run on a context.

The practical predicate is **a resource needs a private copy when the host
touches it while a previous run's device work may still read it**. The GM heap
illustrates the distinction: the host uses its address to relocate pointers in
the uploaded image, but does not read or write heap content, so the heap is not
multiplied by pipeline depth. The measured cost of an extra slot is on the
order of a few tenths of a megabyte, not another 256 MB heap.

### It has no resource pool or admission counter

A stream's queue can accept many submissions, but the per-run device resources
are finite. When no slot's resource set is available, something must block the
caller before it prepares into a slot that is still in use. That blocking is
the run pipeline's admission; the selector is the run-resource lease
(`{slot_id, generation}`). The stream cannot supply either because it does not
know those buffers exist.

### A run has a host head and a host tail

A run is not only a device segment. It has:

1. a host head: resolve arguments, acquire resources, build or stage the
   runtime image, and publish the launch state;
2. a device segment: the AICPU/AICore work submitted on the stream pair;
3. a host tail: observe completion, validate results, copy back outputs, drain
   DFX, release the runtime image and return the resource lease.

A stream has no host completion callback. The host state machine that spans
these three parts is real work, and the number of simultaneously live states is
the pipeline depth. A complete run also needs a place to keep its host-owned
state while the previous run's device segment is still executing.

### One run uses two streams that cannot be merged

A run submits on an AICPU stream and an AICore stream. The AICPU Run kernel
spins in the handshake waiting for the AICore workers to report. If both were
serialized onto one stream, the AICore submission would sit behind a spin that
can never end. The two streams therefore stay distinct and have no mutual
ordering; the run's exclusive execution claim — held from launch through drain
and released by finalize — is what makes runs reach the device one at a time.

## What overlaps today

The current onboard program pipeline overlaps **host preparation** with device
execution. At a granted depth of two, run N can be executing while run N+1 is
prepared; run N+2 blocks before its host callback until an admission slot is
free. This is not concurrent device execution. Whole-run FIFO admission and the
exclusive execution claim keep the device segments serialized.

The op-execute watchdog is not a reason to avoid queueing. It counts execution
time, not time spent waiting behind another operation: on a2a3, three two-second
operations queued on one stream under a three-second budget all completed, while
a single six-second operation was reaped with `507018`.

## Why the common depth is two

A run occupies the whole device — a run is never narrower than the device — so
adding submission depth does not add device parallelism. The resource that can
be idle is the host: while run N executes, the host could be preparing run N+1.
One executing run plus one prepared run covers that overlap. A third slot buys
no additional host overlap and costs another copy of every `HOST_PER_RUN`
resource.

Depth is a **grant**, not a global constant. A backend may publish a different
figure: A5 host-build-graph publishes depth one and serializes the second
submission instead of preparing it. The pipeline contract says what a runtime
supports and what a context was granted; it does not itself promise concurrent
device execution.

## Target direction, not current behavior

The following is the direction from the [run pipeline design
discussion](https://github.com/hw-native-sys/simpler/issues/1853) and the
interface review recorded on
[#2223](https://github.com/hw-native-sys/simpler/issues/2223); it is not a
description of the current implementation.

- Unify device submission across Program and Kernel modes on a
  `kernel_launch`-style layer, while preserving Program mode's concurrent host
  orchestration.
- A successor's submission should not normally wait for the predecessor's
  host-side finalize. Shared mutable device resources should be ordered on the
  device instead.
- Classify resources by whether different versions must coexist and by each
  version's last use, rather than by “the host touches it”. Host-initialized
  immutable data can be shared; a device destination updated in stream order
  may be reusable even when each invocation needs a distinct upload source.
- Order simpler-owned uploads, clears and rebuilds—not just kernels—and keep
  the caller-stream predecessor/successor relationships explicit. Accepting a
  request into a host queue is not the same as submitting it to the caller's
  stream.

Device execution parallelism, host preparation concurrency and submission depth
are separate quantities. A run occupying the whole device does not imply a
two-slot lifetime limit; removing that limit also does not make the RTS queue or
device memory unbounded.

## Related

- [task-flow.md](task-flow.md) — pipeline resource leases and whole-run FIFO
  admission in the current execution leaf.
- [chip-level-arch.md](chip-level-arch.md) — the host/AICPU/AICore split and
  the Layer 1/2 `prepare -> launch -> wait -> finalize` lifecycle.
- [#1853](https://github.com/hw-native-sys/simpler/issues/1853) — what the
  pipeline does not overlap and the open design questions.
- [#1717](https://github.com/hw-native-sys/simpler/issues/1717) — intra-run
  launch/orchestration overlap.
