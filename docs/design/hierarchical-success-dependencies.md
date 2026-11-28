# Explicit predecessor success

A hierarchical `Worker.submit(..., success_predecessor=handle)` can build and
stage its graph while the predecessor executes. Its activation remains held
until the predecessor's complete result succeeds. Physical END, launch
acceptance, and independent communication storage do not establish that success.
Ordinary independent submissions do not acquire this dependency implicitly.

The synchronous `completion_callback(error)` extends the predecessor's completion boundary to
an outer runtime's cleanup and output publication. It executes once after
Simpler finalization and before terminal publication or successor authorization.
An earlier error is preserved; a callback failure becomes the result when no
earlier error exists. The callback remains owned through close and cannot close
its own Worker recursively. A `result()` or `wait()` from the callback that
depends on this result fails immediately. A callback returning an awaitable
fails the completion. After a synchronous graph construction failure, the
callback runs after native cleanup and after the submit lock has been released.
A failed predecessor cancels unactivated staged
successors; their leases remain live until cancellation is acknowledged.
Waiting for a successor also resolves its predecessor, so reverse wait order
does not bypass the success condition. An exception stored as the predecessor's
final result cancels its successor, including a callback-raised
`KeyboardInterrupt` or `SystemExit`. An interruption while observing the
predecessor can propagate to the waiter without authorizing the successor.
Close drains the accepted runs before tearing down the native tree. A waiter
that reaches successor resolution after that drain skips the stale run ID.

Native preparation conservatively retains every eligible communication domain,
including contexts passed through scalar arguments. The Python lifetime records
retain those domains before entering the graph callback, independently of the
graph's declared resource uses. A separate backing table includes logically
released domains until physical reclamation. Admission snapshots and physical
release share a lifetime lock; per-task provenance validation never takes it.
Failure may revoke a domain immediately, but physical reclamation waits until
all such native users finish. These lifetime holds do not add execution
conflicts or authorize activation.

A held graph callback cannot issue synchronous device allocation or copy
controls: those controls require activation and would wait on their own held
run. Prepare those resources before submitting a dependent graph.

`Worker.wait_any(handles)` requests native result observation and waits for any
selected parent to become terminal. Its returned handle still needs `result()`
to run cleanup and the outer callback. A single completion consumer can use
`wake_wait_any()` to refresh its set and `cancel_wait_any()` to stop; both wake
the native condition rather than requiring periodic polling. Worker close
cancels observation waits and drains their operation leases before teardown.

`RunHandle.finalized` reports whether the final result and cleanup outcome have
been published. Unlike `done`, it excludes physical completion awaiting cleanup
or an outer callback. A wait interruption or execution-context refusal may
leave it false; an adapter must keep that run's frame and arguments alive for
another waiter. The query does not consume errors. A true value may represent
failure and does not authorize reuse of quarantined resources.

## Preparation order and resource capacity

A held FIFO head may prepare after all earlier native runs have retired, even
while its success predecessor's outer callback is pending. The parent publishes
an identity- and generation-bound preparation authorization. This authorizes
preparation only; cancellation and failure take precedence, and activation
still requires the predecessor's complete success. Unauthorised staged calls
retain the existing ordering protection against an earlier dispatch arriving
later through another mailbox frame.

`Worker.run_identity_capacity` is a public read-only directory upper bound,
available before initialization. It is not a native admission grant, current
free capacity, or memory reservation. Adapters combine it with their own byte
budgets; actual execution remains governed by native resource admission.
