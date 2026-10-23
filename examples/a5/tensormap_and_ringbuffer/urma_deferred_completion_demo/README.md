# A5 URMA deferred completion

Each of two ranks reads the peer's Device window into `out`. A separate
consumer depends on `out` and computes `result = out + 1`. Both outputs are
checked against nonzero golden data. Completion is polled by the TMR scheduler;
submission does not synchronously wait on the AIV.

The suite contains four pytest items:

- `TestUrmaDeferredCompletionDemo`: one read and dependent consumer per rank.
- `TestUrmaRepeatedPosts`: four slices posted to the same peer queue.
- `TestUrmaDomainLifecycle`: multiple domains, logical release while another
  domain is live, and three complete runs that recreate domains on the same
  worker. Control operations precede task submission in each run.
- `test_urma_native_teardown`: real base and child HCCL communicators, active
  and retired domains, and explicit `comm_destroy` success before device reset.

Use the explicit URMA build and strict verification commands in
[the A5 URMA guide](../../../../docs/a5-urma.md). The PR and Daily URMA jobs
require all four items to execute. The default SDMA build intentionally skips
these tests; that skip is not URMA acceptance.

The consumer kernel is shared in behavior with
[the SDMA demo](../sdma_async_completion_demo/). The runtime build selects one
workspace overlay, so comparing transports requires rebuilding.
