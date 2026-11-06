# Public SDMA post-id and retained workspace qualification

This A5-only supplement leaves the original three async_completion SDMA cases unchanged. Run66 independent ordinary AIV transfer tasks per rank, each with its own dependent +7 consumer, then2 tasks in a later fully retired run using the exact same retained SDMA-only domain and fixed public queue count48. Every transfer has a distinct aligned data slice and full guards. One transfer per run is4MiB; the rest are32B.

Each task has capacity64 but issues only one SDK request. The public owner partition48 yields one submission group, so receipt addresses and complete uint64 post-id continuity can demonstrate session/flag-slot reuse past the pinned64-slot depth. The test checks every result, source, destination, guard, receipt and full native capture. It does not read or mutate private queue/flag addresses.

Actual prior-event pending samples and actual AIV submitter changes are saved separately. Their absence is reported as missing overlap/handoff evidence.66 posts does not mean66 simultaneously in flight; neither kernel duration nor a successful test proves actual SDK backpressure. There is no speed threshold, blocking probe or new runtime protocol.

The next run starts only after both peers retire the prior run. Same domain configuration, public identity and receipt-prefix/post-id continuity must all hold. Overlapping epochs, queue-count changes and automatic fault recovery are outside this test.

The candidate is unexecuted until root applies and runs it. A5 orchestration/submit/consumer compile and link pass. A5sim orchestration/consumer compile, but its real SDMA submit fails because the pinned CPU SDK has no BuildAsyncSession; there is no simulated-DMA fallback. Real peer SDMA qualification requires two A5 devices. Use the normal queue and isolated outputs. No cancellation or unsafe cleanup behavior is added.
