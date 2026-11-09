# A5 resident peer COUNTER qualification

`test_peer_counter.py` contains two manual two-device L3 tests. It combines business
behavior checks with diagnostics for the current A5 resident implementation;
the entire module is not an implementation-independent oracle. The existing
`async_completion` tests keep their own DAGs and oracles.

Both `a5sim` and `a5` are supported test targets. The simulator maps each peer
window through shared memory and executes the payload stores, atomic
notifications, resident waits and consumers across two chip workers. An
`a5sim` pass provides end-to-end functional and diagnostic evidence; it does not
qualify A5 device cache visibility or physical peer transport. M03 hardware
qualification still requires these same two cases on actual A5 devices.

| Test | Business behavior |
| ---- | ----------------- |
| `test_peer_counter_epoch_boundaries` | Three serial rounds use `(signal epoch, payload tag, reset) = (1,101,true), (2,102,false), (1,103,true)`. Actual peer payload reaches a real consumer, stale data cannot satisfy the current tag, and reset/reuse happens only after both ranks complete. |
| `test_peer_counter_c_plus_one_progress` | The receiver registers `C+1` finite AIV waiters, where `C` comes from `rt_available_aiv_count()`. All registrations precede the source publication, and every real consumer produces the expected output. This proves progress and physical core reuse without claiming simultaneous occupancy of all cores. |

`_check_business_behavior` checks actual peer data, `payload + 7` consumer
outputs, source immutability, signal epochs, packet fields, guards, and progress.
The local registration counter is a fixture barrier, not a peer completion ACK.
Its successful snapshot travels in a real peer ready packet followed by a real
notification. The source's resident ready waiter must finish before its main
publisher reads that packet and sends the main payload.

`_check_resident_qualification` separately requires actual native schema10/11,
the full causality audit and explicit execution identities, fresh runtime
epochs, topology agreement, and the expected successful single-condition
COUNTER groups. Every round must include at least one receiver group with
`observations > registered_count` (`> 1` here). For an audited successful
single-condition group this proves that resident CHECK read an unmet counter
before its final successful observation. `pending_count > 0` alone cannot
prove this: acquire or budget yields also count as PENDING service slices.

Both categories fail the test strictly. A behavior mismatch is reported as a
behavior failure. Missing observations are reported as missing diagnostic
coverage; that outcome does not by itself prove a runtime business error.
Passing business checks alone does not satisfy this module's resident
qualification. Captures and complete Host diagnostics remain available on
failure, with an explicit caveat that Host buffers after a failed invocation do
not prove device copyback completed.

After validating the ready packet, the source main publisher executes a fixed
20 ms clock-bounded calculation loop before its first actual TSTORE/notify.
The fixture uses the same CPU-stub steady-clock nanoseconds and hardware system
counter semantics as `async_completion/kernels/device.h`. This occupies one
source AIV while the already-registered receiver has independent cores for
CHECK. It reads no peer event or private scheduler counter and adds no DAG edge,
API or configuration parameter. It creates an intentional observation window,
but does not guarantee a particular schedule or relax the strict observation
assertion. The delay is fixture latency, not a performance threshold. The
registration gate remains bounded at 2 seconds and the enclosing wait at
30 seconds.

Both channels publish their payload before the COUNTER notification, without
resource or consumer declarations. The publisher retires MTE3 and completes its
release barriers before notify; actual
consumers rely on the common business-kernel entry for local DCache acquisition.
Shared communication allocations stay alive until the joint two-rank
invocation completes, and are not overwritten before the consumers finish.
The two manual cases require no peer rescue from AIC, blocking comparison or
extra communication workspace.

`test_signal_payload.py` adds six A5 cases adapted from the peer implementation.
They cover two/four ranks, four independently allocated payload fields, full-pool
cache warmup, clean/dirty release, AIC/AIV scalar and AIV TLOAD consumers, and
scope release or retained-domain reuse. Every active consumer validates the
complete payload; inactive outputs retain their poison values. The existing
pending-close test covers Worker.close draining accepted work separately.
