# A5 HBG synchronous completion and admission ownership

## Decision

Use Host-generated fanout CSR by default. Keep synchronous wake-list resolution
for valid images without a dependency table. Remove budgeted completion and
route continuations from ordinary, native-control and asynchronous FINISH paths.
Asynchronous condition CHECK remains bounded; it must not wait for an unmet
external condition while holding a business execution slot.

Ready pop or direct resolution already transfers a task to one owner. Retaining
that candidate in private pending storage does not need a second per-task claim
lock or a shared PREPARED phase. The global commit lock remains: different tasks
still compete for admission order. INITIALIZED, record epoch/order, run errors
and capacity are checked before committing; only the successful lock owner can
release it. Record, log entry, ADMITTED and tail retain their publication order.

## Measurement

DSV4.1 PyPTO-lib SWA, Ascend 950DT, CANN 9.3.0, on the same device. L0 uses five
warmups and 100 measured runs; full L3 uses one warmup and three captures, with
function names mapped and no dropped scalar records. Values below are medians
in microseconds. Host latency is excluded.

| Variant | L0 Device | L3 full completion | L3 resolve |
| ------- | --------: | -----------------: | ---------: |
| Synchronous CSR before admission cleanup | 821.674 | 11.352 | 1.710 |
| Synchronous CSR after admission cleanup | **811.472** | **9.247** | 1.786 |
| Synchronous wake-list before admission cleanup | 839.677 | 13.299 | 2.834 |
| Synchronous wake-list after admission cleanup | 830.289 | 9.197 | 1.856 |

CSR Device time improves by 1.24%; the final CSR version is 2.27% faster than the
final wake-list version. Both final variants use byte-identical AICore and
AICPU binaries; only the Host dependency representation differs. This supports
selecting CSR without attributing shared admission improvements to wake lists.

Full completion includes physical completion handling, successor resolution and
any immediate successor admission. The CSR last required predecessor is task 21
in all six before/after captures. For wake lists it changes from task 21 to task
15, so that full-completion delta cannot be wholly attributed to admission
cleanup. These are sequential measurements on a shared host, not interleaved
samples. The **2-microsecond full-completion target remains unmet**; a resolve
interval below two microseconds does not satisfy it.

## Layout and correctness constraints

- Separate concurrently accessed GM control groups by 128-byte sectors. This
  follows the documented SIMT sector model and local AIC/AIV scalar atomic and
  bypass-load probes; it is not a general scalar ordering guarantee. Scalar
  DCache operations retain their 64-byte granularity.
- TaskControl grows from 384 to 640 bytes. Completion timing shares a sector
  with its quiescent completion counter, in a different DCache line. Queue head
  and tail use separate sectors; ticket entries retain their 64-byte stride.
- Removing FINISH continuation storage saves 192 bytes per asynchronous task;
  removing the native-control continuation saves 160 private bytes per core.
  LocalState is 208 bytes. Shared GM alignment is not imposed on private stack
  objects.
- The final AICore cold build takes 136.671 seconds with unchanged `-O3 -g`,
  16-way concurrency and ccache disabled. All 14 objects pass the actual SP and
  local-symbol audit. IR checking reports no findings in 13/14 translation
  units; the known `kernel.cpp` decoder failure is retained as an explicit
  limitation, with its machine code verified unchanged.
- CPU tests cover unique Ready ownership, concurrent CSR joins with queued,
  empty-direct and occupied-direct destinations, duplicate commit rejection,
  retry/error gates and current protocol fields. Hardware checks cover SWA,
  native topology, asynchronous peer counters and inactive-AIC MIX execution.
  CPU results do not establish hardware cache ordering.

Local raw evidence is under `outputs/swa-next-20261112/`: `csrall` and `allsync`
are the pre-cleanup variants; `csrowner` and `ownerclaim` are the final variants.
Their provenance, L0 samples, complete L3 records, comparison reports and
per-case hardware logs are preserved there. The source base is `72fba2e75`.
