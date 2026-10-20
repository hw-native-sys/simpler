# Kernel integration validation

## Scope and frozen PR heads

This records local integration of every submitted PR in the supplied kernel
pipeline. It does not merge or change the GitHub PRs. The first audit froze
these heads on 2026-09-11; the heads were re-surveyed on 2026-09-14, and the
"Now" column is what each PR carried then, except #2190, which was re-read on
2026-09-16 and is shown at that head. A head that moved is not by itself
a change this line took — D15 in the integration log says which of the moves
were adopted and which were deferred.

| PR | Contribution | Audited head (2026-09-11) | Now (2026-09-14) |
| -- | ------------ | ------------------------- | ---------------- |
| #2064 | K1 ABI, lifecycle, invocation header, scalar signature | `2ab04b1b1ef76a88743950a4f3e6c78d1bd8be26` | merged to `main` as `1d1ddc815` |
| #2171 | H1 host graph build and H2D separation | `458c0243ccdb9933c9c4bf3928021f83c446b117` | `caf9d9dae7f4` — restructured, deferred |
| #2172 | HBG resources and stream contract | `7523052fee0066c0d289a688b48f76b10c7d54ab` | `f182ec80e801` — restructured, deferred |
| #2173 | HBG context resource preparation and freeze | `15741ac05b24908c71703c72f2549f5fe0e9fa53` | unchanged |
| #2174 | H2 immutable graph packets | `1fe6bf53f0af9b6a02eeb2edff080a8bc7e7467a` | unchanged |
| #2175 | H3 execution slot sealing and validation | `89b00a9cad3bdc32dee3ec2ab1a65d21a8a23c20` | unchanged |
| #2176 | K2 persistent execution resources and capture test | `533f67a13f4c11a02face67a4fc8609926b71ef8` | `abbe07f53cfa` — adopted, including its event topology |
| #2177 | TMR resource contract and initialization admission | `2153406a0420e8de17cf7397d30864939b1bca0c` | `b569474b2833` — adopted |
| #2180 | K4 per-invocation snapshots | `b0943525dd8eaeb486bca92bbf5480a37eb61fcb` | unchanged |
| #2185 | C++, nanobind and Python kernel entry points | `32dd9442f79bb936541cf41a6a08d7838d450134` | `7058de9f9f4d` — adopted |
| #2187 | Three-stream launch binder and compensation | `136e9712d22c495ac921a6900c8f24fa9b8ebcf3` | unchanged; its binder rewritten to the chained topology |
| #2189 | K5 resident and per-invocation execution state | `b6435e4858b1e0443966d664cca478276baefa55` | unchanged |
| #2190 | Callable cache, residency and generation validation | `2c876478dbf41fad5c0019f081261d912e9b687e` | `a8b19e8a0a70` — fully adopted, including the block allocator and capture-callable preparation |
| #2193 | K3 capacity refusal without releasing existing resources | `b8e739d9d8143ca6f35bf2177241976f900e2ab7` | unchanged |

The two audited HBG heads were force-pushed away and are no longer fetchable;
their contributions on this line are the ones the 2026-09-11 audit integrated.

Conflicting contracts and adaptations are explained in
[the integration log](../INTEGRATION-LOG.md). Tests exercise the integrated
behavior, including common ABI adaptation, rather than treating successful
source import as runtime evidence.

## Acceptance coverage

- Public TMR C API: caller-owned ACL/device/stream, initialization, immutable
  callable preparation, two real 16,384-element numerical launches with
  different addresses and scalar values, stable memory, and complete close.
- Lifecycle failures: failed initialization, event/stream cleanup retry,
  forgotten close, fatal-device abandonment, invalid current device and
  device-query errors. Rejected work preserves the context and can recover.
- Registration failure: an orchestration image the AICPU cannot load is
  admitted by `prepare_callable`, which waits for nothing; the caller's own
  drain reports the refusal, and the context still closes clean.
- In-capture preparation: two registrations under an open GLOBAL and an open
  THREAD_LOCAL capture, the borrowed thread mode restored each time, and the
  recorded graph replayed with committed memory unmoved.
- Invocation transport: 24 gated asynchronous minimum/maximum packets with
  host buffers overwritten before device consumption; common header, residency,
  malformed input, stale generation and reserved-field rejection.
- TMR execution: serial and parallel A/B/A invocation isolation, multi-thread
  error propagation, pre-window AICore cancellation and recovery.
- HBG submitted modules: graph build, capacity planning, preparation/freeze,
  immutable serialization, slot identity and forged-pointer rejection.
- Revised K1: historical callable padding, signature-derived scalar counts,
  fixed header offsets, conditional runtime symbol loading and failed-init retry.
- K2 capture: both internal streams, 100 replays and zero forbidden API calls.
- Regression: C++ and Python unit suites, both simulation architectures,
  supported a2a3 onboard scenes, and a separate SDMA phase.

## Execution record

Validated indexed source/test snapshot:
`0f56798f645a607e1f4a10bded08644d60be0bff` (delta-18), on branch
`skx/kernel-collect-all` above HEAD `422621cc`. Runtime build snapshot
`865a4a067d394fea7b1fc5c08feda8ff26799e89` has identical `src/` and `python/`
files; subsequent deltas only corrected test expectations, a fake-DSO resource
contract and this report. The remote checkout has no Git metadata; a Git blob
comparison verified all 2,513 tracked files and both intended source deletions.

The isolated validation checkout uses a project-local system-site-packages
venv, explicit dependency paths, and PTO-ISA pinned to
`5a4f74cbf627d4aac2e0ce10d5e0d8b118343265`. Hardware runs pass the architecture
precheck and acquire devices through `task-submit --device auto`.

All log names below have prefix `kernel-integration-20260911-` and live in
the remote workspace's `skx_log_output/` directory. Every completed group below
returned exit code 0.

- Native builds: a2a3, a5, a2a3sim and a5sim succeeded, including the updated
  nanobind extension. Logs: `install-8.log`, `simbuild-3.log`, `build-a5-2.log`.
- C++ unit tests: **167/167 targets passed**. Logs: `cpp-build-10.log`,
  `cpp-build-11.log`, `cpp-test-7.log`.
- C++ hardware tests: **2/2 targets passed**, covering kernel context ownership,
  close/recreate and the caller's continued use of its device/stream.
  Log: `cpp-hardware-2.log`.
- Python unit tests: **2318 passed, 40 skipped**, including all seven dynamic
  library capability/retry cases. Logs: `pyut-4.log`, `dso-1.log`.
- a2a3sim scenes: **75 passed, 8 skipped**. Log: `scene-a2a3sim-3.log`.
- a5sim scenes: **71 passed**. Log: `scene-a5sim-2.log`.
- a2a3 onboard main scenes: **159 passed, 1 skipped**. The scheduler executed
  61 resource jobs, 40 HBG scenes with one skip, and 58 TMR scenes. K2 capture
  appears explicitly among the 61 jobs. Log: `scene-a2a3-3.log`.
- Separate SDMA scenes: **3 passed**. Log: `sdma-1.log`.
- Public kernel C API on a2a3: **12 hardware cases passed**, including exact
  numerical results for two 16,384-element launches, fresh pointers/scalars,
  prior-output preservation, stable allocation and zero committed memory on
  close. Log: `capi-3.log`.
- K2 standalone capture: **100 replays passed**, both internal streams included,
  zero forbidden calls. Log: `capture-1.log`.
- Invocation transport: **24 gated asynchronous snapshots passed**, including
  minimum/maximum packets and overwritten/released host argument buffers.
  Log: `snapshot-2.log`.
- clang-tidy passed on the repository's supported compilation-unit scope.
  Log: `clang-tidy-2.log`. Onboard and AICore compilation is covered by native
  builds rather than that lint script. Headers, English-only policy, retired
  names, wire isolation, clang-format, cpplint, Ruff, pyright, Markdown and
  whitespace checks also passed.
- Source identity: **2513 files matched, zero mismatches**.
  Log: `source-verify-4.log`.

The main hardware sweep uses `-m "not sdma" --exclude-level 4` and a
1200-second scene timeout, matching CI's isolation requirements. SDMA runs
separately afterwards with a 600-second scene timeout.

## Mainline merge revalidation

The integration line was merged with `feat/kernel-mode-integration-test` at
`748c39fbe`, which is identical to `main` and carries the merged K1 (#2064) as
`1d1ddc815`. The contract decisions for that merge are D14 in the integration
log. After the merge, `PTO_RUNTIME_ERR_INVALID_ARGUMENT` is `BASE - 4`, and the
callable and capacity codes are `BASE - 5` through `BASE - 9`.

The merged tree was rebuilt and re-run on an a2a3 host. Hardware work passed
the architecture precheck and acquired devices through
`task-submit --device auto`.

| Suite | Result |
| ----- | ------ |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | 168/168 passed |
| C++ hardware tests, `^requires_hardware(_a2a3)?$` | 2/2 passed |
| Python unit tests for the kernel entry, worker, task interface and launch source guard | 222 passed, 23 skipped |
| a2a3 hardware: `tests/ut/py/test_kernel_mode_c_api.py` and `tests/st/a2a3/kernel_capture` | 13/13 passed |
| clang-tidy 18 on changed C++ files | passed, zero diagnostics |
| Other pre-commit hooks on changed files | passed |
| `mkdocs build --strict` | passed |

The Python unit run covered `test_chip_worker.py`, `test_task_interface.py`,
`test_kernel_mode_c_api.py` and `test_kernel_launch_source_guard.py`.

Enabling hardware test targets adds two no-hardware targets,
`test_a5_kernel_topology_lifecycle` and `test_kernel_launch_native`; both
passed. The two C++ hardware targets were `test_comm_lifecycle` and
`test_kernel_mode_entry`.

The 13 hardware cases were:

- the eager numerical launch, two rounds of 16,384 elements;
- seven lifecycle scenarios: `repeat_init`, `init_failure`, `stream_close`,
  `event_close`, `destroy_unclosed`, `prepare` and `fatal_device`;
- two device-query rejections: `device_query_error` and `device_mismatch`;
- kernel init on a borrowed device for `host_build_graph` and
  `tensormap_and_ringbuffer`;
- the K2 capture probe with 100 replays.

### Full sweeps on the merged tree

A second pass on the same merged tree ran the remaining suites, including every
Python hardware unit test. Hardware work again acquired devices through
`task-submit --device auto`.

`tests/ut/py/test_worker/test_kernel_mode_entry.py` asserted that every runtime
refuses kernel init, and three of its jobs failed against TMR, which claims the
borrowed device. Its refusal cases load `host_build_graph`, which reports no
kernel capability. The `init_claims_borrowed_stream` case matches the C++ twin:
TMR init succeeds, reports kernel support, and refuses a second worker on the
same device while the first stays initialized.

| Suite | Result |
| ----- | ------ |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 30 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2335 passed |
| Invocation transport probe, `tools/cann-examples/tmr-invocation-snapshot` | 24 gated asynchronous snapshots passed |
| a2a3 onboard scenes, `-m "not sdma" --exclude-level 4` | 167 passed, 1 skipped |
| a2a3 SDMA scenes | 3 passed |
| a2a3sim scenes | 82 passed, 8 skipped |
| a5sim scenes | 78 passed |

## PR refresh revalidation (2026-09-14)

`main` was merged into the integration line, bringing it to `5186d8c7`, and the
four adopted changes from D15 were applied on top. Hardware work passed the
architecture precheck and acquired devices through `task-submit --device auto`.

| Suite | Result |
| ----- | ------ |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | 168/168 passed |
| C++ hardware tests, `^requires_hardware(_a2a3)?$` | 2/2 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2428 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 31/31 passed |
| a2a3 hardware, `tests/ut/py/test_kernel_mode_c_api.py` | 13/13 passed |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, 100 replays |
| a2a3sim scenes | 82 passed, 8 skipped |
| a5sim scenes | 78 passed |
| pre-commit hooks on changed files | passed |

The C++ hardware run needs two devices and a CTest resource spec, as
`.github/workflows/_ut-npu-a2a3.yml` builds one; `test_comm_lifecycle` reports
"need 2 NPU devices; run with --resource-spec-file" and fails without it.

The Python hardware run collects 31 cases but its resource phase schedules 21.
The ten it leaves out — the runtime ABI symbol export, runtime builder,
device-memory-info and two-rank comm/alloc cases — were run directly against
the same device pool and all pass. None of them is on a path this refresh
touches.

`persistent_free_close`, the new lifecycle scenario, asserts that the retry
after a failed release re-attempts and completes rather than that it makes
exactly one more call. A kernel context on this line releases several device
blocks and the injected failure stops the first pass partway, so the retry
covers the failed block plus everything the first pass never reached — eight
`rtFree` calls across the two passes where the first made three. The
count-exact form the source PR uses holds only for its own smaller allocation
set.

`mkdocs build --strict` was not re-run: mkdocs is not installed in this
worktree and PyPI is unreachable from this host. The refresh adds no page and
changes no nav entry.

## Target call-flow rebase revalidation (2026-09-14)

The line was rebased onto the target call-flow design — registration mints the
id, no callable generation anywhere, and the chained caller/AICPU/AICore launch
topology. The decisions are D16 in the integration log. Hardware work passed the
architecture precheck and acquired devices through `task-submit`.

| Suite | Result |
| ----- | ------ |
| Native builds: a2a3, a5, a2a3sim, a5sim, nanobind extension | all succeeded |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | 168/168 passed |
| C++ hardware tests, `^requires_hardware(_a2a3)?$` | 2/2 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2428 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 31/31 passed |
| a2a3 hardware, `tests/ut/py/test_kernel_mode_c_api.py` | 13/13 passed |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, 100 replays |
| a2a3 onboard scenes, `-m "not sdma" --exclude-level 4` | 167 passed, 1 skipped |
| a2a3 SDMA scenes | 3 passed |
| a2a3sim scenes | 82 passed, 8 skipped |
| a5sim scenes | 78 passed |
| pre-commit hooks on changed files | passed |

The capture probe now drives the chained sequence — caller records `Start`,
AICPU waits it and records `AicoreStart`, AICore waits that and records
`AicoreDone`, AICPU joins it and records `AicpuDone`, the caller joins that — and
still completes 100 captured replays with verified buffers on a2a3. That is the
hardware evidence for the topology; the public `prepare -> launch` entries are
still not exercised inside a captured graph by any test, so two-hop capture
propagation through the real entries remains unproven.

Two expectations moved with the contract rather than with a defect. A second
registration of the same image now mints a second id and uploads a second copy,
but `committed_device_memory_ctx` does not grow: the code arena and its
descriptor prefix are committed once on first use, so a later registration
spends arena budget instead of new device memory. And `persistent_free_close`
asserts that the retry after a failed release re-attempts and completes rather
than that it makes exactly one more `rtFree` call.

`mkdocs build --strict` was not re-run: mkdocs is not installed in this worktree
and PyPI is unreachable from this host. The change adds no page and changes no
nav entry.

## Device-entry sync revalidation (2026-09-14)

The device entry was brought onto #2190's restored shape: the launch packet
carries the callable's device image address and extent, and the residency
descriptor is deleted. The decisions are D17 in the integration log. Hardware
work passed the architecture precheck and acquired devices through
`task-submit`.

| Suite | Result |
| ----- | ------ |
| Native builds: a2a3, a5, a2a3sim, a5sim, nanobind extension | all succeeded |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | 168/168 passed |
| C++ hardware tests, `^requires_hardware(_a2a3)?$` | 2/2 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2428 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 31/31 passed |
| a2a3 hardware, `tests/ut/py/test_kernel_mode_c_api.py` | 13/13 passed |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, 100 replays |
| a2a3 onboard scenes, `-m "not sdma" --exclude-level 4` | 167 passed, 1 skipped |
| a2a3 SDMA scenes | 3 passed |
| a2a3sim scenes | 82 passed, 8 skipped |
| a5sim scenes | 78 passed |
| pre-commit hooks on changed files | passed |

Every scene and unit count matches the pre-change baseline exactly.

Two divergences from #2190 remain deliberate. The packet prefix keeps
`binding_address`, `context_generation`, `sm_bytes` and `arena_bytes`, which
this line's TMR consumer reads and #2190 has no source for, so it is 88 bytes
against #2190's 56. And `consume_kernel_invocation` receives the whole prefix
rather than the invocation header alone, for the same reason.

`mkdocs build --strict` was not re-run: mkdocs is not installed in this
worktree and PyPI is unreachable from this host. The change adds no page and
changes no nav entry.

## Callable id cap revalidation (2026-09-14)

`MAX_REGISTERED_CALLABLE_IDS` moves from 64 to 8192, matching #2190. This was
the one part of #2190's contract this line had not taken, and registration is
already pure on both, so every prepare spends an id permanently and 64 was the
binding limit. Hardware work passed the architecture precheck and acquired
devices through `task-submit`.

| Suite | Result |
| ----- | ------ |
| Native builds: a2a3, a5, a2a3sim, a5sim, nanobind extension | all succeeded |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | 168/168 passed |
| C++ hardware tests, `^requires_hardware(_a2a3)?$` | 2/2 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2428 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 31/31 passed |
| a2a3 hardware, `tests/ut/py/test_kernel_mode_c_api.py` | 13/13 passed |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, 100 replays |
| a2a3 onboard scenes, `-m "not sdma" --exclude-level 4` | 167 passed, 1 skipped |
| a2a3 SDMA scenes | 3 passed |
| a2a3sim scenes | 82 passed, 8 skipped |
| a5sim scenes | 78 passed |
| pre-commit hooks on changed files | passed |

The AICPU cost is real and measured. `orch_so_table_[MAX_REGISTERED_CALLABLE_IDS]`
is a fixed array of ~296-byte entries in the TMR AICPU executor, so
`libaicpu_kernel.so`'s `.bss` grows from 450 KiB to 2.73 MiB. The device loads
and runs it: the kernel C API and capture suites pass unchanged on a2a3. The
entry is 86% `char path[256]`, which the kernel path never uses — it registers
by `dev_orch_so_addr` — so moving `path` out of the resident table would return
about 2 MiB. That is not done here.

Four cases failed on the first pass and all pass on re-run:
`test_kernel_device_query_rejections_keep_context_reusable[device_query_error]`
and `[device_mismatch]`, and `TestWorkerAsyncWholeRunFifo::test_run` and
`::test_prepared_run_device_control_waits_for_the_active_run`. The scene log
names the cause: `rtStreamQuery (AICore) failed: 507901`, `EL9999 ... reason=hdc
disconnect` on `dev=13` — the host lost its channel to that device mid-sweep.
The change cannot reach those paths: it edits one constant and one unit test,
and `MAX_REGISTERED_CALLABLE_IDS` appears nowhere under
`src/*/runtime/host_build_graph/` or `src/common/host_build_graph/`, which is
where `worker_async_fifo` runs. A targeted re-run of all four plus the rest of
`worker_async_fifo` passed 9/9 on a fresh device pair, and the same two kernel
cases had already passed 14/14 on this same code earlier in the session. The
onboard row above records the re-run result.

## Callable cache alignment revalidation (2026-09-14)

The callable cache and the shared entry validation are now #2190's: device
memory comes in 2 MiB blocks up to a 2 GiB budget instead of one 512 MiB arena
committed on first use, and the structural image validation lives once in
`validate_kernel_callable_image` rather than being duplicated between the entry
and the cache. Hardware work passed the architecture precheck and acquired
devices through `task-submit`.

| Suite | Result |
| ----- | ------ |
| Native builds: a2a3, a5, a2a3sim, a5sim, nanobind extension | all succeeded |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | 168/168 passed |
| C++ hardware tests, `^requires_hardware(_a2a3)?$` | 2/2 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2428 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 31/31 passed |
| a2a3 hardware, `tests/ut/py/test_kernel_mode_c_api.py` | 13/13 passed |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, 100 replays |
| a2a3 onboard scenes, `-m "not sdma" --exclude-level 4` | 167 passed, 1 skipped |
| a2a3 SDMA scenes | 3 passed |
| a2a3sim scenes | 82 passed, 8 skipped |
| a5sim scenes | 78 passed |
| pre-commit hooks on changed files | passed |

Every scene and unit count matches the pre-change baseline, with no re-runs.

The first registration no longer commits the whole budget: it takes
`max(charged, 2 MiB)`, and a registration larger than a block gets a block
sized exactly to it while earlier blocks keep their usable tails. Block slack
is charged against the 2 GiB budget through `allocated_bytes()`, while
`resident_bytes()` still counts only charged image bytes, and a published
`device_address` never moves as blocks are added.

Which admission limit binds first now crosses over at 256 KiB, since
2 GiB / 8192 is exactly that: below it the id count binds, above it the byte
budget does.

Two things stay divergent from #2190 on purpose.
`validate_kernel_callable_image` additionally rejects a child `func_id` that is
out of range or repeated within one image, two checks #2190's extraction lost;
without them a malformed image is caught only by the device consumer at launch,
which poisons the context instead of failing the registration cleanly. And the
simulation `prepare_callable` calls the image validator too, so both entries
agree that structural checks precede the lifecycle refusal; #2190's simulation
entry reports `INVALID_STATE` for an image whose size clears the header floor
but whose variable tail does not add up. The first of the two is no longer a
divergence — #2190 restored those checks at `08f7f9218728`, in a better form
this line then took; the newest-head section below has it.

`kernel_arena_change_is_forbidden` is a third difference, but it is not one
against #2190: it is mainline code from the merged K1 (#2064), so every branch
off `main` carries it, #2190 included. This line **deleted** it — D14 item 4 — and
routes the same rule through K3's `commit_static_arena_bank`, which applies it
to onboard and simulation from one place instead of an inline scan in
`setup_static_arena`. The consolidation is sound on its own terms, but it
removes a guard `main` uses today in favour of a mechanism that exists only in
an unmerged PR: `static_arena_bank.h` is on this line and on #2193, and nowhere
else. If #2193 does not land in that shape, the rule has no carrier here. This
is recorded as a risk, not a resolved question.

`mkdocs build --strict` was not re-run: mkdocs is not installed in this
worktree and PyPI is unreachable from this host. The change adds no page and
changes no nav entry.

## Newest-head review revalidation (2026-09-15)

Re-reading #2190 at `08f7f9218728` closed two gaps in opposite directions.

The first was this line's. `simpler_aicpu_kernel_exec` treats `packet_bytes` as
the length of the region starting at `arg`, and its upper bound compared that
`uint64_t` field against `SIZE_MAX` — never true on a 64-bit AICPU, so
`arg + packet_bytes` could wrap. It is now bounded by the space actually left
above `arg`, the shape #2190 reached first.

The second was #2190's, and it came back improved. The lost child `func_id`
range and duplicate checks, previously recorded here as a divergence, are back
in that PR, bounded by a named `KERNEL_MAX_FUNC_ID` tied to
`RUNTIME_MAX_FUNC_ID` by a `static_assert` in both `device_runner_base.cpp`
files, where this line had used the extent of `ChipCallable::child_func_ids_`.
That array's capacity is how many children an image may carry, not the size of
the table the id indexes; they agree at 1024 only by coincidence, so widening
the child capacity would have silently widened the accepted id range. #2190's
expression is the correct one and is now this line's, along with the test it
added. The accepted set is unchanged, so this is a provenance fix, not a
behavior change.

One divergence from #2190 remains, and it is deliberate. The simulation
`prepare_callable` fallback validates the image before reporting that no
kernel context is live, so a malformed image reports an argument error on both
platforms; #2190's fallback reports `INVALID_STATE` for an image whose size
clears the header floor but whose variable tail does not add up. The
`kernel_arena_change_is_forbidden` question is unchanged and is still a
difference against `main`, not against #2190 — see the section above.

Hardware work passed the architecture precheck and acquired devices through
`task-submit`.

| Suite | Result |
| ----- | ------ |
| Native builds: a2a3, a5, a2a3sim, a5sim, nanobind extension | all succeeded |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | 168/168 passed |
| C++ hardware tests, `^requires_hardware(_a2a3)?$` | 2/2 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2428 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 21/21 scheduled cases passed |
| The ten cases that marker run drops, invoked by path | 10 passed |
| a2a3 hardware, `tests/ut/py/test_kernel_mode_c_api.py` | 13/13 passed |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, 100 replays |
| a2a3 onboard scenes, `-m "not sdma" --exclude-level 4` | 167 passed, 1 skipped |
| a2a3 SDMA scenes | 3 passed |
| a2a3sim scenes | 82 passed, 8 skipped |
| a5sim scenes | 78 passed |
| pre-commit hooks on changed files | passed |

Every count matches the pre-change baseline, with no re-runs for failure.

Two invocation details are load-bearing and cost a false red each before being
recognised as harness form rather than regression. The C++ unit build must be
configured with **no** `CMAKE_BUILD_TYPE`, as `docs/testing.md` shows it:
`Release` defines `NDEBUG`, `debug_assert` becomes a no-op, and the three
recording-refusal tests that assert on it stop throwing. And the C++ hardware
tests need `--resource-spec-file`, built from `TASK_DEVICE` the way
`.github/workflows/_ut-npu-a2a3.yml` builds it; without it `test_comm_lifecycle`
sees an empty device pool and fails on the count rather than on behavior.

`mkdocs build --strict` was not re-run: mkdocs is not installed in this
worktree and PyPI is unreachable from this host. The change adds no page and
changes no nav entry.

## Registration wait revalidation (2026-09-15)

Another divergence from #2190 was found after the newest-head review, in the one
thing that review did not compare: not what registration sends, but whether
`prepare_callable` waits for it. #2176 and #2190 both reach registration through
`register_callable_on_device`, whose three steps are launch, synchronize with
`PLATFORM_STREAM_SYNC_TIMEOUT_MS`, commit. This line inlined the first and third
into `prepare_kernel_callable` and dropped the second, so registration was
enqueued and the call reported success without it having run. The kernel path
now calls `register_callable_on_device`, which program mode already used, and
the inlined copy is gone. The decisions are D19 in the integration log.

Three documents described the behaviour and did not agree with each other.
`runtime_c_api.h` said preparation never synchronizes and that registration
errors surface through the caller's own warmup plus synchronize; D14 item 5 said
callable registration synchronizes the context's own AICPU stream;
`docs/zh-cn/kernel-mode-integration-test.md` said both, four sections apart. The
implementation matched the first. All three now state the wait.

Ordering never rested on the divergence: the AICPU stream's FIFO orders
registration ahead of every launch either way, which is what D16's chained
topology relies on. What it cost was the report.
`simpler_aicpu_register_callable` returns its `load_orch_so` status on both
arches, so a device-side `dlopen` of the orchestration SO that failed had
nowhere to surface — `prepare_callable` returned 0, nothing poisoned the
context, and the failure reappeared later as a launch fault on a context that
looked healthy.

A regression case covers exactly that.
`test_kernel_lifecycle_retry[register_failure]` registers a `ChipCallable` whose
orchestration binary is 4 KiB of zeroes. Host admission reads sizes, offsets and
names, and `register_callable_impl` copies the image without parsing it, so the
first refusal is the AICPU's own `dlopen`. The case asserts that the refusal is
`prepare_callable`'s status, that `out_callable_id` is -1, that the poisoned
context refuses the next registration with `INVALID_STATE`, and that close still
returns committed device memory to zero. Against a build with the inlined path
restored it fails on its first assertion, with `prepare_callable` returning 0.

| Suite | Result |
| ----- | ------ |
| Native builds: a2a3, a5, a2a3sim, a5sim, nanobind extension | all succeeded |
| C++ unit tests, no hardware | 166/166 passed |
| a2a3 hardware, `tests/ut/py/test_kernel_mode_c_api.py` | 14/14 passed, the new case included |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, 100 replays |
| The new case against a build with the inlined path restored | failed as intended |
| pre-commit hooks on changed files | passed |

The suites this section does not list were not re-run: the full Python unit
sweep, the onboard and simulation scene phases, the SDMA phase, the C++ hardware
targets, and the hardware-enabled C++ unit build. The change is confined to the
onboard kernel registration path and the documents describing it, and the two
hardware suites that reach that path are the two listed.

`mkdocs build --strict` was not re-run: mkdocs is not installed in this
worktree and PyPI is unreachable from this host. The change adds no page and
changes no nav entry.

## Entry-layer hardening revalidation (2026-09-15)

The remainder of #2185's refreshed head is adopted: a status-carrying
`ChipWorkerError` with its nanobind translators and `PTO_RUNTIME_ERR_*` module
attributes, a non-throwing `kernel_mode_supported()`, the owed-teardown
bookkeeping for a refused init or a failed close, and a kernel-only callable
registry. The decisions are D20 in the integration log.

One of those closes a leak rather than changing a preference. This line's
`kernel_init` dropped the device context on any refused init, on the stated
ground that a refused kernel init has adopted nothing. `init_kernel_context`
creates the context's two streams and five events before four steps that can
still fail, and unwinds only its context claim, so a refusal from any of them
arrived at `ChipWorker` with live resources — where `destroy_device_context`
refuses the context and returns void and the library holding their release
routines is then unloaded. Registration now calls `finalize_device` first and,
when that fails, keeps the handle, the bindings and the library and records the
teardown as owed.

Four things stay as this line has them, each an artifact of #2185's stub
platform rather than a divergence worth closing: the D14 item 6 capability gate
(so the adopted fake runtimes declare support to reach the entry), a
program-mode teardown status reported on stderr instead of dropped, the
successful-`kernel_init` scenario in the a2a3 hardware test, and the Python twin
loading `host_build_graph` for a genuine `UNSUPPORTED`.

| Suite | Result |
| ----- | ------ |
| Native builds: a2a3, a5, a2a3sim, a5sim, nanobind extension | all succeeded |
| C++ unit tests, no hardware | 166/166 passed |
| C++ unit tests with `SIMPLER_ENABLE_HARDWARE_TESTS=ON`, no hardware | the two added targets passed |
| C++ hardware, `test_kernel_mode_entry` with a CTest resource spec | passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2448 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 21/21 scheduled cases passed |
| a2a3sim scenes | 82 passed, 8 skipped |
| pre-commit hooks on changed files | passed |

The Python unit count is the 2432 baseline this line reached after PR #2242,
plus the sixteen cases #2185 adds to `tests/ut/py/test_chip_worker.py`; every
other count matches the baseline exactly. The suites ran twice: once on
`29a1cd40`, and again after rebasing onto `4a5f28c9`, the head that carries the
merged PRs #2245 and #2242.

Not re-run: the a2a3 onboard scene phases, the SDMA phase, a5sim scenes, and
`tests/ut/py/test_kernel_mode_c_api.py`. The change is confined to the
`ChipWorker` entry layer, its bindings and its tests, and does not reach the
platform entries those suites exercise.

`mkdocs build --strict` was not re-run: mkdocs is not installed in this
worktree and PyPI is unreachable from this host. The change adds no page and
changes no nav entry.

## Capture scene-test revalidation (2026-09-15)

`tests/st/a2a3/tensormap_and_ringbuffer/kernel_mode_capture` was failing 23 of
its 24 scenarios on a2a3 before this change, every one on
`prepare/launch performed an internal sync` — the observer forbade the AICPU
stream wait that #2245 deliberately added to registration. The decisions are D21
in the integration log: the wait stays, the guard splits into a launch scope
that refuses every synchronize and a registration scope that refuses only the
caller's streams, and the blocking gate moves from registration to launch.

| Suite | Result |
| ----- | ------ |
| a2a3 hardware, `tests/st/.../kernel_mode_capture` | 24/24 scenarios passed |
| a2a3 hardware, `tests/st/a2a3/kernel_capture` | passed, unchanged |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2458 passed |
| Python hardware unit tests, `tests/ut -m requires_hardware --platform a2a3` | 23/23 scheduled cases passed |
| pre-commit hooks on changed files | passed |

Before the change the same invocation reported 1 of 24. Splitting the two scopes
alone took it to 22; the remaining two, `blocked_same` and `stream_busy`, are the
scenarios whose gate assumed registration returns while it is still outstanding,
and they pass once the gate is installed ahead of a launch instead.

This is a test-side change only: no native source differs, so the entry
behaviour every earlier row in this document recorded is unchanged.

## Capture-callable preparation revalidation (2026-09-17)

The #2190 head moved from `08f7f9218728` to `a8b19e8a0a70`, making
`prepare_callable` callable inside an ACLGraph capture. Taking it required
removing preparation's two AICPU stream waits, which exposed a dependency they
had been masking: K7's registration `dlopen`s the orchestration SO, dispatch
refuses a callable whose slot is empty, and the two are ordered only by the
hidden AICPU stream's FIFO — which a replayed graph does not join. D22 in the
integration log moves the load into the launch round's leader-only phase, where
no cross-stream edge is needed.

| Suite | Result |
| ----- | ------ |
| a2a3 hardware, `tests/st/.../kernel_mode_capture` | 30/30 scenarios passed |
| a2a3 hardware, `tests/ut -m requires_hardware` | 30/30 passed |
| C++ unit tests, `ctest -LE requires_hardware` | 177 passed |
| Python unit tests, `tests/ut -m "not requires_hardware"` | 2529 passed |
| pre-commit hooks on changed files | passed |

Three scenarios are new or changed. `prepare_capture_global` and
`prepare_capture_thread_local` hold a capture of each mode open across two
registrations, check that each upload restored the thread mode it borrowed, then
end the capture, replay and confirm committed device memory did not move.
`prepare_in_capture` registers a second callable inside the capture that records
its launches. `register_failure` now asserts that an image the AICPU cannot load
is refused by neither preparation nor a drain, because nothing loads it until a
launch round asks for the orchestration.

`cold_unsynced` is the case that pinned the defect: with both waits removed and
the load still in registration it reported `InvalidBinding` six times, 192 µs
after the registration executed. Restoring only the callable-registration wait
fixed it and restoring only the context-handshake wait did not, which is what
identified `load_orch_so` as the dependency. A five-second host sleep before the
first replay also fixed it, confirming a race rather than a lost task. Moving
preparation inside the capture did not, which ruled out "prepare and launch in
the same capture window" as a sufficient rule.

Two defects were found by these suites while making the move, both now fixed:
`load_orch_so` reset the whole table entry, dangling the `KernelCallableView` a
dispatch already held (device-side `Scheduler fatal error (code=5)`); and a
revoke clears residency without clearing the `dlopen` handle, so a non-null
handle did not imply the current image was loaded.

`test_worker_kernel_mode_hw.py::[prepare_twice]` failed twice early in this work
with `507901`, the hdc-disconnect cascade
[already documented](troubleshooting/a2a3-507899-aicpu-shared-so-fault.md) as an
intermittent AICPU shared-SO device fault. It did not reproduce in ten
controlled runs — five on this branch and five on the unmodified integration
head, interleaved on one device — so it is neither attributed to nor cleared by
this change.

## Kernel-mode DFX validation (2026-09-17)

Covers D23: the chip swimlane and dep_gen collecting inside a kernel context,
bracketed by `begin_dfx` / `end_dfx`.

**Environment.** a2a3 (`Ascend910_9392`), one device per case through
`task-submit`; branch `feat/kernel-mode-chip-swimlane` on `4f162da09`.

| What | Result |
| ---- | ------ |
| `test_worker_kernel_mode_hw` (7 cases, incl. new `dfx_bracket`) | 7/7 pass |
| One launch inside a bracket, level 4 + dep_gen | `aicore_tasks=1`, `scheduler_tasks=1`, 3 scheduler-phase streams, 1 orchestrator phase |
| Round boundaries | `run_boundaries=[[0, …]]`, `dropped_run_boundaries=0` |
| Unbalanced brackets | close-without-open and double-open both refused |
| `swimlane_converter` on the pair | labels resolve from `task(t0)` to `func_0_a(t0)`; `Func_ID` 0 rather than -1 |

**Multi-launch follow-up (2026-09-18).** The hardware Worker suite passes 9/9
cases, including 32 launches in one bracket at levels 1 and 4. Both capture
32 AICore records; level 4 also captures 32 scheduler records and 32 boundaries,
with no count-mismatch or empty-record warnings. Empty task and phase buffers
remain device-owned across launches. AICore flush counts use the current
buffer's dispatch count independently of the window total and buffer sequence.
CPU regression tests cover both a2a3 and a5 pool reuse, partial flushes, full
buffer rotation and recovery after a rejected rotation.

**Window isolation follow-up (2026-09-18).** The device DFX flag is disabled
after context registration, enabled at `begin_dfx`, and disabled at `end_dfx`.
The AICPU round leader refreshes it before initializing each round; AICore
already reads it at launch entry. `begin_dfx(caller_stream)` drains the
caller stream before resetting counters and enabling capture, matching
`end_dfx(caller_stream)`. Prior work on other streams must be joined onto the
caller stream. These operations remain outside ACLGraph capture and
add no synchronization to the launch path.

The eager hardware regression at levels 1 and 4 enqueues an asynchronous
warm-up, captures windows with 3, 0 and 2 launches, and runs 16 unbracketed
launches after each window. AICore, scheduler and dependency task counts match
each bracket exactly; the empty bracket produces no swimlane artifact and an
empty dependency graph. Launch boundaries restart at zero in each window and
are exported even without phase metadata. The Python fake runtime also exports
the bracket symbols required by the kernel entry loader.

Evidence is under `outputs/swimlane-window-fix/`: `after-2.log` covers window
isolation, `cpp-tests-final.log` records 9 passing native test targets, and
`python-tests-final.log` records 378 passing Python tests. The baseline run in
`before.log` fails with an empty AICore record and stream timeout after an
unbracketed asynchronous warm-up.

`worker-regression-retry.log` records 11/11 passing kernel Worker hardware
cases, including both 32-launch cases. The initial combined regression had
one level-4 case stall on its first task (`S1:running-stalled`, core 26), then
export an empty slot during timeout cleanup. Its isolated rerun and the full
Worker rerun passed on the same device; the trigger remains unconfirmed.
`hardware-regression.log` and `sim-regression.log` also record passing TMR
vector/mixed and HBG scheduler-phase program captures. All changed-file
pre-commit checks passed (`pre-commit.log`).

**Caller-stream API follow-up (2026-09-18).** Begin requires the caller stream
through the Python wrappers, native binding and C API. Null streams are
rejected before opening the window. Evidence under
`outputs/swimlane-caller-stream/` records 381 passing Python tests and passing
pre-commit checks. The hardware suite passes 10/11 cases, including both
window-isolation levels; the level-4 32-launch case repeats the first-task
stall described above, then passes in isolation on the same device
(`hardware-retry.log`). This API change does not resolve that intermittent
stall.

**How the four gates were found.** Each was opened in turn and the symptom moved
rather than cleared, which is what identified the next one:

| Gate | Symptom while closed |
| ---- | -------------------- |
| `KernelStaticConfig::validate` | `kernel init: rc=-1001` (UNSUPPORTED) |
| `register_kernel_context` | `rc=507018`, device log `simpler_aicpu_prepare_tmr_context ... ret:2` |
| `aicore_executor.cpp` `KernelMode ? 0 : …` | init passes, launches run |
| `kernel_mode_kernel.cpp` weak stubs | `aicore_tasks=0` with `scheduler_tasks=1`; AICPU counted the dispatch, AICore's record slot stayed all-zero |

**Converter segmentation** is covered by CPU regression tests: repeated task
and register IDs join within each `run_boundaries` epoch, and dependency flows,
SPMD grouping and overhead counters are isolated per launch. See the
[converter guide](../simpler_setup/tools/README.md#kernel-mode-launch-rounds).

**Not covered.**

- **a5.** Its collector mgmt thread calls `aclrtMemcpy` on every poll. CANN's
  capture-mode definition names `aclrtMemcpy` an unsafe function and
  `ACL_MODEL_RI_CAPTURE_MODE_GLOBAL` (enum value 0) forbids it from every thread,
  including one that never called `CaptureBegin` and cannot know to switch modes.
  Whether such a call fails alone or invalidates the caller's capture is not
  stated in the CANN docs and was not measured. a2a3 is unaffected: its mgmt
  thread reads the `halHostRegister` SVM mapping and calls no ACL memory API.
- **ACLGraph capture.** The bracket entries are documented as outside-capture and
  were exercised eagerly only.

## Remaining boundaries

- H4 has no submitted PR in the supplied pipeline. HBG kernel capability is
  zero; init returns `UNSUPPORTED`, and prepare/launch without a kernel claim
  return `INVALID_STATE`. H1-H3 module tests do not establish public HBG execution.
- H1 (#2171) and 2B (#2172) were force-pushed onto a different decomposition
  after the 2026-09-11 audit. This line still carries the audited shape, so the
  HBG contribution is the one thing the target rebase did not cover.
- A5 is compiled and exercised through unit/simulation tests. No A5 hardware
  is available in this validation environment.
- Capture primitives and the public eager TMR path are separately tested.
  This is not an end-to-end PyTorch ACLGraph adapter validation.
- Kernel-context ownership is enforced within one loaded host runtime SO.
  Separate copies/processes, and program/kernel work sharing one resident
  device SO, require caller serialization.
- An outer packet rejected before a trusted Runtime is established cannot
  safely locate the AICore cancellation target. Ordinary invalid host arguments
  are rejected before enqueue; recovery from a corrupted outer device packet
  requires a future prepared-binding registry and explicit revocation.

## TMR and HBG kernel-mode async prefetch

Based on `feat/kernel-mode-integration-test` at `17e342455`, the a2a3 onboard
TMR and HBG paths accept `Worker(level=2, execution_mode="kernel", enable_sdma=True)`
and the equivalent `ChipWorker.kernel_init(..., enable_sdma=True)`.
This validates simpler's `TPREFETCH_ASYNC` runtime support; it does not qualify
an end-to-end PyPTO `pl.prefetch.async_prefetch` frontend integration.

Initialization provisions the SDMA provider and runs its optional warmup ELF
on the context's hidden AICore stream, outside capture. Both runtimes copy DMA workspace
addresses into its Runtime device descriptor for both program and kernel modes.
The scheduler reads that descriptor instead of the resident SO's mutable DMA
globals. Launch/replay reuse the persistent addresses. Close follows the existing
external-quiescence and graph-destruction contract, without resetting the borrowed
device. Kernel fatal teardown skips the program-only reset handoff delay.

The default SDMA flag stays off. A5, simulation kernel mode, and provider-disabled
builds reject explicit SDMA enablement. The native init C ABI has three appended
SDMA arguments; downstream native callers and runtime binaries must be rebuilt
together. HBG's a2a3 scheduler initializes both payload slots with the Runtime's
DMA address snapshot. Its `DispatchPayload` remains 512 bytes.

### Functional coverage

`tests/st/a2a3/tensormap_and_ringbuffer/kernel_mode_capture/test_kernel_prefetch.py`
is marked `sdma` and runs each case in an isolated onboard subprocess:

| Case | Data and assertion |
| ---- | ------------------ |
| `eager` | 100 launches; 128 FP32 elements each; input changes every round and output starts at `-999`; exact output comparison |
| `replay` | First invocation is captured without eager priming; 100 ACLGraph replays with new input values and reset output sentinels |
| `recreate` | 100 replays, graph destruction, context close, then new contexts with SDMA disabled and enabled on the same borrowed device |
| `disabled` | 100 launches leave the output sentinel intact because the kernel observes a zero workspace |
| `program_isolation` | Program SDMA init, kernel init without SDMA, kernel close, then 100 successful program prefetch runs; reproduces the address-clobber sequence without violating the resident executor's program/kernel mutual exclusion |

The reused prefetch kernel writes output only after observing a non-null SDMA
workspace and successful `AsyncEvent.Wait`. Eager/capture/replay instrumentation
checks zero internal synchronization and zero `rtMalloc` calls during invocation,
plus stable resident argument addresses and committed allocator memory. Successful
close is followed by a caller allocation to check borrowed-device usability.
C++ tests check initial zero addresses, independent Runtime snapshots, and the
bytes copied to the device image. C ABI tests cover malformed SDMA arguments and
unsupported targets rejecting before resource acquisition.

### Initial TMR validation results

- Editable build of bindings and available runtime variants: passed.
- Python unit tests: **2560 passed, 74 skipped**.
- C++ unit targets: **179 passed, 2 failed**. Both failures are the existing
  A2A3/A5 `aicore_pre_window_cancel` targets; all four cases in each also fail
  with exit status 4 on the unchanged `17e342455` baseline. The existing
  `test_a2a3_thread_scheduling` target needs the shared platform include directory
  supplied through `CPLUS_INCLUDE_PATH` to build locally.
- Isolated single-device SDMA sweep: **7 passed**, comprising all five new
  prefetch cases, the program-mode prefetch demo, and the existing bounded
  SDMA fault-teardown case. The two-device SDMA completion scene was excluded.
- Existing TMR onboard capture regression: **4 passed** (`cold_unsynced`,
  `fresh_inputs`, `eager_batch`, `eager_dag`), each with 100 rounds.
- Before HBG support was added, its native SDMA rejection before device adoption passed.
- Complete a2a3 onboard TMR L2 scenes with `-m "not sdma"`: **58 passed**.
- Complete a2a3sim L2 scenes: **42 passed, 8 skipped**.
- The broader a2a3sim sweep did not finish: its multi-device resource phase
  timed out after 600 seconds in `TestAllreduceIbingRejectNonSum`. This is an
  unresolved broad-regression result, not a passing full simulation sweep.
- The broader all-runtime onboard L2 sweep was stopped after HBG failures.
  HBG `benchmark_bgemm` also failed alone with AICPU synchronization status
  507018 after approximately 45 seconds. The HBG extension validation below
  supersedes this isolated failure after a clean AICore rebuild and records
  the separate runtime-switching limitation found in the combined sweep.
- Changed-file Python formatting/lint, C++ formatting, header, language,
  platform-literal, retired-name and kernel-wire checks: passed.

A5 onboard and PyPTO frontend integration are not covered. Multi-device onboard
sweeps require more than the single available card and are not covered by these
single-device results.

Local build/test logs are retained under
`outputs/tmr_kernel_prefetch_validation/`. The hardware test commands used
`task-submit` device locks after the a2a3 silicon precheck. Reproduction commands
inside that locked environment (with SDK environment and the worktree venv active):

```bash
python -m pytest tests/st/a2a3/tensormap_and_ringbuffer/kernel_mode_capture/test_kernel_prefetch.py \
    --platform a2a3 --device "$TASK_DEVICE" -v --pto-session-timeout 600
python -m pytest examples tests/st -m sdma --exclude-level 3 \
    --platform a2a3 --device "$TASK_DEVICE" -v --pto-session-timeout 600
python -m pytest examples tests/st -m "not sdma" --level 2 \
    --runtime tensormap_and_ringbuffer --platform a2a3 --device "$TASK_DEVICE" \
    -v --require-pto-isa --pto-session-timeout 600
```

### HBG prefetch extension

The a2a3 HBG kernel path uses the same context-owned SDMA provider and warmup
lifecycle as TMR. `Runtime` retains its own address snapshot; the scheduler
copies it into both dispatch payload buffers. User kernels obtain it through
`get_dma_workspace(args, DMA_WORKSPACE_SDMA)` without adding user arguments.

The HBG scene in
`tests/st/a2a3/host_build_graph/kernel_mode_capture/test_kernel_prefetch.py`
uses two dependent prefetch/copy tasks and a runtime-owned intermediate tensor.
Each of its five cases reuses the 128-element FP32, 100-round, changing-input
oracles above. The native observer handles HBG's HostArgs launch API and checks
stable resident argument addresses as well as zero invocation synchronization,
zero `rtMalloc` calls, and no committed-memory growth.

Validation of the HBG extension:

- HBG prefetch: **5 passed** (eager, first-use capture/replay, recreation,
  disabled workspace, and program/kernel address isolation).
- Isolated single-device SDMA sweep: **12 passed**, including the five HBG
  and five TMR prefetch cases, the program prefetch demo and bounded fault teardown.
- Existing HBG capture/replay suite: **20 passed**, including internal tensors,
  input changes, chained graphs and callable preparation inside capture.
- HBG onboard L2 scenes in isolation: **41 passed, 1 skipped**.
- TMR onboard L2 scenes in isolation: **58 passed**.
- HBG/kernel C++ test targets: **66 passed**.
- Python unit tests: **2560 passed, 73 skipped**.
- a2a3sim L2 scenes: **42 passed, 8 skipped**.
- Runtime builds and changed-file formatting/lint: passed.

An initial HBG handshake timeout also affected the non-prefetch capture control.
That control passed on a clean `17e342455` build; cleaning and rebuilding the
current HBG AICore binary removed the timeout. The AICore custom build commands
omitted HBG Runtime/context header dependencies. Both onboard architecture build
files include those dependencies, and Ninja's dependency graph was checked.
The previously failing HBG program benchmark also passed after the rebuild.

The combined all-runtime onboard L2 sweep reported **94 passed, 1 skipped,
5 failed**. Failures occurred on the first program-mode run following a runtime
switch, with AICPU synchronization status 507018. The two standalone matrix
retries and both complete runtime-isolated sweeps passed. This combined sweep
used `--level 2` without `--runtime`, which bypasses the test dispatcher's
runtime isolation. A clean `17e342455` baseline reproduced the same failure:
two HBG scenes passed, then the first TMR scene failed with 507018. The same
three baseline scenes passed when dispatched into separate processes. This
matches the existing [onboard runtime isolation constraint](testing.md#runtime-isolation-constraint-onboard).
Use one `--runtime` per L2 invocation, or let the dispatcher isolate runtimes;
the failing combined invocation is not a supported full-regression recipe.
Baseline logs are `baseline-mixed-same-process.log` and
`baseline-mixed-isolated.log` in the directory below.

Logs are under `outputs/hbg_kernel_prefetch_validation/`. All onboard tests used
`task-submit` after silicon precheck. A5 onboard, multi-device SDMA and PyPTO DSL
end-to-end execution are not qualified by these results.
