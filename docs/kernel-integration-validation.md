# Kernel integration validation

## Scope and frozen PR heads

This records local integration of every submitted PR in the supplied kernel
pipeline. It does not merge or change the GitHub PRs. The final audit froze
these heads on 2026-09-11; later PR updates are outside this validation run.

| PR | Contribution | Audited head |
| -- | ------------ | ------------ |
| #2064 | K1 ABI, lifecycle, invocation header, scalar signature | `2ab04b1b1ef76a88743950a4f3e6c78d1bd8be26` |
| #2171 | H1 host graph build and H2D separation | `458c0243ccdb9933c9c4bf3928021f83c446b117` |
| #2172 | HBG resources and stream contract | `7523052fee0066c0d289a688b48f76b10c7d54ab` |
| #2173 | HBG context resource preparation and freeze | `15741ac05b24908c71703c72f2549f5fe0e9fa53` |
| #2174 | H2 immutable graph packets | `1fe6bf53f0af9b6a02eeb2edff080a8bc7e7467a` |
| #2175 | H3 execution slot sealing and validation | `89b00a9cad3bdc32dee3ec2ab1a65d21a8a23c20` |
| #2176 | K2 persistent execution resources and capture test | `533f67a13f4c11a02face67a4fc8609926b71ef8` |
| #2177 | TMR resource contract and initialization admission | `2153406a0420e8de17cf7397d30864939b1bca0c` |
| #2180 | K4 per-invocation snapshots | `b0943525dd8eaeb486bca92bbf5480a37eb61fcb` |
| #2185 | C++, nanobind and Python kernel entry points | `32dd9442f79bb936541cf41a6a08d7838d450134` |
| #2187 | Three-stream launch binder and compensation | `136e9712d22c495ac921a6900c8f24fa9b8ebcf3` |
| #2189 | K5 resident and per-invocation execution state | `b6435e4858b1e0443966d664cca478276baefa55` |
| #2190 | Callable cache, residency and generation validation | `2c876478dbf41fad5c0019f081261d912e9b687e` |
| #2193 | K3 capacity refusal without releasing existing resources | `b8e739d9d8143ca6f35bf2177241976f900e2ab7` |

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

## Remaining boundaries

- H4 has no submitted PR in the supplied pipeline. HBG kernel capability is
  zero; init returns `UNSUPPORTED`, and prepare/launch without a kernel claim
  return `INVALID_STATE`. H1-H3 module tests do not establish public HBG execution.
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
