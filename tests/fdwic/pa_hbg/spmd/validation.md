# Resident scheduler validation

The [implementation contract](implementation-contract.md) is checked at three
layers: bounded CPU models, production-helper C++ tests, and native scene tests.
No single layer substitutes for device visibility or original business golden
validation. All commands below run from the repository root after building this
checkout; use the project's device precheck/reservation procedure for A5 runs.

## Coverage map

| Contract | CPU / C++ coverage | Native coverage under this directory |
| -------- | ------------------ | ------------------------------------ |
| Original N, exact task/block/slot execution, waves across capacity | `test_hbg_spmd_layout`, `test_hbg_payload_context`, Python shapes/graphs | `test_spmd_contract.py` |
| Unique admission, no publication holes, member projection | `test_hbg_admission`, `test_hbg_ordered_execution`, Python interleavings | `test_ordered_execution.py`, `test_ready_order_execution.py` |
| Fast participant advances, real dependencies still block | Ordered C++ tests and positive/negative CPU models | Ordered and GM Mix overlap probes |
| GM pairing, inactive resource lanes, final-group seal, reuse | `test_hbg_gm_mix`, `test_hbg_gm_mix_startup`, `test_hbg_direct_mix_handoff` | `test_gm_mix_execution.py`, `test_aiv_mix_handoff.py` |
| Common residency, one claim per position, early return | Admission/order tests and capacity rejection | `test_sync_start.py`, `test_sync_ordered_execution.py`; ST `sync_start_capacity` |
| Bypass publication and cache-domain separation | Layout/admission/error C++ tests | `test_admission_signal_publication.py` |
| Deferred completion remains unsupported | Public API fixture and invalid-context checks | `test_async_boundary.py` does not issue asynchronous work |
| Exact first-error identity, stale generation and failed reuse | `test_hbg_error_identity`, submit poison and late-error tests | `test_error_identity.py`; ST `resident_failure` |
| Dummy/inline service, predicate and direct-refill behavior | Inline-services, ready and dispatch tests | Existing ST predicate, early-enqueue and fallback suites |
| Frozen CLA-shaped dependency structure | Python CLA model/fixture tests | `test_cla_hbg.py`; structural probe, not original CLA numerical kernels |
| Original CSA kernels, all seven outputs, original blocks | CSA fixture/graph/business/scene contracts | `../csa/test_csa_business.py`, strict CSA L4 |
| Identity, native causal proof, complete trace export | `test_hbg_protocol_export`, Python native protocol/causality/converter tests | L4 native captures joined to independent dep-gen |
| Compiler image validation, cache handle identity, sim TLS lifetime | ELF, scene registration/cache and sim platform tests | Shared runtime regression suite |

Tests include invalid identities, repeated completion, missing members, paused
publishers, stale slot/group generations, overcapacity, zero-work members,
exhausted tickets with an unconfirmed valid claim, delayed lane release and
cross-task overlap. Counterexample models intentionally permit wrong ordering
or oversubscribed barriers and must detect a nonterminal no-progress state.

## CPU commands

```bash
python -m pytest tests/ut/py -q
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build -j 4
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure
```

The Python model lives in `tests/ut/py/hbg_spmd_contracts/`. It abstracts ownership
and admission publication as atomic transitions; it does not establish hardware
cache behavior, arbitrary kernel progress or physical fault recovery. Native
fixture tests ensure the probes exercise the intended protocol rather than
accepting a matching execution count alone.

## Native commands

```bash
python -m pytest tests/fdwic/pa_hbg/spmd --platform a5sim --pto-session-timeout 600
python -m pytest tests/fdwic/pa_hbg/spmd --platform a5 --device 0 --pto-session-timeout 1200
```

Case selection restricts common-residency tests to the actual simulator/device
capacity. Existing ST tests under `tests/st/a5/host_build_graph/` remain part of
regression coverage; do not replace their rejection/fallback expectations with
an assertion that every shape is supported. Runtime-mode checks require a new
publication for each invocation, not a stale mode from an earlier success.

For CSA, first import the SHA256-checked fixture using the
[example guide](../../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md).
Keep the original kernels, input, block counts and seven comparison rules:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python tests/fdwic/pa_hbg/csa/test_csa_business.py \
  -p a5 -d 0 --manual include --case TestCSABusiness:: --rounds 3
python tests/fdwic/pa_hbg/csa/test_csa_business.py \
  -p a5 -d 0 --manual include --case TestCSAResident::
python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py l4 \
  -d 0 --require-aicore-scheduler
```

Business variants remap physical pages consistently across pools, tables and
slot mappings; index payload and scale use the same mapping. A write-sentinel
variant detects missing index-scale writes. Unmapped regions remain protected.
Arbitrary batch permutation is not an oracle-preserving transformation because
the frozen snapshot contains cross-request physical-state aliases.

`csa/test_csa_snapshot.py` additionally checks real fixture/oracle corruption on
CPU. It requires fixture data; missing files fail that test. Probe-only runs do
not establish successful CSA numerical execution.

## Capture acceptance

For the fixed B4S2 snapshot, expect 79 kernel-bearing tasks, two Dummy tasks and
2308 physical executions (864 AIC, 1444 AIV). Derive the identity set from the
frozen graph instead of accepting these totals alone. Strict L4 requires:

- Seven outputs pass for both dependency and timing captures.
- Scheduler producer is AICore; every task/block/slot identity appears once.
- Original kernels, shapes, tensor views and fan-in agree with independent dep-gen.
- Native protocol causal audit passes, without globally sorting core-local clocks.
- Scheduler and Host capture streams have no dropped or truncated records.

Old traces remain readable but cannot gain missing identity/protocol evidence by
defaulting fields. L1 and a correct golden result are not substitutes for L4 proof.
Fault-injection runs check failure and buffer/Worker lifetime explicitly; a
failed run must not be forced through a successful-terminal-state validator.
