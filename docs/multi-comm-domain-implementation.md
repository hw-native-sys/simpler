# Multi-Communication-Domain Implementation

This document tracks the implementation status for multiple communication
domains.  It complements the design note in
[`multi-comm-domain.md`](multi-comm-domain.md) and is intentionally written as
a completion ledger: what exists, what is deliberately absent, and what has
been validated.

## Current Surface

The only parent-level communication API is `CommDomainPlan`:

```python
comm_plan = CommDomainPlan(
    domains=[
        CommDomain(
            name="tp",
            worker_indices=[0, 1],
            window_size=tp_window_size,
            buffers=[
                ChipBufferSpec("scratch", "float32", count, nbytes),
            ],
        ),
    ],
)

worker = Worker(level=3, device_ids=device_ids, comm_plan=comm_plan)
```

The low-level explicit path is still available when a caller must add
per-chip bootstrap data, such as host staging:

```python
plan = CommDomainPlan([...])
cfgs = [
    ChipBootstrapConfig(
        comm=plan.bootstrap_for_worker(i),
        host_inputs=[
            HostBufferStaging(
                domain_name="default",
                name="notify_counter",
                shm_name=rank_local_shm.name,
                size=4,
            ),
        ],
    )
    for i in range(nranks)
]

worker = Worker(level=3, device_ids=device_ids, chip_bootstrap_configs=cfgs)
```

`ChipBootstrapConfig.comm` accepts `list[ChipDomainBootstrapConfig]` or
`None`.  The old public `ChipCommBootstrapConfig` surface is removed.

## Feature Status

Implemented:

- `CommDomain`: name, ordered `worker_indices`, window size, and symmetric
  buffers.
- `CommDomainPlan`: parent-level source of truth for all domains in an L3
  worker.
- `bootstrap_for_worker(i)`: derives only the domains containing worker index
  `i`.
- Domain rank mapping: order in `worker_indices` defines `domain_rank`.
- Missing domain behavior: `ctx.domains[name]` is a normal dict lookup and
  raises `KeyError`.
- Symmetric buffer layout: every participant in one domain receives the same
  named buffer layout.
- Multiple domains per chip: a chip can publish more than one
  `ChipCommDomainContext`.
- Overlapping domains: the same chip may have independent ranks in different
  domains.
- `ChipBootstrapConfig.comm`: takes the derived list from
  `CommDomainPlan.bootstrap_for_worker`.
- Explicit host staging: stays on `ChipBootstrapConfig`, keyed by
  `(domain_name, buffer_name)`.
- `Worker(..., comm_plan=...)`: creates per-chip bootstrap configs inside the
  L3 parent.
- Explicit `chip_bootstrap_configs`: still supported for host staging and
  manual bootstrap control.
- Hidden base communicator/window: one base communicator and one base window
  per L3 worker.
- Domain-local `CommContext`: each visible domain gets its own
  kernel-visible `CommContext*`.
- HCCL base-window slicing: domain windows are slices of the hidden base
  window.
- Sim backend slicing: sim mirrors the domain slicing contract for tests and
  examples.
- Bootstrap mailbox domains: publishes named domain results and rejects
  duplicate names.
- Domain-aware Python context: communication access is through
  `ChipContext.domains[name]`.
- Public old single-domain config removal: no public
  `ChipCommBootstrapConfig` compatibility path remains.

Not implemented by design:

- HCCL collective kernels.  PTO-ISA RMA kernels use HCOMM/HCCL windows only.
- Visible meta domain.  No all-device logical domain is exposed without
  buffers.
- Independent root-info per domain.  The implementation derives domain views
  from one base setup.
- Automatic window sizing.  Domain `window_size` remains explicit, matching
  current style.

## Bootstrap Flow

```text
L3 parent
  owns CommDomainPlan
  validates all domain names, worker indices, windows, and buffers
  for each chip worker i:
    derives comm = plan.bootstrap_for_worker(i)
    creates ChipBootstrapConfig(comm=comm, host staging=optional)
    attaches private base communicator metadata
        |
        v
chip child
  initializes one hidden base communicator
  allocates one hidden base communication window
  derives one domain-local CommContext for each received domain config
  carves named buffers inside that domain's window slice
  stages requested host inputs
  publishes named domain contexts through ChipBootstrapChannel
        |
        v
L3 orchestration
  domain = ctx.domains["tp"]
  passes domain.buffer_ptrs["scratch"] and domain.device_ctx to kernels
```

Workers outside a domain receive no config for that domain and publish no
`ctx.domains[name]` entry.

## Host Staging

Host staging is not part of a communication domain.  A domain describes a
symmetric communication contract: membership, dense rank order, window size,
and buffer layout.  Host staging describes per-chip movement between
parent-owned POSIX shared memory and one chip's already declared domain
buffer.

The split is:

- `CommDomain`: symmetric domain contract shared by participants;
- `CommDomainPlan`: parent-level source of truth for all domains;
- `ChipBootstrapConfig`: per-chip transport object sent to one chip child;
- `HostBufferStaging`: per-chip shared-memory source or destination.

Therefore host staging stays on `ChipBootstrapConfig`.  Each staging entry
that targets a communication buffer includes `domain_name`; the effective key
is `(domain_name, buffer_name)`, so two domains can both own `"scratch"`.

## Why Sim Was Extended

The sim backend is not only a compatibility target.  It is the cheapest place
to validate parent-side derivation, child bootstrap, missing-domain behavior,
domain rank order, duplicate-name rejection, and host-staging scoping without
needing hardware.  The sim extension mirrors the same visible contract:
`ctx.domains[name]` contains a domain-local context and buffer pointers.

It does not attempt to model HCCL transport performance or collective
behavior.  It only validates the runtime contract that kernels and
orchestration code consume.

## Example Coverage

Two new L3 examples now cover the multi-domain surface:

- `domain_rank_map`: a small communication example.  It shows the difference
  from single-domain usage by checking domain-local ranks, absent domains,
  separate slices for overlapping memberships, and one real allreduce per
  domain.
- `dual_domain_overlap`: a real data example.  It runs two overlapping
  domains, performs domain-local allreduce in both, then runs affine compute
  and checks real outputs against host goldens.

Existing one-domain communication examples were also migrated to the new
surface.  They are still important because they prove the single-domain case
is expressed through the same API rather than through a compatibility path.

## Validation Status

Validation date: 2026-05-13.

### Build And Unit Tests

- Editable build: pass.

  ```bash
  pip install --no-build-isolation -e .
  ```

- Focused unit and sim tests: pass, 30 tests.

  ```bash
  pytest \
      tests/ut/py/test_worker/test_comm_domain_plan.py \
      tests/ut/py/test_worker/test_worker_distributed_sim.py \
      tests/ut/py/test_worker/test_bootstrap_context_sim.py \
      tests/ut/py/test_worker/test_bootstrap_channel.py \
      -q
  ```

- Hardware bootstrap unit test: pass.

  ```bash
  pytest tests/ut/py/test_worker/test_bootstrap_context_hw.py \
      -q -s --platform a2a3 --device 3-4
  ```

- Docs lint: pass.

  ```bash
  markdownlint-cli2 \
      docs/multi-comm-domain.md \
      docs/multi-comm-domain-implementation.md
  ```

### Example Results

- `workers/l3/domain_rank_map`
  - Sim: not applicable.
  - Hardware: pass with `-p a2a3 -d 3-5`.
  - New small domain-rank and per-domain communication example.

- `workers/l3/dual_domain_overlap`
  - Sim: not applicable.
  - Hardware: pass with `-d 3-5`.
  - New two-domain data and compute example.

- `workers/l3/allreduce_distributed`
  - Sim: not applicable.
  - Hardware: pass with `-d 3-4`.
  - One-domain baseline through `CommDomainPlan`.

- `workers/l3/ffn_tp_parallel`
  - Sim: not applicable.
  - Hardware: pass with `-d 3-4`.
  - One-domain tensor-parallel compute plus reduce.

- `workers/l3/ep_dispatch_combine`
  - Sim: not applicable.
  - Hardware: pass with `-d 3-4`.
  - One-domain EP dispatch/combine.

- `a2a3/async_notify_demo`
  - Sim: pass with `-p a2a3sim`.
  - Hardware: not run.
  - Explicit bootstrap plus host staging.

- `a2a3/deferred_notify_demo`
  - Sim: pass with `-p a2a3sim`.
  - Hardware: not run.
  - Explicit bootstrap plus deferred notify staging.

- `a2a3/sdma_async_completion_demo`
  - Sim: not run.
  - Hardware: fail with `-p a2a3 -d 3-4`.
  - Both chips bootstrap, then `worker.run()` fails with AICPU 507015.

- `a5/async_notify_demo`
  - Sim: pass with `-p a5sim`.
  - Hardware: not run.
  - A5 sim explicit bootstrap path.

- `a5/deferred_notify_demo`
  - Sim: pass with `-p a5sim`.
  - Hardware: not run.
  - A5 sim deferred notify path.

The SDMA failure happens after both chip bootstrap messages are ready.  The
observed error is `aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507015`,
followed by 507901 during stream teardown.  This is tracked as a remaining
SDMA runtime/data-plane issue, not as a communication-domain bootstrap
failure.

## Grep Gates

The migrated public examples should stay clean for these stale surfaces:

```bash
rg -n "ChipCommBootstrapConfig" examples
rg -n "comm=ChipCommBootstrapConfig|rootinfo_path" examples
rg -n "ctx\\.buffer_ptrs|ctx\\.device_ctx|ctx\\.rank|ctx\\.nranks" examples
```

Repository-wide checks should also keep generated or local-only paths out of
the design documents.

## Non-Goals

- no visible meta communication domain;
- no independent root-info communicator per domain;
- no HCCL collective kernels;
- no implicit current domain inside kernels;
- no lazy first-use domain creation;
- no automatic `window_size` derivation in this implementation.
