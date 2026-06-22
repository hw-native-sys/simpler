# Ascend Chip Architecture

This page describes the **single-chip hardware substrate** the runtime
in this repo targets: how a host attaches to an Ascend NPU chip via a
system bus, how the chip is organized into AICPU and AICore clusters,
and the off-chip / on-chip boundaries that shape every design decision
under `src/{arch}/`. Concepts here are shared across the chip
generations supported in this repo (a2a3, a5). Chip-specific counts,
sizes, and bus versions live in `src/a2a3/docs/` and `src/a5/docs/`.

For multi-chip / multi-host server topology, a future
`server-architecture.md` will sit alongside this page.

For the *software* three-program model layered on top (host `.so` +
AICPU `.so` + AICore `.o`), see [chip-level-arch.md](../chip-level-arch.md).

## System overview: host + chip

A deployment is a **host** (CPU + DDR) attached to one or more
**Ascend chips** via a system bus.

```text
┌───────────────────────┐           ┌────────────────────────────┐
│   Host                │           │   Ascend chip              │
│   ┌────────────────┐  │           │   ┌────────────────────┐   │
│   │ Host CPU       │  │   bus     │   │ AICPU clusters     │   │
│   │  x86 or        │──┼──────────▶│   │  (control)         │   │
│   │  Kunpeng       │  │           │   └─────────┬──────────┘   │
│   │ + Host DDR     │  │           │             │ on-chip bus  │
│   └────────────────┘  │           │             │ + MMIO regs  │
└───────────────────────┘           │   ┌─────────▼──────────┐   │
                                    │   │ AICore clusters    │   │
                                    │   │  L2 (per cluster)  │   │
                                    │   │  AIC: L1, L0A/B/C  │   │
                                    │   │  AIV: UB           │   │
                                    │   └────────────────────┘   │
                                    │                            │
                                    │   GM (shared HBM)          │
                                    └────────────────────────────┘
```

The host↔chip bus depends on the host CPU:

| Host CPU | Bus | Naming on a2a3 | Naming on a5 |
| -------- | --- | -------------- | ------------ |
| x86 (Intel / AMD) | PCIe | PCIe | PCIe |
| Kunpeng (aarch64) | UB (Unified Bus) | UB 1.0 / HCCS | UB 2.0 / UB |

PCIe and UB are interchangeable from the runtime's perspective — the
CANN driver hides the difference. Both have microsecond-scale latency;
the design pressure to minimize host round-trips applies equally.

A chip may comprise one or more dies, and may present to the host as
one or more device IDs (the die ↔ device-id mapping is chip-specific —
see `src/{a2a3,a5}/docs/`).

**GM is "shared" only within one `device_id`** — shared across that device's
AICPU and AICore tiers. It is **exclusive per `device_id`**, never shared across
device IDs. The die↔device mapping differs by arch but does not change this:

- **a5**: one `device_id` owns the chip's dies/clusters, managed as one unit by
  the AICPU; one orch runs per `device_id` (an invariant). There is no
  "two-die concurrency" inside a device.
- **a2a3**: a card presents two `device_id`s; they are independent.

So multiple device IDs (e.g. pytest xdist `gw0`/`gw1`, or `--device 2-3`) run on
**independent HBM with no cross-device contention**. A per-device OOM comes from
that device's own workload, not from an "adjacent die". (The chip-shared
contention model was removed in PR #990 — see
`.claude/rules/running-onboard.md`.)

## Identifying which chip generation you have

CANN ships per-SoC platform configs under
`$ASCEND_HOME_PATH/{aarch64,x86_64}-linux/data/platform_config/<SoC>.ini`.
The `Short_SoC_version=` line in those ini files is the authoritative
discriminator. This repo's arch names map to CANN's families as:

| This repo | Product family | `Short_SoC_version` | `AIC_version` | `NpuArch` | Example `SoC_version` values |
| --------- | -------------- | ------------------- | ------------- | --------- | ---------------------------- |
| **a2** | Atlas A2 (`800T A2`, `300T A2`, `200T A2 Box16`, `900 A2 PoD`, …) | `Ascend910B` | `AIC-C-220` | 2201 | `Ascend910B1`, `Ascend910B2`, `Ascend910B3`, `Ascend910B4` |
| **a3** | Atlas A3 | `Ascend910_93` | `AIC-C-220` | 2201 | `Ascend910_9362`, `9372`, `9381`, `9382`, `9391`, `9392` |
| **a5** | Atlas 950 (presumed; confirm against ini for your SoC) | `Ascend950` | `AIC-C-310` | 3510 | `Ascend950DT_9571…9599`, `Ascend950PR_957x…` |

a2 and a3 share `AIC-C-220` because a3 is a chiplet-pair packaging of
two a2-equivalent dies (publicly described as the "910C" chip — two
910B dies under one heat-spreader, presenting as two device IDs that
share an AICPU OS). a5 is a new `AIC-C-310` AICore microarchitecture.

**Sources for the Atlas A2 / A3 ↔ SoC family mapping**:

- vllm-ascend FAQ #21 documents `export SOC_VERSION="ascend910b1"` for
  Atlas A2 and `export SOC_VERSION="ascend910_9391"` for Atlas A3
  (<https://docs.vllm.ai/projects/ascend/en/latest/faqs.html>).
- Same FAQ lists kernel packages by family: `Ascend-cann-kernels-910b`
  for Atlas A2, `Atlas-A3-cann-kernels` for Atlas A3.
- The repo↔CANN AICore bridge is closed by `simpler_setup/toolchain.py`
  - `src/{a2a3,a5}/platform/onboard/aicore/CMakeLists.txt`: `a2a3`
  compiles with `--cce-aicore-arch=dav-c220-…`, `a5` with `dav-c310-…`,
  which line up 1:1 with the `CCEC_AIC_version=` / `CCEC_VECTOR_version=`
  fields in every matching `platform_config/Ascend*.ini`.

To check a live machine, run `tools/cann-examples/query device <id>` —
the tool reads the SoC name via ACL, looks up the matching ini, and
prints `detected_arch: a2|a3|a5`.

## The three execution tiers

The runtime works with three execution tiers, each compiled with a
different toolchain.

| Tier | Where it runs | Role | Code in this repo |
| ---- | ------------- | ---- | ----------------- |
| Host CPU | Off-chip (host server) | Orchestration; allocates memory, loads kernels, submits tasks | `src/{arch}/platform/onboard/host/` |
| AICPU | On-chip control cluster | Task scheduler; dispatches tasks to AICore, tracks completion | `src/{arch}/platform/onboard/aicpu/` + `src/{arch}/runtime/*/aicpu/` |
| AICore | On-chip compute clusters (AIC + AIV per cluster) | Compute kernels (matmul on AIC, vector ops on AIV) | `src/{arch}/platform/onboard/aicore/` + `src/{arch}/runtime/*/aicore/` |

## Host

The host is a server-class CPU running a Linux process. It owns:

- **Host CPU** — x86_64 (Intel / AMD) or aarch64 (Kunpeng).
- **Host DDR** — system RAM; all host-side allocations live here.
- One or more **Ascend chips**, attached via the bus listed in the
  table above.

The host:

- **Role**: orchestration, not compute. Allocates GM (device memory),
  copies kernel binaries to the chip, packs task descriptors, and
  submits them via the driver.
- **Cannot directly dispatch tasks to AICore.** All on-chip work goes
  through AICPU.
- **Latency floor**: every host → chip operation pays a driver
  round-trip (microsecond scale, PCIe or UB alike). The runtime
  submits a *graph*, not per-task dispatches.

Code lives in `src/{arch}/platform/onboard/host/` (production) and
`src/{arch}/platform/sim/host/` (thread-based simulation).

## AICPU (on-chip control)

AICPU is the on-chip control complex. Organized as **AICPU clusters**:

- A chip contains multiple AICPU clusters (count chip-specific — see
  `src/{a2a3,a5}/docs/`).
- Each cluster contains multiple AICPU cores.
- All cores together run the AICPU `.so` that the host loads onto the
  chip.

Properties:

- **Role**: task scheduler. Receives a `Runtime` struct from the host
  (containing the task graph + buffer pointers), dispatches ready
  tasks to AICore units, and tracks completion.
- **Why it exists**: per-task dispatch must not pay host-bus latency.
  AICPU sits on-die with AICore and talks to it over the on-chip bus,
  signaling through MMIO control / completion registers (see
  [cache-coherency.md](cache-coherency.md) for the COND / FIN
  protocol).
- **Constraints**:
  - Its own data cache, **not coherent** with host DMA writes — see
    [cache-coherency.md](cache-coherency.md).
  - No virtual memory in the host sense; all GM addresses are physical
    handles supplied by the host via the `Runtime` struct.

Code lives in `src/{arch}/platform/onboard/aicpu/` (driver-facing
entry point) and `src/{arch}/runtime/*/aicpu/` (the actual scheduler
implementation, swappable per runtime variant).

## AICore (on-chip compute)

AICore is the compute substrate, organized as **AICore clusters**:

- A chip contains multiple AICore clusters (count chip-specific).
- Each cluster contains **1 AIC** (cube) and **multiple AIV** (vector)
  — typical ratio 1C : 2V.
- All units in a cluster share one **L2 cache**.

Within a cluster:

| Unit | Abbreviation | Compute | Private memory |
| ---- | ------------ | ------- | -------------- |
| Cube | **AIC** | Matrix-multiply / convolution | L1 + L0A / L0B / L0C |
| Vector | **AIV** | Element-wise SIMD (add, mul, activation, reductions) | UB (Unified Buffer) |

Sizes of L1 / L0 / UB / L2 are chip-specific — see
`src/{a2a3,a5}/docs/`.

Properties:

- **Role**: compute, not control. AICore units do not self-schedule;
  they execute a task descriptor handed to them by AICPU.
- **Cache asymmetry**: AICore's data cache is **not coherent** with
  GM in the AICore→GM direction. Code that publishes results AICPU
  will read must emit `dcci` — see
  [cache-coherency.md](cache-coherency.md).
- **Terminology**: "AICore" is the umbrella; "AIC" / "AIV" refer to
  the unit types inside a cluster, per
  [.claude/rules/ascend.md](../../.claude/rules/ascend.md).

Code lives in `src/{arch}/platform/onboard/aicore/` (build glue,
ccec-compiled) and `src/{arch}/runtime/*/aicore/` (the kernel
executors).

## How the three tiers cooperate (end-to-end task flow)

A complete `Worker::run()` call traverses all three tiers:

```text
1. Host (C++ via ChipWorker)
   - allocates GM for inputs / outputs
   - copies inputs H2D (over PCIe or UB)
   - packs Runtime + TaskArgs in GM
   - rtMemcpy a small Runtime header to a known GM address
   - writes the AICPU start register to signal "go"

2. AICPU (.so running on the chip)
   - cache_invalidate_range on the Runtime header   (host DMA wrote it)
   - reads task graph + buffer pointers
   - for each ready task, picks an idle AICore unit (AIC or AIV)
   - writes task descriptor to that unit over the on-chip bus
   - writes the unit's start / COND register to signal "go"

3. AICore (AIC or AIV kernel)
   - reads descriptor, runs the kernel
   - dcci any result that AICPU or peer unit must see
   - writes FIN to its completion register

4. AICPU
   - polls AICore completion registers (Device-nGnRnE MMIO)
   - rmb() before reading any AICore-produced slot
                                       (see cache-coherency.md)
   - releases dependents into the ready queue
   - on the last task, writes FIN back to the host

5. Host
   - polls host-visible completion flag
   - copies outputs D2H
   - frees GM
```

The detailed wire formats and queue shapes are documented in
[task-flow.md](../task-flow.md). Cache invalidation and load-load
ordering at the AICore→AICPU and host→AICPU handoffs are documented
in [cache-coherency.md](cache-coherency.md).

## Off-chip vs on-chip boundary

The host↔chip link is **the** boundary that dominates runtime design.

| Boundary | Typical cost | Implication |
| -------- | ------------ | ----------- |
| Host ↔ Chip (PCIe or UB + driver) | µs | One host-driven dispatch per task is too slow; the runtime batches a task graph and hands it off in a single submission. |
| AICPU ↔ AICore (on-chip bus + MMIO control / completion regs) | ns | AICPU can dispatch and reap thousands of small tasks per second without host involvement. |
| AICore ↔ GM (cache + bus) | sub-µs | Compute-side bottlenecks are about bandwidth, not latency. |

The runtime is structured as **host submits a graph, AICPU runs the
graph** — not **host dispatches each task** — precisely because the
on-chip scheduler exists to keep per-task control off the host link.

## Memory at a glance

Each tier sees a different memory landscape:

- **Host**: Host DDR, plus a window into GM through the driver.
- **AICPU**: GM (cached), AICPU-cluster SRAM, MMIO registers.
- **AICore cluster**: L2 cache (shared by all units in the cluster).
- **AIC** (per unit): L1, L0A / L0B / L0C — not addressable from host
  or AICPU.
- **AIV** (per unit): UB — not addressable from host or AICPU.

The asymmetric coherency rules between these caches are in
[cache-coherency.md](cache-coherency.md). Sizes are chip-specific —
see `src/{a2a3,a5}/docs/`.

## What to read next

| You want to know… | Read |
| ----------------- | ---- |
| When to insert `dcci` / `cache_invalidate_range` / `rmb()` | [cache-coherency.md](cache-coherency.md) |
| AIC_CTRL MMIO attribute / cost / how to pick a notification path | [mmio-performance.md](mmio-performance.md) |
| Where the public CANN driver / HCCL / HCOMM source lives | [cann-source-references.md](cann-source-references.md) |
| Software three-program model layered on this hardware | [../chip-level-arch.md](../chip-level-arch.md) |
| End-to-end task data flow | [../task-flow.md](../task-flow.md) |
| Chip-specific counts, bus version, die / device-id mapping | [`src/a2a3/docs/hardware.md`](../../src/a2a3/docs/hardware.md), [`src/a5/docs/hardware.md`](../../src/a5/docs/hardware.md) |
