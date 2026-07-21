# a2a3 Hardware Layout

Chip-specific hardware facts for the a2a3 generation. For the cross-chip
hardware model (host / AICPU / AICore tiers, cluster structure, memory
hierarchy concepts) see
[docs/hardware/chip-architecture.md](../../../docs/hardware/chip-architecture.md).
For the cache coherency rules see
[docs/hardware/cache-coherency.md](../../../docs/hardware/cache-coherency.md).

a2 and a3 share the same runtime / platform code under `src/a2a3/`; the
two generations differ only in chip packaging (die count, device-id
count, shared resources).

## a2 vs a3 packaging

| Property | a2 | a3 |
| -------- | -- | -- |
| Dies per chip | 1 | 2 (each die = one a2-equivalent) |
| Device IDs per chip | 1 | 2 (each die exposed as a separate device) |
| Shared across dies | n/a | AICPU operating system (a single AICPU OS spans both dies) |

From the runtime's perspective an a3 chip is **two devices that happen
to share an AICPU OS** — allocation, kernel launch, and per-task
dispatch all treat the two dies independently.

## Per-die layout (applies to a2, and to each a3 die)

| Component | Count |
| --------- | ----- |
| AICPU clusters | 2 |
| AICPU cores per cluster | 4 |
| AICPU cores per die | 8 |
| AICore clusters | 24 |
| Units per AICore cluster | 1 AIC + 2 AIV |
| AIC per die | 24 |
| AIV per die | 48 |
| GM (HBM) per die | 64 GiB (≈ 61.3 GiB user-visible after driver reserve) |

L1 / L0A / L0B / L0C (per AIC), UB (per AIV), and L2 (per AICore
cluster) exist per the cross-chip model — sizes are not documented in
this repo.

The counts above are the **spec view** (what's contractually delivered
and what user code can address). Below the spec view there is an extra
silicon layer that HAL exposes — **8 AICPU bit positions** (1 owned by
the AICPU OS scheduler, 1 fab-disabled by Partial-Good binning, 6
visible to runtime) and **25 AICore clusters in silicon** (1
fab-disabled, 24 visible to runtime). See "Three views of how many
cores" below for which API returns which view, and the device-side
probe (`tools/cann-examples/aicpu-device-query/`) that closed the
"OS-reservation vs PG" question with direct evidence.

## Verifying against real hardware

`tools/cann-examples/query` reads these counts via CANN ACL.

- **Generation discriminator**: a die belongs to this family iff CANN's
  `platform_config/<SoC>.ini` has `Short_SoC_version=Ascend910B` (a2)
  or `Short_SoC_version=Ascend910_93` (a3). See the canonical mapping
  in [docs/hardware/chip-architecture.md](../../../docs/hardware/chip-architecture.md#identifying-which-chip-generation-you-have).
- **Expected per-die output** (top-bin SKU): 24 AIC + 48 AIV,
  ~61 GiB HBM user-visible (64 GiB raw).
- **Multi-device example**: on an a3 host with 8 chips you see 16
  device IDs (2 dies per chip).

## Three views of "how many cores": observation + device-side ground truth

Different APIs return different counts for the same die. They are not
contradictory — they measure different things. The **observed
discrepancy on this generation** is:

- AICPU: HAL's `AICPU+OCCUPY` is an **8-bit mask** with value
  `0xfc = 0b11111100` (6 bits set, bits 0+1 cleared); ACL / CANN ini /
  `rtGetAiCpuCount` all report **6 user-accessible**.
- AICore: HAL's `AICORE+CORE_NUM` is **25**; ACL `aclrtGetStreamResLimit`
  and CANN ini `cube_core_cnt` are **24**. AIV mirrors at 50 → 48 (1:2 ratio).

Both gaps are **deterministic across all 16 dies on this host** (sampled
via `tools/cann-examples/aicpu-topo`): every die reports the same
`0xfc` mask and the same 25 AICore count.

### Device-side probe resolves the AICPU question

CANN's `halGetDeviceInfo` exposes some queries (notably
`MODULE_TYPE_AICPU + INFO_TYPE_OS_SCHED`) that are flagged "used in
device" in the header — they only succeed when called from device-side
AICPU code, not from the host. The `tools/cann-examples/aicpu-device-query/`
companion tool uploads a small inner SO via the dispatcher bootstrap path,
runs HAL queries from inside an AICPU OS process, and reads results
back through GM. On this a3 host with local device id 0 it returns:

| Query | Result | Interpretation |
| ----- | ------ | -------------- |
| `AICPU + OS_SCHED` | `0x1` | **AICPU OS owns exactly cpu_id 0** (single bit) |
| `AICPU + OCCUPY` | `0xfc` | cpu_id 2..7 in the AICPU subsystem's view |
| `AICPU + PF_OCCUPY` | `0xfc` | identical to OCCUPY → no SR-IOV / vNPU slicing |
| `AICPU + PF_CORE_NUM` | `6` | PF-view count matches user view → no virtualization |

Combined with the absence of any vNPU mode (`is_virtual: no` via ACL),
the AICPU side splits as:

| Slot | Owner | Evidence |
| ---- | ----- | -------- |
| cpu_id 0 | AICPU OS scheduler | OS_SCHED bit 0 = 1 (device-side probe) |
| cpu_id 1 | Partial-Good fab-disabled | absent from OS_SCHED, OCCUPY, PF_OCCUPY, and from every other CPU module's OCCUPY (CCPU/DCPU/TSCPU all checked); a3 not in vNPU mode rules out virtualization remapping |
| cpu_id 2..7 | user-schedulable (6) | OCCUPY bits 2..7 set; matches `rtGetAiCpuCount=6` |

The 8 → 6 gap on a3 is therefore **1 AICPU OS-reserved + 1 PG
fab-disabled**, not "2 OS-reserved" as earlier inferred from HAL alone.

### AICore +1 by inference, same chip

There is no analogous "AICore OS_SCHED" query — AICore is bare-metal
compute. The 25 → 24 gap is **Partial-Good silicon binning** (chip is
designed with 25 clusters per die, sold as a 24-cluster part by
qualifying any die with ≥ 24 working clusters). The confidence is
high because the same chip we just probed has confirmed PG on the AICPU
side (cpu_id 1 above), making "same fab, same binning policy on the
AICore side" the parsimonious read. The exact disabled cluster index
is not exposed by any host or device API we have located.

### Which API gives which view (observed, this generation, this host)

| API | AICPU | AIC | AIV |
| --- | ----- | --- | --- |
| `rtGetAiCpuCount` | **6** | — | — |
| `aclrtGetDeviceInfo(ACL_DEV_ATTR_AICPU_CORE_NUM)` | **6** | — | — |
| `aclrtGetStreamResLimit(ACL_RT_DEV_RES_CUBE_CORE)` | — | **24** | — |
| `aclrtGetStreamResLimit(ACL_RT_DEV_RES_VECTOR_CORE)` | — | — | **48** |
| CANN ini `ai_cpu_cnt` / `cube_core_cnt` / `vector_core_cnt` | **6** | **24** | **48** |
| `halGetDeviceInfo(AICPU, CORE_NUM)` | **6** | — | — |
| `halGetDeviceInfo(AICPU, OCCUPY)` | `0xfc` (8-bit, 6 set) | — | — |
| `halGetDeviceInfo(AICORE, CORE_NUM)` | — | **25** | — |
| `halGetDeviceInfo(VECTOR_CORE, CORE_NUM)` | — | — | **50** |

**Important: these mappings are observed on a3 (`Ascend910_9392`).
On a5 the HAL semantics shift** — `halGetDeviceInfo(AICPU, CORE_NUM)`
returns 8 there (not 6), and the OCCUPY mask is 9-bit `0x1fe`. The
underlying reason (an extra layer of reservation, plus 1 hyperthread
logical CPU) is described in `src/a5/docs/hardware.md`. Do not assume
HAL's CORE_NUM semantics are stable across generations; for cross-gen
correctness, prefer ACL or CANN ini.

### When to use which value

| You are doing… | Use |
| -------------- | --- |
| Configuring runtime `aicpu_thread_num` (per-case config) | **user-accessible** (6 — never request more) |
| Setting kernel `block_dim` for AICore launch | **user-accessible** (24 — runtime rejects > 24) |
| Reading the spec sheet / product datasheet | **spec / user-accessible** (24 AIC, 6 AICPU) |
| Asking "what is in silicon" | **HAL physical** (25 AIC, 8 AICPU slots) — 1 PG-disabled + 1 OS-reserved on AICPU; 1 PG-disabled on AICore |
| Debugging "I set `aicpu_thread_num=8`, got error" | gap is **cpu_id 0 (AICPU OS) + cpu_id 1 (PG)** — both unreachable from runtime; cap is 6 |
| Debugging "I set `block_dim=25`, got error" | runtime cap is **24**; the +1 slot is PG fab-disabled |
| Building a chip-management dashboard | use HAL OCCUPY + OS_SCHED to expose the split; cite `tools/cann-examples/aicpu-device-query/` for the OS_SCHED query |

For everyday kernel and runtime work: **always use the user-accessible
view** (ACL, CANN ini, `rtGetAiCpuCount`). The HAL physical-view counts
are diagnostic only and are exposed by `tools/cann-examples/aicpu-topo/`
(host side) and `tools/cann-examples/aicpu-device-query/` (device side,
for the OS_SCHED bitmap that needs an AICPU OS context).

## Host bus

| Host CPU | Bus |
| -------- | --- |
| x86 (Intel / AMD) | PCIe |
| Kunpeng (aarch64) | UB 1.0 — also known as HCCS |
