# a5 Hardware Layout

Chip-specific hardware facts for a5. For the cross-chip hardware model
(host / AICPU / AICore tiers, cluster structure, memory hierarchy
concepts) see
[docs/hardware/chip-architecture.md](../../../docs/hardware/chip-architecture.md).
For the cache coherency rules see
[docs/hardware/cache-coherency.md](../../../docs/hardware/cache-coherency.md).

## Chip packaging

a5 is a single chip composed of **2 dies** that present to the host as
**1 device ID** — from the runtime's perspective an a5 chip is one
device, regardless of die count.

## Per-die layout

| Component | Per die | Per chip (×2 dies) |
| --------- | ------- | ------------------ |
| AICPU clusters | 2 | 4 |
| AICPU cores per cluster | 2 | 2 |
| AICPU cores | 4 | 8 |
| AICore clusters | 18 | 36 |
| Units per AICore cluster | 1 AIC + 2 AIV (1C2V) | 1C2V |
| AIC | 18 | 36 |
| AIV | 36 | 72 |

L1 / L0A / L0B / L0C (per AIC), UB (per AIV), and L2 (per AICore
cluster) exist per the cross-chip model — sizes are not documented in
this repo.

## Host bus

| Host CPU | Bus |
| -------- | --- |
| x86 (Intel / AMD) | PCIe |
| Kunpeng (aarch64) | UB 2.0 |

## Verifying against real hardware

`tools/cann-examples/query` reads device info via CANN ACL.

- **Generation discriminator**: a die belongs to a5 iff CANN's
  `platform_config/<SoC>.ini` has `Short_SoC_version=Ascend950` (and
  `AIC_version=AIC-C-310`). See the canonical mapping in
  [docs/hardware/chip-architecture.md](../../../docs/hardware/chip-architecture.md#identifying-which-chip-generation-you-have).
- **Per-die layout above is one a5 variant**. CANN's a5 ini files span
  multiple SKUs (e.g. `Ascend950DT_9571…9599`, `Ascend950PR_957x…`)
  with `ai_core_cnt` ranging from 8 to ~28 per die — the 18 listed in
  the spec table is the variant this repo's runtime targets. Check
  the actual `Ascend950*.ini` for your SoC to confirm.

## Three views of "how many cores": observation + device-side ground truth

a5's HAL exposes more layers than a3 does. The same `halGetDeviceInfo`
call surface has **different semantics** on a5 vs a3 — do not assume
HAL counts mean the same thing across generations.

### Observed on a5 (one device, one chip = 2 dies)

| API | AICPU | AIC | AIV |
| --- | ----- | --- | --- |
| `rtGetAiCpuCount` | **6** | — | — |
| `aclrtGetDeviceInfo(ACL_DEV_ATTR_AICPU_CORE_NUM)` | **6** | — | — |
| CANN ini `ai_cpu_cnt` / `ai_core_cnt` / `vector_core_cnt` | (per-SKU, see ini) | (per-SKU) | (per-SKU) |
| `halGetDeviceInfo(AICPU, CORE_NUM)` host-side | **8** | — | — |
| `halGetDeviceInfo(AICPU, OCCUPY)` host-side | `0x1fe` (**9-bit** mask, 8 set: bits 1..8) | — | — |
| `halGetDeviceInfo(AICPU, IN_USED)` | **8** | — | — |
| `halGetDeviceInfo(AICORE, CORE_NUM)` | — | **36** (per device, = 2 dies × 18) | — |
| `halGetDeviceInfo(AICORE, DIE_NUM)` | — | **2** | — |
| `halGetDeviceInfo(VECTOR_CORE, CORE_NUM)` | — | — | **72** (per device) |
| DSMI `SOC_INFO+CPU_TOPO` | **9 logical CPUs** (8 physical + 1 hyperthread on phy_cpu_id 1) | — | — |

### Device-side probe resolves the AICPU question

CANN's `halGetDeviceInfo` exposes some queries (notably
`MODULE_TYPE_AICPU + INFO_TYPE_OS_SCHED`) that are flagged "used in
device" in the header — they only succeed when called from device-side
AICPU code, not from the host. The `tools/cann-examples/aicpu-device-query/`
companion tool uploads a small inner SO via the dispatcher bootstrap path,
runs HAL queries from inside an AICPU OS process, and reads results
back through GM. On this a5 host (`Ascend950PR_9599`) with local device
id 0 it returns:

| Query | Result | Interpretation |
| ----- | ------ | -------------- |
| `AICPU + OS_SCHED` | `0x1` | **AICPU OS owns exactly cpu_id 0** (single bit) |
| `AICPU + OCCUPY` (device-side) | `0x1f8 = 0b111111000` | **6 cores in the AICPU user pool at cpu_id 3..8** — not the `0x1fe` seen host-side. The 2-bit divergence (bits 1, 2) is the key new finding. |
| `AICPU + PF_OCCUPY` | `0x1f8` | identical to device-side OCCUPY → no SR-IOV / vNPU slicing |
| `AICPU + PF_CORE_NUM` | `6` | PF-view count matches user view → no virtualization |
| `AICPU + CORE_NUM` (device-side) | rc=3 | unlike a3, a5 restricts this query device-side — use `PF_CORE_NUM` instead |
| `CCPU + OCCUPY` | `0x1` | CCPU owns 1 core in its own namespace |
| `DCPU/TSCPU + OCCUPY`, `+ CORE_NUM` | rc=3 | module-level access restricted device-side (same as a3) |

The host-side / device-side OCCUPY divergence is **a5-specific**: on a3
both views return the same `0xfc`. On a5 host-side reports 8 enabled
cores (`0x1fe`) but the device-side AICPU OS exposes only 6 to its user
kernel pool (`0x1f8`). The 2-bit gap (bits 1, 2) exactly matches DSMI
CPU_TOPO's lone hyperthread pair on phy_cpu_id 1 — the AICPU OS keeps
the SMT-paired logical CPUs for itself rather than dispatching user
kernels onto them.

Combined with the absence of any vNPU mode (`is_virtual: no` via ACL),
the AICPU side splits as:

| Slot | Owner | Evidence |
| ---- | ----- | -------- |
| cpu_id 0 | AICPU OS scheduler | OS_SCHED bit 0 = 1 (device-side probe); cleared in host-side OCCUPY by design (OS scheduler is exposed via OS_SCHED, not OCCUPY) |
| cpu_id 1, 2 | Hyperthread pair on phy_cpu_id 1, withheld from the user pool by the AICPU OS | present in host-side OCCUPY (`0x1fe`) so they are **not** PG fab-disabled — that would clear them everywhere as cpu_id 1 was on a3. Absent from device-side AICPU OCCUPY (`0x1f8`), absent from CCPU OCCUPY (`0x1`). DSMI CPU_TOPO labels exactly this pair as the chip's only SMT pair. AICPU OS withholds SMT pairs from user dispatch to avoid intra-pair contention. |
| cpu_id 3..8 | user-schedulable (6) | device-side OCCUPY bits 3..8 set; matches `rtGetAiCpuCount=6` and `PF_CORE_NUM=6` |

The 9 → 6 gap on a5 is therefore **1 AICPU OS-reserved (cpu_id 0) + 2
SMT-pair withheld from user (cpu_id 1, 2)**, not "AICPU-OS-reserved
or PG fab-disabled" as the earlier inference from HAL host-side data
alone suggested. PG fab-disable can be ruled out on a5 by the host-side
OCCUPY containing both gap slots.

### Key semantic differences from a3

| Observation | a3 (Ascend910_93xx) | a5 (Ascend950) |
| ----------- | ------------------- | -------------- |
| `halGetDeviceInfo(AICPU, CORE_NUM)` host-side | 6 (matches user-visible) | **8** (does NOT match user-visible) |
| `halGetDeviceInfo(AICPU, CORE_NUM)` device-side | 6 (succeeds) | **rc=3** (restricted) |
| `halGetDeviceInfo(AICPU, OCCUPY)` host-side | 8-bit `0xfc` | **9-bit `0x1fe`** |
| `halGetDeviceInfo(AICPU, OCCUPY)` device-side | `0xfc` (matches host) | **`0x1f8` (differs from host)** — AICPU OS withholds the SMT pair |
| `AICPU` gap composition (HAL → user) | 1 OS-reserved + 1 PG fab-disabled | **1 OS-reserved + 2 SMT-pair withheld** (no PG-disable) |
| Logical vs physical AICPU | no hyperthread evidence | **1 phy core hyperthreaded → 9 logical** |
| `halGetDeviceInfo(AICORE, DIE_NUM)` | fails (rc=3) | works, returns **2** |
| `halGetDeviceInfo(AICORE, CORE_NUM)` | 25 per die | **36 per device** (aggregates both dies) |
| DSMI `SOC_INFO+CPU_TOPO` (sub=2) | fails (rc=8) | **works**, returns 9-CPU layout |

**Why per-die vs per-device differs**: on a3 each device ID maps to one
die, so HAL's "per-device" counts are per-die. On a5 each device ID
maps to one chip (= 2 dies), so HAL's "per-device" counts aggregate
both dies. ACL and CANN ini are stable across both — they consistently
report what user code can address.

### When to use which value (a5)

| You are doing… | Use |
| -------------- | --- |
| Configuring runtime `aicpu_thread_num` | **user-visible** (6) |
| Setting kernel `block_dim` for AICore | **user-visible** (per CANN ini for your specific SKU) |
| Counting cores in a multi-die a5 device | **per-device** HAL CORE_NUM (= 2 × per-die) |
| Reasoning about hyperthreading on AICPU | **DSMI CPU_TOPO** (only it shows the hyperthread pair on cpu_id 1+2) |
| Writing code expected to also work on a3 | **ACL or CANN ini only** — HAL semantics differ |
| Debugging "I requested N AICPU, only 6 ran" | gap is **1 AICPU OS scheduler (cpu_id 0) + 2 SMT-pair (cpu_id 1, 2) withheld by AICPU OS**; cap is 6 |

For cross-generation portable code: **always go through ACL or CANN
ini, never HAL**. HAL's CORE_NUM semantics shift between a3 and a5 in
ways that have no public documentation.
