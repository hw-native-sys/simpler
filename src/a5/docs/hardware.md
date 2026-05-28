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

## Three views of "how many cores": observation + calibrated inference

a5's HAL exposes more layers than a3 does. The same `halGetDeviceInfo`
call surface has **different semantics** on a5 vs a3 — do not assume
HAL counts mean the same thing across generations.

### Observed on a5 (one device, one chip = 2 dies)

| API | AICPU | AIC | AIV |
| --- | ----- | --- | --- |
| `rtGetAiCpuCount` | **6** | — | — |
| `aclrtGetDeviceInfo(ACL_DEV_ATTR_AICPU_CORE_NUM)` | **6** | — | — |
| CANN ini `ai_cpu_cnt` / `ai_core_cnt` / `vector_core_cnt` | (per-SKU, see ini) | (per-SKU) | (per-SKU) |
| `halGetDeviceInfo(AICPU, CORE_NUM)` | **8** | — | — |
| `halGetDeviceInfo(AICPU, OCCUPY)` | `0x1fe` (**9-bit** mask, 8 set) | — | — |
| `halGetDeviceInfo(AICPU, IN_USED)` | **8** | — | — |
| `halGetDeviceInfo(AICORE, CORE_NUM)` | — | **36** (per device, = 2 dies × 18) | — |
| `halGetDeviceInfo(AICORE, DIE_NUM)` | — | **2** | — |
| `halGetDeviceInfo(VECTOR_CORE, CORE_NUM)` | — | — | **72** (per device) |
| DSMI `SOC_INFO+CPU_TOPO` | **9 logical CPUs** (8 physical + 1 hyperthread on phy_cpu_id 1) | — | — |

### Two-layer AICPU reservation on a5

`9 → 8 → 6` shows two distinct reservations stacked:

1. **9 logical CPUs** (DSMI CPU_TOPO total): 8 physical Taishan cores
   on this die, one of which is hyperthreaded into 2 logical CPUs.
2. **8 in HAL OCCUPY mask** (`0x1fe = 0b111111110`): bit 0 is cleared,
   bits 1–8 set. Whatever owns cpu_id 0 — likely the lowest-level
   firmware / hypervisor — is below the HAL's view entirely.
3. **6 in `rtGetAiCpuCount`**: the additional 2 cores between HAL's
   "occupied 8" and runtime's "user-visible 6" are most plausibly
   AICPU-OS-reserved or PG-disabled, by analogy with a3 where a
   device-side probe confirmed `OS_SCHED = 0x1` (1 OS core) + the
   remaining gap cpu_id is PG fab-disabled.
   (See [`src/a2a3/docs/hardware.md`](../../a2a3/docs/hardware.md#device-side-probe-resolves-the-aicpu-question)
   for the technique that resolved the a3 question.)

**The a3-equivalent question on a5 is not yet resolved**:
`tools/cann-examples/aicpu-device-query/` should be run on a5 hardware
to read `AICPU + OS_SCHED` from inside an AICPU OS process — that one
bit pattern will tell us how many of the 2 "missing" cores between
HAL's 8 and runtime's 6 are OS-reserved (the rest being PG-disabled).
Until that probe runs on a5, the two-layer breakdown above is
**inference by analogy**, not direct measurement. Likewise the role of
cpu_id 0 (cleared in OCCUPY) — firmware-only / RAS / boot — remains
inferred until a device-side query covers it.

### Key semantic differences from a3

| Observation | a3 (Ascend910_93xx) | a5 (Ascend950) |
| ----------- | ------------------- | -------------- |
| `halGetDeviceInfo(AICPU, CORE_NUM)` | 6 (matches user-visible) | **8** (does NOT match user-visible) |
| `halGetDeviceInfo(AICPU, OCCUPY)` | 8-bit `0xfc` | **9-bit `0x1fe`** |
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
| Debugging "I requested N AICPU, only 6 ran" | gap is the **AICPU OS + lowest-level reservation**, totalling 3 cores between physical 9 and user 6 |

For cross-generation portable code: **always go through ACL or CANN
ini, never HAL**. HAL's CORE_NUM semantics shift between a3 and a5 in
ways that have no public documentation.
