# MMIO Performance (AIC_CTRL register window)

This page is the authoritative reference for the **AICPU↔AICore
control-register MMIO** path — its ARM memory attribute, its observed
latency and throughput, and what those numbers mean for designs that
poll or signal through it. Read it before proposing any optimisation
that touches DMB / COND / FAST_PATH_ENABLE or any other AIC_CTRL
register. Measurements below are sampled on a2a3 silicon; a5 uses the
same driver mapping path and is expected to behave identically modulo
clock frequency.

## Memory attribute — Device-nGnRE, proven from driver source

The mapping is established when host code calls
`halMemCtl(ADDR_MAP_TYPE_REG_AIC_CTRL)` (see
`src/{arch}/platform/onboard/host/host_regs.cpp`). That call sends a
channel message to the device-side driver, which mmaps the chip's AIC
control-register window into both the host process and the AICPU
process at the same VA. The mmap uses Linux's standard
`pgprot_device()`. Paths below reference the public `cann/driver`
source on gitcode — see
[`cann-source-references.md`](cann-source-references.md) for the
clone-to-`build/` convention if you want to grep across the tree:

| Step | File (gitcode.com/cann/driver) |
| ---- | ------------------------------ |
| HAL ioctl dispatch | `src/ascend_hal/svm/v2/devmm/devmm_map_dev_reserve.c:184` (`devmm_ctrl_map_mem` for `ADDR_MAP_TYPE_REG_AIC_CTRL`) |
| Master kernel → slave kernel msg | `src/sdk_driver/svm/v2/master/comm/svm_master_addr_map.c:33` (`DEVMM_CHAN_MAP_DEV_RESERVE_H2D_ID`) |
| Slave-side remap_pfn_range | `src/sdk_driver/svm/v2/master/pmaster/svm_master_remote_map.c:1673` (`devmm_remap_addrs_with_palist` → `devmm_make_nocache_pgprot`) |
| pgprot factory | `src/sdk_driver/svm/v2/common/svm_mem_mng.c:34` (`devmm_make_nocache_pgprot` → `ka_mm_pgprot_device`) |
| Linux kernel adapter | `src/sdk_driver/kernel_adapt/include/ka_memory_pub.h:183` (`ka_mm_pgprot_device(prot) → pgprot_device(prot)` on kernel ≥ 3.18) |

On aarch64, `pgprot_device()` resolves to `MT_DEVICE_nGnRE` — the
PTE's `AttrIndx[2:0]` is set to the MAIR_EL1 slot encoded with **n**on-
Gathering, **n**on-Reordering, Early-write-ack semantics. This is the
ground truth — not the looser `Device-nGnRnE` previously claimed in
some comments in this repo (corrected as part of the same change as
this doc).

### Attribute bits — each independently confirmed by measurement

| Bit | Meaning | Behavioural evidence |
| --- | ------- | -------------------- |
| **n** Gather | CPU does not combine adjacent stores | a2a3 experiment: 64-bit STR to DMB hammered with `hi == ~lo` patterns showed a ~0.5% tear rate as the bus split it into 2×32-bit beats. A Normal/cacheable mapping would not tear; gathering would also hide the split. |
| **n** Reorder | LDR/STR cannot be reordered, single LDR is in-flight at a time | 1 AICPU thread doing 10000 LDR of COND — same per-LDR cost (~95–105 ns) whether all LDRs target the same core or rotate across 9 cores. Cross-target switching is free, but no outstanding LDR pipelining. |
| **E** Early-ack | STR is posted; many can be in flight | Burst 1000 STR to DMB completes in ~5 µs ≈ 5 ns/STR ≈ 200M STR/s, far exceeding the single-LDR round trip — implies ~19 STR concurrently in the bus's outstanding queue. nGnRnE would force ~95 ns/STR. |

The driver source + the asymmetry between LDR and STR are
**independent** witnesses. The attribute is nGnRE, not nGnRnE.

## Cost table (single AICPU thread, a2a3, ~50 MHz sys counter)

| Operation | Typical | Notes |
| --------- | ------- | ----- |
| Single STR (posted retire) | < 20 ns | CPU enqueues into write buffer |
| Burst 1000 STR | **~5 ns / STR** | Issue-rate bound; many in flight |
| Single LDR COND | **95–105 ns** | Bus round trip; no caching |
| LDR rotating N targets (single thread) | same ~95 ns / LDR | Switching target free; outstanding still 1 |
| STR + LDR back-to-back round trip | 240–300 ns | LDR drains the in-flight STR queue |
| AICPU→AICore E2E (write DMB, AICore sees) | ~140 ns | Phase 3 measurement |
| AICore→AICPU E2E via COND | **avg ~600 ns / min ~180 ns** | Phase 14 — includes AICore set_cond + AICPU poll cycle |
| AICore→AICPU E2E via GM+dcci | **avg ~1040 ns / min ~980 ns** | Phase 14 — includes dcci + HBM commit + coherency invalidate |

## Concurrency model — single-thread LDR is strictly serial, multi-thread is fully parallel

**Inside one AICPU thread**, the `nR` attribute serialises every LDR
against every other LDR/STR in the same region. The CPU cannot issue
LDR N+1 until LDR N has returned. So one thread polling N cores' COND
takes **N × 95 ns** with no way around it. This is a hardware contract,
not a microarchitectural budget.

**Across AICPU threads**, separate LDRs go through independent paths
to their per-core SPR slots — they do not serialise at the bus. A
3-thread polling experiment on a3 (each thread spinning a different
AIC's COND) measured per-thread cost identical to a single thread
(~95 ns / LDR), giving 3× aggregate throughput.

So the only way to shrink "one polling round across many cores" is to
**add threads**. Don't reach for outstanding-LDR tricks or speculative
prefetch — the attribute forbids them.

The colloquial "polling COND is sequential" claim refers to the
single-thread fact. The multi-thread case is the opposite. Both must
be stated when designing a scheduler that polls completions.

## DATA_MAIN_BASE is hardware-unidirectional

Don't conflate "the SPR is 64-bit" with "AICore can write to it".

| Direction | Path | Status |
| --------- | ---- | ------ |
| AICPU → AICore | MMIO STR at `reg_addr + DMB_OFFSET` | Production dispatch; works. 64-bit STR is split at the bus (2×32, see `n`-Gather above) so don't rely on a 64-bit STR being atomic. |
| AICore reads own DMB | SPR instruction `MOV %0, DATA_MAIN_BASE`, constraint `=l` (uint32 or uint64) | Works |
| AICore reads peer core's DMB | None | The peer's reg_addr is a chip-internal MMIO address. An AICore-side `LDR` to that address **hangs the chip** — the LSU does not route to the SPR MMIO window. CCECPU monitor kills `aicpu-sd` after the 50 s op-timeout. Verified on a3 with 9 cores simultaneously stuck at the first cross-core LDR. |
| AICore writes own DMB via SPR | None | `MOV DATA_MAIN_BASE, x` is **rejected at compile time** by the CCEC backend ("invalid operand for instruction") — the SPR mnemonic only encodes DMB as a source, not a destination. |
| AICore writes own DMB via MMIO | None | Same hang as cross-core LDR. The LSU can't reach the SPR window in either direction. |

If you are designing a protocol that needs an AICore to mutate
DATA_MAIN_BASE — directly or indirectly — change the design. There is
no hardware path. The closest workaround is to write a flag in GM with
`dcci` and have an AICPU thread observe it and forward the value into
DMB via MMIO.

The same MMIO-window restriction applies to peer cores' COND and any
other AIC_CTRL register read from inside an AICore.

## When to pick MMIO COND vs GM+dcci as a notification channel

Both paths can carry "AICore done" → "AICPU notice" signals; they
trade off differently.

| Property | COND (MMIO Device-nGnRE) | GM + dcci (Normal cacheable, coherent) |
| -------- | ------------------------ | -------------------------------------- |
| Per-event E2E latency, single producer / consumer | **600 ns avg / 180 ns min** (Phase 14) | 1040 ns avg / 980 ns min |
| Per-LDR consumer cost when value is unchanged | ~100 ns | ~3 ns (L1 hit) |
| Single-thread polling-round latency across 24 cores | ~24 × 100 = **2400 ns** | ~24 × 41 = **~1000 ns** (Phase 13, rotating cache lines) |
| AICore producer cost per event | set_cond ≈ 5–10 ns | write field + dcci(SINGLE_CACHE_LINE) ≈ 150–300 ns |
| Burst many events from one AICore | Bound by AICore SPR issue rate (~ns) | Bound by dcci rate (~hundreds of ns each) |

Rule of thumb:

- **Latency-critical single event** (a FIN signal you want to catch
  fast): COND.
- **Wide polling sweep** (one thread checking many cores for any
  activity): GM-coherent — the AICPU's cache stays warm and the per-LDR
  cost drops by ~25× as long as no AICore has just written.
- **High AICore-side write rate** (per-task ACK/FIN, profiling
  records): COND — `dcci` is much more expensive per event than
  `set_cond` and would dominate the AICore-side budget.

Production scheduler uses COND for ACK / FIN. The GM-coherent option
remains open for *hint-style* paths where the producer rate is much
lower than the consumer poll rate.

## How to extend or rerun these measurements

The cleanest reproduction path is the pair of standalone tools under
`tools/cann-examples/`:

- [`aicpu-mmio-probes/`](../../tools/cann-examples/aicpu-mmio-probes/)
  — Phase 4 (STR DMB burst + round trip) and Phase 12 (LDR COND
  single-thread serial + multi-thread parallel). No AICore involvement.
- [`aicore-notification-perf/`](../../tools/cann-examples/aicore-notification-perf/)
  — Phase 13 + Phase 14 (GM vs COND notification comparison). Runs an
  AICore producer and AICPU consumer concurrently on two streams.

Both build with CMake against `ASCEND_HOME_PATH` and run end-to-end
against a free NPU via `task-submit`. Add new measurements there.

The full set of original probes (Phase 0–14, including those that
ended in chip hangs) lives on the `experiment/dmb-64bit-probe`
branch — inside the `#ifndef __CPU_SIM` block of
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp`
and the matching producer code in
`src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`.
LOG_WARN lines like `[P12-A] ... per=5.29 ticks (~105 ns/LDR)` land in
the AICPU device log at `~/ascend/log/debug/device-<id>/`. Each phase
is gated by a `hammer_go` value so that the AICore-side handshake is
explicit; the AICPU-side `scheduler_cold_path::handshake_all_cores` is
the host of the test. Phase 10 and Phase 11 are kept under `#if 0`
because they hang the chip on entry — see
`docs/investigations/2026-06-aicore-mmio-to-spr.md` for the verdict.

## Related docs

- [cache-coherency.md](cache-coherency.md) — when AICPU must
  invalidate before reading; complements the MMIO ordering rules here.
- [chip-architecture.md](chip-architecture.md) — the tier model these
  registers live on.
- [`docs/investigations/2026-06-aicore-mmio-to-spr.md`](../investigations/2026-06-aicore-mmio-to-spr.md)
  — why we don't try to make AICore write DMB.
- [`docs/investigations/2026-06-cond-vs-gm-notification.md`](../investigations/2026-06-cond-vs-gm-notification.md)
  — full data + trade-off table for COND vs GM notification paths.
