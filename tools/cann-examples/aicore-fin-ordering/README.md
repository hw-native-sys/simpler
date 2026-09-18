# aicore-fin-ordering

Tests whether a consumer that has observed an AICore task's **FIN
notification** can then read data that task published — the ordering
question behind
[#2233](https://github.com/hw-native-sys/simpler/issues/2233).

Three standalone programs, no dependency on this repo's runtime
(scheduler, ringbuffer, task dispatch):

```text
host launch.cpp                 AICPU consumer.cpp                AICore producer.cce
------------------------------  --------------------------------  --------------------------
allocate GM, resolve            request round r                   wait for ctl.round_request == r
  aic_ctrl_reg_base                                               payload = r          (one store)
register + launch both                                            dcci(payload)
                                wait for COND == FIN|r             [P1] dsb(DSB_DDR)
                                apply consumer variant             set_cond(FIN|r)
                                FIRST payload load -> classify
                                diagnostics
                                ACK (round_request = r+1)          hold payload frozen until ACK
D2H result + records, print
```

## Why this is not the `aicore-notification-perf` tool

That tool measures the **latency** of two notification paths, so its
producer must run free. This one tests **ordering correctness**, which
requires the opposite: the payload must be frozen across the consumer's
whole read sequence. A throttle delay is not a synchronisation guarantee —
with a free-running producer a mismatch cannot be told apart from "the
producer moved on", which is precisely the flaw that made an earlier
measurement of this question uninterpretable. Its README names copying the
directory as the way to add a new mechanism; this is that copy.

## The frozen handoff is what makes a result interpretable

The producer writes `payload` exactly once per round and then must not
touch it until the consumer bumps `round_request`, which the consumer only
does after finishing every read and diagnostic. So a first-load value
*below* the round the FIN carried cannot be explained by the producer
having advanced.

Consequently the first load is classified three ways, not two:

| outcome | meaning |
| ------- | ------- |
| `equal` | the published value was visible |
| `older` | **the only failure** — published data not visible |
| `newer` | must not occur under the frozen protocol; reported as a protocol / foreign-writer anomaly, never folded into a failure count |

Two further properties the protocol depends on:

- The wait **returns the MMIO word that ended it**, and the round is
  derived from that value. Re-reading COND after the wait would let the
  producer advance in between, so "expected" would name a different round.
- No 64 B maintenance block is written by both sides. `ctl` is
  AICPU-written / AICore-read; `rpt` and `pay` are AICore-written /
  AICPU-read. A block is only ever cleaned by its writer and only ever
  invalidated by its reader.

## The matrix

Producer side, differing **only** in the barrier:

| arm | publish sequence |
| --- | ---------------- |
| P0 | `store` → `dcci` → `set_cond` |
| P1 | `store` → `dcci` → `dsb(DSB_DDR)` → `set_cond` |

Consumer side, applied after capturing the round's FIN and before the first
payload load:

| arm | operation |
| --- | --------- |
| C0 | bare load |
| C1 | `dsb sy`, load |
| C2 | invalidate the payload line (`dc civac; dsb sy; isb`), load |
| C3 | `dsb sy`, invalidate, load |
| C4 | delay matched to the measured invalidate cost, load |
| C5 | `dmb ld`, load |
| C6 | acquire the flag itself (`ldar` on COND) inside the wait, bare load |
| C7 | delay matched to the measured `dsb sy` cost, load |

Crossed with a cache-state axis: `pre = yes` has the consumer deliberately
read `payload` before the publish, so the line is provably resident with
pre-publish content going into the round.

C4 and C7 are the controls that separate "the operation ordered something"
from "the operation merely took time". C2 already ends in `dsb sy; isb`, so
**C3 differs from C2 only by a leading `dsb`** — report them as near-
equivalent rather than as two independent treatments.

## Build

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

cmake -B device-aicore/build -S device-aicore -DCCE_AICORE_ARCH=dav-c220-cube
cmake --build device-aicore/build

cmake -B device-aicpu/build -S device-aicpu \
    -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++
cmake --build device-aicpu/build

cmake -B host/build -S host
cmake --build host/build
```

## Run

```bash
export SIMPLER_DISPATCHER_SO=<repo>/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
export FIN_ORDER_CONSUMER_SO=$(pwd)/device-aicpu/build/libfin_order_consumer.so
export FIN_ORDER_PRODUCER_O=$(pwd)/device-aicore/build/fin_order_producer.o
export FIN_ORDER_RECORDS=/path/to/records.csv        # optional per-trial dump

# Always hold a device lock on a shared box (.claude/rules/running-onboard.md).
# argv: <device> <stage> <rounds per cell>
task-submit --device auto --device-num 1 \
    --run "./host/build/launch_fin_order \$TASK_DEVICE 0 50"
```

`stage` is a bisect gate for bring-up failures, not a measurement mode:
`1` returns after touching the result block, `2` after the producer reports
its core id, `3` after cost calibration, `0` runs the matrix.

## Two build/registration constraints worth knowing

Both cost a debugging round-trip to rediscover:

- **`rtRegisterAllKernel` needs a linked device ELF carrying both halves of
  the mix-kernel symbol pair** (`*_0_mix_aic` and `*_0_mix_aiv`). A raw
  `ccec -c` object, or one holding only the cube half, is rejected with
  `107000`. Hence the two-arch compile plus the
  `ld.lld -m aicorelinux -Ttext=0 -static -n` link, mirroring
  `src/a2a3/platform/onboard/aicore/CMakeLists.txt`. The vector half must
  not include `kernel_operator.h` — AscendC's non-weak globals
  (`g_tilingKey` among them) would then be defined in both halves and the
  combined link fails on duplicate symbols.
- **`get_coreid()` is not a bare core index.** It must be masked with
  `0x0FFF` before indexing the AIC_CTRL window, the way
  `get_physical_core_id()` does. Using the raw value (observed `0x8010`)
  produces an address outside the window, and reading it faults the AICPU op
  — surfacing as `507018 / aicpu exception`, not as a failed match.

## Reading the output

`older` is the only column that counts as a failure. `to/o` is a wait
timeout, which means the producer never published that round — a protocol
or bring-up problem, not a visibility result. `re-read` and `post-inv` are
named for what recovered the value (ordinary re-reads with no cache
maintenance, versus an invalidate) and deliberately not for a presumed
hardware cause. `unres` never matched within the bound and is reported
without attribution.

`round-echo-mismatch` in the anomaly line is expected: the producer
publishes `rpt.published_round` *after* the FIN so that write can never be
what made the payload visible, so the consumer can reach its cross-check
before that value lands. It is a property of this protocol, not a finding.
