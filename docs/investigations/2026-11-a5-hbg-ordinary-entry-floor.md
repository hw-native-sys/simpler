# A5 HBG ordinary entry with a MIX completion proof

Investigated on 2026-11-21. The candidate was dropped after its seven-case AICore L0 time increased by 0.528%.

## What was tested

The baseline was `aaad77245`. The candidate extended the ordinary Host graph planner to propagate per-lane prefixes of completed MIX tasks along data edges. It added a lane mask to each catalog order row, increasing the row from 24 to 28 bytes. At ordinary execution entry, a lane with a complete proof and an idle local MIX bank could skip the MIX floor check after observing its fanin. Claims retained the original floor. Conditional or asynchronous paths without a completion proof kept the entry check.

The seven native `pypto_lib_latest` cases ran with HBG AICore resident scheduling on A5. Each case used one device allocation for baseline, candidate, candidate, baseline. Each segment ran one warmup and five timed launches. Dual-rank launches used a Device start rendezvous and the larger of the two AICore wall times.

## Measurements

| AICore L0 sum, µs | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Five single-device cases | 887.837 | 890.795 | +0.333% |
| Two communication cases | 2451.344 | 2466.024 | +0.599% |
| All seven | 3339.181 | 3356.819 | +0.528% |

MoE increased by 1.904 µs and layers 19–21 by 12.776 µs. A separate L2 capture on devices 3 and 6 confirmed the Host generated an entry mask of 6 for MoE t28 on both ranks and retained a claim floor of 27 on both AIV lanes. READY to pickup shortened from 0.829 to 0.385 µs on rank 0 and from 0.673 to 0.394 µs on rank 1. The full t27-to-t28 interval moved from 2.669 to 2.341 µs and from 2.637 to 2.927 µs respectively.

The retention criterion was the seven-case L0 sum. A future version would need to demonstrate an aggregate gain under that same comparison.
