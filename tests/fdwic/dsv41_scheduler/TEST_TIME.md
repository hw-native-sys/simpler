# DSV4.1 scheduler test execution time

Measured on A5 on 2026-11-15, based on the requested HBG revision `8db8a15fc`, with the test-runner changes published as `6d050abbd`.
These are whole-test seconds, including process startup, input preparation, device setup, warmup, measurement, validation where enabled, and teardown. They are not AICore kernel latency.

The final 16-case run completed in **68.55 seconds** with up to eight automatically allocated devices, from first submission to all results returned. The native weight cache was warm.
The sum of individual job durations fell from **382.53 to 226.05 seconds (40.9% less)**. This sum is not parallel wall time. Each job excludes queue time; the matrix wall time includes queueing and submission/collection. No comparable complete baseline matrix wall time was measured.

| Case | Implementation | Before (s) | After, warm cache (s) |
| ---- | -------------- | ---------: | --------------------: |
| swa | Native PyPTO-lib | 16.55 | 10.47 |
| c2a_full | Initial PyPTO-lib | 15.54 | 11.35 |
| c2a_full | DSL half-sync | 12.93 | 14.68 |
| c2a_full | DSL all-sync <=64 | 18.94 | 13.05 |
| c2a_full | Native PyPTO-lib | 14.34 | 12.34 |
| c1a_reindex | Native PyPTO-lib | 17.09 | 20.32 |
| c1a_reuse | Native PyPTO-lib | 22.31 | 17.20 |
| moe | Initial PyPTO-lib | 8.28 | 6.28 |
| moe | DSL half-sync | 8.28 | 6.41 |
| moe | DSL all-sync <=64 | 11.70 | 16.16 |
| moe | Native PyPTO-lib | 139.92 | 11.03 |
| engram | Native PyPTO-lib | 17.08 | 13.73 |
| layers_19_21 | Initial PyPTO-lib | 19.20 | 19.41 |
| layers_19_21 | DSL half-sync | 21.28 | 17.53 |
| layers_19_21 | DSL all-sync <=64 | 22.88 | 19.13 |
| layers_19_21 | Native PyPTO-lib | 16.20 | 16.99 |

## Measurement and changes

- Both runs use the same compiled HBG runtime and business binaries, one warmup and ten measured rounds. No target workload was shortened. All two-device launches retain the device-side start rendezvous; L0 AICore endpoints and rank aggregation are checked.
- Replay cases retain every reference/output comparison. Bulk file reads and bulk one-byte initializer fills reduce host input and validation overhead without changing bytes.
- Native MoE is a performance test for `72/72` tokens. It retains live Host control reads and original nonzero weights/routing, but skips `0/6`, `6/0`, numerical golden calculation and output comparison. Results explicitly mark `numerical_validation: false`.
- Fixed native weights and scales are cached by model revision, Torch version and layout. The cache is 2,479,476,799 bytes (2.48 GB; 2.31 GiB), for both ranks combined. The measured runner verified the file digest and tensor layouts; first creation verified serialized bytes against generated tensors. The integrated runner checks identity, file size and tensor layouts without file-digest verification. These timings remain measurements of the original runner, not a rerun of that integration change.
- First cache creation took 40.87 seconds for input preparation. A cache-hit run took 1.50 seconds for that stage and 10.57 seconds overall; the final concurrent matrix measured 11.03 seconds for native MoE. The final performance-only cold-start total was not separately measured.
- The before/after table is one observed run per case, with different automatic allocations. Driver startup and process-exit delays vary by several seconds. Separate matched-pair replay runs reduced cumulative input setup/reset/validation from 83.23 to 45.89 seconds while retaining all comparisons; this isolates the host-side improvement more clearly than individual total durations.

## Reproduction and evidence

Use the usual `production/run.sh` command with level `0`, warmup `1`, rounds `10`, and already prepared bundles. The native cache is created and reused automatically in `build/native-weights/`. Reserve devices with `task-submit --device auto --device-num N`; use the returned `TASK_DEVICE`. At most eight devices were occupied across concurrent jobs.

Local evidence is under `artifacts/turnaround/` in the dedicated test-turnaround worktree: `comparison.json` records actual before/after devices, `runtime-audit.json` records the verified reused runtime, `final-state.json` records matrix time and completed jobs, and per-case `wall.json`, timing files and logs retain the measurements. Prepared business bundles were reused from the existing local benchmark artifacts.

Validation: all 16 final cases completed with ten valid L0 samples; replay numerical comparisons passed, native MoE timing-only execution passed. Current CPU tests cover cache byte fidelity, fresh dynamic inputs, model-revision separation, truncated-cache rejection and timing/report validation.
