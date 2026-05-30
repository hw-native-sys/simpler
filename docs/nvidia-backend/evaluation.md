# CUDA Backend Evaluation

This is the review entry point for CUDA backend evaluation evidence. The
numbers are microbenchmarks for runtime launch and device scheduling behavior,
not end-to-end model serving results.

## Current Status

- Latest full paired capture: commit `743709f3`, `1350` A100/H200 samples.
- Latest compact review gate: commit `743709f3`, `108` A100/H200 samples.
- Current GPU targets: A100 with `compute_80`, H200 with `compute_90`.
- Current tensor descriptor shape: `16x16x16`.
- Current vector sizes: `1024`, `65536`, `1048576`.
- Current source-paper setup is preserved under `tmp/sources/`.

Use the static viewer for the human-reviewable benchmark matrix:

- [Benchmark viewer](benchmark-viewer/index.html)
- [Current capture summary](evaluation-current.md)
- [Historical capture archive](history/index.md)
- [Changelog reports](changelog/index.md)

## What To Review

The current review should focus on three questions:

1. Does `cuda/host_schedule` expose the basic CUDA host runtime path clearly?
2. Does `cuda/persistent_device` make the missing-AICPU design constraint
   explicit through a device-side persistent scheduler?
3. Do benchmark claims point to implementation, validation, and artifact
   evidence instead of only prose?

The implementation evidence map is intentionally checked by
`.agents/checks/check_nvidia_review_ready.py`.

## Raw Artifacts

Raw benchmark artifacts are generated under `tmp/cuda-backend/` and are not
committed. The latest reviewed captures are:

- `tmp/cuda-backend/current-head-full-layered-cross-fixed/`
- `tmp/cuda-backend/layered-cross-selected-current-fixed/`

Source PDFs and extracted text used for the CUDA design and evaluation notes
remain under:

- `tmp/sources/arxiv-2605.03190-vdcores.pdf`
- `tmp/sources/arxiv-2605.03190-vdcores.txt`
- `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.pdf`
- `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt`
