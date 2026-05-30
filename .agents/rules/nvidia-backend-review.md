# NVIDIA Backend Review Rules

- Keep docs synchronized with code in the same change.
- Every described CUDA backend feature must point to implementation evidence:
  a file path plus the symbol, class, function, or constant that implements it.
- Keep `docs/nvidia-backend/evaluation.md` and
  `docs/nvidia-backend/evaluation-current.md` short enough for review.
- Put old capture narratives under `docs/nvidia-backend/history/captures/`.
- Put review-facing change summaries under `docs/nvidia-backend/changelog/`.
- Keep benchmark viewer data in JSON under
  `docs/nvidia-backend/benchmark-viewer/data/`.
- Do not commit raw benchmark dumps from `tmp/cuda-backend/`; commit distilled
  review data and preserve the raw artifact path.
- Do not describe planned runtime behavior as implemented unless the evidence
  map points at current code.
- When code behavior changes, update the viewer data and changelog in the same
  slice.
