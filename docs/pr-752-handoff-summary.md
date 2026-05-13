# PR 752 Handoff Summary

This document only records the continuation work done on top of PR 752 in
the `pr-752-continuation` branch.

## Completed In This Handoff

- Revalidated the normal multi-domain path on hardware.  With CANN 8.5,
  `workers/l3/domain_rank_map` passes on devices `12,13,14`.
- Compared the failing `sdma_async_completion_demo` with `origin/main`.
  The original demo also fails in the tested CANN 8.5 and CANN 9.0 beta.2
  setups, so the current SDMA failure is not yet proven to be caused by the
  PR 752 multi-domain changes.
- Cleaned up the temporary `origin/main` worktree after comparison testing.

## Code Changes In This Handoff

- `.gitignore` now ignores `.ccache/`, because local compiler cache contents
  were showing up in git status.
- `src/a2a3/platform/onboard/host/comm_hccl.cpp` no longer runs an extra
  `HcclBarrier` immediately before `HcclAllocComResourceByTiling`.  That
  barrier caused a 507018 failure on the current branch; the existing file
  barrier remains the rendezvous point before HCCL resource allocation.
- `src/a2a3/platform/onboard/host/CMakeLists.txt` now links against the
  resolved `hcomm` library path from the active CANN install.  This avoids
  accidentally mixing CANN 8.5 and CANN 9.0 libraries through the default
  linker search path.
- `sdma_async_completion_demo` now checks the derived domain `CommContext`
  after `worker.init()`.  If `workSpace` or `workSpaceSize` is missing, the
  test fails before launch with a direct message to rebuild with
  `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1` and `PTO_ISA_ROOT`.
- `docs/multi-comm-domain-implementation.md` records the concrete validation
  results for CANN 8.5, CANN 9.0 beta.2, and the `origin/main` comparison.

## Problems Found

- CANN 8.5 with `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1` cannot initialize the
  SDMA workspace because the installed `libopapi.so` does not export
  `aclnnShmemSdmaStarsQuery`.
- CANN 8.5 with the workspace disabled can bootstrap the original
  `sdma_async_completion_demo`, but it still fails in `worker.run()` with
  AICPU 507015 and then 507901 during stream teardown.
- CANN 9.0 beta.2 removes the missing-symbol problem, but
  `HcclAllocComResourceByTiling` still fails.  On the current branch the C++
  log reports return code 15; on `origin/main` the original demo reports
  `comm_alloc_windows failed with code -1`.
- The CANN 9.0 beta.2 HCCL allocation failure reproduced on tested card
  pairs `10,11`, `12,13`, and `14,15`.  It also affects non-SDMA
  `domain_rank_map`, so it should be diagnosed as a CANN 9.0 HCCL resource
  allocation issue before treating the SDMA demo as a multi-domain bug.

## Verification Run

- `python -m py_compile` passed for
  `sdma_async_completion_demo/test_sdma_async_completion_demo.py`.
- Focused unit and sim tests passed: 30 tests.
- `git diff --check` passed.
- Hardware `domain_rank_map` passed with CANN 8.5 on devices `12,13,14`.
- SDMA demo remains blocked by the CANN/HCCL issues above.
