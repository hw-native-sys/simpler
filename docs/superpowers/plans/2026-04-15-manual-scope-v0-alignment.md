# Manual Scope V0 Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the `manual_scope_v0` implementation with the current design doc by decoupling submit-result task ids from output tensors and removing TensorMap lookup/insert for manual-local tensors.

**Architecture:** Keep tensor-side producer provenance unchanged in `Tensor`, add a standalone task-id field to `TaskOutputTensors`, and narrow the submit hot path so manual-local tensors use only explicit deps instead of TensorMap. Validate with focused C++ unit tests before broader reruns.

**Tech Stack:** C++ runtime code, gtest/ctest, Markdown docs

---

### Task 1: Lock The Required Behaviors With Focused Tests

**Files:**
- Modify: `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp`

- [ ] **Step 1: Add a failing zero-output task-id test**
- [ ] **Step 2: Add a failing manual-local TensorMap-insert bypass test**
- [ ] **Step 3: Run the focused runtime unit test binary and verify the new tests fail for the current implementation**

### Task 2: Implement Submit-Result Task IDs

**Files:**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- Modify: `tests/ut/cpp/test_a2a3_pto2_manual_scope_api.cpp`

- [ ] **Step 1: Add standalone task-id storage to `TaskOutputTensors`**
- [ ] **Step 2: Set that task id on both submit and alloc paths**
- [ ] **Step 3: Update API/unit tests to assert direct task-id behavior independent of output materialization**

### Task 3: Remove Manual-Local TensorMap Lookup/Insert

**Files:**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- Modify: `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp`

- [ ] **Step 1: Skip TensorMap lookup for all manual-local tensors in manual scope**
- [ ] **Step 2: Skip TensorMap insert for all manual-local tensors in manual scope**
- [ ] **Step 3: Keep creator retention and boundary-tensor behavior unchanged**

### Task 4: Verify And Refresh The Design Note

**Files:**
- Modify: `docs/manual-scope-v0-design.md`

- [ ] **Step 1: Run focused C++ tests until green**
- [ ] **Step 2: Update the design note’s implementation-gap section if behavior has changed**
- [ ] **Step 3: Prepare for follow-up real-device reruns after unit-level alignment is complete**
