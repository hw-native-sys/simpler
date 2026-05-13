/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file dep_gen_replay.h
 * @brief Host-side replay of in-memory DepGenRecord stream → deps.json.
 *
 * Takes the records the host collector drained from the device ring buffer
 * (``DepGenCollector::records()``) and runs them back through a host-resident
 * PTO2TensorMap using the same ``compute_task_fanin`` / ``register_task_outputs``
 * primitives the device orchestrator uses, emitting the full
 * predecessor → successor edge list to deps.json.
 *
 * The records buffer is passed in directly — there is no intermediate
 * ``submit_trace.bin`` on disk. The host already has the records once the
 * device run completes, so going through the filesystem would just be
 * extra I/O and an extra file in the output directory.
 *
 * deps.json supersedes ``L2PerfRecord::fanout[]`` for tools that need the
 * *complete* dependency graph: fanout is sealed when a producer finishes, so
 * consumers submitted after a fast producer retires never get attributed to
 * it (the race window that motivated dep_gen). Replay sees every submit and
 * so reconstructs the graph the runtime would have built if no producer ever
 * raced ahead.
 *
 * Output format (deps.json, v2):
 *
 *   {"version":2,
 *    "tasks":   [{"task_id":<u64>, "scope":"auto|manual"}, ...],
 *    "tensors": [{"tensor_id":<u64>, "buffer_addr":<u64>, "version":<i32>,
 *                 "dtype":"FLOAT32", "ndims":<u32>, "raw_shapes":[...]}, ...],
 *    "edges":   [{"pred":<u64>, "succ":<u64>, "arg":<i32>,
 *                 "source":"explicit|creator|tensormap",
 *                 "overlap":"covered|other" (tensormap only),
 *                 "tensor_id":<u64> (non-explicit),
 *                 "consumer_dtype":"...", "consumer_shape":[...], "consumer_offset":[...],
 *                 "producer_shape":[...] (tensormap), "producer_offset":[...] (tensormap)},
 *                ...]}
 *
 *   - All task ids are ``PTO2TaskId::raw`` values (``(ring_id << 32) | local_id``).
 *   - ``tensor_id`` is a stable FNV-1a hash of ``(buffer_addr, version)``.
 *   - Distinct producers / arg indices / sources keep their own edges; per-record
 *     deduplication of producer ids mirrors the runtime
 *     ``PTO2FaninBuilder::append_fanin_or_fail`` semantics so the set of
 *     ``(pred, succ)`` pairs in v2 is identical to what the runtime would have
 *     recorded.
 *
 * Self-checking: the replay runs two parallel tensormap instances per record —
 * an "oracle" map driven by the canonical ``compute_task_fanin`` template, and
 * an "annotated" map driven by an inlined mirror that captures the per-edge
 * tensor metadata. If the producer-id set on the two passes ever diverges,
 * deps.json is NOT written and the function returns a non-zero error code.
 * This is the guarantee against silent shotgun modifications: anyone who
 * changes ``compute_task_fanin`` semantics has to mirror the change here too
 * or the gate fires immediately.
 *
 * The replay is single-threaded and pure CPU: no device handle is required.
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_HOST_DEP_GEN_REPLAY_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_HOST_DEP_GEN_REPLAY_H_

#include <stddef.h>
#include <stdint.h>

// Opaque forward decl — the canonical layout lives in common/dep_gen.h, but
// replay's API only needs to take a pointer + count. Callers who construct
// the buffer must include common/dep_gen.h themselves.
struct DepGenRecord;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Replay an in-memory DepGenRecord stream and write deps.json.
 *
 * Per-ring task window sizes are auto-derived from the trace itself so each
 * ring's window covers its observed max local_id without slot aliasing.
 *
 * @param records            Pointer to a contiguous DepGenRecord array
 *                           (typically ``DepGenCollector::records().data()``).
 * @param num_records        Number of records in the array.
 * @param deps_json_path     Output path; truncated if it exists.
 * @return 0 on success; negative on error (see source for codes).
 */
int dep_gen_replay_emit_deps_json(const struct DepGenRecord *records, size_t num_records, const char *deps_json_path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_HOST_DEP_GEN_REPLAY_H_
