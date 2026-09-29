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
 * (``DepGenCollector::window_records()``) and runs them back through a host-resident
 * ChipTensorMap using the same ``compute_task_fanin`` / ``register_task_outputs``
 * primitives the device orchestrator uses, emitting the full
 * predecessor → successor edge list to deps.json.
 *
 * The records buffer is passed in directly — there is no intermediate
 * ``submit_trace.bin`` on disk. The host already has the records once the
 * device run completes, so going through the filesystem would just be
 * extra I/O and an extra file in the output directory.
 *
 * deps.json is the sole source of truth for fanout: the chip swimlane hot
 * path no longer records ``ChipSwimlaneAicpuTaskRecord::fanout[]`` (taking the per-task
 * 1 KB GM store off the scheduler critical path). Replay sees every
 * submit and reconstructs the complete dependency graph.
 *
 * Output format (deps.json, strided tensor representation):
 *
 *   {"tasks":   [{"task_id":<u64>, "scope":"auto|manual", "early_dispatch":<bool>,
 *                 "args":[{"idx":<i32>, "type":"<arg_type>",
 *                          "tensor_id":<u64>, "dtype":"...", "shape":[...],
 *                          "start_offset":<u64>, "strides":[...]}, ...]}, ...],
 *    "tensors": [{"tensor_id":<u64>, "buffer_addr":<u64>, "version":<i32>,
 *                 "dtype":"FLOAT32", "buffer_numel":<u64>}, ...],
 *    "edges":   [{"pred":<u64>, "succ":<u64>, "arg":<i32>,
 *                 "source":"explicit|creator|tensormap",
 *                 "overlap":"covered|other" (tensormap only),
 *                 "tensor_id":<u64> (non-explicit),
 *                 "consumer_dtype":"...", "consumer_shape":[...],
 *                 "consumer_start_offset":<u64>, "consumer_strides":[...],
 *                 "producer_shape":[...] (tensormap),
 *                 "producer_start_offset":<u64> (tensormap),
 *                 "producer_strides":[...] (tensormap)},
 *                ...]}
 *
 *   - All task ids are encoded words (``TaskId::to_uint64``), i.e.
 *     ``(ring_id << 32) | local_id``.
 *   - ``tensor_id`` is a stable FNV-1a hash of ``(buffer_addr, version)``.
 *   - ``buffer_numel`` is the underlying storage element count; tensor shapes
 *     are carried per-arg / per-edge alongside ``start_offset`` + ``strides``.
 *   - Distinct producers / arg indices / sources keep their own edges; per-record
 *     deduplication of producer ids mirrors the runtime
 *     ``FaninBuilder::append_fanin_or_fail`` semantics so the set of
 *     ``(pred, succ)`` pairs is identical to what the runtime would have
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

#pragma once

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
 * A budget every allocation this replay makes is charged against.
 *
 * `charge` is called before an allocation and refuses by returning false, at
 * which point the replay fails rather than allocating; `credit` is called after
 * the matching deallocation. Both the container growth transient (the new
 * block is charged while the old one still is) and the tensormap arena go
 * through this, so nothing the replay allocates escapes the caller's bound.
 *
 * A null budget means unbounded, which is what the default synchronous path
 * passes: it has no retained budget, so only a real allocation failure stops
 * it, exactly as before.
 */
struct DepGenReplayBudget {
    void *ctx;
    bool (*charge)(void *ctx, size_t bytes);
    void (*credit)(void *ctx, size_t bytes);
};

/**
 * Replay an in-memory DepGenRecord stream and write deps.json.
 *
 * Per-ring task window sizes are auto-derived from the trace itself so each
 * ring's window covers its observed max local_id without slot aliasing. Every
 * record's layout is validated against its own kind first — a base record by
 * its counts, an overflow slot by its `dep_count` — because the counts are
 * device-written and a corrupted one must refuse the graph rather than size an
 * allocation or index an array.
 *
 * @param records            Pointer to a contiguous DepGenRecord array
 *                           (typically ``DepGenCollector::window_records()->data()``).
 * @param num_records        Number of records in the array.
 * @param deps_json_path     Output path; truncated if it exists.
 * @param budget             Charged against for every allocation, or null for
 *                           unbounded.
 * @return 0 on success, or a negative code: -1 bad arguments, -3 the replay's
 *         working storage could not be reserved, -4 the runtime's own fanin
 *         computation reported fatal, -5 a record layout, overflow-chain
 *         structure or size computation was rejected, -6 the dual-pass
 *         self-check diverged, -7 an invalid explicit dep-flag byte, -8 a
 *         charge refusal or an unexpected host failure, -9 the file could not
 *         be written. Every non-zero code means no graph was published — but
 *         not that the path is untouched: this writes `deps_json_path`
 *         directly, so a -9 raised mid-write leaves a truncated file there.
 *         A caller for whom a partial file is indistinguishable from a
 *         complete one must publish through a temporary of its own, which is
 *         what the retained background writer does.
 */
int dep_gen_replay_emit_deps_json_budgeted(
    const struct DepGenRecord *records, size_t num_records, const char *deps_json_path,
    const struct DepGenReplayBudget *budget
);

/** `dep_gen_replay_emit_deps_json_budgeted` with no budget. */
int dep_gen_replay_emit_deps_json(const struct DepGenRecord *records, size_t num_records, const char *deps_json_path);

#ifdef __cplusplus
}  // extern "C"
#endif
