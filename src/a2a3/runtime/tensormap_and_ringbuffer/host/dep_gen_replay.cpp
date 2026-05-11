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
 * @file dep_gen_replay.cpp
 * @brief Replay in-memory DepGenRecord stream → deps.json via a host-resident
 *        PTO2TensorMap.
 *
 * Input is the host collector's in-memory record buffer (no disk I/O — the
 * collector drains the device ring directly into the buffer this function
 * reads). Per-record processing mirrors pto_orchestrator::submit_task
 * exactly so the edges replay emits are bit-equivalent to the ones the
 * device would have recorded if no producer had retired before its
 * consumers were submitted:
 *
 *   STEP 1 — explicit_deps: emitted at the call site (per pto_dep_compute.h's
 *            "kept at call site" note; both runtime and replay do this loop
 *            themselves because STEP 3's reuse / dedup is subtly different).
 *   STEP 3 — compute_task_fanin: creator retention (Tensor::owner_task_id) +
 *            tensormap.lookup for INPUT / INOUT, with INOUT+COVERED removal.
 *   STEP 4 — register_task_outputs: insert INOUT and OUTPUT_EXISTING outputs
 *            into the tensormap so subsequent records can find them.
 *
 * Pool sizing: replay never advances last_task_alive, so the entry pool must
 * accommodate every output write across the whole trace. We scan the record
 * buffer once to count INOUT + OUTPUT_EXISTING slots and size the pool
 * accordingly.
 */

#include "dep_gen_replay.h"

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "common/dep_gen.h"
#include "common/unified_log.h"
#include "pto_dep_compute.h"
#include "pto_task_id.h"
#include "pto_tensormap.h"
#include "tensor.h"
#include "tensor_arg.h"

namespace {

int32_t next_pow2(int32_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// Count INOUT + OUTPUT_EXISTING slots across the record buffer —
// register_task_outputs only inserts those, and skips entries with manual_dep
// set. Counting both without inspecting manual_dep is a conservative upper
// bound (manual_dep is rare; the small over-allocation pays for itself in
// avoided pool exhaustion).
int32_t count_outputs(const DepGenRecord *records, size_t n) {
    int32_t total = 0;
    for (size_t i = 0; i < n; i++) {
        const DepGenRecord &r = records[i];
        for (uint16_t j = 0; j < r.tensor_count; j++) {
            auto t = static_cast<TensorArgType>(r.arg_types[j]);
            if (t == TensorArgType::INOUT || t == TensorArgType::OUTPUT_EXISTING) {
                total++;
            }
        }
    }
    return total;
}

bool write_deps_json(const char *path, const std::vector<std::pair<uint64_t, uint64_t>> &edges) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        LOG_ERROR("dep_gen replay: failed to open '%s' for write", path);
        return false;
    }
    out << "{\"version\":1,\"edges\":[";
    for (size_t i = 0; i < edges.size(); i++) {
        if (i > 0) {
            out << ",";
        }
        out << "[" << edges[i].first << "," << edges[i].second << "]";
    }
    out << "]}\n";
    return static_cast<bool>(out);
}

}  // namespace

extern "C" int dep_gen_replay_emit_deps_json(
    const DepGenRecord *records, size_t num_records, const char *deps_json_path, const int32_t *task_window_sizes_in
) {
    if (deps_json_path == nullptr) {
        LOG_ERROR("dep_gen replay: null deps_json_path");
        return -1;
    }
    if (num_records > 0 && records == nullptr) {
        LOG_ERROR("dep_gen replay: num_records=%zu but records pointer is null", num_records);
        return -1;
    }
    LOG_INFO_V0("dep_gen replay: processing %zu in-memory records", num_records);

    // Per-ring task window sizes — tensormap masks slot indices and requires
    // each to be a power of two. When the caller passes null we auto-size from
    // the records themselves so each ring's window comfortably covers its
    // observed max local_id (no slot aliasing during INOUT+COVERED
    // remove_from_task).
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    if (task_window_sizes_in != nullptr) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            int32_t v = task_window_sizes_in[r];
            if (v < 1) v = 1;
            if ((v & (v - 1)) != 0) v = next_pow2(v);
            task_window_sizes[r] = v;
        }
    } else {
        uint32_t max_local[PTO2_MAX_RING_DEPTH] = {0};
        for (size_t i = 0; i < num_records; i++) {
            PTO2TaskId tid{records[i].task_id};
            uint8_t ring = tid.ring();
            uint32_t local = tid.local();
            if (ring < PTO2_MAX_RING_DEPTH && local > max_local[ring]) {
                max_local[ring] = local;
            }
        }
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            // +1 for slot needed by max_local; floor at 16 to avoid tiny allocs.
            int32_t need = static_cast<int32_t>(max_local[r] + 1);
            int32_t v = next_pow2(need < 16 ? 16 : need);
            task_window_sizes[r] = v;
        }
    }

    int32_t output_count = count_outputs(records, num_records);
    int32_t pool_size = output_count + (output_count / 10) + 64;
    if (pool_size < PTO2_TENSORMAP_POOL_SIZE) {
        pool_size = PTO2_TENSORMAP_POOL_SIZE;
    }
    int32_t num_buckets = next_pow2(pool_size / 16);
    if (num_buckets < PTO2_TENSORMAP_NUM_BUCKETS) {
        num_buckets = PTO2_TENSORMAP_NUM_BUCKETS;
    }

    PTO2TensorMap tensor_map;
    std::memset(&tensor_map, 0, sizeof(tensor_map));
    if (!tensor_map.init(num_buckets, pool_size, task_window_sizes)) {
        LOG_ERROR("dep_gen replay: tensormap.init failed (buckets=%d, pool=%d)", num_buckets, pool_size);
        return -3;
    }

    std::vector<std::pair<uint64_t, uint64_t>> edges;
    edges.reserve(num_records * 2);

    TensorRef tref_buf[CORE_MAX_TENSOR_ARGS];
    TensorArgType atype_buf[CORE_MAX_TENSOR_ARGS];

    for (size_t rec_i = 0; rec_i < num_records; rec_i++) {
        const DepGenRecord &rec = records[rec_i];
        PTO2TaskId task_id{rec.task_id};
        bool in_manual_scope = (rec.flags & DEP_GEN_FLAG_IN_MANUAL_SCOPE) != 0;

        int32_t tc = static_cast<int32_t>(rec.tensor_count);
        if (tc > CORE_MAX_TENSOR_ARGS) {
            tc = CORE_MAX_TENSOR_ARGS;
        }
        for (int32_t i = 0; i < tc; i++) {
            tref_buf[i].ptr = reinterpret_cast<const Tensor *>(&rec.tensors[i][0]);
            atype_buf[i] = static_cast<TensorArgType>(rec.arg_types[i]);
        }

        int32_t dc = static_cast<int32_t>(rec.explicit_dep_count);
        if (dc > DEP_GEN_MAX_EXPLICIT_DEPS) {
            dc = DEP_GEN_MAX_EXPLICIT_DEPS;
        }

        DepInputs inputs;
        inputs.tensor_count = tc;
        inputs.tensors = tref_buf;
        inputs.arg_types = atype_buf;
        inputs.explicit_dep_count = dc;
        // PTO2TaskId is a struct of one uint64_t; record stores raw uint64s for
        // explicit_deps. Layout-compatible (PTO2TaskId is verified 8 bytes).
        inputs.explicit_deps = reinterpret_cast<const PTO2TaskId *>(rec.explicit_deps);

        // STEP 1: explicit deps — single linear emit, matches runtime call site.
        for (int32_t i = 0; i < dc; i++) {
            edges.emplace_back(rec.explicit_deps[i], rec.task_id);
        }

        // STEP 3: creator retention + tensormap lookup.
        bool ok = compute_task_fanin(inputs, tensor_map, in_manual_scope, [&](PTO2TaskId producer) -> bool {
            edges.emplace_back(producer.raw, rec.task_id);
            return true;
        });
        if (!ok) {
            LOG_ERROR("dep_gen replay: compute_task_fanin returned fatal at task_id=%" PRIu64, rec.task_id);
            tensor_map.destroy();
            return -4;
        }

        // STEP 4: publish this task's outputs.
        register_task_outputs(inputs, task_id, tensor_map, in_manual_scope);
    }

    tensor_map.destroy();

    if (!write_deps_json(deps_json_path, edges)) {
        return -5;
    }
    LOG_INFO_V0("dep_gen replay: wrote %zu edges to %s", edges.size(), deps_json_path);
    return 0;
}
