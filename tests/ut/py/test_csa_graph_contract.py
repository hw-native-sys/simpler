# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Frozen real CSA graph shape/edge/view regression gates without large tensors."""

import copy

import pytest

from examples.a5.host_build_graph.deepseek_v4_pro_csa import fixture, graph_contract

FROZEN = graph_contract.frozen_graph()
NODES = FROZEN["graph"]["nodes"]


def dependency_capture():
    # Reconstruct only fields retained by the graph projection, with arbitrary tensor identities.
    tasks = []
    for node in NODES:
        task = {
            key: copy.deepcopy(node[key]) for key in ("task_id", "kernel_ids", "block_num", "early_dispatch", "args")
        }
        for arg in task["args"]:
            arg["tensor_id"] = "address-independent"
        tasks.append(task)
    edges = [{"pred": pred, "succ": node["task_id"]} for node in NODES for pred in node["deps"]]
    edges += [{"pred": ref, "succ": NODES[0]["task_id"]} for ref in FROZEN["graph"]["host_ready_refs"]]
    return {"runtime": "host_build_graph", "tasks": tasks, "edges": edges}


def test_graph_capture_provenance_and_roundtrip():
    assert FROZEN["source_manifest_sha256"] == fixture.sha256(fixture.CASE_DIR / "manifest.json")
    assert FROZEN["fixture_id"] == fixture.manifest()["fixture_id"]
    assert graph_contract.validate_graph(dependency_capture()) == FROZEN["graph"]


@pytest.mark.parametrize("ref", [node["task_id"] for node in NODES if node["block_num"] > 1])
def test_each_real_spmd_task_must_keep_its_original_block_count(ref):
    deps = dependency_capture()
    next(t for t in deps["tasks"] if t["task_id"] == ref)["block_num"] = 1
    with pytest.raises(ValueError, match="frozen graph"):
        graph_contract.validate_graph(deps)


@pytest.mark.parametrize("pred,succ", [(p, n["task_id"]) for n in NODES for p in n["deps"]])
def test_each_real_dependency_edge_cannot_disappear(pred, succ):
    deps = dependency_capture()
    deps["edges"].remove({"pred": pred, "succ": succ})
    with pytest.raises(ValueError, match="frozen graph"):
        graph_contract.validate_graph(deps)


@pytest.mark.parametrize(
    "ref", [node["task_id"] for node in NODES if node["kernel_ids"][0] >= 0 and node["kernel_ids"][1] >= 0]
)
def test_each_real_mix_task_must_keep_its_vector_payload(ref):
    deps = dependency_capture()
    next(t for t in deps["tasks"] if t["task_id"] == ref)["kernel_ids"][1] = -1
    with pytest.raises(ValueError, match="frozen graph"):
        graph_contract.validate_graph(deps)


@pytest.mark.parametrize("ref", list(range(59, 105, 3)))
def test_sixteen_same_kernel_pack_tasks_keep_distinct_tensor_views(ref):
    deps = dependency_capture()
    task = next(t for t in deps["tasks"] if t["task_id"] == ref)
    target = next(a for a in task["args"] if a["type"].startswith("OUTPUT"))
    target["start_offset"] = str(int(target["start_offset"]) + 1)
    with pytest.raises(ValueError, match="frozen graph"):
        graph_contract.validate_graph(deps)


@pytest.mark.parametrize("mutation", ["duplicate_task", "runtime", "early_dispatch"])
def test_capture_metadata_corruption_is_rejected(mutation):
    deps = dependency_capture()
    if mutation == "duplicate_task":
        deps["tasks"].append(copy.deepcopy(deps["tasks"][0]))
    elif mutation == "runtime":
        deps["runtime"] = "tensormap_and_ringbuffer"
    else:
        deps["tasks"][0]["early_dispatch"] = True
    with pytest.raises(ValueError):
        graph_contract.validate_graph(deps)
