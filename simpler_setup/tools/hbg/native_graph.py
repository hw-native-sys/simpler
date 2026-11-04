# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reconstruct native nodes from original submissions and recorded Definition rows."""

from collections.abc import Iterable
from typing import Any

from .native_protocol import _array, _error, _indices, _number, _object

TOP, CHILD, DUMMY, ENTER, EXIT, HOST_DONE = range(6)
NO_CHILD = (1 << 32) - 1
EXECUTABLE, INLINE, PREDICATE = 1, 32, 64


def _deps(row, limit, path):
    values = row["dependencies"]
    _array(values, path, signed=True)
    if len(set(values)) != len(values) or any(not 0 <= dep < limit for dep in values):
        _error(f"invalid original dependencies at {path}")


def read_graph_source(section):
    source = section["graph_source"]
    _object(source, {"tasks", "definitions"}, "graph_source")
    if not isinstance(source["tasks"], list) or not isinstance(source["definitions"], list):
        _error("invalid original source arrays")
    definitions = {}
    for definition in source["definitions"]:
        _object(definition, {"graph_key", "children"}, "original Definition")
        key, children = definition["graph_key"], definition["children"]
        _number(key, "graph_key")
        if key in definitions or not isinstance(children, list) or not 1 <= len(children) <= 1024:
            _error("duplicate or invalid original Definition")
        definitions[key] = definition
        for local, child in enumerate(children):
            _object(
                child,
                {"active_mask", "block_num", "task_attrs", "predicate_slot", "kernel_ids", "dependencies"},
                "original child",
            )
            for name in ("active_mask", "block_num", "task_attrs", "predicate_slot"):
                _number(child[name], name)
            _array(child["kernel_ids"], "kernel_ids", 3, signed=True)
            _deps(child, local, "child dependencies")
            if not 0 <= child["active_mask"] < 8 or not 1 <= child["block_num"] <= 32767:
                _error("invalid original child shape")
    for task_id, task in enumerate(source["tasks"]):
        _object(task, {"task_id", "kind", "graph_key", "dependencies"}, "original task")
        for name in ("task_id", "kind", "graph_key"):
            _number(task[name], name)
        if task["task_id"] != task_id or task["kind"] not in (0, 1, 2, 3):
            _error("invalid original task identity or kind")
        if task["kind"] == 1 and task["graph_key"] not in definitions:
            _error("original Graph has no Definition")
        _deps(task, task_id, "outer dependencies")
    _indices(section["node_map"], section["task_count"], "node_map", key="node_ref", dense=True)
    for row in section["node_map"]:
        _object(row, {"node_ref", "kind", "public_task_id", "original_task_id", "local_child_id"}, "node_map row")
        for name, value in row.items():
            _number(value, name)
    return definitions


def audit_graph_expansion(section):
    """Compare observed accessors with an expansion that uses no planner-produced edges."""
    definitions = read_graph_source(section)
    expected, completed, shapes = [], {}, {}

    def append(kind, original, deps, local=NO_CHILD):
        ref = len(expected)
        public = original if local == NO_CHILD else (1 << 62) | (original << 32) | local
        expected.append(
            (
                dict(node_ref=ref, kind=kind, public_task_id=public, original_task_id=original, local_child_id=local),
                deps,
            )
        )
        return ref

    for outer in section["graph_source"]["tasks"]:
        original = outer["task_id"]
        deps = [completed[dep] for dep in outer["dependencies"]]
        if outer["kind"] != 1:
            completed[original] = append({0: TOP, 2: DUMMY, 3: HOST_DONE}[outer["kind"]], original, deps)
            continue
        children = definitions[outer["graph_key"]]["children"]
        enter = append(ENTER, original, deps)
        first = enter + 1
        consumed = set()
        for local, child in enumerate(children):
            consumed.update(child["dependencies"])
            deps = [first + dep for dep in child["dependencies"]] or [enter]
            ref = append(CHILD if child["active_mask"] else DUMMY, original, deps, local)
            shapes[ref] = child
        completed[original] = append(
            EXIT, original, [first + local for local in range(len(children)) if local not in consumed]
        )
    if len(expected) != section["task_count"]:
        _error("expanded node count disagrees with original source")
    nodes = {row["node_ref"]: row for row in section["node_map"]}
    tasks = {row["task_ref"]: row for row in section["tasks"]}
    for ref, (identity, deps) in enumerate(expected):
        if nodes[ref] != identity:
            _error(f"native public identity or node kind mismatch at {ref}")
        if sorted(tasks[ref]["dependencies"]) != sorted(deps):
            _error(f"native dependencies disagree with original source at {ref}")
        if ref in shapes and identity["kind"] == CHILD:
            child, task = shapes[ref], tasks[ref]
            kernels = [
                value if child["active_mask"] & (1 << lane) else 65535 for lane, value in enumerate(child["kernel_ids"])
            ]
            if (task["active_mask"], task["block_num"], task["kernel_ids"]) != (
                child["active_mask"],
                child["block_num"],
                kernels,
            ):
                _error(f"native child shape disagrees with original source at {ref}")
            if bool(task["flags"] & 16) != bool(child["task_attrs"] & 2) or bool(task["flags"] & 64) != bool(
                child["predicate_slot"]
            ):
                _error(f"native child attributes disagree with original source at {ref}")
    return nodes


def export_kernel_dependencies(section: dict[str, Any], executed_task_ids: Iterable[int]) -> dict[str, Any]:
    """Project a verified native DAG onto the kernels executed in one run.

    ENTER, EXIT, DUMMY, host services and skipped predicates are transparent.
    Their predecessor frontier preserves cross-Graph and join dependencies.
    This projection is specific to the observed predicate outcomes.
    """
    if section.get("schema") != "hbg-native-protocol-v6":
        _error("kernel dependency export requires native Graph v6 source")
    nodes = audit_graph_expansion(section)
    tasks = {row["task_ref"]: row for row in section["tasks"]}
    kernels = {
        row["public_task_id"]: ref
        for ref, row in nodes.items()
        if row["kind"] in (TOP, CHILD) and tasks[ref]["flags"] & EXECUTABLE and not tasks[ref]["flags"] & INLINE
    }
    observed = set(executed_task_ids)
    if not observed <= kernels.keys():
        _error("kernel dependency export has unknown execution identities")
    for public, ref in kernels.items():
        if public not in observed and not tasks[ref]["flags"] & PREDICATE:
            _error("kernel dependency export has an unexecuted non-predicated task")
    frontier, output, edges = {}, [], []
    for ref in range(section["task_count"]):
        task, node = tasks[ref], nodes[ref]
        predecessors = set().union(*(frontier[dep] for dep in task["dependencies"]))
        public = node["public_task_id"]
        if kernels.get(public) == ref and public in observed:
            output.append(
                dict(
                    task_id=str(public),
                    kernel_ids=[kernel if kernel != 65535 else -1 for kernel in task["kernel_ids"]],
                    block_num=task["block_num"],
                    native_node_ref=ref,
                )
            )
            edges.extend(dict(pred=str(pred), succ=str(public), source="native_graph") for pred in sorted(predecessors))
            frontier[ref] = {public}
        else:
            frontier[ref] = predecessors
    return dict(
        runtime="host_build_graph",
        tasks=output,
        tensors=[],
        edges=edges,
        projection=dict(schema="hbg-native-kernel-deps-v1", run_epoch=section["run_epoch"]),
    )
