# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Output-root routing without creating device contexts."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pytest
from _task_interface import CallConfig
from simpler import task_interface as task_interface_mod
from simpler import worker as worker_mod
from simpler.orchestrator import _callback_run

from simpler_setup.tools import swimlane_converter


def test_host_log_output_root_is_fixed_and_persistent(tmp_path, monkeypatch):
    current = [""]
    monkeypatch.setattr(task_interface_mod, "_HOST_LOG_OUTPUT_ROOT", None)
    monkeypatch.setattr(task_interface_mod, "_HOST_LOG_SESSION_DIRECTORY", tmp_path / "old-spool")
    monkeypatch.setattr(task_interface_mod, "_native_host_log_directory", lambda: current[0])
    monkeypatch.setattr(task_interface_mod, "_native_set_host_log_directory", lambda path: current.__setitem__(0, path))

    root = task_interface_mod._configure_host_log_output_root(tmp_path / "output")
    target = root / "host_logs" / task_interface_mod._HOST_LOG_SESSION_ID
    assert target.is_dir()
    assert task_interface_mod._bind_host_log_session_directory() == str(target)
    (target / "host.123.log").write_text("record\n")
    task_interface_mod._cleanup_host_log_session_directory()
    assert (target / "host.123.log").read_text() == "record\n"
    assert task_interface_mod._configure_host_log_output_root(root) == root
    with pytest.raises(RuntimeError, match="already use output_root"):
        task_interface_mod._configure_host_log_output_root(tmp_path / "other")


def test_invalid_prewarm_does_not_claim_process_output_root(tmp_path, monkeypatch):
    bound = [""]
    monkeypatch.setattr(task_interface_mod, "_HOST_LOG_OUTPUT_ROOT", None)
    monkeypatch.setattr(task_interface_mod, "_HOST_LOG_SESSION_DIRECTORY", tmp_path / "old-spool")
    monkeypatch.setattr(task_interface_mod, "_native_host_log_directory", lambda: bound[0])
    monkeypatch.setattr(task_interface_mod, "_native_set_host_log_directory", lambda path: bound.__setitem__(0, path))
    monkeypatch.setattr(worker_mod.Worker, "_init_hierarchical", lambda self: None)
    monkeypatch.setattr(worker_mod.Worker, "_start_hierarchical", lambda self: None)

    invalid = CallConfig()
    invalid.enable_dep_gen = True
    with pytest.raises(ValueError, match="output_prefix must be set"):
        worker_mod.Worker(3, output_root=tmp_path / "A").init(prewarm_config=invalid)
    assert task_interface_mod._HOST_LOG_OUTPUT_ROOT is None
    assert bound[0] == ""

    worker_mod.Worker(3, output_root=tmp_path / "B").init()
    assert task_interface_mod._HOST_LOG_OUTPUT_ROOT == tmp_path / "B"
    assert bound[0] == str(tmp_path / "B" / "host_logs" / task_interface_mod._HOST_LOG_SESSION_ID)


@pytest.mark.parametrize(
    ("level", "config", "message"),
    [
        (4, {"device_ids": [0]}, "device_ids are only supported on L3"),
        (4, {"pipeline_depth": 3}, "pipeline_depth=3 is not supported"),
        (2, {"platform": "a2a3", "workspace_budget_bytes": 0}, "positive byte count"),
    ],
)
def test_invalid_startup_parameters_do_not_claim_output_root(tmp_path, monkeypatch, level, config, message):
    bound = [""]
    monkeypatch.setattr(task_interface_mod, "_HOST_LOG_OUTPUT_ROOT", None)
    monkeypatch.setattr(task_interface_mod, "_HOST_LOG_SESSION_DIRECTORY", tmp_path / "old-spool")
    monkeypatch.setattr(task_interface_mod, "_native_host_log_directory", lambda: bound[0])
    monkeypatch.setattr(task_interface_mod, "_native_set_host_log_directory", lambda path: bound.__setitem__(0, path))
    monkeypatch.setattr(worker_mod.Worker, "_init_hierarchical", lambda self: None)
    monkeypatch.setattr(worker_mod.Worker, "_start_hierarchical", lambda self: None)

    with pytest.raises((RuntimeError, ValueError), match=message):
        worker_mod.Worker(level, output_root=tmp_path / "A", **config).init()
    assert task_interface_mod._HOST_LOG_OUTPUT_ROOT is None
    assert bound[0] == ""

    worker_mod.Worker(3, output_root=tmp_path / "B").init()
    assert task_interface_mod._HOST_LOG_OUTPUT_ROOT == tmp_path / "B"


@pytest.mark.parametrize("level", [2, 3])
def test_automatic_capture_directories_do_not_mutate_call_config(tmp_path, monkeypatch, level):
    monkeypatch.setattr(worker_mod, "_configure_host_log_output_root", lambda path: Path(path))
    worker = worker_mod.Worker(level, output_root=tmp_path)
    config = CallConfig()
    config.enable_chip_swimlane = True
    first = worker._diagnostic_config_for_submit(config)
    second = worker._diagnostic_config_for_submit(config)
    assert config.output_prefix == ""
    assert Path(first.output_prefix).is_dir()
    assert Path(second.output_prefix).is_dir()
    assert first.output_prefix != second.output_prefix
    assert worker.capture_output_prefix == second.output_prefix
    assert Path(second.output_prefix).parent == tmp_path / "captures" / worker._owner_id


def test_case_capture_reserves_one_prefix_for_multiple_submissions(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_configure_host_log_output_root", lambda path: Path(path))
    worker = worker_mod.Worker(3, output_root=tmp_path)
    config = CallConfig()
    config.enable_chip_swimlane = True
    with worker._diagnostic_case_output("case one") as prefix:
        first = worker._diagnostic_config_for_submit(config)
        second = worker._diagnostic_config_for_submit(config)
        assert first.output_prefix == second.output_prefix == prefix
        assert Path(prefix).parent == tmp_path
        assert Path(prefix).name.startswith("case_one_")
    assert worker._diagnostic_config_for_submit(config).output_prefix != prefix


def test_parallel_submissions_have_distinct_directories_under_one_root(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_configure_host_log_output_root", lambda path: Path(path))
    worker = worker_mod.Worker(3, output_root=tmp_path)

    def submit_one(_index):
        config = CallConfig()
        config.enable_chip_swimlane = True
        return worker._diagnostic_config_for_submit(config).output_prefix

    with ThreadPoolExecutor(max_workers=4) as pool:
        prefixes = list(pool.map(submit_one, range(8)))
    assert len(set(prefixes)) == len(prefixes)
    assert all(Path(path).parent == tmp_path / "captures" / worker._owner_id for path in prefixes)


@pytest.mark.parametrize("level", [4, 5])
def test_local_descendants_inherit_output_root_when_they_enable_diagnostics(tmp_path, monkeypatch, level):
    monkeypatch.setattr(worker_mod, "_configure_host_log_output_root", lambda path: Path(path))
    root = worker_mod.Worker(level, output_root=tmp_path)
    leaf = worker_mod.Worker(3)
    if level == 5:
        middle = worker_mod.Worker(4)
        middle.add_worker(leaf)
        root.add_worker(middle)
        assert middle._output_root == str(tmp_path)
    else:
        root.add_worker(leaf)
    config = CallConfig()
    config.enable_chip_swimlane = True
    capture = leaf._diagnostic_config_for_submit(config).output_prefix
    assert leaf._output_root == str(tmp_path)
    assert Path(capture).parent == tmp_path / "captures" / leaf._owner_id


def test_local_child_rejects_different_output_root(tmp_path):
    parent = worker_mod.Worker(4, output_root=tmp_path / "parent")
    child = worker_mod.Worker(3, output_root=tmp_path / "child")
    with pytest.raises(ValueError, match="output_root must match"):
        parent.add_worker(child)
    assert child._topology_parent is None


def test_child_diagnostics_inherit_parent_capture_and_reject_external_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_configure_host_log_output_root", lambda path: Path(path))
    worker = worker_mod.Worker(3, output_root=tmp_path)
    child = CallConfig()
    child.enable_chip_swimlane = True
    with _callback_run(1, worker):
        first = worker._diagnostic_config_for_next_level(child)
        second = worker._diagnostic_config_for_next_level(child)
        assert first.output_prefix == second.output_prefix == worker.capture_output_prefix
        assert child.output_prefix == ""
        child.output_prefix = str(tmp_path / "outside")
        with pytest.raises(ValueError, match="managed by Simpler"):
            worker._diagnostic_config_for_next_level(child)


def test_child_diagnostics_use_case_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_configure_host_log_output_root", lambda path: Path(path))
    worker = worker_mod.Worker(3, output_root=tmp_path)
    child = CallConfig()
    child.enable_dep_gen = True
    with worker._diagnostic_case_output("group") as prefix, _callback_run(1, worker):
        resolved = worker._diagnostic_config_for_next_level(child)
        assert resolved.output_prefix == prefix
        assert worker.capture_output_prefix == prefix


def test_public_submit_rejects_caller_selected_prefix(tmp_path):
    worker = worker_mod.Worker(2, output_root=tmp_path)
    config = CallConfig()
    config.enable_dep_gen = True
    config.output_prefix = str(tmp_path / "chosen")
    with pytest.raises(ValueError, match="output_prefix is managed by Simpler"):
        worker.submit(None, config=config)


def test_internal_child_submission_accepts_inherited_capture(tmp_path, monkeypatch):
    worker = worker_mod.Worker(3, output_root=tmp_path)
    config = CallConfig()
    config.enable_dep_gen = True
    config.output_prefix = str(tmp_path / "captures" / "run" / "L3w0")
    monkeypatch.setattr(worker, "_operation_lease", lambda _api: nullcontext())
    monkeypatch.setattr(worker, "_submit_locked", lambda *_args, **_kwargs: "accepted")
    monkeypatch.setattr(worker, "_teardown_delegated_fatal_if_safe", lambda *_args: None)
    with pytest.raises(ValueError, match="output_prefix is managed by Simpler"):
        worker.submit(None, config=config)
    with worker._inherited_capture_output(config.output_prefix):
        assert worker.submit(None, config=config) == "accepted"


def test_capture_sidecar_points_to_persistent_host_logs(tmp_path, monkeypatch):
    capture = tmp_path / "captures" / "case" / "rank0" / "d0"
    logs = tmp_path / "host_logs" / "session"
    logs.mkdir(parents=True)
    (logs / "host.123.log").write_text("host record\n")
    monkeypatch.setattr(worker_mod, "_persistent_host_log_directory", lambda: str(logs))
    worker_mod._write_dispatch_identity_sidecar(
        str(capture),
        frame_identity=(worker_mod._TASK_PROTOCOL_VERSION, 17, 1, 23, 41, 5, 0, 1),
        chip_rank=0,
        capture_index=0,
        callable_digest=b"\xab" * 32,
        owner_instance_id=b"\xcd" * 16,
    )
    identity = json.loads((capture / "dispatch_identity.json").read_text())
    assert identity["host_pid"] == os.getpid()
    assert (capture / identity["host_log_directory"]).resolve() == logs
    assert swimlane_converter._referenced_host_log_directory(capture) == logs
    records = capture / "chip_swimlane_records.json"
    records.write_text("{}")
    assert swimlane_converter._discover_host_logs(capture.parent.parent, [], [(0, records)]) == [logs / "host.123.log"]
    (logs / "host.123.log").unlink()
    logs.rmdir()
    local_alignment = capture / "host_clock_alignment.123.log"
    local_alignment.write_text("alignment record\n")
    assert swimlane_converter._discover_host_logs(capture.parent.parent, [], [(0, records)]) == [local_alignment]
