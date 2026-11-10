# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The native fixture validates every replay sample without requiring hardware."""

import ast
import json
import struct
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _validator() -> Callable[[Path, int], None]:
    path = Path(__file__).resolve().parents[3] / "tests/st/a2a3/kernel_capture/test_hbg_capture.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "check_args_dump_output"
    )
    namespace: dict[str, Any] = dict(Path=Path, json=json, Counter=Counter, struct=struct)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def _outputs(root, replays):
    for leaf, initial, increment, rounds, recorded in (
        ("a", 10, -3, replays, True),
        ("b", 1000, 5, replays, True),
        ("eager", -24680, 17, 1, False),
    ):
        directory = root / leaf / "args_dump"
        directory.mkdir(parents=True)
        payload = bytearray()
        entries = []
        for index in range(rounds):
            seed = initial + 100 * index
            for task, scalar, before, after in (
                (1, increment, seed, seed + increment),
                (2, 7, seed + increment, seed + increment + 7),
            ):
                base = dict(
                    task_id=f"0x{task:016x}",
                    func_id=[task],
                    run_epoch=1,
                    role="inout",
                    dtype="INT32",
                    is_contiguous=True,
                    start_offset=0,
                    truncated=False,
                )
                entries.append(
                    base
                    | dict(
                        kind="scalar",
                        arg_index=1,
                        stage="before_dispatch",
                        value=scalar,
                        bin_offset=0,
                        bin_size=0,
                        shape=[],
                        strides=[],
                    )
                )
                for stage, value in (("before_dispatch", before), ("after_completion", after)):
                    entries.append(
                        base
                        | dict(
                            kind="tensor",
                            arg_index=0,
                            stage=stage,
                            bin_offset=len(payload),
                            bin_size=64,
                            shape=[16],
                            strides=[1],
                        )
                    )
                    payload.extend(struct.pack("<16i", *(value + 7 * i for i in range(16))))
        manifest = dict(
            args=entries,
            total_args=len(entries),
            bin_file="samples.bin" if payload else None,
            dump_args_level=3 if recorded else 1,
            dropped_records=None if recorded else 0,
        )
        if recorded:
            manifest.update(
                recorded_window=True,
                counts_unknown=True,
                dropped_records_complete=False,
                collection_verdict="aggregate_counts_unknown",
                host_discarded_args=0,
                metadata_discarded_records=0,
            )
        (directory / "args_dump.json").write_text(json.dumps(manifest))
        if payload:
            (directory / "samples.bin").write_bytes(payload)


@pytest.mark.parametrize("replays", [0, 3])
def test_args_dump_native_validator_accepts_complete_samples(tmp_path, replays):
    _outputs(tmp_path, replays)
    _validator()(tmp_path, replays)


@pytest.mark.parametrize("corruption", ["missing_scalar", "extra_tensor"])
def test_args_dump_native_validator_rejects_lost_or_duplicate_occurrences(tmp_path, corruption):
    _outputs(tmp_path, 3)
    directory = tmp_path / "a/args_dump"
    manifest_path = directory / "args_dump.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "missing_scalar":
        manifest["args"].pop(0)
    else:
        entry = next(item for item in manifest["args"] if item["kind"] == "tensor").copy()
        binary = directory / "samples.bin"
        payload = binary.read_bytes()
        entry["bin_offset"] = len(payload)
        binary.write_bytes(payload + payload[:64])
        manifest["args"].append(entry)
    manifest["total_args"] = len(manifest["args"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        _validator()(tmp_path, 3)
