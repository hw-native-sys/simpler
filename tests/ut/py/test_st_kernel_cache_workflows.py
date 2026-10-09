# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = (
    (
        "_st-deepseek-a2a3.yml",
        ("Compile DeepSeek kernels (a2a3)",),
        "Run DeepSeek pytest smokes (a2a3)",
    ),
    (
        "_st-npu-a2a3.yml",
        ("Compile scene-test kernels (a2a3)", "Compile Qwen standalone kernels (a2a3)"),
        "Run pytest scene tests (a2a3)",
    ),
    (
        "_st-npu-a5.yml",
        ("Compile scene-test kernels (a5)", "Compile Qwen standalone kernels (a5)"),
        "Run pytest scene tests (a5)",
    ),
)


@pytest.mark.parametrize(("workflow_name", "compile_names", "test_name"), WORKFLOWS)
def test_kernel_cache_is_saved_after_compile_and_before_device_test(
    workflow_name: str, compile_names: tuple[str, ...], test_name: str
) -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows" / workflow_name).read_text())
    steps = workflow["jobs"]["run"]["steps"]
    by_name = {step["name"]: (index, step) for index, step in enumerate(steps)}

    restore_index, restore = next(value for name, value in by_name.items() if name.startswith("Restore scene-test"))
    save_index, save = next(value for name, value in by_name.items() if name.startswith("Save scene-test"))
    test_index, _test = by_name[test_name]
    compile_indexes = [by_name[name][0] for name in compile_names]

    assert restore_index < min(compile_indexes)
    assert max(compile_indexes) < save_index < test_index
    assert restore["uses"] == "actions/cache/restore@v5"
    assert restore["id"] in {"deepseek-kernel-cache", "scene-kernel-cache"}
    key = restore["with"]["key"]
    assert "-cases-v1-${{ hashFiles(" in key
    assert "github.sha" not in key
    assert "github.run_id" not in key
    if workflow_name == "_st-deepseek-a2a3.yml":
        assert "host_build_graph/deepseek_v4_flash_decode/**" in key
        assert "tensormap_and_ringbuffer/deepseek_v4_flash_decode/**" in key
        restore_keys = restore["with"]["restore-keys"].splitlines()
        assert restore_keys[0].startswith("scene-kernels-a2a3-ds-")
        assert restore_keys[1].startswith("scene-kernels-a2a3-${{ runner.os }}-${{ runner.arch }}-")
        assert "hashFiles('examples/**', 'tests/st/**')" in restore_keys[1]
        assert "cache-matched-key == ''" in save["if"]
        assert "startsWith(" in save["if"]
        assert "scene-kernels-a2a3-ds-" in save["if"]
    else:
        assert "hashFiles('examples/**', 'tests/st/**')" in key
    assert save["uses"] == "actions/cache/save@v5"
    assert f"steps.{restore['id']}.outputs.cache-hit != 'true'" in save["if"]
    assert save["with"]["key"] == f"${{{{ steps.{restore['id']}.outputs.cache-primary-key }}}}"
