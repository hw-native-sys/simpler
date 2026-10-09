# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github/workflows/_st-deepseek-a2a3.yml"


def test_deepseek_cache_is_saved_after_compile_and_before_device_test() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["run"]["steps"]
    by_name = {step["name"]: (index, step) for index, step in enumerate(steps)}

    restore_index, restore = by_name["Restore scene-test kernel cache (a2a3)"]
    compile_index, _compile = by_name["Compile DeepSeek kernels (a2a3)"]
    save_index, save = by_name["Save scene-test kernel cache (a2a3)"]
    test_index, _test = by_name["Run DeepSeek pytest smokes (a2a3)"]

    assert restore_index < compile_index < save_index < test_index
    assert restore["uses"] == "actions/cache/restore@v5"
    assert restore["id"] == "deepseek-kernel-cache"
    assert "${{ github.run_id }}-${{ github.run_attempt }}" in restore["with"]["key"]
    assert save["uses"] == "actions/cache/save@v5"
    assert save["with"]["key"] == "${{ steps.deepseek-kernel-cache.outputs.cache-primary-key }}"
