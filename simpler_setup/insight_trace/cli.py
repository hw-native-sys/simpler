# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from .arg_resolver import resolve_args  # noqa: F401
from .case_loader import load_scene_case  # noqa: F401
from .cli_pkg import build_parser as _build_parser
from .cli_pkg import run_simpler
from .cli_pkg.config import hw_block_num as _hw_block_num  # noqa: F401
from .cli_pkg.config import spmd_meta as _spmd_meta  # noqa: F401
from .kernel_analyzer import select_kernel, validate_single_task_kernel  # noqa: F401
from .runner import run_workspace  # noqa: F401
from .workspace import create_workspace  # noqa: F401

# Re-export for testability via monkeypatch
_run_simpler = run_simpler


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_simpler(args)
    except Exception as exc:  # noqa: BLE001
        print(f"insight trace failed: {exc}")
        return 1
    print(f"Insight trace workspace: {result.workspace_dir}")
    if result.simulator_dir is not None:
        print(f"MindStudio Insight input: {result.simulator_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
