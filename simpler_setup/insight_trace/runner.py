# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .models import TraceConfig, TraceResult


def run_workspace(config: TraceConfig) -> TraceResult:
    script = config.output_dir / "run_collect.sh"
    bash_path = shutil.which("bash")
    if bash_path is None:
        raise RuntimeError("bash executable not found in PATH")
    env = os.environ.copy()
    if config.cann_home is not None:
        env["CANN_HOME"] = str(config.cann_home)
    if config.pto_isa_root is not None:
        env["PTO_ISA_ROOT"] = str(config.pto_isa_root)
    env["REPO_ROOT"] = str(config.repo_root)
    try:
        result = subprocess.run(
            [bash_path, str(script)],
            cwd=str(config.output_dir),
            env=env,
            check=False,
            text=True,
            timeout=config.timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Trace collection timed out after {config.timeout}s; check logs at {config.output_dir / 'msprof_collect'}"
        )
    if result.returncode != 0:
        raise RuntimeError(f"insight trace collection failed; see {config.output_dir / 'msprof_collect'}")
    simulator_dir = find_simulator_dir(config.output_dir / "insight_export")
    validate_simulator_dir(simulator_dir)
    return TraceResult(
        workspace_dir=config.output_dir,
        simulator_dir=simulator_dir,
        collect_log=config.output_dir / "msprof_collect" / "msprof_collect.log",
        export_log=config.output_dir / "insight_export" / "msprof_export.log",
    )


def find_simulator_dir(export_root: Path) -> Path:
    candidates = sorted(export_root.glob("OPPROF_*/simulator"))
    if not candidates:
        raise FileNotFoundError(f"No OPPROF simulator directory under {export_root}")
    return candidates[-1]


def validate_simulator_dir(simulator_dir: Path) -> None:
    required = [simulator_dir / "trace.json", simulator_dir / "visualize_data.bin"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Insight trace artifacts: {names}")
    if not list(simulator_dir.glob("core*/*instr_exe*.csv")):
        raise FileNotFoundError(f"No instr_exe CSV files under {simulator_dir}")
