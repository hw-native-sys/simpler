# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import json
import stat

from .models import TraceConfig, TraceResult
from .templates import render_cmake, render_config, render_host, render_kernel, render_launch, render_run_collect


def create_workspace(config: TraceConfig) -> TraceResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "replay_kernel.cpp": render_kernel(config),
        "replay_launch.cpp": render_launch(config),
        "replay_host.cpp": render_host(config),
        "CMakeLists.txt": render_cmake(config),
        "run_collect.sh": render_run_collect(config),
        "insight_trace_config.json": json.dumps(render_config(config), indent=2) + "\n",
    }
    for name, content in files.items():
        path = config.output_dir / name
        path.write_text(content)
        if name == "run_collect.sh":
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return TraceResult(
        workspace_dir=config.output_dir,
        simulator_dir=None,
        collect_log=config.output_dir / "msprof_collect" / "msprof_collect.log",
        export_log=config.output_dir / "insight_export" / "msprof_export.log",
    )
