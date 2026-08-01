# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free RemoteCallable callbacks for the MPI L3 integration smoke."""

from __future__ import annotations

import json
import os

from .task_interface import CallConfig, TaskArgs


def record_rank_value(_orch, args: TaskArgs, _config: CallConfig) -> None:
    if args.scalar_count() != 1:
        raise ValueError("MPI group smoke callback expects one scalar")
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    value = int(args.scalar(0))
    if value == 0xFFFF:
        raise ValueError(f"injected callback failure on MPI rank {rank}")
    output_dir = os.environ["SIMPLER_MPI_SMOKE_DIR"]
    path = os.path.join(output_dir, f"rank-{rank}.json")
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump({"rank": rank, "value": value}, output_file, sort_keys=True)
