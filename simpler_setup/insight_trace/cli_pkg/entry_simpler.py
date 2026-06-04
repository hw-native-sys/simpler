# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simpler backend entry point for insight trace."""

from __future__ import annotations

from .config import build_trace_config


def run_simpler(args):
    # Defer import to avoid circular dependency at module load time
    from .. import cli as cli_module  # noqa: PLC0415

    if args.test_module is None or args.case is None:
        raise ValueError("simpler backend requires test_module and --case")
    context = cli_module.load_scene_case(args.test_module, args.case)
    kernel = cli_module.select_kernel(context, args.kernel, args.func_id, args.kernel_source)
    cli_module.validate_single_task_kernel(kernel)
    # PR792 format: task_id filters by task, func_id is for kernel selection only
    task_id = getattr(args, "task_id", None)
    trace_args = cli_module.resolve_args(context, kernel, args.arg_spec, args.dump_dir, task_id)
    config = build_trace_config(args, context, kernel, trace_args)
    result = cli_module.create_workspace(config)
    if args.dry_run:
        return result
    return cli_module.run_workspace(config)
