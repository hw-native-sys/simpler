# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Percounter's nested execution functions on the shared Scalar interval renderer."""

from simpler_setup.tools.hbg import scalar_profile as scalar


def analyze_streams(streams, metadata, task_rows, execution_identities=None):
    return scalar.analyze_streams(streams, metadata, task_rows, execution_identities, nested_kernels=True)


def trace_events(profiles, **kwargs):
    names = {row["kind"]: row["kind"].removeprefix("scalar_") for profile in profiles for row in profile["rows"]}
    names["scalar_run_percounter_lane"] = "run_percounter_lane: setup"
    return scalar.trace_events(profiles, function_names=names, **kwargs)
