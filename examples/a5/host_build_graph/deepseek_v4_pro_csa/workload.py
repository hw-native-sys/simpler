# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Self-contained, frozen HBG CSA workload."""

from pathlib import Path

if __package__:
    from . import fixture, kernel_config
else:
    import fixture
    import kernel_config

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[3]
RUNTIME = fixture.RUNTIME
SOURCE = CASE_DIR / "kernels/vendor/orchestration/attention_csa_test.cpp"
CALLABLE = {"orchestration": kernel_config.ORCHESTRATION, "incores": kernel_config.KERNELS}


def checkout_identity():
    identity = fixture.checkout_identity()
    if not identity["case_sources_match_baseline"]:
        raise ValueError("HBG CSA workload no longer matches the frozen source manifest")
    return {
        **identity,
        "runtime": RUNTIME,
        "hbg_orchestration_sha256": fixture.sha256(SOURCE),
        "host_only_arguments": ["hc_attn_scale"],
    }
