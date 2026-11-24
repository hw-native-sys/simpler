# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 Host-source URMA gather callable shared by the driver and L3 scene."""

from pathlib import Path

from simpler.task_interface import ArgDirection as D

from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key

RUNTIME = "tensormap_and_ringbuffer"
_KERNELS = Path(__file__).resolve().parent / "kernels"
GATHER = {
    "name": "gather",
    "orchestration": {
        "source": str(_KERNELS / "orchestration/gather.cpp"),
        "function_name": "host_comm_domain",
        "config_name": "host_comm_domain_config",
        "signature": [D.IN, D.OUT, D.OUT, D.IN, D.IN],
    },
    "incores": [
        {
            "func_id": 0,
            "source": str(_KERNELS / "aiv/gather.cpp"),
            "core_type": "aiv",
            "signature": [D.IN, D.OUT, D.IN],
        },
        {
            "func_id": 1,
            "source": str(_KERNELS / "aiv/consume.cpp"),
            "core_type": "aiv",
            "signature": [D.IN, D.OUT],
        },
    ],
}


def build_callable():
    # The L3 precompiler keys each entry by its collected module and class.
    key = l3_compile_cache_key(
        "examples.a5.tensormap_and_ringbuffer.host_comm_domain.test_host_comm_domain",
        "TestHostCommDomain",
        GATHER["name"],
        "a5",
        RUNTIME,
    )
    return compile_chip_callable_spec(GATHER, "a5", RUNTIME, key)
