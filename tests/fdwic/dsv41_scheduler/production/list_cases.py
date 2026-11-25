#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""List the active variant/case paths for a complete scheduler benchmark run."""

import argparse

MODEL_CASES = ("swa", "c2a_full", "c1a_reindex", "c1a_reuse", "engram")
FULL_WAVE = "synthetic/full_wave_mix_chain"
DIAMOND = "synthetic/mpmd_diamond"


def selected_cases(flavor="all"):
    if flavor in ("all", "pypto_lib_latest"):
        for name in MODEL_CASES:
            yield f"pypto_lib_latest/{name}"
    if flavor in ("all", "synthetic"):
        yield FULL_WAVE
        yield DIAMOND


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("host_build_graph", "percounter"), default="host_build_graph")
    parser.add_argument("--flavor", choices=("all", "pypto_lib_latest", "synthetic"), default="all")
    print("\n".join(selected_cases(parser.parse_args().flavor)))
