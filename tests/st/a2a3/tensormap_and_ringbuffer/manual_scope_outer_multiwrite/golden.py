# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch


__outputs__ = ["out", "result", "check"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> list:
    del params
    size = 128 * 128
    a = torch.full((size,), 1.0, dtype=torch.float32)
    b = torch.full((size,), 2.0, dtype=torch.float32)
    out = torch.zeros(size, dtype=torch.float32)
    result = torch.zeros(size, dtype=torch.float32)
    check = torch.zeros(4, dtype=torch.float32)
    return [
        ("a", a),
        ("b", b),
        ("out", out),
        ("result", result),
        ("check", check),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    del params
    out = torch.as_tensor(tensors["out"])
    result = torch.as_tensor(tensors["result"])
    check = torch.as_tensor(tensors["check"])

    out.fill_(5.0)
    result.fill_(7.0)
    check[0] = 5.0
    check[1] = 7.0
    check[2] = 5.0
