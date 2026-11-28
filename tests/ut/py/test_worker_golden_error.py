# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side absolute errors for finite worker example goldens."""

import math

import pytest

from simpler_setup.goldens import max_abs_error


@pytest.mark.parametrize("errors", [(math.nan, 0.0), (0.0, math.nan), (0.0, -math.inf)])
def test_nonfinite_error_is_reported(errors):
    assert max_abs_error(errors) == math.inf


def test_finite_error_preserves_magnitude():
    assert max_abs_error([0.0, -3.0, 1.0]) == 3.0
