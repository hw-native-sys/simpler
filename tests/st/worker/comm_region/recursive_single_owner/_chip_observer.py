# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Install the test's provider observer before the production chip bootstrap."""

import sys

from _helpers import _install_provider_recorder, _LifecycleRecorder
from simpler.chip_process import _main

if __name__ == "__main__":
    _install_provider_recorder(_LifecycleRecorder(sys.argv[1]))
    _main(int(sys.argv[2]), int(sys.argv[3]))
