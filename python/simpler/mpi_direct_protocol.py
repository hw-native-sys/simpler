# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed MPI tag lanes for the direct-MPI transport."""

import enum

MPI_DIRECT_STARTUP_TOKEN_ENV = "SIMPLER_MPI_DIRECT_STARTUP_TOKEN"
MPI_DIRECT_GATE_MAX_BYTES = 64 * 1024


class MpiDirectTag(enum.IntEnum):
    COMMAND_REQUEST = 1
    COMMAND_REPLY = 2
    HEALTH = 3
    LIFECYCLE = 4
