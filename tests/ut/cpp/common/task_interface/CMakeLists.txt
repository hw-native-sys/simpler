# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/common/task_interface/.
#
# These cover the public argument and callable types, so they need only
# task_interface on the include path and link no runtime.

macro(task_interface_case src)
    simpler_ut_plain_case(${src}
        INCLUDES ${SIMPLER_SRC}/common/task_interface
        LABEL no_hardware ${ARGN})
endmacro()

# Pulls both task_interface (callable.h) and src/common (utils/fnv1a_64.h).
task_interface_case(test_chip_callable_upload_immutable.cpp
    INCLUDES ${SIMPLER_SRC}/common)

simpler_ut_glob_cases(task_interface_case)
