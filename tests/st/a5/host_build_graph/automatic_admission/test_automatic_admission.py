#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard default automatic admission through public Worker.submit."""

from simpler_setup import SceneTestCase, scene_test
from tests.st.automatic_admission_support import (
    AutomaticAdmissionScene,
    AutomaticDeviceDemandScene,
    AutomaticOutstandingCloseScene,
    AutomaticTwoEndpointScene,
    callable_spec,
)


@scene_test(level=3, runtime="host_build_graph", automatic_admission=True)
class TestAutomaticAdmission(AutomaticAdmissionScene):
    CALLABLE = callable_spec("a5", "host_build_graph")
    CASES = [
        {
            "name": "automatic_admission",
            "platforms": ["a5"],
            "config": {"device_count": 1, "num_sub_workers": 1},
            "params": {},
        }
    ]


@scene_test(level=3, runtime="host_build_graph", automatic_admission=True)
class TestAutomaticDeviceDemand(AutomaticDeviceDemandScene):
    CALLABLE = callable_spec("a5", "host_build_graph")
    CASES = [
        {
            "name": "automatic_devicedemand",
            "platforms": ["a5"],
            "config": {"device_count": 1, "num_sub_workers": 1},
            "params": {},
        }
    ]


@scene_test(level=3, runtime="host_build_graph", automatic_admission=True)
class TestAutomaticOutstandingClose(AutomaticOutstandingCloseScene):
    CALLABLE = callable_spec("a5", "host_build_graph")
    CASES = [
        {
            "name": "automatic_outstandingclose",
            "platforms": ["a5"],
            "config": {"device_count": 1, "num_sub_workers": 1},
            "params": {},
        }
    ]


@scene_test(level=3, runtime="host_build_graph", automatic_admission=True)
class TestAutomaticTwoEndpointFold(AutomaticTwoEndpointScene):
    CALLABLE = callable_spec("a5", "host_build_graph")
    CASES = [
        {
            "name": "automatic_twoendpointfold",
            "platforms": ["a5"],
            "config": {"device_count": 2, "num_sub_workers": 1},
            "params": {},
        }
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
