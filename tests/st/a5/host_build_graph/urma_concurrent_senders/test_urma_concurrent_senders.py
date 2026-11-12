#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HBG sender isolation uses the same transfers and numerical oracles as TMR."""

import json

from simpler_setup import scene_test
from tests.st.a5.tensormap_and_ringbuffer.urma_concurrent_senders.test_urma_concurrent_senders import (
    _UrmaConcurrentSenders,
)


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaConcurrentSenders(_UrmaConcurrentSenders):
    CASES = [
        {
            "name": name,
            "platforms": ["a5"],
            "config": {"device_count": peers + 1, "num_sub_workers": 0},
            "params": {"slots": 2, "peers": peers, "waves": 3, "repeats": 4, "independent": mode},
        }
        for name, peers, mode in (
            ("same_peer", 1, 0),
            ("concurrent_tasks", 1, 1),
            ("mix_siblings", 1, 2),
            ("two_peers", 2, 0),
        )
    ]

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        records = test_args.records.reshape(params["waves"] + 1, params["slots"], 32)
        owners = {}
        for row in sorted(records.reshape(-1, 32).tolist(), key=lambda row: row[8]):
            sender = row[1]
            if sender in owners:
                assert row[12] >= owners[sender], "sender queue counter moved backwards across tasks"
            owners[sender] = row[13]
        print("HBG URMA raw sender records: " + json.dumps(records.tolist()))


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaSenderLatency(TestHbgUrmaConcurrentSenders):
    CASES = [
        {
            "name": f"{shape}_{'synchronous' if synchronous else 'deferred'}",
            "platforms": ["a5"],
            "manual": True,
            "config": {"device_count": 3, "num_sub_workers": 0},
            "params": {
                "slots": 2,
                "peers": 2,
                "waves": 11,
                "repeats": 1,
                "elems": 262144,
                "independent": mode,
                "synchronous": synchronous,
            },
        }
        for shape, mode in (("spmd", 0), ("mix", 2))
        for synchronous in (False, True)
    ]
