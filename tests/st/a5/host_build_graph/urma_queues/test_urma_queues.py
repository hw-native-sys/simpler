#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the shared Device URMA kernels and their queue/data assertions on the A5 HBG calendar."""

from simpler_setup import SceneTestCase, scene_test
from tests.st.a5.tensormap_and_ringbuffer.urma_concurrent_senders import test_urma_concurrent_senders as senders
from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion import test_urma_deferred_completion as transfers
from tests.st.a5.tensormap_and_ringbuffer.urma_multi_jetty import test_urma_multi_jetty as queues


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaSenders(senders._UrmaConcurrentSenders):
    CASES = [
        *senders.TestUrmaSamePeer.CASES,
        *senders.TestUrmaConcurrentTasks.CASES,
        *senders.TestUrmaMixSiblings.CASES,
    ]


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaTwoPeers(senders._UrmaConcurrentSenders):
    CASES = senders.TestUrmaTwoPeers.CASES


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaQueues(queues._UrmaMultiJetty):
    CASES = [
        *queues.TestUrmaMultiJettySmallRequests.CASES,
        *queues.TestUrmaMultiJettySenders.CASES,
        queues._case("below", elems=queues.CHUNK_ELEMS - queues.TILE_ELEMS),
        queues._case("exact", elems=queues.CHUNK_ELEMS),
        queues._case("above", elems=queues.CHUNK_ELEMS + queues.TILE_ELEMS),
        queues._case("uneven_put", elems=2 * queues.CHUNK_ELEMS + queues.TILE_ELEMS, put=1),
        queues._case("full_64_conditions", elems=queues.TILE_ELEMS, repeats=64),
    ]


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaRingWrap(queues._UrmaMultiJetty):
    CASES = [
        queues._case(
            "same_row_wrap",
            elems=queues.TILE_ELEMS,
            put=1,
            waves=513,
            repeats=64,
            slots=1,
            fixed_senders=True,
            wrap=True,
        )
    ]


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaSegments(transfers._UrmaRegression):
    CASES = transfers.TestUrmaGatherScatter.CASES


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
