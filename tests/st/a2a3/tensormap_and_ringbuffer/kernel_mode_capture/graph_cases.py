# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Order-sensitive device oracles; all tensor addresses and shapes stay fixed."""

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _COUNT

REPLAYS = 100
POISON = -999.0


def batch_expected(sequence, initial=0.0):
    """Two feedback graphs with independent execution counters."""
    x, y = initial, POISON
    counts = [0, 0]
    for index in sequence:
        if index == 0:
            y = x + 1.25
        else:
            assert index == 1
            x = y + 2.75
        counts[index] += 1
    return x, y, *counts


def _run_dependency_case(scenario, io, record, replay, sync, buffers, initial):
    x, y, z, count_a = buffers
    if scenario == "tmr_dag":
        graph = record([(0, (x, y), 1.25)])
        for iteration in range(REPLAYS):
            values = [v + iteration for v in initial]
            io.write(x, values)
            io.write(y, [POISON] * _COUNT)
            replay(graph)
            sync()
            io.verify(y, [2 * v + 3.75 for v in values])
    else:
        graph = record([(0, (x, y), 1.0), (1, (y, z), 2.0), (0, (z, x), 4.0), (0, (count_a, count_a), 1.0)])
        for _ in range(REPLAYS):
            replay(graph)
        sync()
        io.verify(x, initial)
        io.verify(y, [2.0 - v for v in initial])
        io.verify(z, [v - 4.0 for v in initial])
        io.verify(count_a, [float(REPLAYS)] * _COUNT)


def run_graph_case(scenario, io, record, replay, launch, sync, destroy):
    x, y, z, count_a, count_b = [io.allocate() for _ in range(5)]
    initial = [float(i % 127) for i in range(_COUNT)]
    io.write(x, initial)
    for address in (y, z):
        io.write(address, [POISON] * _COUNT)
    for address in (count_a, count_b):
        io.write(address, [0.0] * _COUNT)
    if scenario in ("tmr_dag", "multi_callable"):
        _run_dependency_case(scenario, io, record, replay, sync, (x, y, z, count_a), initial)
        return

    if scenario == "graph_recreate":
        for iteration in range(REPLAYS):
            scalar = float(iteration + 1)
            graph = record([(0, (x, y), scalar)])
            replay(graph)
            sync()
            io.verify(y, [value + scalar for value in initial])
            destroy(graph)
    elif scenario == "long_chain":
        nodes = [(0, (x, y), 1.0)]
        nodes.extend((0, (y, y), 1.0) for _ in range(15))
        nodes.append((0, (count_a, count_a), 1.0))
        graph = record(nodes)
        for _ in range(REPLAYS):
            replay(graph)
        sync()
        io.verify(y, [value + 16.0 for value in initial])
        io.verify(count_a, [float(REPLAYS)] * _COUNT)
    elif scenario == "feedback_batch":
        graph_a = record([(0, (x, y), 1.25), (0, (count_a, count_a), 1.0)])
        graph_b = record([(0, (y, x), 2.75), (0, (count_b, count_b), 1.0)])
        sequence = (0, 1) * (REPLAYS // 2)
        for index in sequence:
            replay((graph_a, graph_b)[index])
        sync()
        expected_x, expected_y, a, b = batch_expected(sequence)
        io.verify(x, [v + expected_x for v in initial])
        io.verify(y, [v + expected_y for v in initial])
        io.verify(count_a, [float(a)] * _COUNT)
        io.verify(count_b, [float(b)] * _COUNT)
    elif scenario == "eager_replay":
        graph = record([(0, (x, y), 1.25), (0, (count_a, count_a), 1.0)])
        # Eager changes addresses/scalar in the same callable's encoding cache.
        # Feedback makes replay/eager reordering observable at the final sync.
        for _ in range(REPLAYS):
            replay(graph)
            launch(0, (y, x), scalar=2.75)
        sync()
        io.verify(x, [v + 4.0 * REPLAYS for v in initial])
        io.verify(y, [v + 4.0 * (REPLAYS - 1) + 1.25 for v in initial])
        io.verify(count_a, [float(REPLAYS)] * _COUNT)
    else:
        nodes = [(0, (x, y), 1.25)]
        if scenario == "chain":
            nodes.append((0, (y, z), -3.5))
        graph = record(nodes)
        for iteration in range(REPLAYS):
            values = [v + iteration * 257 for v in initial]
            io.write(x, values)
            io.write(y, [POISON] * _COUNT)
            io.write(z, [POISON] * _COUNT)
            replay(graph)
            sync()
            io.verify(y, [v + 1.25 for v in values])
            io.verify(z, [v - 2.25 for v in values] if scenario == "chain" else [POISON] * _COUNT)
