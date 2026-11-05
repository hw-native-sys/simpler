# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A causal DAG with private worker clocks and bounded publication linearization points."""

from bisect import bisect_left, bisect_right
from collections import defaultdict, deque

from .native_protocol import STAMPS, _error, _event_records


class CausalOrder:
    def __init__(self, section):
        self.edges = defaultdict(set)
        self.clocks = {worker["index"]: [0] * worker["next_sequence"] for worker in section["workers"]}
        for record, kind, _ in _event_records(section):
            for name in STAMPS[kind].split():
                cycles, sequence = record[name]
                if sequence:
                    self.clocks[record["worker_index"]][sequence - 1] = cycles
        for worker, cycles in self.clocks.items():
            self.edges[("event", worker, 0)]
            for sequence in range(1, len(cycles) + 1):
                self.add(("event", worker, sequence - 1), ("event", worker, sequence))

    @staticmethod
    def key(point):
        return ("event", point[0], point[2]) if type(point[0]) is int else point

    def add(self, left, right):
        left, right = self.key(left), self.key(right)
        if left == right:
            _error(f"self-dependent causal point {left}")
        self.edges[left].add(right)
        self.edges[right]

    def before(self, left, right, message):
        if type(left[0]) is int and type(right[0]) is int and left[0] == right[0]:
            if left[1] > right[1] or left[2] >= right[2]:
                _error(message)
        self.add(left, right)

    def bounded(self, name, left, right):
        self.before(left, right, f"invalid publication interval {name}")
        self.add(left, name)
        self.add(name, right)
        return name

    def snapshot(self, name, worker, cycles):
        # Equal cycle readings do not identify which instruction came first.
        clock = self.clocks[worker]
        lower = bisect_left(clock, cycles)
        upper = bisect_right(clock, cycles)
        self.add(("event", worker, lower), name)
        if upper < len(clock):
            self.add(name, ("event", worker, upper + 1))
        return name

    def validate(self):
        incoming = dict.fromkeys(self.edges, 0)
        for targets in self.edges.values():
            for target in targets:
                incoming[target] += 1
        ready = deque(node for node, count in incoming.items() if count == 0)
        visited = 0
        while ready:
            node = ready.popleft()
            visited += 1
            for target in self.edges[node]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
        if visited != len(incoming):
            remaining = [node for node, count in incoming.items() if count][:8]
            _error(f"contradictory causal cycle near {remaining}")
