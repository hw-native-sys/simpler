# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Completion propagation across broad and reproducible generated DAGs."""

import random

import pytest

from .helpers import make_protocol, run_task
from .trace_audit import SCHEMA, audit_trace
from .types import ContractViolation, Task


def check_complete(model):
    specs = tuple(state.spec for state in model.tasks.values())
    payload = {
        "schema": SCHEMA,
        "epoch": model.epoch,
        "producer": "reference",
        "complete": True,
        "dropped": 0,
        "events": model.events,
    }
    result = audit_trace(model.topology, specs, payload, model.epoch, expected_producer="reference")
    assert result["tasks"] == len(specs)
    assert result["executions"] == sum(t.blocks * len(t.slots) for t in specs)
    assert model.can_exit()
    assert all(state.releases == 1 for state in model.tasks.values())


@pytest.mark.parametrize("width", [2, 7, 32, 65, 128])
def test_fanout_fanin_diamond_releases_consumer_only_after_the_last_branch(width):
    tasks = [Task(0, 2, 1)]
    tasks += [Task(i, 3, (1, 2, 7)[i % 3], deps=(0,)) for i in range(1, width + 1)]
    tasks += [Task(width + 1, 1, 0, deps=tuple(range(1, width + 1)), service=True)]
    model = make_protocol(*tasks)
    run_task(model, 0)
    for ref in reversed(range(1, width + 1)):
        assert not model.ready(width + 1)
        run_task(model, ref)
    assert model.ready(width + 1)
    model.complete_service(width + 1)
    check_complete(model)


@pytest.mark.parametrize("seed", range(20))
def test_seeded_dags_preserve_all_identities_and_exactly_once_logical_completion(seed):
    rng = random.Random(seed)
    tasks = []
    for ref in range(24):
        deps = tuple(sorted(rng.sample(range(ref), min(ref, rng.randrange(4)))))
        service = ref % 7 == 0
        tasks.append(
            Task(
                ref,
                1 if service else rng.randrange(1, 8),
                0 if service else rng.choice((1, 2, 3, 5, 6, 7)),
                deps,
                service=service,
            )
        )
    model = make_protocol(*tasks)
    pending = set(range(len(tasks)))
    while pending:
        ready = [ref for ref in sorted(pending) if model.ready(ref)]
        assert ready, f"valid DAG stalled at seed={seed}, pending={sorted(pending)}"
        ref = rng.choice(ready)
        run_task(model, ref)
        pending.remove(ref)
    check_complete(model)


def test_two_pending_slots_are_not_extra_execution_positions():
    model = make_protocol(Task(0, 4, 1))
    model.admit(0)
    first = model.claim(0, 0)
    second = model.claim(0, 0)
    with pytest.raises(ContractViolation, match="PENDING_CAPACITY"):
        model.claim(0, 0)
    model.publish(first)
    model.publish(second)
    model.start(first, 0)
    with pytest.raises(ContractViolation, match="WORKER_BUSY"):
        model.start(second, 0)


def test_sync_zero_work_members_can_close_after_distinct_owners_exhaust_claims():
    model = make_protocol(Task(0, 2, 2, sync_start=True))
    model.admit(0)
    model.claim(0, 1)
    model.claim(0, 2)
    model.close(0, 4)
    model.close(0, 5)
    assert not model.state(0).done and model.state(0).completed == 0
