# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L2 registration handles must not leak through scene inheritance or cross Worker/callable boundaries."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from simpler_setup import SceneTestCase, TaskArgsBuilder

scene = importlib.import_module("simpler_setup.scene_test")


class _Worker:
    def __init__(self):
        self.registered = []
        self.ran = []
        self.expected = None
        self.fail_registration = False

    def register(self, callable_obj):
        if self.fail_registration:
            raise RuntimeError("registration failed")
        handle = SimpleNamespace(owner=self, callable_obj=callable_obj)
        self.registered.append(handle)
        return handle

    def run(self, handle, args, *, config):
        assert handle.owner is self, "handle belongs to another Worker"
        assert handle.callable_obj is self.expected, "handle belongs to another callable"
        self.ran.append(handle)


@pytest.fixture
def cases(monkeypatch):
    class Parent(SceneTestCase):
        CALLABLE = {"orchestration": {"signature": []}}

        def generate_args(self, params):
            return TaskArgsBuilder()

        def compute_golden(self, args, params):
            pass

    class Child(Parent):
        CALLABLE = {"orchestration": {"signature": []}}

    monkeypatch.setattr(scene, "_child_memory_args", lambda *args: nullcontext(SimpleNamespace(tensors={})))
    monkeypatch.setattr(scene, "_build_l2_ref_args", lambda *args, **kwargs: (object(), []))
    monkeypatch.setattr(Parent, "_build_config", lambda *args, **kwargs: object())
    return Parent, Child


def _run(case, worker, callable_obj, *, rounds=1):
    worker.expected = callable_obj
    case._run_and_validate_l2(worker, callable_obj, {"name": "registration"}, rounds=rounds)


@pytest.mark.parametrize("order", [(0, 1), (1, 0), (0, 1, 0, 1)])
def test_parent_and_child_never_reuse_each_others_handle(cases, order):
    instances = [cls() for cls in cases]
    callables = [object(), object()]
    worker = _Worker()
    for index in order:
        _run(instances[index], worker, callables[index])
    assert len(worker.registered) == 2
    assert [handle.callable_obj for handle in worker.ran] == [callables[index] for index in order]


def test_repeated_cases_and_rounds_reuse_one_registration(cases):
    case = cases[0]()
    worker, callable_obj = _Worker(), object()
    _run(case, worker, callable_obj, rounds=3)
    _run(case, worker, callable_obj, rounds=2)
    assert len(worker.registered) == 1
    assert worker.ran == [worker.registered[0]] * 5


@pytest.mark.parametrize("new_instance", [False, True])
def test_replacement_worker_requires_its_own_registration(cases, new_instance):
    case = cases[0]()
    first, second, callable_obj = _Worker(), _Worker(), object()
    _run(case, first, callable_obj)
    _run(cases[0]() if new_instance else case, second, callable_obj)
    assert len(first.registered) == len(second.registered) == 1


def test_replacement_callable_requires_its_own_registration(cases):
    case = cases[0]()
    worker, first, second = _Worker(), object(), object()
    _run(case, worker, first)
    _run(case, worker, second)
    assert [handle.callable_obj for handle in worker.registered] == [first, second]


def test_registration_failure_does_not_cache_the_failed_request(cases):
    case = cases[0]()
    worker, first, second = _Worker(), object(), object()
    _run(case, worker, first)
    worker.fail_registration = True
    with pytest.raises(RuntimeError, match="registration failed"):
        _run(case, worker, second)
    worker.fail_registration = False
    _run(case, worker, first)
    _run(case, worker, second)
    assert len(worker.registered) == 2


def test_independent_instances_do_not_share_mutable_class_registration(cases):
    first, second = cases[0](), cases[0]()
    worker, first_callable, second_callable = _Worker(), object(), object()
    _run(first, worker, first_callable)
    _run(second, worker, second_callable)
    _run(first, worker, first_callable)
    assert len(worker.registered) == 2
    assert worker.ran[0] is worker.ran[2]
