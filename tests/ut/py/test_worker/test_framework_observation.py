# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual facade ownership over a recording native transport, without a device."""

import gc
import os
import threading
import weakref
from types import SimpleNamespace

import pytest
from simpler import worker as worker_module
from simpler.buffer import _DeviceBufferSource
from simpler.task_interface import (
    ArgDirection,
    ChipCallable,
    DataType,
    ExecutionContextRefused,
    TaskArgs,
    TensorTransfer,
)

from . import test_native_submission

prepared = test_native_submission.prepared
chip_fixture = test_native_submission.chip_fixture


@pytest.fixture
def transferred(prepared, monkeypatch, request):
    chip, pending = prepared
    env = chip.impl._execution_environment
    chip.impl._execution_environment = (*env[:4], True)
    state = SimpleNamespace(complete=False, transfers=0, own_waits=0, observe_rc=0, all_observed=True, observes=0)
    state.control_buffer = None
    args = None
    if getattr(request, "param", False):
        state.control_buffer = chip.worker.create_buffer(64)
        chip.callable = chip.worker.register(
            ChipCallable.build(signature=[ArgDirection.IN], func_name="host_control", binary=b"\0", children=[])
        )
        args = TaskArgs()
        args.add_tensor(state.control_buffer.tensor((4,), DataType.INT32), transfer=TensorTransfer.NONE)

    class Stream:
        raw = 123

    stream = Stream()
    handle = chip.worker.prepare_submission(chip.callable, args, caller_stream=123)
    state.run_id = handle._run_id
    native = pending[0]
    native._framework_claimed = False
    native._framework_transferred = False

    def claim():
        if native._framework_claimed or native.ready.is_set():
            raise RuntimeError("already consumed")
        native._framework_claimed = True

    def retire(wait_for_own_completion, timeout=-1):
        if native._framework_transferred:
            return True
        if wait_for_own_completion:
            state.own_waits += 1
        if not native.ready.is_set() or not state.complete:
            return False
        state.transfers += 1
        native._framework_transferred = True
        chip.live.discard(native.run)
        native.recovery_required = False
        return True

    def observe():
        state.observes += 1
        return state.observe_rc, state.all_observed

    monkeypatch.setattr(native, "_claim_framework", claim, raising=False)
    monkeypatch.setattr(native, "_retire_framework", retire, raising=False)
    monkeypatch.setattr(native, "_wait_framework", native.wait, raising=False)
    monkeypatch.setattr(chip.impl, "_observe_framework_submissions", observe, raising=False)
    identity = (env[2], env[3], 0, 123)
    handle._claim_framework_observation(stream, identity, lambda: stream.raw)
    yield chip, handle, native, state, stream, identity
    if handle._framework_source_release_unknown:
        # This fixture has no device/native carrier. Discard its deliberately
        # quarantined recording state after assertions so it cannot escape into
        # another test. This is not a product cleanup/recovery qualification.
        chip.live.clear()
        with chip.worker._registry_lock:
            for records in (
                chip.worker._chip_runs,
                chip.worker._chip_source_uses,
                chip.worker._chip_call_values,
                chip.worker._chip_run_touched_identities,
                chip.worker._chip_run_callable_ids,
            ):
                records.pop(handle._run_id, None)
        chip.worker._accepted_run_handles.discard(handle)
        worker_module._native_submission_roots.pop(handle, None)
        handle._publish_terminal(handle._error)
    state.observe_rc, state.all_observed, stream.raw = 0, True, 123
    try:
        if not handle._terminal:
            handle.cancel()
            if native.run is not None:
                state.complete = True
                handle._retire_framework()
        chip.worker._observe_framework_submissions()
    except Exception:
        pass
    # Recording transport cleanup only, after assertions; no native work exists.
    worker_module._native_submission_roots.pop(handle, None)
    if state.control_buffer is not None and not state.control_buffer.closed:
        chip.worker._chip_import_registry.host_mappings.release(state.run_id)
        chip.worker.release_buffer(state.control_buffer)


@pytest.mark.parametrize("transferred", [True], indirect=True)
@pytest.mark.parametrize("completion", ["retire", "wait", "cancel", "close"])
def test_framework_host_control_releases_uses_after_proven_completion(transferred, completion):
    chip, handle, native, state, _, _ = transferred
    buffer = state.control_buffer
    run_id = state.run_id
    registry = chip.worker._chip_import_registry.host_mappings
    assert registry._uses[run_id] == frozenset((buffer.identity,))
    assert not handle._retire_framework()
    with pytest.raises(RuntimeError, match="in-flight"):
        chip.worker.release_buffer(buffer)
    if completion == "cancel":
        assert handle.cancel()
    else:
        native.submit()
        assert not handle._retire_framework()
        assert not handle.safe_to_release and run_id in registry._uses
        state.complete = True
        if completion == "wait":
            native.recovery_required = False
            handle._wait_framework()
        elif completion == "close":
            chip.worker.close()
        else:
            assert handle._retire_framework()
    assert handle.safe_to_release and not chip.live
    assert run_id not in registry._uses
    if completion != "close":
        chip.worker.release_buffer(buffer)
        chip.worker.close()
    assert buffer.closed and chip.impl.finalized


@pytest.mark.parametrize("transferred", [True], indirect=True)
def test_framework_host_control_retains_uses_when_completion_is_unknown(transferred, monkeypatch):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    run_id = state.run_id
    registry = chip.worker._chip_import_registry.host_mappings

    def unknown(*_):
        raise RuntimeError("completion unknown")

    with monkeypatch.context() as patch:
        patch.setattr(native, "_retire_framework", unknown)
        with pytest.raises(RuntimeError, match="completion unknown"):
            handle._retire_framework()
    assert not handle.safe_to_release and run_id in registry._uses
    with pytest.raises(RuntimeError, match="in-flight"):
        chip.worker.release_buffer(state.control_buffer)
    state.complete = True
    with pytest.raises(RuntimeError, match="completion unknown"):
        handle._retire_framework()
    assert handle.safe_to_release and run_id not in registry._uses
    chip.worker.release_buffer(state.control_buffer)
    chip.worker.close()
    assert chip.impl.finalized


@pytest.mark.parametrize("transferred", [True], indirect=True)
@pytest.mark.parametrize("released_before_error", [False, True])
def test_framework_host_release_retry_does_not_repeat_device_decrement(transferred, monkeypatch, released_before_error):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    run_id = handle._run_id
    device_releases = []
    chip.worker._chip_source_uses[run_id] = ("source",)
    chip.worker._device_buffer_source = SimpleNamespace(release=device_releases.append)
    registry = chip.worker._chip_import_registry.host_mappings
    release = registry.release

    def fail_release(run):
        if released_before_error:
            release(run)
        raise RuntimeError("Host release interrupted")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(registry, "release", fail_release)
            with pytest.raises(RuntimeError, match="Host release interrupted"):
                handle._retire_framework()
        assert device_releases == [("source",)]
        assert run_id not in chip.worker._chip_source_uses
        assert run_id in chip.worker._chip_call_values and run_id in chip.worker._chip_runs
        assert not handle.safe_to_release and not native.retired
        assert not handle._framework_source_release_unknown
        with pytest.raises(RuntimeError, match="Host release interrupted"):
            handle._retire_framework()
        assert device_releases == [("source",)]
        assert handle.safe_to_release and run_id not in registry._uses
    finally:
        chip.worker._device_buffer_source = None
    chip.worker.release_buffer(state.control_buffer)
    chip.worker.close()
    assert chip.impl.finalized


def test_private_retirement_does_not_call_public_wait_and_releases_call_roots(transferred):
    chip, handle, native, state, _, _ = transferred
    assert not handle._retire_framework()
    native.submit()
    assert not handle._retire_framework()
    with pytest.raises(RuntimeError, match="publicly observed"):
        _ = handle.done
    with pytest.raises(RuntimeError, match="publicly observed"):
        handle.wait()
    state.complete = True
    assert handle._retire_framework()
    assert handle.safe_to_release and state.transfers == 1
    assert not native.entered_wait.is_set()
    assert handle not in worker_module._native_submission_roots
    assert not chip.worker._chip_runs
    assert chip.worker._framework_observation_roots
    chip.worker._observe_framework_submissions()
    assert not chip.worker._framework_observation_roots


def test_source_release_failure_keeps_records_without_replaying_an_unknown_decrement(transferred):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    run_id = handle._run_id
    chip.worker._chip_source_uses[run_id] = ("source",)
    release_calls = []

    def release(identities):
        release_calls.append(identities)
        if len(release_calls) == 1:
            raise RuntimeError("source release refused")

    chip.worker._device_buffer_source = SimpleNamespace(release=release)
    with pytest.raises(RuntimeError, match="source release refused"):
        handle._retire_framework()
    assert native._framework_transferred and not handle.safe_to_release
    assert chip.worker._chip_source_uses[run_id] == ("source",)
    assert run_id in chip.worker._chip_runs
    assert state.transfers == 1
    with pytest.raises(RuntimeError, match="source release refused"):
        handle._retire_framework()
    assert state.transfers == 1 and len(release_calls) == 1
    assert not handle.safe_to_release and run_id in chip.worker._chip_source_uses
    assert handle._framework_source_release_unknown
    # The recording leaf is now settled by fixture teardown, not a product
    # permission to retry this ambiguous source release.
    chip.worker._chip_source_uses.pop(run_id)
    chip.worker._device_buffer_source = None


def test_empty_ticket_observation_retains_unknown_pair_and_clears_fact_before_error(transferred):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    assert handle._retire_framework()
    assert not chip.worker._accepted_run_handles
    state.observe_rc, state.all_observed = -901, False
    with pytest.raises(RuntimeError, match="-901"):
        chip.worker._observe_framework_submissions()
    assert chip.worker._framework_observation_roots
    state.observe_rc, state.all_observed = -901, True
    with pytest.raises(RuntimeError, match="-901"):
        chip.worker._observe_framework_submissions()
    assert not chip.worker._framework_observation_roots
    with pytest.raises(RuntimeError, match="-901"):
        chip.worker.close()
    assert chip.impl.finalized


def test_binding_ack_cannot_drop_pending_token_stream_owner(transferred):
    chip, handle, _, state, stream, identity = transferred
    weak = weakref.ref(stream)
    chip.worker._observe_framework_submissions()
    assert state.observes == 1 and identity in chip.worker._framework_observation_roots
    gc.collect()
    assert weak() is not None
    assert handle.cancel()
    assert handle.safe_to_release
    chip.worker._observe_framework_submissions()
    assert identity not in chip.worker._framework_observation_roots


def test_changed_mapping_refuses_native_observation_and_preserves_roots(transferred):
    chip, _, _, state, stream, identity = transferred
    stream.raw = 999
    with pytest.raises(ExecutionContextRefused, match="mapping changed"):
        chip.worker._observe_framework_submissions()
    assert state.observes == 0 and identity in chip.worker._framework_observation_roots


def test_destroyed_with_cleanup_error_retires_roots_but_preserves_first_error(transferred, monkeypatch):
    _, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    retire = native._retire_framework

    def fail_after_destroy(wait_for_own_completion, timeout=-1):
        assert retire(wait_for_own_completion, timeout)
        raise RuntimeError("returned cleanup error after destruction")

    monkeypatch.setattr(native, "_retire_framework", fail_after_destroy)
    with pytest.raises(RuntimeError, match="returned cleanup error"):
        handle._retire_framework()
    assert handle.safe_to_release and native.retired
    with pytest.raises(RuntimeError, match="returned cleanup error"):
        handle._retire_framework()
    assert state.transfers == 1


@pytest.mark.parametrize("transferred", [False, True], indirect=True)
def test_actual_two_identity_partial_release_is_never_replayed(transferred):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    run_id = handle._run_id
    chip.worker._chip_source_uses[run_id] = ("first", "missing")
    # Drive the real source method: it decrements the first identity before
    # discovering the second is absent. A fake atomic refusal cannot prove this.
    source = _DeviceBufferSource.__new__(_DeviceBufferSource)
    source._lock = threading.Lock()
    source._pid = os.getpid()
    source._records = {"first": SimpleNamespace(uses=1)}
    chip.worker._device_buffer_source = source
    with pytest.raises(KeyError):
        handle._retire_framework()
    assert source._records["first"].uses == 0
    assert native._framework_transferred and handle._framework_source_release_unknown
    with pytest.raises(KeyError):
        handle._retire_framework()
    with pytest.raises(KeyError):
        handle._wait_framework()
    assert source._records["first"].uses == 0 and state.transfers == 1
    for _ in range(2):
        with pytest.raises(KeyError):
            chip.worker._close_framework()
        assert not chip.worker._close_completion._outcome.cleanup_complete
    assert source._records["first"].uses == 0 and state.observes == 0
    assert not handle.safe_to_release and handle in worker_module._native_submission_roots
    if state.control_buffer is not None:
        assert run_id in chip.worker._chip_import_registry.host_mappings._uses
        with pytest.raises(RuntimeError, match="in-flight"):
            chip.worker.release_buffer(state.control_buffer)
    chip.worker._chip_source_uses.pop(run_id)
    chip.worker._device_buffer_source = None


def test_close_performs_own_retirement_then_one_owner_observation(transferred):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    chip.worker.close()
    assert handle.safe_to_release and native.retired
    assert state.own_waits == 1 and state.observes == 1
    assert not native.entered_wait.is_set()
    assert chip.impl.finalized


def test_failed_close_does_not_retry_its_original_pair_inside_the_same_operation(transferred):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    state.observe_rc, state.all_observed = -904, False
    with pytest.raises(RuntimeError, match="-904"):
        chip.worker.close()
    assert state.observes == 1 and state.own_waits == 1
    assert not native.entered_wait.is_set()
    assert not chip.impl.finalized and chip.worker._framework_observation_roots
    assert handle.safe_to_release
    state.observe_rc, state.all_observed = 0, True
    with pytest.raises(RuntimeError, match="-904"):
        chip.worker.close()
    assert state.observes == 2 and chip.impl.finalized
    assert not chip.worker._framework_observation_roots


@pytest.mark.parametrize("fault_type", [MemoryError, KeyboardInterrupt])
def test_uncached_observer_failure_without_roots_keeps_native_journal(transferred, monkeypatch, fault_type):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    assert handle._retire_framework()
    original_observe = chip.worker._observe_framework_submissions
    original_observe()
    assert not chip.worker._framework_observation_roots
    assert chip.worker._framework_observation_error is None
    fault = fault_type("observer failed before its error was cached")

    def fail_before_cache():
        raise fault

    monkeypatch.setattr(chip.worker, "_observe_framework_submissions", fail_before_cache)
    try:
        with pytest.raises(fault_type) as failure:
            chip.worker._close_framework()
        assert failure.value is fault
        assert not chip.impl.finalized and chip.worker._chip_worker is chip.impl
        assert not chip.worker._cleanup_journal.empty
        assert not chip.worker._close_completion._outcome.cleanup_complete
        assert chip.worker._framework_observation_error is None
    finally:
        monkeypatch.setattr(chip.worker, "_observe_framework_submissions", original_observe)
    assert chip.worker._close_framework() is None
    assert chip.impl.finalized and chip.worker._cleanup_journal.empty
    assert chip.worker._close_completion._outcome.cleanup_complete


def test_private_close_separates_recovered_cleanup_from_the_same_first_error(transferred):
    chip, _, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    state.observe_rc, state.all_observed = -904, False
    with pytest.raises(RuntimeError, match="-904") as first:
        chip.worker._close_framework()
    original = first.value
    first_attempt = chip.worker._close_completion
    assert not first_attempt._outcome.cleanup_complete
    assert chip.worker._framework_observation_roots and not chip.impl.finalized
    tail = []
    chip.worker._cleanup_journal.add("domain-retirement", "confirmed tail", lambda: tail.append(True))
    state.observe_rc, state.all_observed = 0, True
    assert chip.worker._close_framework() is original
    assert chip.impl.finalized and not chip.worker._framework_observation_roots
    assert tail == [True] and chip.worker._cleanup_journal.empty
    assert chip.worker._close_completion._outcome.cleanup_complete
    assert chip.worker._close_completion is not first_attempt
    with pytest.raises(RuntimeError) as public:
        chip.worker.close()
    assert public.value is original
    assert chip.worker._close_framework() is original


def test_private_close_refuses_an_unconfirmed_domain_tail(transferred):
    chip, _, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    fault = RuntimeError("domain retirement remains unproved")
    confirmed = False

    def retire():
        if not confirmed:
            raise fault

    chip.worker._cleanup_journal.add("domain-retirement", "unconfirmed tail", retire)
    with pytest.raises(RuntimeError) as failure:
        chip.worker._close_framework()
    assert failure.value is fault
    assert chip.impl.finalized and not chip.worker._has_native_tree()
    assert not chip.worker._cleanup_journal.empty
    assert not chip.worker._close_completion._outcome.cleanup_complete
    confirmed = True
    assert chip.worker._close_framework() is None
    assert chip.worker._cleanup_journal.empty
    assert chip.worker._close_completion._outcome.cleanup_complete


def test_private_close_does_not_infer_cleanup_from_an_absent_native_tree():
    worker = worker_module.Worker(level=3, num_sub_workers=0)
    handle = worker_module.RunHandle(worker, 1, (object(),))
    worker._accepted_run_handles.add(handle)
    worker._publish_abandoned_run(handle, RuntimeError("unknown original run"))
    assert not worker._has_live_resources() and worker._abandoned_run_handles
    try:
        with pytest.raises(RuntimeError, match="cleanup.*unconfirmed"):
            worker._close_framework()
        assert worker._abandoned_run_handles == [handle]
        assert handle._keepalive is not None
        attempt = worker._close_completion
        assert attempt is not None and attempt._outcome is not None
        assert not attempt._outcome.cleanup_complete
    finally:
        # No native run exists in this fixture. This disposal is not proof
        # that an unknown production carrier can be reclaimed by another API.
        worker._abandoned_run_handles.clear()
        handle._keepalive = None


def test_private_close_pins_the_joined_attempt_when_a_real_retry_finishes_first(monkeypatch):
    worker = worker_module.Worker(level=3, num_sub_workers=0)
    fault = RuntimeError("original stream verdict")
    worker._framework_observation_error = fault
    entered = threading.Event()
    release = threading.Event()
    joiner_waiting = threading.Event()
    created = []
    nested = []
    results = []
    cleanup_calls = []
    original_attempt = worker_module._CloseAttempt

    def retire():
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            entered.set()
            assert release.wait(5)
            raise fault

    worker._cleanup_journal.add("domain-retirement", "retryable tail", retire)

    class ObservedAttempt(original_attempt):
        def __init__(self):
            super().__init__()
            created.append(self)

        @property
        def done(self):
            done = super().done
            if not done and threading.current_thread().name == "framework-close-joiner":
                joiner_waiting.set()
            return done

        def publish(self, error, incomplete, *, cleanup_complete=False):
            outcome = super().publish(error, incomplete, cleanup_complete=cleanup_complete)
            if self is created[0] and not nested:
                # The old outcome is committed. A real reentrant retry completes
                # the same owner's journal before the old callers return.
                nested.append(worker._close_framework())
            return outcome

    monkeypatch.setattr(worker_module, "_CloseAttempt", ObservedAttempt)

    def close():
        try:
            results.append(("returned", worker._close_framework()))
        except BaseException as exc:  # noqa: BLE001
            results.append(("raised", exc))

    owner = threading.Thread(target=close, name="framework-close-owner")
    joiner = threading.Thread(target=close, name="framework-close-joiner")
    owner.start()
    try:
        assert entered.wait(5)
        joiner.start()
        assert joiner_waiting.wait(5)
        release.set()
        owner.join(5)
        joiner.join(5)
        assert not owner.is_alive() and not joiner.is_alive()
        assert results == [("raised", fault), ("raised", fault)]
        assert nested == [fault] and len(created) == 2
        assert worker._close_completion is created[1]
        assert not created[0]._outcome.cleanup_complete
        assert created[1]._outcome.cleanup_complete
        assert created[0].error is created[1].error is fault
        assert worker._close_framework() is fault
        with pytest.raises(RuntimeError) as public:
            worker.close()
        assert public.value is fault
    finally:
        release.set()
        owner.join(5)
        if joiner.ident is not None:
            joiner.join(5)


def test_interrupted_framework_election_releases_only_its_own_wait(transferred, monkeypatch):
    _, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    cv = handle._cv
    interrupt = KeyboardInterrupt("interrupted after election")

    class InterruptedExit:
        def __enter__(self):
            return cv.__enter__()

        def __exit__(self, *exc):
            cv.__exit__(*exc)
            if handle._wait_in_progress:
                raise interrupt

        def notify_all(self):
            cv.notify_all()

    with monkeypatch.context() as patch:
        patch.setattr(handle, "_cv", InterruptedExit())
        try:
            with pytest.raises(KeyboardInterrupt) as failure:
                handle._retire_framework()
            assert failure.value is interrupt
            assert not handle._wait_in_progress
            assert state.transfers == 0
        finally:
            handle._wait_in_progress = False
    assert handle._retire_framework()


def test_interrupted_framework_waiter_does_not_clear_another_election(transferred, monkeypatch):
    _, handle, _, _, _, _ = transferred

    def interrupt_wait(_):
        raise KeyboardInterrupt("interrupted before election")

    with monkeypatch.context() as patch:
        patch.setattr(handle._cv, "wait", interrupt_wait)
        handle._wait_in_progress = True
        try:
            with pytest.raises(KeyboardInterrupt, match="before election"):
                handle._wait_framework(timeout=1)
            assert handle._wait_in_progress
        finally:
            handle._wait_in_progress = False


def test_framework_context_refusal_can_retry_without_caching_run_error(transferred, monkeypatch):
    _, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    refusal = ExecutionContextRefused("wrong execution context")

    def refuse(*_):
        raise refusal

    with monkeypatch.context() as patch:
        patch.setattr(native, "_retire_framework", refuse)
        with pytest.raises(ExecutionContextRefused) as failure:
            handle._retire_framework()
        assert failure.value is refusal
    assert not handle.safe_to_release
    assert handle._error is None
    assert handle._retire_framework()
    handle._wait_framework()


def test_framework_unregistration_refusal_preserves_last_handle(transferred, monkeypatch):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    assert handle._retire_framework()
    calls = []
    unregister = chip.impl.unregister_callable

    def refuse_while_binding_is_owed(cid):
        calls.append(cid)
        if chip.worker._framework_observation_roots:
            raise RuntimeError("original stream binding remains")
        unregister(cid)

    monkeypatch.setattr(chip.impl, "unregister_callable", refuse_while_binding_is_owed)
    with pytest.raises(RuntimeError, match="original stream binding remains"):
        chip.worker.unregister(chip.callable)
    handle_id, _, registration = chip.worker._coerce_handle_state(chip.callable)
    assert handle_id in chip.worker._live_handles and registration.ref_count == 1
    chip.worker._observe_framework_submissions()
    chip.worker.unregister(chip.callable)
    assert calls == [registration.slot_id, registration.slot_id]


@pytest.mark.parametrize("claim_during_observation", [False, True])
def test_binding_root_outlives_observation_racing_transfer(
    transferred, prepared, monkeypatch, claim_during_observation
):
    chip, handle, native, state, stream, identity = transferred
    native.submit()
    state.complete = True
    observe = chip.impl._observe_framework_submissions
    if claim_during_observation:
        assert handle._retire_framework()

    def observe_then_transfer():
        outcome = observe()
        if claim_during_observation:
            later = chip.worker.prepare_submission(chip.callable, caller_stream=123)
            later_native = prepared[1][-1]

            def retire_later(*_):
                assert later_native.ready.is_set()
                chip.live.discard(later_native.run)
                return True

            monkeypatch.setattr(later_native, "_framework_claimed", False, raising=False)
            monkeypatch.setattr(
                later_native,
                "_claim_framework",
                lambda: setattr(later_native, "_framework_claimed", True),
                raising=False,
            )
            monkeypatch.setattr(later_native, "_retire_framework", retire_later, raising=False)
            later._claim_framework_observation(stream, identity, lambda: stream.raw)
            later_native.submit()
            assert later._retire_framework()
        else:
            assert handle._retire_framework()
        return outcome

    with monkeypatch.context() as patch:
        patch.setattr(chip.impl, "_observe_framework_submissions", observe_then_transfer)
        chip.worker._observe_framework_submissions()
    assert not chip.worker._accepted_run_handles
    assert identity in chip.worker._framework_observation_roots
    chip.worker._observe_framework_submissions()
    assert identity not in chip.worker._framework_observation_roots


@pytest.mark.parametrize("failure_stage", ["destroyed", "poll_transient", "poll_persistent"])
def test_close_recovers_framework_carrier_and_retains_first_error(transferred, monkeypatch, failure_stage):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    primary = RuntimeError("framework transfer" if failure_stage == "destroyed" else "framework completion poll")
    retry_error = RuntimeError("later poll error")
    attempts = []
    retire = native._retire_framework
    close_lane = chip.impl._close_chip_run_lane
    allow_recovery = failure_stage != "poll_persistent"

    def retire_with_error(wait_for_own_completion, timeout):
        assert wait_for_own_completion and 0 <= timeout <= worker_module._ROLLBACK_GRACEFUL_TIMEOUT_S
        attempts.append(timeout)
        if len(attempts) == 1:
            if failure_stage == "destroyed":
                assert retire(wait_for_own_completion, timeout)
            raise primary
        if not allow_recovery:
            raise retry_error
        return retire(wait_for_own_completion, timeout)

    def close_poisoned_lane(report_run_error=True):
        close_lane()
        if report_run_error:
            raise primary

    monkeypatch.setattr(native, "_retire_framework", retire_with_error)
    monkeypatch.setattr(chip.impl, "_close_chip_run_lane", close_poisoned_lane)
    if failure_stage == "poll_persistent":
        with pytest.raises(RuntimeError) as failed:
            chip.worker._close_framework()
        assert failed.value is primary
        assert not chip.worker._close_completion._outcome.cleanup_complete
        assert not handle.safe_to_release and handle in worker_module._native_submission_roots
        assert not chip.impl.finalized and state.observes == 0
        assert len(attempts) == 2 and attempts[1] <= attempts[0]
        allow_recovery = True

    assert chip.worker._close_framework() is primary
    assert chip.worker._close_completion._outcome.cleanup_complete
    assert handle.safe_to_release and native.retired
    assert state.transfers == state.observes == 1
    assert chip.impl.finalized and not native.entered_wait.is_set()
    assert len(attempts) == {"destroyed": 1, "poll_transient": 2, "poll_persistent": 3}[failure_stage]


@pytest.mark.parametrize("recover_between_attempts", [False, True])
def test_close_retry_preserves_first_error_when_later_ticket_is_already_safe(
    transferred, prepared, monkeypatch, recover_between_attempts
):
    chip, original, native, state, stream, identity = transferred
    later = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    later_native = prepared[1][-1]
    monkeypatch.setattr(later_native, "_framework_claimed", False, raising=False)
    monkeypatch.setattr(
        later_native, "_claim_framework", lambda: setattr(later_native, "_framework_claimed", True), raising=False
    )
    later._claim_framework_observation(stream, identity, lambda: stream.raw)
    native.submit()
    later_native.submit()
    native_by_handle = {original: native, later: later_native}
    first, second = chip.worker._accepted_run_handles
    first_error = RuntimeError("first ticket poll failed")
    second_error = RuntimeError("second ticket cleanup returned an error after transfer")
    allow_first = False
    close_lane = chip.impl._close_chip_run_lane

    def close_poisoned_lane(report_run_error=True):
        close_lane()
        if report_run_error:
            raise second_error

    monkeypatch.setattr(chip.impl, "_close_chip_run_lane", close_poisoned_lane)

    def retire_carrier(item):
        item._framework_transferred = True
        chip.live.discard(item.run)
        item.recovery_required = False

    def retire_first(*_):
        if not allow_first:
            raise first_error
        retire_carrier(native_by_handle[first])
        return True

    def retire_second(*_):
        retire_carrier(native_by_handle[second])
        raise second_error

    monkeypatch.setattr(native_by_handle[first], "_retire_framework", retire_first, raising=False)
    monkeypatch.setattr(native_by_handle[second], "_retire_framework", retire_second, raising=False)
    monkeypatch.setattr(later_native, "_framework_transferred", False, raising=False)
    with pytest.raises(RuntimeError) as first_close:
        chip.worker._close_framework()
    assert first_close.value is first_error
    assert not chip.worker._close_completion._outcome.cleanup_complete
    assert not first.safe_to_release and second.safe_to_release
    assert state.observes == 0

    allow_first = True
    if recover_between_attempts:
        with pytest.raises(RuntimeError) as retired:
            first._retire_framework()
        assert retired.value is first_error and first.safe_to_release
    assert chip.worker._close_framework() is first_error
    assert chip.worker._close_completion._outcome.cleanup_complete
    assert first.safe_to_release and second.safe_to_release
    assert state.observes == 1 and chip.impl.finalized


def test_close_journal_retries_keep_selected_framework_error(transferred, monkeypatch):
    chip, handle, native, state, _, _ = transferred
    native.submit()
    state.complete = True
    primary = RuntimeError("framework transfer result")
    cleanup_errors = [RuntimeError("first native finalize refusal"), RuntimeError("second native finalize refusal")]
    finalize_calls = 0
    original_retire = native._retire_framework
    original_close_lane = chip.impl._close_chip_run_lane
    original_finalize = chip.impl.finalize

    def retire_with_error(*args):
        assert original_retire(*args)
        raise primary

    def close_poisoned_lane(report_run_error=True):
        original_close_lane()
        if report_run_error:
            raise primary

    def finalize():
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls <= len(cleanup_errors):
            raise cleanup_errors[finalize_calls - 1]
        original_finalize()

    monkeypatch.setattr(native, "_retire_framework", retire_with_error)
    monkeypatch.setattr(chip.impl, "_close_chip_run_lane", close_poisoned_lane)
    monkeypatch.setattr(chip.impl, "finalize", finalize)
    for index, cleanup_error in enumerate(cleanup_errors, start=1):
        with pytest.raises(RuntimeError) as failed:
            chip.worker._close_framework()
        assert failed.value is primary
        assert finalize_calls == index
        assert cleanup_error in chip.worker._cleanup_journal.errors
        assert not chip.worker._close_completion._outcome.cleanup_complete
        assert handle.safe_to_release and not chip.impl.finalized

    assert chip.worker._close_framework() is primary
    assert finalize_calls == 3
    assert chip.impl.finalized and chip.worker._close_completion._outcome.cleanup_complete
    assert state.observes == state.transfers == 1
