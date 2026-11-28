# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import socket

import pytest
from simpler.mpi_direct_service import ServiceConfig, _Controller, _read_frame


def test_service_coordination_manifest_rejects_router_launch_arguments(tmp_path):
    config = {
        "controller_host": "localhost",
        "router_port": 18421,
        "services": [
            {"id": "p", "role": "prefill", "host": "localhost", "http_url": "http://p", "argv": ["serving"]},
            {
                "id": "d",
                "role": "decode",
                "host": "localhost",
                "http_url": "http://d",
                "argv": ["serving"],
                "bind_port": 18422,
            },
        ],
        "coordination": {"router_argv": ["router"]},
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid service coordination configuration"):
        ServiceConfig.load(str(path))


@pytest.fixture
def deployment(monkeypatch):
    controller = _Controller(
        0,
        {
            "port": 0,
            "token": "router",
            "session_timeout_s": 10,
            "startup_timeout_s": 10,
            "coordination": {"prepare_timeout_s": 10, "drain_timeout_s": 10, "prebuild": []},
            "identities": [{"id": "p", "role": "prefill"}, {"id": "d", "role": "decode"}],
        },
    )
    commands = []
    monkeypatch.setattr(controller, "send", lambda *message: commands.append(message))
    try:
        yield controller, commands
    finally:
        controller.close()


def _ready(controller):
    for rank, (name, role) in enumerate((("p", "prefill"), ("d", "decode")), 1):
        descriptor = {
            "node_id": name,
            "role": role,
            "boot": name,
            "registry": name,
            "healthy": True,
            "contract": {},
            "provider": "hixl",
            "version": 1,
        }
        controller.events.put(
            (
                "mpi",
                rank,
                "status",
                "ready",
                {
                    "boot": name,
                    "registry_digest": name,
                    "registry": "opaque",
                    "descriptor": descriptor,
                },
            )
        )


def _exit(controller, rank, code):
    controller.events.put(("mpi", rank, "status", "exited", {"returncode": code}))


def test_decode_exit_stops_surviving_prefill_after_startup(deployment):
    controller, commands = deployment
    _ready(controller)
    _exit(controller, 2, 7)
    _exit(controller, 1, 0)

    assert controller.run() == 1
    assert commands == [(1, "shutdown"), (1, "finish"), (2, "finish")]


def test_abnormal_prefill_exit_does_not_authorize_decode_shutdown(deployment):
    controller, commands = deployment
    _ready(controller)
    controller.events.put(("supervisor_shutdown",))
    _exit(controller, 1, 7)
    _exit(controller, 2, 0)

    assert controller.run() == 1
    assert commands == [(1, "shutdown"), (1, "finish"), (2, "finish")]


def _read_drain(router):
    while True:
        frame = _read_frame(router)
        if frame["type"] == "drain":
            return frame


def test_reconnected_router_can_confirm_drain_by_detaching(deployment):
    controller, commands = deployment
    first_bridge, first_router = socket.socketpair()
    next_bridge, next_router = socket.socketpair()
    with first_bridge, first_router, next_bridge, next_router:
        for conn in (first_router, next_router):
            conn.settimeout(3)
        controller.events.put(("connected", first_bridge))
        _ready(controller)
        controller.events.put(("supervisor_shutdown",))
        controller.events.put(("disconnected",))
        controller.events.put(("connected", next_bridge))
        controller.events.put(("ipc", {"type": "detached", "id": "detached"}))
        _exit(controller, 1, 0)
        _exit(controller, 2, 0)
        controller.events.put(("error", AssertionError("drain stayed pending after Router detached")))

        assert controller.run() == 0
        assert commands == [(1, "shutdown"), (2, "shutdown"), (1, "finish"), (2, "finish")]
        original, replayed = _read_drain(first_router), _read_drain(next_router)
        assert replayed["id"] == original["id"]
        assert 0 < replayed["timeout_s"] <= original["timeout_s"]
