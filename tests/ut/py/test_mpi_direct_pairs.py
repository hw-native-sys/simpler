# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from simpler import mpi_direct_pairs


def registration(name, role, boot):
    return {
        "boot": boot,
        "registry_digest": boot + "-registry",
        "registry": "opaque",
        "descriptor": {
            "node_id": name,
            "role": role,
            "boot": boot,
            "registry": boot + "-registry",
            "healthy": True,
            "contract": {"model": "model"},
            "provider": "hixl",
            "version": 1,
        },
    }


def deployment():
    commands = []
    identities = {
        1: {"id": "p", "role": "prefill"},
        2: {"id": "d", "role": "decode"},
        3: {"id": "other", "role": "decode"},
    }
    directory = mpi_direct_pairs.PairDirectory(
        identities,
        [{"prefill": "p", "decode": "d", "device_pairs": [[0, 0]]}],
        10,
        lambda *message: commands.append(message),
    )
    for rank, node in identities.items():
        directory.status(rank, "ready", registration(node["id"], node["role"], node["id"] + "-boot"))
    return directory, commands


def states(directory):
    return {row["decode"]: row for row in directory.snapshot()["pairs"]}


def result(command, boot, **outcome):
    return {"id": command[2], "boot": boot, "registry_digest": boot + "-registry", **outcome}


def test_delivery_does_not_publish_prebuild_and_old_generation_results_are_ignored():
    directory, commands = deployment()
    (prepare_receiver,) = commands
    assert prepare_receiver[0] == 2
    assert states(directory)["other"]["state"] == "on_demand"
    assert states(directory)["d"]["state"] == "preparing"
    directory.result(2, result(prepare_receiver, "d-boot", ok=True, receiver_ready=True))
    prepare_sender = commands[-1]
    assert prepare_sender[:2] == (1, "prepare_pair")
    assert states(directory)["d"]["state"] == "preparing"

    directory.status(2, "ready", registration("d", "decode", "d-new"))
    directory.result(1, result(prepare_sender, "p-boot", ok=True, domain_bound=True, device_connected=True))
    assert states(directory)["d"]["state"] == "preparing"
    directory.result(2, result(commands[-1], "d-new", ok=True, receiver_ready=True))
    directory.result(1, result(commands[-1], "p-boot", ok=True, domain_bound=True, device_connected=True))
    ready = states(directory)["d"]
    assert ready["state"] == "prebuilt"
    assert ready["receiver_ready"] and ready["domain_bound"] and ready["device_connected"]
    directory.status(2, "disconnected", {})
    assert states(directory)["d"]["state"] == "retired"
    assert states(directory)["other"]["state"] == "on_demand"


def test_prebuild_timeout_and_partial_failure_do_not_retire_other_pair(monkeypatch):
    current = 0
    monkeypatch.setattr(mpi_direct_pairs.time, "monotonic", lambda: current)
    directory, commands = deployment()
    current = 11
    assert directory.expire()
    assert states(directory)["d"]["state"] == "failed"
    directory.status(2, "ready", registration("d", "decode", "d-new"))
    directory.result(2, result(commands[-1], "d-new", ok=True, receiver_ready=True))
    directory.result(1, result(commands[-1], "p-boot", ok=False, domain_bound=True, error="link unavailable"))
    failed = states(directory)["d"]
    assert failed["state"] == "failed" and failed["domain_bound"] and failed["device_connected"] is False
    assert states(directory)["other"]["state"] == "on_demand"
    directory.drain()
    assert all(pair["state"] == "retired" for pair in states(directory).values())


def test_decode_result_received_after_deadline_does_not_start_prefill(monkeypatch):
    current = 0
    monkeypatch.setattr(mpi_direct_pairs.time, "monotonic", lambda: current)
    directory, commands = deployment()
    (prepare_receiver,) = commands
    current = 11

    directory.result(2, result(prepare_receiver, "d-boot", ok=True, receiver_ready=True))

    assert commands == [prepare_receiver]
    assert states(directory)["d"]["state"] == "failed"
