import time

import pytest

from simpler.distributed.l3_daemon import L3Daemon
from simpler.distributed.remote_proxy import RemoteUnavailable
from simpler.task_interface import TaskArgs
from simpler.worker import Worker


def test_remote_proxy_marks_down_after_heartbeat_failures():
    daemon = L3Daemon(0, lambda: Worker(level=3, num_sub_workers=0))
    endpoint = f"127.0.0.1:{daemon.start()}"
    w4 = Worker(level=4, num_sub_workers=0)
    w4.register(lambda orch, args, config: None)
    w4.add_remote_worker(endpoint, heartbeat_interval=0.05, heartbeat_failures=1, heartbeat_timeout=0.05)
    w4.init()
    w4.run(lambda orch, args, config: None)
    daemon.stop()
    try:
        time.sleep(0.2)

        def l4_orch(orch, args, config):
            orch.submit_next_level(0, TaskArgs(), config)

        with pytest.raises(RuntimeError, match="unavailable|dispatch RPC failed"):
            w4.run(l4_orch)
    finally:
        w4.close()
