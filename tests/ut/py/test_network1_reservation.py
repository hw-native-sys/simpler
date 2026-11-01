# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the network CI action against host-local queue and SSH doubles."""

import importlib.util
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_ACTION = _ROOT / ".github/actions/network1-run-pytest"


def _executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


@pytest.fixture
def pair(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    binary = tmp_path / "bin"
    queue = tmp_path / "queue"
    events = tmp_path / "events.jsonl"
    control = tmp_path / "control.json"
    control.write_text("{}")
    for checkout in (local, remote):
        shutil.copytree(_ROOT / ".github", checkout / ".github")
        (checkout / ".venv/bin").mkdir(parents=True)
        (checkout / ".venv/bin/activate").write_text(f"export PATH={binary}:$PATH\n")
    (tmp_path / "set_env.sh").write_text(":\n")
    (tmp_path / "github-env").touch()
    (tmp_path / "runner").mkdir()
    (tmp_path / "runner/network1-job-start").write_text(str(time.time()))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    _executable(
        binary / "ssh",
        "import json,os,subprocess,sys\n"
        "if json.load(open(os.environ['FAKE_CONTROL'])).get('ssh_failure'):sys.exit(255)\n"
        "args=sys.argv[1:]\n"
        "while args and args[0].startswith('-'):\n"
        " args=args[2:] if args[0] in ('-o','-p') else args[1:]\n"
        "args=args[1:]\n"
        "control=json.load(open(os.environ['FAKE_CONTROL']))\n"
        "if control.get('fail_first_stop') and not control.get('stop_failed') and ' stop ' in ' '.join(args):\n"
        " control['stop_failed']=True;json.dump(control,open(os.environ['FAKE_CONTROL'],'w'));sys.exit(255)\n"
        "env={**os.environ,'FAKE_HOST':'remote'}\n"
        "sys.exit(subprocess.call(['bash','-c',' '.join(args)],env=env))\n",
    )
    _executable(
        binary / "rsync",
        "import shutil,sys\nsrc=sys.argv[-2].split(':',1)[-1]\nshutil.copytree(src,sys.argv[-1],dirs_exist_ok=True)\n",
    )
    _executable(binary / "pkill", "import sys\nsys.exit(0)\n")
    _executable(
        binary / "npu-smi",
        "import json,os,subprocess,sys,time\n"
        "c=json.load(open(os.environ['FAKE_CONTROL']))\n"
        "if os.environ.get('FAKE_HOST')=='remote':time.sleep(c.get('remote_preflight_delay',0))\n"
        "if c.get('diagnostic_child'):\n"
        " p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)\n"
        " event=json.dumps(dict(kind='diagnostic_child',pid=p.pid))+'\\n'\n"
        " with open(os.environ['FAKE_EVENTS'],'a') as f:f.write(event)\n"
        "print('memory/process snapshot')\n"
        "sys.exit(1 if os.environ.get('FAKE_HOST')=='local' and c.get('local_preflight_failure') else 0)\n",
    )
    _executable(
        binary / "task-submit",
        r"""import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['FAKE_QUEUE'])/os.environ.get('FAKE_HOST','local')
root.mkdir(parents=True,exist_ok=True)
args=sys.argv[1:]
def event(kind,**kw):
 with open(os.environ['FAKE_EVENTS'],'a') as f:f.write(json.dumps(dict(kind=kind,host=root.name,**kw))+'\n')
def read(p):return json.loads(p.read_text())
if os.environ.get('TASKQUEUE_INSIDE'):
 event('nested_queue_call',args=args);raise SystemExit('nested task-submit invocation refused')
if args in (['--version'],['--list']):
 c=json.loads(Path(os.environ['FAKE_CONTROL']).read_text())
 running=any(read(p)['status']=='running' for p in root.glob('task_*') if not p.name.endswith('.log'))
 if c.get('queue_after_failure') and running:
  event('queue_query_failure',args=args);sys.exit(7)
if args==['--version']:print('queue-double');sys.exit(0)
if args==['--list']:print('\n'.join(p.name for p in root.glob('task_*')));sys.exit(0)
if args[:1]==['--find']:
 for p in root.glob('task_*'):
  t=read(p)
  if t['status'] in ('pending','running') and args[1] in t['command']:print(p.name)
 sys.exit(0)
if args[:1] in (['--status'],['--cancel'],['--kill'],['--log']):
 p=root/args[1]
 if not p.exists():print('not_found');sys.exit(0 if args[0]=='--status' else 1)
 with open(root/'mutex','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);t=read(p)
  if args[0]=='--status':print(t['status'])
  elif args[0]=='--log':print((root/(p.name+'.log')).read_text() if (root/(p.name+'.log')).exists() else '')
  elif args[0]=='--cancel':
   event('cancel',task=p.name)
   c=json.loads(Path(os.environ['FAKE_CONTROL']).read_text())
   if c.get('cancel_race') and t['status']=='pending':
    t['delay']=0;p.write_text(json.dumps(t));sys.exit(1)
   if t['status']=='pending':t['status']='not_found';p.write_text(json.dumps(t))
   elif t['status']=='running':sys.exit(1)
  else:
   event('kill',task=p.name)
   t['stop']=True;p.write_text(json.dumps(t))
 sys.exit(0)
if args[:1]==['--device']:
 devices=args[1];limit=args[3];command=args[4]
 p=root/f'task_20261101_000000_{os.getpid()}'
 c=json.loads(Path(os.environ['FAKE_CONTROL']).read_text())
 t=dict(status='pending',command=command,devices=devices,delay=c.get(root.name+'_delay',0),stop=False)
 p.write_text(json.dumps(t));event('submit',task=p.name,command=command,devices=devices,max_time=limit)
 subprocess.Popen([sys.executable,os.environ['FAKE_QUEUE_RUNNER'],str(p)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 if c.get('drop_submit_ack') and root.name=='remote':sys.exit(7)
 print(p.name);sys.exit(0)
raise SystemExit('unsupported queue args '+repr(args))
""",
    )
    runner = tmp_path / "queue_runner.py"
    runner.write_text(
        r"""import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);root=p.parent;began=time.monotonic()
def event(kind,**kw):
 with open(os.environ['FAKE_EVENTS'],'a') as f:f.write(json.dumps(dict(kind=kind,host=root.name,task=p.name,**kw))+'\n')
while True:
 with open(root/'mutex','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);t=json.loads(p.read_text())
  if t['status']!='pending':sys.exit(0)
  if time.monotonic()-began>=t['delay']:
   t['status']='running';p.write_text(json.dumps(t));break
 time.sleep(.02)
event('grant')
with open(root/(p.name+'.log'),'w') as out:
 child=subprocess.Popen(['bash','-c',t['command']],env={**os.environ,'TASKQUEUE_TASK_ID':p.name,'TASK_DEVICE':t['devices'],'TASKQUEUE_INSIDE':'1'},stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 with open(root/'mutex','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);t=json.loads(p.read_text());t['pid']=child.pid;p.write_text(json.dumps(t))
 while child.poll() is None:
  with open(root/'mutex','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);current=json.loads(p.read_text())
  if current.get('stop'):
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
  time.sleep(.02)
 rc=child.wait()
with open(root/'mutex','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);t=json.loads(p.read_text())
 t['status']=f'completed (exit={rc})';p.write_text(json.dumps(t))
event('terminal',rc=rc)
"""
    )
    _executable(
        binary / "python",
        r"""import json,os,signal,socket,sys,time
from pathlib import Path
host=os.environ.get('FAKE_HOST','local')
def event(kind,**kw):
 with open(os.environ['FAKE_EVENTS'],'a') as f:f.write(json.dumps(dict(kind=kind,host=host,**kw))+'\n')
if not os.environ.get('TASKQUEUE_TASK_ID'):
 event('unreserved_work',argv=sys.argv);sys.exit(93)
control=json.loads(Path(os.environ['FAKE_CONTROL']).read_text())
if sys.argv[1:3]==['-m','simpler.remote_l3_worker']:
 event('daemon',task=os.environ['TASKQUEUE_TASK_ID'])
 if control.get('daemon_failure'):sys.exit(17)
 sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 sock.bind(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1])));sock.listen()
 def stopped(*_):
  signal.signal(signal.SIGTERM,signal.SIG_IGN);event('daemon_stopping')
  time.sleep(control.get('daemon_stop_delay',0));event('daemon_exit');sys.exit(0)
 signal.signal(signal.SIGTERM,stopped)
 while True:
  conn,_=sock.accept();conn.close()
elif sys.argv[1:3]==['-m','pytest']:
 files=Path(os.environ['FAKE_QUEUE']).glob('*/task_*')
 running=[json.loads(p.read_text()) for p in files if not p.name.endswith('.log')]
 if sum(t['status']=='running' for t in running)!=2:event('work_without_pair');sys.exit(94)
 event('pytest',task=os.environ['TASKQUEUE_TASK_ID'])
 if control.get('ssh_fail_after_work'):
  control['ssh_failure']=True;Path(os.environ['FAKE_CONTROL']).write_text(json.dumps(control))
 sys.exit(control.get('pytest_exit',0))
else:raise SystemExit('unexpected python '+repr(sys.argv))
""",
    )
    helper = tmp_path / "ssh.sh"
    helper.write_text('network1_ssh() { ssh "$REMOTE_TARGET" "$@"; }\nRSYNC_SSH=ssh\n')
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "FAKE_QUEUE": str(queue),
        "FAKE_EVENTS": str(events),
        "FAKE_CONTROL": str(control),
        "FAKE_QUEUE_RUNNER": str(runner),
        "FAKE_HOST": "local",
        "GITHUB_WORKSPACE": str(local),
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "RUNNER_TEMP": str(tmp_path / "runner"),
        "RUN_DIR": str(local / "output/network1-ci-42-1"),
        "REMOTE_WORKDIR": str(remote),
        "REMOTE_TARGET": "peer",
        "REMOTE_DAEMON_HOST": "127.0.0.1",
        "NETWORK1_REMOTE_HOST": "127.0.0.1",
        "NETWORK1_REMOTE_SSH_PORT": "22",
        "NETWORK1_SSH_HELPER": str(helper),
        "NETWORK1_MACHINE": "125",
        "NETWORK1_REMOTE_MACHINE": "126",
        "NETWORK1_LOCAL_DEVICES": "12,13",
        "NETWORK1_REMOTE_DEVICES": "14,15",
        "NETWORK1_L3_DAEMON_PORT": str(port),
        "NETWORK1_DAEMON_WAIT_S": "3",
        "NETWORK1_SMOKE_TIMEOUT_S": "5",
        "NETWORK1_CANN_ENV": str(tmp_path / "set_env.sh"),
        "NETWORK1_REMOTE_CANN_ENV": str(tmp_path / "set_env.sh"),
        "SIMPLER_SCHEDULER_TIMEOUT_MS": "2000",
        "SIMPLER_OP_EXECUTE_TIMEOUT_US": "3000000",
        "SIMPLER_STREAM_SYNC_TIMEOUT_MS": "6000",
        "PYTEST_ARGS": "examples tests/st --level 4 --device 12,13",
    }
    yield dict(local=local, remote=remote, env=env, events=events, control=control, queue=queue)
    for path in queue.glob("*/task_*"):
        if path.name.endswith(".log"):
            continue
        try:
            task = json.loads(path.read_text())
            if task["status"] in ("pending", "running"):
                task["stop"] = True
                task["status"] = "not_found" if task["status"] == "pending" else task["status"]
                path.write_text(json.dumps(task))
                if task.get("pid"):
                    try:
                        os.killpg(task["pid"], 9)
                    except ProcessLookupError:
                        pass
        except (OSError, ValueError):
            pass


def _run_action(pair):
    action = yaml.load((_ACTION / "action.yml").read_text(), Loader=yaml.BaseLoader)
    result = None
    for step in action["runs"]["steps"]:
        if step.get("if") == "always()":
            continue
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
            cwd=pair["local"],
            env=pair["env"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode:
            break
        for line in Path(pair["env"]["GITHUB_ENV"]).read_text().splitlines():
            key, _, value = line.partition("=")
            pair["env"][key] = value
    assert result is not None
    return result


def _events(pair):
    return [json.loads(line) for line in pair["events"].read_text().splitlines()]


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
@pytest.mark.parametrize("reverse", [False, True])
def test_action_reserves_both_hosts_before_device_work(pair, reverse):
    if reverse:
        pair["env"].update(NETWORK1_MACHINE="126", NETWORK1_REMOTE_MACHINE="125")
    result = _run_action(pair)
    assert result.returncode == 0, result.stdout + result.stderr
    events = _events(pair)
    assert not any(e["kind"] in ("unreserved_work", "work_without_pair", "nested_queue_call") for e in events)
    assert {e["host"] for e in events if e["kind"] == "grant"} == {"local", "remote"}
    assert len([e for e in events if e["kind"] == "pytest"]) == 1
    cleanup = json.loads((Path(pair["env"]["RUN_DIR"]) / "reservations/cleanup.json").read_text())
    assert cleanup["complete"]
    assert all(record["retired"] for record in cleanup["records"])
    assert (Path(pair["env"]["RUN_DIR"]) / "reservations/peer/before.log").exists()
    journal = Path(pair["env"]["RUN_DIR"]) / "reservations"
    for machine in (pair["env"]["NETWORK1_MACHINE"], pair["env"]["NETWORK1_REMOTE_MACHINE"]):
        controls = (journal / f"control-{machine}.log").read_text()
        assert controls.count("task-submit --version") >= 2
        assert controls.count("task-submit --list") >= 2
    for host in ("local", "peer"):
        for phase in ("before", "after"):
            snapshot = (journal / host / f"{phase}.log").read_text()
            assert '["npu-smi", "info"]' in snapshot and '["ps",' in snapshot
            assert not re.search(r'(?m)^\d+\.\d+ \["task-submit",', snapshot)


def _start_controller(pair, queue_wait=3, cleanup_wait=6):
    source = (
        "import sys;sys.path.insert(0,'.github/actions/network1-run-pytest');import run;"
        f"run._QUEUE_WAIT_S={queue_wait};run._CLEANUP_WAIT_S={cleanup_wait};run._POLL_S=.03;"
        "sys.exit(run.main())"
    )
    return subprocess.Popen(
        [sys.executable, "-c", source, "run"],
        cwd=pair["local"],
        env=pair["env"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _finish(process):
    try:
        stdout, stderr = process.communicate(timeout=20)
    except BaseException:
        process.kill()
        process.wait()
        raise
    return process.returncode, stdout + stderr


def _wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(0.02)
    pytest.fail("timed out waiting for the lifecycle oracle")


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
@pytest.mark.parametrize("control,expected", [({"pytest_exit": 9}, 9), ({"daemon_failure": True}, 1)])
def test_failures_retain_raw_logs_and_retire_both_tasks(pair, control, expected):
    pair["control"].write_text(json.dumps(control))
    code, output = _finish(_start_controller(pair))
    assert code == expected, output
    journal = Path(pair["env"]["RUN_DIR"]) / "reservations"
    cleanup = json.loads((journal / "cleanup.json").read_text())
    assert cleanup["complete"]
    assert all(record["retired"] for record in cleanup["records"])
    assert (journal / "peer/work.log").exists()
    if control.get("daemon_failure"):
        assert not any(event["kind"] == "pytest" for event in _events(pair))


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
def test_after_queue_query_failures_do_not_prevent_proved_pair_release(pair):
    pair["control"].write_text(json.dumps({"queue_after_failure": True}))
    code, output = _finish(_start_controller(pair))
    assert code == 0, output
    events = _events(pair)
    assert {event["host"] for event in events if event["kind"] == "queue_query_failure"} == {"local", "remote"}
    assert not any(event["kind"] == "nested_queue_call" for event in events)
    journal = Path(pair["env"]["RUN_DIR"]) / "reservations"
    cleanup = json.loads((journal / "cleanup.json").read_text())
    assert cleanup["complete"] and all(record["retired"] for record in cleanup["records"])
    timeline = [json.loads(line) for line in (journal / "timeline.jsonl").read_text().splitlines()]
    snapshots = [event for event in timeline if event["event"] == "queue_snapshot"]
    assert len(snapshots) == 4 and all(event["exit_code"] == 7 for event in snapshots)
    assert min(event["time"] for event in snapshots) >= next(
        event["time"] for event in timeline if event["event"] == "pair_drained"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
def test_queue_wait_has_its_own_budget_before_daemon_readiness(pair):
    pair["control"].write_text(json.dumps({"local_delay": 2.2}))
    pair["env"]["NETWORK1_DAEMON_WAIT_S"] = "2"
    code, output = _finish(_start_controller(pair, queue_wait=6))
    assert code == 0, output
    assert any(event["kind"] == "pytest" for event in _events(pair))


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
@pytest.mark.parametrize("race", [False, True])
def test_queue_timeout_cancels_only_the_owned_second_task_and_drains_racing_grant(pair, race):
    pair["control"].write_text(json.dumps({"remote_delay": 60, "cancel_race": race}))
    unrelated = pair["queue"] / "remote/task_19990101_000000_1"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text(json.dumps(dict(status="pending", command="unrelated-work", devices="14,15")))
    code, output = _finish(_start_controller(pair, queue_wait=1))
    assert code == 1, output
    events = _events(pair)
    owned = {e["task"] for e in events if e["kind"] == "submit"}
    assert len(owned) == 2
    assert {e["task"] for e in events if e["kind"] in ("cancel", "kill")} <= owned
    assert json.loads(unrelated.read_text())["status"] == "pending"
    assert not any(e["kind"] in ("daemon", "pytest") for e in events)
    cleanup = json.loads((Path(pair["env"]["RUN_DIR"]) / "reservations/cleanup.json").read_text())
    assert cleanup["complete"], output
    assert all(record["retired"] for record in cleanup["records"])


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
def test_neither_reservation_releases_until_delayed_peer_drain_finishes(pair):
    pair["control"].write_text(json.dumps({"daemon_stop_delay": 2}))
    process = _start_controller(pair, cleanup_wait=5)
    _wait_for(lambda: any(e["kind"] == "daemon_stopping" for e in _events(pair)))
    journal = Path(pair["env"]["RUN_DIR"]) / "reservations"
    _wait_for(lambda: (journal / "local/drained.json").exists())
    assert not any(e["kind"] == "terminal" for e in _events(pair))
    assert not (journal / "local/release").exists()
    code, output = _finish(process)
    assert code == 0, output
    events = _events(pair)
    peer_exit = next(i for i, e in enumerate(events) if e["kind"] == "daemon_exit")
    assert all(i > peer_exit for i, e in enumerate(events) if e["kind"] == "terminal")


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
def test_transient_stop_failure_releases_after_fresh_pair_drain_proof(pair):
    pair["control"].write_text(json.dumps({"fail_first_stop": True}))
    code, output = _finish(_start_controller(pair, cleanup_wait=6))
    assert code == 0, output
    journal = Path(pair["env"]["RUN_DIR"]) / "reservations"
    timeline = [json.loads(line) for line in (journal / "timeline.jsonl").read_text().splitlines()]
    assert any(event["event"] == "stop_request_failed" for event in timeline)
    assert json.loads((journal / "cleanup.json").read_text())["complete"]
    assert len([event for event in _events(pair) if event["kind"] == "terminal"]) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="owned listener attribution inspects Linux /proc")
def test_foreign_listener_never_counts_as_owned_daemon_readiness(pair):
    pair["control"].write_text(json.dumps({"remote_preflight_delay": 1}))
    pair["env"]["NETWORK1_DAEMON_WAIT_S"] = "5"
    with socket.socket() as foreign:
        foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        foreign.bind(("127.0.0.1", int(pair["env"]["NETWORK1_L3_DAEMON_PORT"])))
        foreign.listen()
        code, output = _finish(_start_controller(pair, cleanup_wait=6))
        assert code == 1, output
        assert not any(e["kind"] == "pytest" for e in _events(pair))
        assert foreign.getsockname()[1] == int(pair["env"]["NETWORK1_L3_DAEMON_PORT"])
        with socket.create_connection(foreign.getsockname(), timeout=1):
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation worker preflight inspects Linux /proc")
def test_local_preflight_failure_is_prompt_and_never_runs_pytest(pair):
    pair["control"].write_text(json.dumps({"local_preflight_failure": True}))
    before = time.monotonic()
    code, output = _finish(_start_controller(pair, cleanup_wait=6))
    assert code == 1, output
    assert time.monotonic() - before < 15
    assert not any(e["kind"] == "pytest" for e in _events(pair))


@pytest.mark.skipif(sys.platform != "linux", reason="diagnostic descendant drain inspects Linux /proc")
def test_after_snapshot_descendants_drain_before_either_reservation_releases(pair):
    pair["control"].write_text(json.dumps({"diagnostic_child": True}))
    code, output = _finish(_start_controller(pair, cleanup_wait=6))
    assert code == 0, output
    pids = [e["pid"] for e in _events(pair) if e["kind"] == "diagnostic_child"]
    assert len(pids) == 4
    for pid in pids:
        try:
            status = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        except FileNotFoundError:
            continue
        assert status == "Z", f"diagnostic process {pid} survived retirement"


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
def test_unreachable_peer_keeps_unknown_ownership_and_staging_pointers(pair):
    pair["control"].write_text(json.dumps({"ssh_fail_after_work": True}))
    code, output = _finish(_start_controller(pair, cleanup_wait=1))
    assert code == 1, output
    cleanup = json.loads((Path(pair["env"]["RUN_DIR"]) / "reservations/cleanup.json").read_text())
    assert not cleanup["complete"]
    assert cleanup["remote_workdir"] == str(pair["remote"])
    assert pair["remote"].exists()
    assert not any(e["kind"] == "terminal" for e in _events(pair))


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
def test_lost_submit_ack_recovers_and_cancels_owned_queue_entry(pair):
    pair["control"].write_text(json.dumps({"remote_delay": 60, "drop_submit_ack": True}))
    code, output = _finish(_start_controller(pair))
    assert code == 1, output
    events = _events(pair)
    peer = next(e["task"] for e in events if e["kind"] == "submit" and e["host"] == "remote")
    assert [e["task"] for e in events if e["kind"] == "cancel"] == [peer]
    cleanup = json.loads((Path(pair["env"]["RUN_DIR"]) / "reservations/cleanup.json").read_text())
    assert cleanup["complete"], output
    assert all(record["retired"] for record in cleanup["records"])


@pytest.mark.skipif(sys.platform != "linux", reason="real reservation workers inspect Linux /proc")
def test_coordinator_cancellation_does_not_leave_queued_peer(pair):
    pair["control"].write_text(json.dumps({"remote_delay": 60}))
    process = _start_controller(pair, queue_wait=10)
    _wait_for(lambda: len([e for e in _events(pair) if e["kind"] == "submit"]) == 2)
    process.terminate()
    code, output = _finish(process)
    assert code == 1, output
    cleanup = json.loads((Path(pair["env"]["RUN_DIR"]) / "reservations/cleanup.json").read_text())
    assert cleanup["complete"], output
    assert all(record["retired"] for record in cleanup["records"])
    assert not any(e["kind"] == "pytest" for e in _events(pair))


def _load_helpers(monkeypatch):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    worker = load("_network1_test_reservation", _ACTION / "reservation.py")
    monkeypatch.setitem(sys.modules, "reservation", worker)
    coordinator = load("_network1_test_run", _ACTION / "run.py")
    return worker, coordinator


def test_insufficient_job_budget_submits_no_tasks(pair, monkeypatch):
    _, coordinator = _load_helpers(monkeypatch)
    for key, value in pair["env"].items():
        monkeypatch.setenv(key, value)
    (Path(pair["env"]["RUNNER_TEMP"]) / "network1-job-start").write_text(str(time.time() - 3590))
    journal = Path(pair["env"]["RUN_DIR"]) / "reservations"
    journal.mkdir(parents=True)
    with pytest.raises(TimeoutError, match="insufficient"):
        coordinator.configure(journal, pair["env"])
    assert not pair["events"].exists()


@pytest.mark.parametrize("status", ["pending", "running", "completed (exit=17)", "not_found"])
def test_status_uses_exact_owned_id_through_real_command_transport(pair, monkeypatch, status):
    _, coordinator = _load_helpers(monkeypatch)
    for key, value in pair["env"].items():
        monkeypatch.setenv(key, value)
    queue = pair["queue"] / "remote"
    queue.mkdir(parents=True)
    task = "task_20261101_000000_123"
    (queue / task).write_text(json.dumps(dict(status=status, command="owned")))
    record = dict(
        machine="126", cwd=str(pair["remote"]), state=str(pair["remote"] / "state"), remote=True, task_id=task
    )
    host = coordinator.Host(pair["local"], record)
    assert host.status() == status


def test_expired_worker_receipt_recovers_lost_ack_and_retires_without_claiming_drain(pair, monkeypatch):
    worker, coordinator = _load_helpers(monkeypatch)
    for key, value in pair["env"].items():
        monkeypatch.setenv(key, value)
    task = "task_20261101_000000_123"
    queue = pair["queue"] / "remote"
    queue.mkdir(parents=True)
    (queue / task).write_text(json.dumps(dict(status="completed (exit=1)", command="owned")))
    state = pair["remote"] / "state"
    state.mkdir()
    for name in ("rejected", "closed"):
        worker.write_json(state / f"{name}.json", dict(task_id=task, reason="admission expired"))
    record = dict(machine="126", cwd=str(pair["remote"]), state=str(state), remote=True, intent=True, key="owned")
    host = coordinator.Host(pair["local"], record)
    coordinator.await_drained(host, time.monotonic() + 1)
    assert record["task_id"] == task
    assert record["retired"] and not record["drained"]
    assert not (state / "granted.json").exists() and not (state / "drained.json").exists()
    assert not pair["events"].exists()


def test_cleanup_gate_blocks_accepted_spawn_then_rejects_late_rank(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    state = tmp_path / "state"
    state.mkdir()
    (state / "gate").touch()
    (state / "start").touch()
    monkeypatch.setattr(
        worker,
        "live_grant",
        lambda _: dict(uid=os.geteuid(), environment={"TASKQUEUE_TASK_ID": "task_1_2", "TASK_DEVICE": "14,15"}),
    )
    accepted = threading.Event()
    spawn = threading.Event()
    closing_finished = threading.Event()
    launched = []

    def popen(argv, env):
        accepted.set()
        assert spawn.wait(5)
        launched.append(env)
        return type("Child", (), {"wait": lambda self: 0})()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    launch = threading.Thread(target=worker.rank, args=(state, ["rank"]))
    launch.start()
    assert accepted.wait(5)

    def close():
        worker.control(state, "stop")
        closing_finished.set()

    closing = threading.Thread(target=close)
    closing.start()
    assert not closing_finished.wait(0.1)
    assert not (state / "closing").exists()
    spawn.set()
    launch.join(5)
    closing.join(5)
    assert not launch.is_alive() and not closing.is_alive()
    assert launched[0]["TASKQUEUE_TASK_ID"] == "task_1_2"
    with pytest.raises(RuntimeError, match="active reservation"):
        worker.rank(state, ["late-rank"])
    assert len(launched) == 1


def test_rank_without_live_grant_never_spawns(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    (tmp_path / "gate").touch()
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("rank spawned without a grant"))
    with pytest.raises(FileNotFoundError):
        worker.rank(tmp_path, ["rank"])
    worker.write_json(tmp_path / "granted.json", dict(task_id="task_1_2", pid=1, start="1"))
    monkeypatch.setattr(worker, "process_info", lambda _: ("Z", 0, "1"))
    with pytest.raises(RuntimeError, match="not live"):
        worker.rank(tmp_path, ["rank"])


@pytest.mark.skipif(sys.platform != "linux", reason="real marker drain inspects Linux /proc")
def test_accepted_independent_rank_and_setsid_grandchild_drain_before_close(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    state = tmp_path / "state"
    state.mkdir()
    (state / "gate").touch()
    (state / "start").touch()
    task_id = f"task_1_{os.getpid()}"
    grant = dict(
        task_id=task_id,
        pid=os.getpid(),
        uid=os.geteuid(),
        start=worker.process_info(os.getpid())[2],
        environment=dict(TASKQUEUE_TASK_ID=task_id, TASK_DEVICE="14,15"),
    )
    worker.write_json(state / "granted.json", grant)
    actual_popen = subprocess.Popen
    unrelated = actual_popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    accepted = threading.Event()
    proceed = threading.Event()
    finished = threading.Event()
    children = []
    errors = []

    def paused(argv, **kwargs):
        accepted.set()
        assert proceed.wait(5)
        child = actual_popen(argv, **kwargs)
        children.append(child)
        observed = _wait_for(lambda: json.loads(pid_file.read_text()))
        assert all(worker.process_info(pid)[0] != "Z" for pid in observed)
        assert {pid for pid, _ in worker.marked_processes(task_id, {os.getpid()}, grant["start"])} >= set(observed)
        return child

    def close():
        try:
            worker.control(state, "stop")
            worker.drain(state, task_id, 5)
            finished.set()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(worker.subprocess, "Popen", paused)
    pid_file = tmp_path / "grandchild.json"
    child_source = (
        "import json,os,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);"
        f"open({str(pid_file)!r},'w').write(json.dumps([os.getpid(),p.pid]));"
        "time.sleep(60)"
    )
    launching = threading.Thread(target=worker.rank, args=(state, [sys.executable, "-c", child_source]))
    closing = threading.Thread(target=close)
    try:
        launching.start()
        assert accepted.wait(5)
        closing.start()
        assert not finished.wait(0.1)
        proceed.set()
        # The spawn is accepted under the shared gate. Its descendants are found
        # by the queue marker even after changing session and parent process.
        closing.join(7)
        launching.join(7)
        assert not errors
        assert finished.is_set() and not launching.is_alive()
        assert children[0].poll() is not None
        assert worker.marked_processes(task_id, {os.getpid()}, grant["start"]) == []
        assert unrelated.poll() is None
    finally:
        proceed.set()
        unrelated.kill()
        unrelated.wait()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
        for pid, start in worker.marked_processes(task_id, {os.getpid()}, grant["start"]):
            worker.signal_owned(pid, start)


@pytest.mark.parametrize("birth", ["100", "101"])
def test_same_uid_unreadable_process_is_unknown_not_drained(tmp_path, monkeypatch, birth):
    worker, _ = _load_helpers(monkeypatch)
    (tmp_path / "123").mkdir()
    effective_uid = (tmp_path / "123").stat().st_uid + 1
    (tmp_path / "123/status").write_text(f"Uid:\t0\t{effective_uid}\t0\t0\n")
    monkeypatch.setattr(worker, "_PROC", tmp_path)
    monkeypatch.setattr(worker.os, "geteuid", lambda: effective_uid)
    monkeypatch.setattr(worker, "process_info", lambda _: ("S", 1, birth))

    def unreadable(_):
        raise PermissionError("owned environment denied")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(RuntimeError, match="ownership is unreadable"):
        worker.marked_processes("task_1_2", set(), "100")


def test_unreadable_same_uid_process_born_before_holder_is_not_potential_work(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    (tmp_path / "123").mkdir()
    monkeypatch.setattr(worker, "_PROC", tmp_path)
    monkeypatch.setattr(worker, "process_info", lambda _: ("S", 1, "99"))
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("old process environment was read"))
    assert worker.marked_processes("task_1_2", set(), "100") == []


def test_reused_process_identity_is_never_signalled(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    descriptor = os.open(tmp_path / "descriptor", os.O_CREAT | os.O_RDWR)
    monkeypatch.setattr(worker.os, "pidfd_open", lambda _: descriptor, raising=False)
    monkeypatch.setattr(worker, "process_info", lambda _: ("S", 1, "new-birth"))
    monkeypatch.setattr(
        worker.signal, "pidfd_send_signal", lambda *_: pytest.fail("reused PID was signalled"), raising=False
    )
    worker.signal_owned(123, "old-birth")


def test_known_owned_handle_does_not_reread_exiting_process_environment(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    descriptor = os.open(tmp_path / "descriptor", os.O_CREAT | os.O_RDWR)
    monkeypatch.setattr(worker.os, "pidfd_open", lambda _: descriptor, raising=False)
    monkeypatch.setattr(worker, "process_info", lambda _: ("S", 1, "100"))
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("known owned environment was reread"))
    sent = []
    monkeypatch.setattr(worker.signal, "pidfd_send_signal", lambda *args: sent.append(args), raising=False)
    worker.signal_owned(123, "100")
    assert sent == [(descriptor, worker.signal.SIGTERM)]


@pytest.mark.parametrize("persistent", [False, True])
def test_drain_retries_unknown_scan_but_never_accepts_persistent_unknown(tmp_path, monkeypatch, persistent):
    worker, _ = _load_helpers(monkeypatch)
    monkeypatch.setattr(worker, "process_info", lambda _: ("S", 1, "100"))
    scans = []

    def scan(*_):
        scans.append(1)
        if persistent or len(scans) == 1:
            raise RuntimeError("process ownership is unreadable for PID 123")
        return []

    monkeypatch.setattr(worker, "marked_processes", scan)
    if persistent:
        with pytest.raises(RuntimeError, match="ownership is unreadable"):
            worker.drain(tmp_path, "task_1_2", 0.1)
    else:
        worker.drain(tmp_path, "task_1_2", 1)
    assert len(scans) >= 2
    assert not (tmp_path / "drained.json").exists()


@pytest.mark.parametrize("condition", ["fresh", "expired", "different_boot"])
def test_configured_admission_is_checked_by_worker_before_grant_or_diagnostics(tmp_path, monkeypatch, condition):
    worker, _ = _load_helpers(monkeypatch)
    monkeypatch.setattr(worker.time, "monotonic", lambda: 463.963392822)
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    boot = proc / "sys/kernel/random/boot_id"
    boot.write_text("same-host-boot\n")
    monkeypatch.setattr(worker, "_PROC", proc)
    state = tmp_path / "state"
    config = dict(devices="14,15", admission_seconds=60, hold_seconds=0, cleanup_seconds=0)
    monkeypatch.setattr(worker.sys, "argv", ["reservation.py", "configure", str(state)])
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps(config)))
    assert worker.main() == 0
    configured = json.loads((state / "config.json").read_text())
    assert configured["boot_id"] == "same-host-boot"
    assert configured["admission_expires"] == configured["admission_started"] + 60
    if condition == "expired":
        configured["admission_expires"] = configured["admission_started"]
        worker.write_json(state / "config.json", configured)
    elif condition == "different_boot":
        boot.write_text("new-host-boot\n")
    monkeypatch.setenv("TASKQUEUE_TASK_ID", "task_1_2")
    monkeypatch.setenv("TASK_DEVICE", "14,15")
    probes = []
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(worker, "check_process_handles", lambda: probes.append("handles"))
    monkeypatch.setattr(worker, "process_info", lambda _: ("S", 1, "100"))
    monkeypatch.setattr(worker, "snapshot", lambda *_a, **_kw: probes.append("diagnostics") or True)
    monkeypatch.setattr(worker, "drain", lambda *_: None)
    (state / "release").touch()
    if condition == "fresh":
        with pytest.raises(TimeoutError, match="lifetime expired"):
            worker.worker(state)
        assert probes == ["handles", "diagnostics", "diagnostics"]
        assert (state / "granted.json").exists()
        assert "rejected" not in worker.read_state(state)
    else:
        assert worker.worker(state) == 1
        assert probes == []
        values = worker.read_state(state)
        assert values["rejected"]["task_id"] == values["closed"]["task_id"] == "task_1_2"
        assert "granted" not in values and "drained" not in values
        assert not (state / "closing").exists()


def test_missing_pidfd_support_fails_before_work(monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    monkeypatch.delattr(worker.os, "pidfd_open", raising=False)
    with pytest.raises(RuntimeError, match="require Linux pidfd"):
        worker.check_process_handles()


def test_reader_accepts_worker_ready_receipt_without_privileged_socket_queries(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    worker.write_json(tmp_path / "granted.json", dict(task_id="task_1_2", uid=0, pid=500, start="77"))
    worker.write_json(
        tmp_path / "ready.json", dict(task_id="task_1_2", pid=123, start="100", address="127.0.0.1", port=19073)
    )
    monkeypatch.setattr(worker, "process_info", lambda pid: ("S", 1, "77" if pid == 500 else "100"))
    monkeypatch.setattr(worker, "owns_listener", lambda *_: pytest.fail("unprivileged reader queried socket ownership"))
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("unprivileged reader queried environment"))
    assert worker.read_state(tmp_path)["ready"]["task_id"] == "task_1_2"
    monkeypatch.setattr(worker, "process_info", lambda pid: ("S", 1, "77" if pid == 500 else "different-birth"))
    assert "ready" not in worker.read_state(tmp_path)
    monkeypatch.setattr(worker, "process_info", lambda pid: ("S", 1, "77" if pid == 500 else "100"))
    worker.write_json(tmp_path / "ready.json", dict(task_id="task_2_3", pid=123, start="100"))
    assert "ready" not in worker.read_state(tmp_path)


def test_ready_child_is_rejected_when_its_reservation_holder_dies_or_is_reused(tmp_path, monkeypatch):
    worker, _ = _load_helpers(monkeypatch)
    worker.write_json(tmp_path / "granted.json", dict(task_id="task_1_2", uid=0, pid=500, start="77"))
    worker.write_json(tmp_path / "ready.json", dict(task_id="task_1_2", pid=123, start="100"))
    for holder_status, holder_birth in (("Z", "77"), ("S", "reused-birth")):
        monkeypatch.setattr(
            worker,
            "process_info",
            lambda pid: (holder_status, 1, holder_birth) if pid == 500 else ("S", 1, "100"),
        )
        assert "ready" not in worker.read_state(tmp_path)
    monkeypatch.setattr(worker, "process_info", lambda pid: ("S", 1, "77" if pid == 500 else "100"))
    (tmp_path / "closing").touch()
    assert "ready" not in worker.read_state(tmp_path)


@pytest.mark.parametrize("complete", [False, True])
def test_teardown_removes_only_after_pair_retirement_receipt(pair, complete):
    journal = Path(pair["env"]["RUN_DIR"]) / "reservations"
    journal.mkdir(parents=True)
    (journal / "cleanup.json").write_text(json.dumps(dict(complete=complete)))
    action = yaml.load((_ROOT / ".github/actions/network1-teardown/action.yml").read_text(), Loader=yaml.BaseLoader)
    result = subprocess.run(
        ["bash", "-e", "-c", action["runs"]["steps"][0]["run"]],
        env=pair["env"],
        cwd=pair["local"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) == complete, result.stdout + result.stderr
    assert pair["remote"].exists() != complete
