# `hello_worker/` — minimal L2 Worker lifecycle

The smallest correct program using the `Worker` API. No kernels, no data —
just prove that your venv, runtime binaries, and device/ACL setup all work.

## What it does

1. Parses `-p <platform> -d <device>`.
2. Constructs `Worker(level=2, platform=..., runtime="tensormap_and_ringbuffer", device_id=...)`.
3. Calls `init()` — this is where runtime binaries are resolved and the device is opened.
4. `malloc(4096)` + `free(ptr)` as a sanity round-trip through the mailbox.
5. `close()` in a `finally` block.

## Why each line matters

| Step | Line (`main.py`) | Why |
| ---- | ---------------- | --- |
| Construction | `Worker(level=2, platform=..., runtime=..., device_id=...)` | Stashes config; no ACL / binary loading yet. Cheap. |
| `init()` | `worker.init()` | Loads `host_runtime.so` + `aicpu.so` + `aicore.o` from `build/lib/<platform>/tensormap_and_ringbuffer/`, calls `aclrtSetDevice`. First place errors show up (wrong platform, missing binaries, device busy). |
| `malloc/free` | smoke test | Exercises the host↔device control path. If this works, a real kernel run will at least reach the dispatcher. |
| `close()` in `finally` | lifecycle end | Releases ACL + device. If you skip this, a self-hosted runner keeps the device locked and the next job hangs. |

## Run

```bash
python examples/workers/l2/hello_worker/main.py -p a2a3sim -d 0
```

Expected output (on `a2a3sim`):

```text
[hello_worker] init on a2a3sim device=0 ...
[hello_worker] malloc/free round-trip OK (ptr=0x...)
[hello_worker] close OK — lifecycle complete.
```

## Next step

Once this runs end to end, move to [`../vector_add/`](../vector_add/) — same
lifecycle, plus a real kernel compiled into a `ChipCallable` and a numpy
golden check.
