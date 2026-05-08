import argparse
import ctypes

from simpler.task_interface import CallConfig, ContinuousTensor, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remotes", default="127.0.0.1:5050")
    args = parser.parse_args()

    result = ctypes.c_int64(0)
    endpoints = [item.strip() for item in args.remotes.split(",") if item.strip()]

    def l3_sub(task_args):
        output = task_args.tensor(1)
        current = ctypes.c_int64.from_address(int(output.data))
        current.value += int(task_args.scalar(0))

    w4 = Worker(level=4, num_sub_workers=0)
    sub_cid = w4.register(l3_sub)

    def l3_orch(orch, task_args, config):
        orch.submit_sub(sub_cid, task_args)

    l3_cid = w4.register(l3_orch)
    for endpoint in endpoints:
        w4.add_remote_worker(endpoint)
    w4.init()
    try:
        def l4_orch(orch, task_args, config):
            for value in (2, 5):
                sub_args = TaskArgs()
                sub_args.add_scalar(value)
                sub_args.add_tensor(
                    ContinuousTensor.make(ctypes.addressof(result), (1,), DataType.INT64),
                    TensorArgType.OUTPUT_EXISTING,
                )
                orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
    finally:
        w4.close()

    print(f"remote result={result.value}")
    return 0 if result.value == 7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
