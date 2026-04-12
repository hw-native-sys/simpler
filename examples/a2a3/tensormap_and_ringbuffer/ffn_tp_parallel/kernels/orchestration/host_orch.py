"""Host-side L3 orchestration for the distributed ffn_tp_parallel example."""

from pathlib import Path

_ORCH_ROOT = Path(__file__).parent

DISTRIBUTED_TASKS = [
    {
        "name": "ffn_local",
        "source": str(_ORCH_ROOT / "ffn_local_orch.cpp"),
        "function_name": "aicpu_orchestration_entry",
        "args": ["x_shard", "w_shard", "partial_local"],
    },
    {
        "name": "allreduce_sum",
        "source": str(_ORCH_ROOT / "allreduce_sum_orch.cpp"),
        "function_name": "aicpu_orchestration_entry",
        "args": ["partial_local", "partial_window", "y", "notify_counter", "deviceCtx"],
    },
]


def distributed_orch(ctx) -> None:
    ctx.submit_task("ffn_local", outputs=["partial_local"])
    ctx.submit_task("allreduce_sum", inputs=["partial_local"], outputs=["y"])
