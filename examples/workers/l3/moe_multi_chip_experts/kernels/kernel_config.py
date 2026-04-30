# Kernel and Orchestration Configuration

from pathlib import Path

_ROOT_DIR = Path(__file__).parent.parent

# Runtime configuration for tensormap_and_ringbuffer
# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3)
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 4,
	"block_dim": 24,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "kernels" / "orchestration" / "moe_multi_chip_orch.cpp"),
	"function_name": "aicpu_orchestration_entry"
}

KERNELS = [
	{"func_id": 0, "name": "moe_demo_incore_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "moe_demo_incore_0.cpp"), "core_type": "aiv"},
	{"func_id": 1, "name": "moe_demo_incore_1", "source": str(_ROOT_DIR / "kernels" / "aiv" / "moe_demo_incore_1.cpp"), "core_type": "aiv"},
	{"func_id": 2, "name": "moe_demo_incore_2", "source": str(_ROOT_DIR / "kernels" / "aiv" / "moe_demo_incore_2.cpp"), "core_type": "aiv"},
]
