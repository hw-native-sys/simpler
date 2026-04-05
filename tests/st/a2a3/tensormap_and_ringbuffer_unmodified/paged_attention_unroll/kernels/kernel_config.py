from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BASE_KERNEL_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "tensormap_and_ringbuffer"
    / "paged_attention_unroll"
    / "kernels"
    / "kernel_config.py"
)
_SPEC = spec_from_file_location("tmr_unmodified_paged_attention_unroll_kernel_config", _BASE_KERNEL_CONFIG)
_MODULE = module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

ORCHESTRATION = deepcopy(_MODULE.ORCHESTRATION)
KERNELS = deepcopy(_MODULE.KERNELS)
RUNTIME_CONFIG = {**_MODULE.RUNTIME_CONFIG, "runtime": "tensormap_and_ringbuffer_unmodified"}
