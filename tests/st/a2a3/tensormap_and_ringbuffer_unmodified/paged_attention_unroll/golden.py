from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BASE_GOLDEN = (
    Path(__file__).resolve().parents[2] / "tensormap_and_ringbuffer" / "paged_attention_unroll" / "golden.py"
)
_SPEC = spec_from_file_location("tmr_unmodified_paged_attention_unroll_golden", _BASE_GOLDEN)
_MODULE = module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

ALL_CASES = _MODULE.ALL_CASES
ATOL = _MODULE.ATOL
DEFAULT_CASE = _MODULE.DEFAULT_CASE
RTOL = _MODULE.RTOL
__outputs__ = _MODULE.__outputs__
compute_golden = _MODULE.compute_golden
generate_inputs = _MODULE.generate_inputs
