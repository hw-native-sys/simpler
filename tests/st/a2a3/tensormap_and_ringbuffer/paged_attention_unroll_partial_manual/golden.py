from pathlib import Path
import sys

_BASE = Path(__file__).resolve().parents[1] / "paged_attention_unroll"
sys.path.insert(0, str(_BASE))

from golden import ALL_CASES, ATOL, DEFAULT_CASE, RTOL, generate_inputs  # noqa: E402,F401
from paged_attention_golden import compute_golden, run_golden_test  # noqa: E402,F401
