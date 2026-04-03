from pathlib import Path
import sys

_BASE = Path(__file__).resolve().parents[1] / "paged_attention"
sys.path.insert(0, str(_BASE))

from golden import compute_golden, generate_inputs  # noqa: E402,F401
