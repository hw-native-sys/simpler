"""Shared fixtures for Python unit tests."""

import sys
from pathlib import Path

import pytest

# Ensure python/ is importable for all unit tests
PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_DIR = PROJECT_ROOT / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT
