"""Unit tests for python/bindings.py — ctypes Python↔C++ bindings."""

import ctypes
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

import env_manager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _clear_env_manager_cache():
    """Clear env_manager and bindings module state."""
    env_manager._cache.clear()
    yield
    env_manager._cache.clear()


@pytest.fixture(autouse=True)
def _reset_bindings_lib():
    """Reset the module-level _lib to None between tests."""
    import bindings
    original = bindings._lib
    bindings._lib = None
    yield
    bindings._lib = original


# =============================================================================
# RuntimeLibraryLoader tests
# =============================================================================

class TestRuntimeLibraryLoader:
    """Tests for RuntimeLibraryLoader initialization."""

    def test_missing_file_raises(self, tmp_path):
        """Non-existent library file raises FileNotFoundError."""
        from bindings import RuntimeLibraryLoader
        with pytest.raises(FileNotFoundError, match="Library not found"):
            RuntimeLibraryLoader(tmp_path / "nonexistent.so")

    def test_valid_path_loads_library(self, tmp_path):
        """Valid .so path attempts to load via CDLL."""
        fake_so = tmp_path / "fake.so"
        fake_so.touch()

        from bindings import RuntimeLibraryLoader

        with patch("bindings.CDLL") as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.return_value = mock_lib
            loader = RuntimeLibraryLoader(str(fake_so))
            assert loader.lib is mock_lib
            mock_cdll.assert_called_once()


# =============================================================================
# Runtime class tests
# =============================================================================

class TestRuntime:
    """Tests for Runtime wrapper class."""

    def _make_mock_lib(self):
        """Create a mock ctypes library."""
        lib = MagicMock()
        lib.get_runtime_size.return_value = 1024
        lib.init_runtime.return_value = 0
        lib.finalize_runtime.return_value = 0
        lib.enable_runtime_profiling.return_value = 0
        return lib

    def test_init_allocates_buffer(self):
        """Runtime __init__ allocates buffer of correct size."""
        from bindings import Runtime
        lib = self._make_mock_lib()
        rt = Runtime(lib)
        lib.get_runtime_size.assert_called_once()
        assert rt._handle is not None

    def test_return_code_checking(self):
        """Non-zero C return code raises RuntimeError."""
        from bindings import Runtime
        lib = self._make_mock_lib()
        lib.init_runtime.return_value = -1
        rt = Runtime(lib)

        with pytest.raises(RuntimeError, match="init_runtime failed"):
            rt.initialize(b"\x00" * 8, "test_func")

    def test_finalize_return_code_checking(self):
        """Non-zero finalize return code raises RuntimeError."""
        from bindings import Runtime
        lib = self._make_mock_lib()
        lib.finalize_runtime.return_value = -1
        rt = Runtime(lib)

        with pytest.raises(RuntimeError, match="finalize_runtime failed"):
            rt.finalize()

    def test_empty_kernel_binaries(self):
        """Empty kernel binaries list is handled correctly."""
        from bindings import Runtime
        lib = self._make_mock_lib()
        rt = Runtime(lib)

        # Should not raise
        rt.initialize(b"\x00" * 8, "test_func", kernel_binaries=[])
        lib.init_runtime.assert_called_once()


# =============================================================================
# Module-level function tests
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level bindings functions."""

    def test_set_device_not_loaded_raises(self):
        """set_device() without loading library raises RuntimeError."""
        from bindings import set_device
        with pytest.raises(RuntimeError, match="not loaded"):
            set_device(0)

    def test_device_malloc_not_loaded_raises(self):
        """device_malloc() without loading library raises RuntimeError."""
        from bindings import device_malloc
        with pytest.raises(RuntimeError, match="not loaded"):
            device_malloc(1024)

    def test_device_malloc_null_returns_none(self):
        """device_malloc returning NULL (0) returns None."""
        import bindings
        mock_lib = MagicMock()
        mock_lib.device_malloc.return_value = 0
        bindings._lib = mock_lib

        result = bindings.device_malloc(1024)
        assert result is None

    def test_device_malloc_valid_returns_ptr(self):
        """device_malloc returning valid address returns integer."""
        import bindings
        mock_lib = MagicMock()
        mock_lib.device_malloc.return_value = 0xDEADBEEF
        bindings._lib = mock_lib

        result = bindings.device_malloc(1024)
        assert result == 0xDEADBEEF


# =============================================================================
# bind_host_binary tests
# =============================================================================

class TestBindHostBinary:
    """Tests for bind_host_binary()."""

    def test_bytes_input_creates_temp_file(self):
        """Bytes input writes to temp file then loads."""
        import bindings

        with patch("bindings.RuntimeLibraryLoader") as MockLoader:
            mock_lib = MagicMock()
            mock_lib.get_runtime_size.return_value = 256
            MockLoader.return_value = MagicMock(lib=mock_lib)

            RuntimeClass = bindings.bind_host_binary(b"\x7FELF" + b"\x00" * 100)
            # Should return a class
            assert RuntimeClass is not None
