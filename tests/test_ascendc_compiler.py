"""Tests for AscendC compiler module (python/ascendc_compiler.py)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


class TestGenerateWrapperSource:
    """Test wrapper C++ source generation."""

    def _gen(self, **kwargs):
        from ascendc_compiler import generate_wrapper_source
        return generate_wrapper_source(**kwargs)

    def test_basic_three_tensors(self):
        """Generate wrapper for a simple 3-tensor AscendC kernel (x, y, z)."""
        src = self._gen(
            ascendc_kernel_symbol="add_custom",
            tensor_args=[
                {"name": "x", "direction": "input"},
                {"name": "y", "direction": "input"},
                {"name": "z", "direction": "output"},
            ],
        )
        assert "add_custom(" in src
        assert "kernel_entry" in src
        assert "t_x" in src and "t_y" in src and "t_z" in src
        assert "gm_x" in src and "gm_y" in src and "gm_z" in src
        assert "nullptr" in src

    def test_with_static_tiling_data(self):
        """Tiling data bytes are embedded as a const array."""
        tiling = bytes([0x10, 0x27, 0x00, 0x00])
        src = self._gen(
            ascendc_kernel_symbol="my_op",
            tensor_args=[
                {"name": "a", "direction": "input"},
                {"name": "b", "direction": "output"},
            ],
            tiling_data=tiling,
        )
        assert "TILING_DATA[4]" in src
        assert "0x10" in src and "0x27" in src
        assert "(__gm__ uint8_t*)TILING_DATA" in src
        call_lines = [l for l in src.splitlines()
                      if "my_op(" in l and "extern" not in l]
        assert len(call_lines) == 1
        assert "TILING_DATA" in call_lines[0]

    def test_with_workspace(self):
        """Workspace parameter adds an extra nullptr in the call."""
        src = self._gen(
            ascendc_kernel_symbol="op",
            tensor_args=[{"name": "a", "direction": "input"}],
            has_workspace=True,
        )
        # Call should have: gm_a, nullptr (workspace), nullptr (tiling)
        call_line = [l for l in src.splitlines() if "op(" in l and "extern" not in l][0]
        assert call_line.count("nullptr") == 2

    def test_empty_tiling_treated_as_none(self):
        """Empty tiling bytes => no TILING_DATA array, nullptr passed."""
        src = self._gen(
            ascendc_kernel_symbol="op",
            tensor_args=[{"name": "a", "direction": "input"}],
            tiling_data=b"",
        )
        assert "TILING_DATA" not in src
        assert "nullptr" in src

    def test_includes_tensor_header(self):
        """Wrapper source includes tensor.h for Tensor struct access."""
        src = self._gen(
            ascendc_kernel_symbol="op",
            tensor_args=[{"name": "a", "direction": "input"}],
        )
        assert '#include "tensor.h"' in src

    def test_correct_arg_indexing(self):
        """Each tensor is unpacked from the correct args index."""
        src = self._gen(
            ascendc_kernel_symbol="op",
            tensor_args=[
                {"name": "a", "direction": "input"},
                {"name": "b", "direction": "input"},
                {"name": "c", "direction": "input"},
                {"name": "d", "direction": "output"},
            ],
        )
        assert "args[0]" in src
        assert "args[1]" in src
        assert "args[2]" in src
        assert "args[3]" in src

    def test_large_tiling_data_formatting(self):
        """Large tiling data is formatted correctly across multiple lines."""
        tiling = bytes(range(48))
        src = self._gen(
            ascendc_kernel_symbol="op",
            tensor_args=[{"name": "a", "direction": "input"}],
            tiling_data=tiling,
        )
        assert "TILING_DATA[48]" in src
        assert "0x00" in src
        assert "0x2f" in src  # last byte = 47 = 0x2f


class TestExtractKernelArtifacts:
    """Test extraction of .o and tiling data from kernel_meta directory."""

    def test_extracts_o_file(self, tmp_path):
        """Finds and reads a .o file from kernel_meta."""
        from ascendc_compiler import extract_kernel_artifacts

        meta_dir = tmp_path / "kernel_meta"
        meta_dir.mkdir()
        (meta_dir / "kernel.o").write_bytes(b"\x7fELFfake")

        kernel_o, tiling = extract_kernel_artifacts(str(meta_dir))
        assert kernel_o == b"\x7fELFfake"

    def test_extracts_tiling_bin(self, tmp_path):
        """Finds and reads a tiling .bin file."""
        from ascendc_compiler import extract_kernel_artifacts

        meta_dir = tmp_path / "kernel_meta"
        meta_dir.mkdir()
        (meta_dir / "kernel.o").write_bytes(b"obj")
        (meta_dir / "tiling_data.bin").write_bytes(b"\x01\x02\x03")

        kernel_o, tiling = extract_kernel_artifacts(str(meta_dir))
        assert tiling == b"\x01\x02\x03"

    def test_extracts_tiling_from_json(self, tmp_path):
        """Falls back to extracting tiling data from JSON metadata."""
        import json
        from ascendc_compiler import extract_kernel_artifacts

        meta_dir = tmp_path / "kernel_meta"
        meta_dir.mkdir()
        (meta_dir / "kernel.o").write_bytes(b"obj")
        meta = {"tiling_data": [0x10, 0x20, 0x30]}
        (meta_dir / "metadata.json").write_text(json.dumps(meta))

        kernel_o, tiling = extract_kernel_artifacts(str(meta_dir))
        assert tiling == bytes([0x10, 0x20, 0x30])

    def test_extracts_tiling_hex_from_json(self, tmp_path):
        """Tiling data as hex string in JSON."""
        import json
        from ascendc_compiler import extract_kernel_artifacts

        meta_dir = tmp_path / "kernel_meta"
        meta_dir.mkdir()
        (meta_dir / "kernel.o").write_bytes(b"obj")
        meta = {"tiling_data": "aabb"}
        (meta_dir / "metadata.json").write_text(json.dumps(meta))

        kernel_o, tiling = extract_kernel_artifacts(str(meta_dir))
        assert tiling == bytes([0xaa, 0xbb])

    def test_missing_dir_raises(self):
        """Raises FileNotFoundError for non-existent directory."""
        from ascendc_compiler import extract_kernel_artifacts

        with pytest.raises(FileNotFoundError):
            extract_kernel_artifacts("/nonexistent/path")

    def test_no_o_file_returns_none(self, tmp_path):
        """Returns (None, ...) when no .o file exists."""
        from ascendc_compiler import extract_kernel_artifacts

        meta_dir = tmp_path / "kernel_meta"
        meta_dir.mkdir()
        (meta_dir / "readme.txt").write_text("no object file")

        kernel_o, tiling = extract_kernel_artifacts(str(meta_dir))
        assert kernel_o is None


class TestAscendCCompilerInit:
    """Test AscendCCompiler initialization."""

    def test_sim_platform_raises_on_compile(self):
        """Simulation platforms raise RuntimeError when trying to compile."""
        from ascendc_compiler import AscendCCompiler

        compiler = AscendCCompiler(platform="a2a3sim")
        with pytest.raises(RuntimeError, match="hardware platform"):
            compiler.compile_ascendc_kernel(
                ascendc_kernel_source="/fake.cpp",
                ascendc_kernel_symbol="op",
                tensor_args=[{"name": "a", "direction": "input"}],
            )

    def test_unknown_platform_raises(self):
        """Unknown platform raises ValueError."""
        from ascendc_compiler import AscendCCompiler

        with pytest.raises(ValueError, match="Unknown platform"):
            AscendCCompiler(platform="unknown_platform")

    def test_must_provide_source_or_o(self):
        """Raises ValueError when neither source nor .o is provided."""
        from ascendc_compiler import AscendCCompiler

        compiler = AscendCCompiler(platform="a2a3sim")
        with pytest.raises(RuntimeError, match="hardware platform"):
            compiler.compile_ascendc_kernel(
                ascendc_kernel_symbol="op",
                tensor_args=[{"name": "a", "direction": "input"}],
            )


class TestAscendCToolchain:
    """Test AscendCToolchain compiler flags and include dirs."""

    @pytest.fixture(autouse=True)
    def _ensure_ascend(self):
        import env_manager
        env_manager.ensure("ASCEND_HOME_PATH")

    def test_flags_use_aicore_lang(self):
        """AscendC flags must use --cce-aicore-lang, not -x cce."""
        from toolchain import AscendCToolchain

        tc = AscendCToolchain(platform="a2a3")
        flags = tc.get_compile_flags(core_type="aiv")
        assert "--cce-aicore-lang" in flags
        assert "-x" not in flags
        assert "cce" not in flags
        assert "--cce-auto-sync" in flags

    def test_flags_contain_arch(self):
        """Flags include correct architecture for platform and core_type."""
        from toolchain import AscendCToolchain

        tc = AscendCToolchain(platform="a2a3")
        aiv_flags = tc.get_compile_flags(core_type="aiv")
        assert "--cce-aicore-arch=dav-c220-vec" in aiv_flags

        aic_flags = tc.get_compile_flags(core_type="aic")
        assert "--cce-aicore-arch=dav-c220-cube" in aic_flags

    def test_include_dirs_are_specific(self):
        """Include dirs are specific paths, not os.walk results."""
        from toolchain import AscendCToolchain

        tc = AscendCToolchain(platform="a2a3")
        dirs = tc.get_ascendc_include_dirs()
        assert len(dirs) > 0
        # All returned dirs must actually exist
        for d in dirs:
            assert Path(d).is_dir(), f"Include dir does not exist: {d}"
        # Must include key AscendC directories
        dir_basenames = [Path(d).name for d in dirs]
        assert "include" in dir_basenames  # asc/include


class TestGenerateMergedSource:
    """Test merged single-TU source generation."""

    def _gen(self, **kwargs):
        from ascendc_compiler import generate_merged_source
        return generate_merged_source(**kwargs)

    def test_basic_structure(self):
        """Merged source has kernel_entry before user source include."""
        src = self._gen(
            ascendc_kernel_source="/path/to/add_custom.cpp",
            ascendc_kernel_symbol="add_custom",
            tensor_args=[
                {"name": "x", "direction": "input"},
                {"name": "y", "direction": "input"},
                {"name": "z", "direction": "output"},
            ],
        )
        lines = src.splitlines()
        # kernel_entry must appear before the user source include
        ke_line = next(i for i, l in enumerate(lines) if "kernel_entry" in l and "extern" in l)
        inc_line = next(i for i, l in enumerate(lines) if "#include" in l and "add_custom.cpp" in l)
        assert ke_line < inc_line, "kernel_entry must be defined before user source"

    def test_global_suppressed(self):
        """__global__ is suppressed around user source include."""
        src = self._gen(
            ascendc_kernel_source="/path/to/kernel.cpp",
            ascendc_kernel_symbol="my_kernel",
            tensor_args=[{"name": "a", "direction": "input"}],
        )
        assert '#define __global__' in src
        assert '#undef __global__' in src

    def test_forward_declaration(self):
        """Forward declares user kernel without __global__."""
        src = self._gen(
            ascendc_kernel_source="/path/to/kernel.cpp",
            ascendc_kernel_symbol="my_kernel",
            tensor_args=[
                {"name": "a", "direction": "input"},
                {"name": "b", "direction": "output"},
            ],
            has_workspace=True,
        )
        # Forward decl has 4 params: a, b, workspace, tiling
        fwd_lines = [l for l in src.splitlines()
                     if "extern" in l and "my_kernel" in l and "kernel_entry" not in l]
        assert len(fwd_lines) == 1
        assert "__global__" not in fwd_lines[0]
        assert fwd_lines[0].count("uint8_t*") == 4

    def test_kernel_entry_not_global(self):
        """kernel_entry is NOT __global__ (uses standard calling convention)."""
        src = self._gen(
            ascendc_kernel_source="/path/to/kernel.cpp",
            ascendc_kernel_symbol="my_kernel",
            tensor_args=[{"name": "a", "direction": "input"}],
        )
        ke_line = [l for l in src.splitlines() if "kernel_entry" in l and "extern" in l][0]
        assert "__global__" not in ke_line
        assert "__aicore__" in ke_line

    def test_includes_kernel_operator(self):
        """Merged source includes kernel_operator.h for AscendC types."""
        src = self._gen(
            ascendc_kernel_source="/path/to/kernel.cpp",
            ascendc_kernel_symbol="my_kernel",
            tensor_args=[{"name": "a", "direction": "input"}],
        )
        assert '#include "kernel_operator.h"' in src
        assert '#include "tensor.h"' in src

    def test_with_tiling_and_workspace(self):
        """Tiling data and workspace are correctly embedded and passed."""
        tiling = bytes([0xAA, 0xBB])
        src = self._gen(
            ascendc_kernel_source="/path/to/kernel.cpp",
            ascendc_kernel_symbol="my_kernel",
            tensor_args=[{"name": "a", "direction": "input"}],
            tiling_data=tiling,
            has_workspace=True,
        )
        assert "TILING_DATA[2]" in src
        assert "0xaa" in src
        call_line = [l for l in src.splitlines()
                     if "my_kernel(" in l and "extern" not in l][0]
        assert "nullptr" in call_line  # workspace
        assert "TILING_DATA" in call_line  # tiling


class TestCodeRunnerAscendCDispatch:
    """Test that CodeRunner dispatches to AscendC compiler for compiler='ascendc' kernels."""

    def test_default_compiler_is_pto(self):
        """Kernels without 'compiler' field default to 'pto'."""
        kernel = {"func_id": 0, "source": "/fake.cpp", "core_type": "aiv"}
        assert kernel.get("compiler", "pto") == "pto"

    def test_ascendc_compiler_detected(self):
        """Kernels with compiler='ascendc' are detected."""
        kernel = {
            "func_id": 0,
            "source": "/fake.o",
            "core_type": "aiv",
            "compiler": "ascendc",
        }
        assert kernel.get("compiler", "pto") == "ascendc"
