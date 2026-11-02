# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Behavioral regressions for sharing C++ UT production sources."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parents[1] / "cpp/cmake/ut_test.cmake"
_SOURCES = {
    "main_flag_disabled.cpp": "extern int flag_value();\nint main() { return flag_value() == 0 ? 0 : 1; }\n",
    "main_flag_enabled.cpp": "extern int flag_value();\nint main() { return flag_value() == 1 ? 0 : 1; }\n",
    "main_weak.cpp": "extern int weak_value();\nint main() { return weak_value(); }\n",
    "test_weak.cpp": "int weak_value() __attribute__((weak));\nint weak_value() { return 0; }\n",
    "main_visibility.cpp": "extern int visibility_value();\nint main() { return visibility_value(); }\n",
    "src/flag_value.cpp": "int flag_value() {\n#ifdef PRODUCTION_FLAG\nreturn 1;\n#else\nreturn 0;\n#endif\n}\n",
    "src/weak_first.cpp": "int weak_value() __attribute__((weak));\nint weak_value() { return 0; }\n",
    "src/weak_second.cpp": "int weak_value() __attribute__((weak));\nint weak_value() { return 1; }\n",
    "src/weak_production.cpp": "int weak_value() __attribute__((weak));\nint weak_value() { return 1; }\n",
    "src/archive_value.cpp": "int weak_value() { return 1; }\n",
    "src/visibility_value.cpp": "int visibility_value() { return 0; }\n",
}
_CMAKE_PROJECT = """\
cmake_minimum_required(VERSION 3.20)
project(simpler_ut_source_sharing_regression CXX)
enable_testing()

set(SIMPLER_SRC "${CMAKE_CURRENT_LIST_DIR}/src")
include("${UT_CMAKE_HELPER}")

add_executable(flag_disabled main_flag_disabled.cpp "${SIMPLER_SRC}/flag_value.cpp")
add_executable(flag_enabled main_flag_enabled.cpp "${SIMPLER_SRC}/flag_value.cpp")
set_target_properties(flag_enabled PROPERTIES COMPILE_FLAGS "-DPRODUCTION_FLAG=1")

add_executable(weak_first main_weak.cpp "${SIMPLER_SRC}/weak_first.cpp" "${SIMPLER_SRC}/weak_second.cpp")
set_source_files_properties("${SIMPLER_SRC}/weak_second.cpp" PROPERTIES COMPILE_FLAGS "-DSECOND_SOURCE_FLAG=1")

add_library(weak_archive STATIC "${SIMPLER_SRC}/archive_value.cpp")
add_executable(archive_order main_weak.cpp "${SIMPLER_SRC}/weak_first.cpp")
target_link_libraries(archive_order PRIVATE weak_archive)

add_executable(test_before_production main_weak.cpp test_weak.cpp "${SIMPLER_SRC}/weak_production.cpp")

add_executable(visibility_default main_visibility.cpp "${SIMPLER_SRC}/visibility_value.cpp")
add_executable(visibility_hidden main_visibility.cpp "${SIMPLER_SRC}/visibility_value.cpp")
set_target_properties(visibility_hidden PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

if(ENABLE_SHARING)
    set_property(GLOBAL PROPERTY SIMPLER_UT_TARGETS
                 flag_disabled flag_enabled weak_first archive_order test_before_production
                 visibility_default visibility_hidden)
    simpler_ut_share_production_sources()

    get_target_property(_flag_sources flag_enabled SOURCES)
    if(CMAKE_VERSION VERSION_LESS 3.21)
        if(NOT "${SIMPLER_SRC}/flag_value.cpp" IN_LIST _flag_sources)
            message(FATAL_ERROR "CMake 3.20 must retain the original source list")
        endif()
    else()
        if("${SIMPLER_SRC}/flag_value.cpp" IN_LIST _flag_sources)
            message(FATAL_ERROR "flag_enabled did not share its production source")
        endif()
        get_target_property(_weak_sources weak_first SOURCES)
        if(NOT "${SIMPLER_SRC}/weak_first.cpp" IN_LIST _weak_sources)
            message(FATAL_ERROR "weak_first must retain all production sources")
        endif()
        get_target_property(_archive_sources archive_order SOURCES)
        if("${SIMPLER_SRC}/weak_first.cpp" IN_LIST _archive_sources)
            message(FATAL_ERROR "archive_order did not share its production source")
        endif()
    endif()
endif()

add_test(NAME flag_disabled COMMAND flag_disabled)
add_test(NAME flag_enabled COMMAND flag_enabled)
add_test(NAME weak_first COMMAND weak_first)
add_test(NAME archive_order COMMAND archive_order)
add_test(NAME test_before_production COMMAND test_before_production)
add_test(NAME visibility_default COMMAND visibility_default)
add_test(NAME visibility_hidden COMMAND visibility_hidden)
"""


@pytest.mark.parametrize("enable_sharing", [False, True])
@pytest.mark.parametrize("generator,build_tool", [("Ninja", "ninja"), ("Unix Makefiles", "make")])
def test_production_source_sharing_preserves_flags_and_link_order(tmp_path, enable_sharing, generator, build_tool):
    if not shutil.which("cmake") or not shutil.which("c++"):
        pytest.skip("CMake and a C++ compiler are required")
    if not shutil.which(build_tool):
        pytest.skip(f"{build_tool} is required for {generator}")

    project = tmp_path / "project"
    for name, content in _SOURCES.items():
        source = project / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
    (project / "CMakeLists.txt").write_text(_CMAKE_PROJECT, encoding="utf-8")

    build = tmp_path / ("shared" if enable_sharing else "original")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(project),
            "-B",
            str(build),
            "-G",
            generator,
            f"-DUT_CMAKE_HELPER={_HELPER}",
            f"-DENABLE_SHARING={'ON' if enable_sharing else 'OFF'}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        check=True,
    )
    commands = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    visibility_commands = [entry["command"] for entry in commands if Path(entry["file"]).name == "visibility_value.cpp"]
    assert len(visibility_commands) == 2
    assert sum("-fvisibility=hidden" in command for command in visibility_commands) == 1
    assert sum("-fvisibility-inlines-hidden" in command for command in visibility_commands) == 1
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
    subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure"], check=True)
