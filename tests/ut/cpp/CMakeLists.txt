cmake_minimum_required(VERSION 3.15)
project(pto_runtime_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable profiling to avoid device_time.h dependency in hot paths
add_compile_definitions(
    PTO2_PROFILING=0
    PTO2_ORCH_PROFILING=0
    PTO2_SCHED_PROFILING=0
    PTO2_TENSORMAP_PROFILING=0
    PTO2_SPIN_VERBOSE_LOGGING=0
    _GLIBCXX_USE_CXX11_ABI=0
)

# GoogleTest: prefer system installation, fallback to FetchContent
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

# Source directories (use a2a3 as the reference arch for UT)
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TMR_RUNTIME_DIR ${PROJECT_ROOT}/src/a2a3/runtime/tensormap_and_ringbuffer/runtime)
set(TMR_ORCH_DIR ${PROJECT_ROOT}/src/a2a3/runtime/tensormap_and_ringbuffer/orchestration)
set(TMR_COMMON_DIR ${PROJECT_ROOT}/src/a2a3/runtime/tensormap_and_ringbuffer/common)
set(PLATFORM_INCLUDE_DIR ${PROJECT_ROOT}/src/a2a3/platform/include)
set(COMMON_TASK_DIR ${PROJECT_ROOT}/src/common/task_interface)

set(COMMON_INCLUDE_DIRS
    ${TMR_RUNTIME_DIR}
    ${TMR_ORCH_DIR}
    ${TMR_COMMON_DIR}
    ${PLATFORM_INCLUDE_DIR}
    ${COMMON_TASK_DIR}
)

# Determine the GTest link target name
if(TARGET GTest::gtest_main)
    set(GTEST_TARGET GTest::gtest_main)
else()
    set(GTEST_TARGET gtest_main)
endif()

# Stub sources
set(STUB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/stubs/test_stubs.cpp)

# Helper: add a test target (only if source file exists)
function(add_gtest_target name)
    cmake_parse_arguments(ARG "" "" "SOURCES;EXTRA_SOURCES" ${ARGN})
    # Check all source files exist
    foreach(src ${ARG_SOURCES})
        if(NOT IS_ABSOLUTE "${src}")
            set(src "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        endif()
        if(NOT EXISTS "${src}")
            message(STATUS "Skipping ${name}: ${src} not found")
            return()
        endif()
    endforeach()
    add_executable(${name} ${ARG_SOURCES} ${STUB_SOURCES} ${ARG_EXTRA_SOURCES})
    target_include_directories(${name} PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${name} ${GTEST_TARGET})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

# --- Header-only tests (no runtime .cpp sources needed) ---

add_gtest_target(test_submit_types SOURCES test_submit_types.cpp)
add_gtest_target(test_core_types SOURCES test_core_types.cpp)
add_gtest_target(test_tensor SOURCES test_tensor.cpp)

# --- Tests requiring runtime .cpp sources ---

add_gtest_target(test_shared_memory
    SOURCES test_shared_memory.cpp
    EXTRA_SOURCES ${TMR_RUNTIME_DIR}/pto_shared_memory.cpp
)

add_gtest_target(test_ring_buffer
    SOURCES test_ring_buffer.cpp
    EXTRA_SOURCES
        ${TMR_RUNTIME_DIR}/pto_shared_memory.cpp
        ${TMR_RUNTIME_DIR}/pto_scheduler.cpp
        ${TMR_RUNTIME_DIR}/pto_ring_buffer.cpp
)

add_gtest_target(test_tensormap
    SOURCES test_tensormap.cpp
    EXTRA_SOURCES
        ${TMR_RUNTIME_DIR}/pto_tensormap.cpp
)

add_gtest_target(test_ready_queue
    SOURCES test_ready_queue.cpp
    EXTRA_SOURCES
        ${TMR_RUNTIME_DIR}/pto_scheduler.cpp
        ${TMR_RUNTIME_DIR}/pto_shared_memory.cpp
)

add_gtest_target(test_scheduler_state
    SOURCES test_scheduler_state.cpp
    EXTRA_SOURCES
        ${TMR_RUNTIME_DIR}/pto_scheduler.cpp
        ${TMR_RUNTIME_DIR}/pto_shared_memory.cpp
)

add_gtest_target(test_pto_types SOURCES test_pto_types.cpp)

add_gtest_target(test_dispatch_payload SOURCES test_dispatch_payload.cpp)
