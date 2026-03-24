cmake_minimum_required(VERSION 3.14)
project(simpler_unit_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# GoogleTest: prefer system installation, fallback to FetchContent
# =============================================================================
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(GTEST_LIBS gtest_main)
else()
    set(GTEST_LIBS GTest::gtest_main)
    # System GoogleTest may use pre-cxx11 ABI; match it
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES ${GTEST_LIBS})
    check_cxx_source_compiles("
        #include <gtest/gtest.h>
        TEST(ABI, Check) { EXPECT_EQ(1,1); }
        int main(int argc, char** argv) { testing::InitGoogleTest(&argc, argv); return 0; }
    " GTEST_ABI_OK)
    if(NOT GTEST_ABI_OK)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

enable_testing()

# =============================================================================
# Project paths
# =============================================================================
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR}/../..)
set(A2A3_PLATFORM_INCLUDE ${PROJECT_ROOT}/src/a2a3/platform/include)
set(HOST_BUILD_GRAPH_RUNTIME ${PROJECT_ROOT}/src/a2a3/runtime/host_build_graph/runtime)
set(TMR_RUNTIME ${PROJECT_ROOT}/src/a2a3/runtime/tensormap_and_ringbuffer/runtime)

# =============================================================================
# Stub library (provides unified_log_*, common.h stubs for host testing)
# =============================================================================
add_library(test_stubs STATIC test_stubs.cpp)
target_include_directories(test_stubs PUBLIC
    ${A2A3_PLATFORM_INCLUDE}
    ${A2A3_PLATFORM_INCLUDE}/common
    ${A2A3_PLATFORM_INCLUDE}/host
)

# Common include paths for tensormap_and_ringbuffer runtime tests
set(TMR_INCLUDE_DIRS
    ${TMR_RUNTIME}
    ${A2A3_PLATFORM_INCLUDE}
    ${A2A3_PLATFORM_INCLUDE}/common
    ${A2A3_PLATFORM_INCLUDE}/host
    ${A2A3_PLATFORM_INCLUDE}/aicpu
)
set(TMR_COMPILE_DEFS PTO2_UNIT_TEST=1 NDEBUG)

# =============================================================================
# Test: Runtime Graph (host_build_graph)
# =============================================================================
add_executable(test_runtime_graph test_runtime_graph.cpp
    ${HOST_BUILD_GRAPH_RUNTIME}/runtime.cpp)
target_include_directories(test_runtime_graph PRIVATE
    ${HOST_BUILD_GRAPH_RUNTIME}
    ${A2A3_PLATFORM_INCLUDE}
    ${A2A3_PLATFORM_INCLUDE}/common
    ${A2A3_PLATFORM_INCLUDE}/host
)
target_compile_definitions(test_runtime_graph PRIVATE PTO2_UNIT_TEST=1)
target_link_libraries(test_runtime_graph ${GTEST_LIBS} test_stubs)
add_test(NAME RuntimeGraph COMMAND test_runtime_graph)

# =============================================================================
# Test: Handshake Protocol (platform_config.h macros)
# =============================================================================
add_executable(test_handshake test_handshake.cpp)
target_include_directories(test_handshake PRIVATE
    ${A2A3_PLATFORM_INCLUDE}
    ${A2A3_PLATFORM_INCLUDE}/common
)
target_compile_definitions(test_handshake PRIVATE PTO2_UNIT_TEST=1)
target_link_libraries(test_handshake ${GTEST_LIBS})
add_test(NAME Handshake COMMAND test_handshake)

# =============================================================================
# Test: HeapRing (ring buffer allocation)
# =============================================================================
add_executable(test_heap_ring test_heap_ring.cpp ${TMR_RUNTIME}/pto_ring_buffer.cpp)
target_include_directories(test_heap_ring PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_heap_ring PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_heap_ring ${GTEST_LIBS} test_stubs)
add_test(NAME HeapRing COMMAND test_heap_ring)

# =============================================================================
# Test: TaskRing (task slot allocation)
# =============================================================================
add_executable(test_task_ring test_task_ring.cpp ${TMR_RUNTIME}/pto_ring_buffer.cpp)
target_include_directories(test_task_ring PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_task_ring PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_task_ring ${GTEST_LIBS} test_stubs)
add_test(NAME TaskRing COMMAND test_task_ring)

# =============================================================================
# Test: DepListPool (dependency list entry pool)
# =============================================================================
add_executable(test_dep_pool test_dep_pool.cpp ${TMR_RUNTIME}/pto_ring_buffer.cpp)
target_include_directories(test_dep_pool PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_dep_pool PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_dep_pool ${GTEST_LIBS} test_stubs)
add_test(NAME DepPool COMMAND test_dep_pool)

# =============================================================================
# Test: Tensor overlap detection
# =============================================================================
add_executable(test_tensor_overlap test_tensor_overlap.cpp)
target_include_directories(test_tensor_overlap PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_tensor_overlap PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_tensor_overlap ${GTEST_LIBS} test_stubs)
add_test(NAME TensorOverlap COMMAND test_tensor_overlap)

# =============================================================================
# Test: TensorMap (hash table + dependency discovery)
# =============================================================================
add_executable(test_tensormap test_tensormap.cpp ${TMR_RUNTIME}/pto_tensormap.cpp)
target_include_directories(test_tensormap PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_tensormap PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_tensormap ${GTEST_LIBS} test_stubs)
add_test(NAME TensorMap COMMAND test_tensormap)

# =============================================================================
# Test: ReadyQueue (lock-free MPMC)
# =============================================================================
add_executable(test_ready_queue test_ready_queue.cpp)
target_include_directories(test_ready_queue PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_ready_queue PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_ready_queue ${GTEST_LIBS} test_stubs pthread)
add_test(NAME ReadyQueue COMMAND test_ready_queue)

# =============================================================================
# Test: Shared Memory layout
# =============================================================================
add_executable(test_shared_memory test_shared_memory.cpp ${TMR_RUNTIME}/pto_shared_memory.cpp)
target_include_directories(test_shared_memory PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_shared_memory PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_shared_memory ${GTEST_LIBS} test_stubs)
add_test(NAME SharedMemory COMMAND test_shared_memory)

# =============================================================================
# Test: Task State Machine
# =============================================================================
add_executable(test_task_state test_task_state.cpp)
target_include_directories(test_task_state PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_task_state PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_task_state ${GTEST_LIBS} test_stubs)
add_test(NAME TaskState COMMAND test_task_state)

# =============================================================================
# Test: Scope mechanism
# =============================================================================
add_executable(test_scope test_scope.cpp)
target_include_directories(test_scope PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_scope PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_scope ${GTEST_LIBS} test_stubs)
add_test(NAME Scope COMMAND test_scope)

# =============================================================================
# Edge-case tests: Ring Buffer system
# =============================================================================
add_executable(test_ring_buffer_edge test_ring_buffer_edge.cpp ${TMR_RUNTIME}/pto_ring_buffer.cpp)
target_include_directories(test_ring_buffer_edge PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_ring_buffer_edge PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_ring_buffer_edge ${GTEST_LIBS} test_stubs pthread)
add_test(NAME RingBufferEdge COMMAND test_ring_buffer_edge)

# =============================================================================
# Edge-case tests: TensorMap system
# =============================================================================
add_executable(test_tensormap_edge test_tensormap_edge.cpp ${TMR_RUNTIME}/pto_tensormap.cpp)
target_include_directories(test_tensormap_edge PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_tensormap_edge PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_tensormap_edge ${GTEST_LIBS} test_stubs)
add_test(NAME TensorMapEdge COMMAND test_tensormap_edge)

# =============================================================================
# Edge-case tests: Scheduler / SharedMemory / TaskState
# =============================================================================
add_executable(test_scheduler_edge test_scheduler_edge.cpp ${TMR_RUNTIME}/pto_shared_memory.cpp)
target_include_directories(test_scheduler_edge PRIVATE ${TMR_INCLUDE_DIRS})
target_compile_definitions(test_scheduler_edge PRIVATE ${TMR_COMPILE_DEFS})
target_link_libraries(test_scheduler_edge ${GTEST_LIBS} test_stubs pthread)
add_test(NAME SchedulerEdge COMMAND test_scheduler_edge)
