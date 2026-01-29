# Build Host runtime for a2a3sim (Simulation)
#
# This builds the simulated host runtime that uses threads instead of
# actual device execution.
#
# CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake

cmake_minimum_required(VERSION 3.16.3)

project(host_runtime_sim LANGUAGES C CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../common")

if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build Host runtime (simulation)")
endif()

# Build complete source list
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pto_runtime_c_api.cpp"
)

if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND HOST_RUNTIME_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(STATUS "CUSTOM_SOURCE_DIRS not set, using only core sources")
endif()

# Create shared library
add_library(host_runtime SHARED ${HOST_RUNTIME_SOURCES})

# Compile options
target_compile_options(host_runtime
    PRIVATE
        -Wall
        -Wextra
        -std=c++17
        -fPIC
        -O3
        -g
)

# Include directories
target_include_directories(host_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
)

# Link against pthread (for std::thread)
target_link_libraries(host_runtime
    PRIVATE
        pthread
        dl
)

set_target_properties(host_runtime PROPERTIES OUTPUT_NAME "host_runtime")
