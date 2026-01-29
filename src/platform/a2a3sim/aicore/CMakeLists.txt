# Build AICore runtime for a2a3sim (Simulation)
#
# This builds the AICore executor as a host-compatible shared library (.so)
# that can be dlopen'd by the simulation runtime.
#
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake

cmake_minimum_required(VERSION 3.16.3)

project(aicore_sim LANGUAGES CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../common")

if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build AICore (simulation)")
endif()

# Build complete source list
set(AICORE_SOURCES "")

# Add local platform-specific sources
file(GLOB LOCAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(APPEND AICORE_SOURCES ${LOCAL_SOURCES})

if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND AICORE_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_SOURCE_DIRS to build AICore (simulation)")
endif()

# Create shared library (host-compatible for dlopen)
add_library(aicore_kernel SHARED ${AICORE_SOURCES})

# Compile options (host g++)
target_compile_options(aicore_kernel
    PRIVATE
        -Wall
        -Wextra
        -std=c++17
        -fPIC
        -O3
        -g
)

# Include directories
target_include_directories(aicore_kernel
    PRIVATE
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
)

set_target_properties(aicore_kernel PROPERTIES OUTPUT_NAME "aicore_kernel")
