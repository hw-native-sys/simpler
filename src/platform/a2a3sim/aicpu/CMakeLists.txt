# Build AICPU runtime for a2a3sim (Simulation)
#
# This builds the AICPU executor as a host-compatible shared library (.so)
# that can be dlopen'd by the simulation runtime.
#
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake

cmake_minimum_required(VERSION 3.16.3)

project(aicpu_sim LANGUAGES CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../common")

if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build AICPU (simulation)")
endif()

# Build complete source list
set(AICPU_SOURCES "")

if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND AICPU_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_SOURCE_DIRS to build AICPU (simulation)")
endif()

# Create shared library (host-compatible for dlopen)
add_library(aicpu_kernel SHARED ${AICPU_SOURCES})

# Compile options (host g++)
target_compile_options(aicpu_kernel
    PRIVATE
        -Wall
        -Wextra
        -std=c++17
        -fPIC
        -O3
        -g
)

# Include directories
target_include_directories(aicpu_kernel
    PRIVATE
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
)

# Link against pthread
target_link_libraries(aicpu_kernel
    PRIVATE
        pthread
)

set_target_properties(aicpu_kernel PROPERTIES OUTPUT_NAME "aicpu_kernel")
