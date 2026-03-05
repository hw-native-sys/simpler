# a2a3sim Platform - Thread-Based Simulation
#
# This platform simulates the Ascend AICPU/AICore execution model using threads.
# It provides the same API as the real a2a3 platform for testing and development.
#
# Build: cmake -B build && cmake --build build
# Or use from examples: cmake -DCUSTOM_INCLUDE_DIRS=... -DCUSTOM_SOURCE_DIRS=...

cmake_minimum_required(VERSION 3.16.3)

project(a2a3sim_platform LANGUAGES CXX)

# Add subdirectories
add_subdirectory(host)
add_subdirectory(aicore)
