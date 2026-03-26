cmake_minimum_required(VERSION 3.15)

# Support both scikit-build and native CMake builds
if(DEFINED SKBUILD_PROJECT_NAME)
    project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})
    set(SKBUILD_MODE ON)
else()
    project(simpler VERSION 0.1.0)
    set(SKBUILD_MODE OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Include Python bindings
add_subdirectory(python/bindings)
