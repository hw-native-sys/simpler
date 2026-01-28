# Build AICore kernel using kernel.cpp + common folder + custom includes and sources (for `Task`)
# BISHENG_CC and BISHENG_LD are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake
cmake_minimum_required(VERSION 3.16.3)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../common")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "-I${INC_DIR}")
    endforeach()
    string(REPLACE ";" " " CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}")
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build with `Task` implementation")
endif()

# Build complete source list: kernel.cpp + common + CUSTOM_SOURCE_DIRS
set(ALL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp")
file(GLOB COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.c")
list(APPEND ALL_SOURCES ${COMMON_SOURCES})
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND ALL_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_SOURCE_DIRS to build with `Task` implementation")
endif()
list(LENGTH ALL_SOURCES NUM_SOURCES)
message(STATUS "AICore kernel: ${NUM_SOURCES} source files")

# Output files
set(AICORE_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/aicore_kernel.o)

# Compiler flags
set(AICORE_FLAGS
    "-c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}"
)
separate_arguments(AICORE_FLAGS)

# Step 1: Compile each source file to individual .o files for AIC and AIV
set(ALL_AIC_OBJECTS "")
set(ALL_AIV_OBJECTS "")

foreach(SRC_FILE ${ALL_SOURCES})
    get_filename_component(SRC_NAME ${SRC_FILE} NAME_WE)
    set(OBJ_AIC "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aic.o")
    set(OBJ_AIV "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aiv.o")

    list(APPEND ALL_AIC_OBJECTS ${OBJ_AIC})
    list(APPEND ALL_AIV_OBJECTS ${OBJ_AIV})

    # Compile for AIC architecture
    add_custom_command(
        OUTPUT ${OBJ_AIC}
        COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube
                -o ${OBJ_AIC} ${SRC_FILE}
        DEPENDS ${SRC_FILE}
        COMMENT "Compiling ${SRC_NAME} for AIC"
    )

    # Compile for AIV architecture
    add_custom_command(
        OUTPUT ${OBJ_AIV}
        COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS} -D__AIV__ --cce-aicore-arch=dav-c220-vec
                -o ${OBJ_AIV} ${SRC_FILE}
        DEPENDS ${SRC_FILE}
        COMMENT "Compiling ${SRC_NAME} for AIV"
    )
endforeach()

# Step 2: Combine all .o files using ld -r (relocatable link)
set(COMBINED_AIC_OBJ "${CMAKE_CURRENT_BINARY_DIR}/aicore_aic_combined.o")
set(COMBINED_AIV_OBJ "${CMAKE_CURRENT_BINARY_DIR}/aicore_aiv_combined.o")

add_custom_command(
    OUTPUT ${COMBINED_AIC_OBJ}
    COMMAND ${BISHENG_LD} -r -o ${COMBINED_AIC_OBJ} ${ALL_AIC_OBJECTS}
    DEPENDS ${ALL_AIC_OBJECTS}
    COMMENT "Combining ${NUM_SOURCES} AIC object files into single .o"
)

add_custom_command(
    OUTPUT ${COMBINED_AIV_OBJ}
    COMMAND ${BISHENG_LD} -r -o ${COMBINED_AIV_OBJ} ${ALL_AIV_OBJECTS}
    DEPENDS ${ALL_AIV_OBJECTS}
    COMMENT "Combining ${NUM_SOURCES} AIV object files into single .o"
)

# Step 3: Final link
add_custom_target(aicore_kernel ALL
    DEPENDS ${COMBINED_AIC_OBJ} ${COMBINED_AIV_OBJ}

    COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n
    -o ${AICORE_KERNEL_OBJ} ${COMBINED_AIC_OBJ} ${COMBINED_AIV_OBJ}

    BYPRODUCTS ${AICORE_KERNEL_OBJ}
    COMMENT "Linking AICore kernel binary"
)
