# nanobind Python bindings for task_interface

set(BINDING_SOURCES
    task_interface.cpp
)

list(TRANSFORM BINDING_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

nanobind_add_module(_task_interface ${BINDING_SOURCES})

target_include_directories(_task_interface PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/task_interface
)

if(SKBUILD_MODE)
    install(TARGETS _task_interface DESTINATION .)
else()
    set_target_properties(_task_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    )
endif()
