cmake_minimum_required(VERSION 3.18)
project(ShuffleCopy LANGUAGES CXX CUDA)

# Specify the CUDA architecture (optional, adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source files
set(SOURCES
    extern.h
    host.cpp
    kernel.cu
    kernel.h
)

# Set source file properties to treat host.cpp as a CUDA file
set_source_files_properties(host.cpp PROPERTIES LANGUAGE CUDA)

# Add the shared library target
add_library(ShuffleCopy SHARED ${SOURCES})

# Define SHUFFLECOPY_EXPORTS when building the DLL
target_compile_definitions(ShuffleCopy PRIVATE SHUFFLECOPY_EXPORTS)

# Set properties for the CUDA compiler
set_target_properties(ShuffleCopy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

# Find the Python3 package
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include Python3 directories and link against Python3 libraries
include_directories(${Python3_INCLUDE_DIRS})
target_link_libraries(ShuffleCopy ${Python3_LIBRARIES})

# Include directories (optional, adjust according to your setup)
target_include_directories(ShuffleCopy PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Find the CUDA package
find_package(CUDA REQUIRED)

# Set CUDA library and include paths
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
set(CUDA_LIB_PATH ${CUDA_LIBRARY_DIRS}/cuda.lib)
set(CUDA_RT_STATIC_LIB_PATH ${CUDA_LIBRARY_DIRS}/cudart_static.lib)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(ShuffleCopy ${CUDA_RT_STATIC_LIB_PATH} ${CUDA_LIB_PATH})

# Add pybind11 for Python bindings
add_subdirectory(
    "${CMAKE_SOURCE_DIR}/../third_party/pybind11"  # Source directory
    "${CMAKE_BINARY_DIR}/third_party/pybind11"     # Binary directory
)
include_directories(../third_party/pybind11/include)
pybind11_add_module(shuffle_copy python_bindings.cpp)
target_link_libraries(shuffle_copy PRIVATE ShuffleCopy ${CUDA_RT_STATIC_LIB_PATH} ${CUDA_LIB_PATH} ${Python3_LIBRARIES})
set_target_properties(shuffle_copy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/mygputest
)
# Set properties specific to Windows DLLs
if (WIN32)
    set_target_properties(ShuffleCopy PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# Install the module and DLL to the virtual environment's site-packages directory
set(VENV_SITE_PACKAGES "${CMAKE_BINARY_DIR}/../../../../.venv/Lib/site-packages/mygputest")

install(TARGETS shuffle_copy
        LIBRARY DESTINATION ${VENV_SITE_PACKAGES}
        RUNTIME DESTINATION ${VENV_SITE_PACKAGES})

install(FILES ${CMAKE_BINARY_DIR}/mygputest/shuffle_copy.cp313-win_amd64.pyd
        DESTINATION ${VENV_SITE_PACKAGES})

# Ensure the ShuffleCopy DLL is also installed
install(TARGETS ShuffleCopy
        LIBRARY DESTINATION ${VENV_SITE_PACKAGES}
        RUNTIME DESTINATION ${VENV_SITE_PACKAGES})

# Add a custom command to run the install step after the build
add_custom_command(TARGET shuffle_copy
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR})

add_executable(TestExecutable test.cpp)
# Link the ShuffleCopy library to the test executable
target_link_libraries(TestExecutable PRIVATE ShuffleCopy ${CUDA_RT_STATIC_LIB_PATH} ${CUDA_LIB_PATH})

# Include directories for the test executable
target_include_directories(TestExecutable PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PYTHON_INCLUDE_DIR})

target_link_libraries(TestExecutable PRIVATE ShuffleCopy ${CUDA_RT_STATIC_LIB_PATH} ${CUDA_LIB_PATH})
install(TARGETS TestExecutable RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin) 