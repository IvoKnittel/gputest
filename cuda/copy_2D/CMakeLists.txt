cmake_minimum_required(VERSION 3.18)
project(ImageCopy LANGUAGES CXX CUDA)

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
add_library(ImageCopy SHARED ${SOURCES})

# Define IMAGECOPY_EXPORTS when building the DLL
target_compile_definitions(ImageCopy PRIVATE IMAGECOPY_EXPORTS)

# Set properties for the CUDA compiler
set_target_properties(ImageCopy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

# Find the Python3 package
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include Python3 directories and link against Python3 libraries
include_directories(${Python3_INCLUDE_DIRS})
target_link_libraries(ImageCopy ${Python3_LIBRARIES})

# Include directories (optional, adjust according to your setup)
target_include_directories(ImageCopy PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Find the CUDA package
find_package(CUDA REQUIRED)

# Set CUDA library and include paths
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
set(CUDA_LIB_PATH ${CUDA_LIBRARY_DIRS}/cuda.lib)
set(CUDA_RT_STATIC_LIB_PATH ${CUDA_LIBRARY_DIRS}/cudart_static.lib)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(ImageCopy ${CUDA_RT_STATIC_LIB_PATH} ${CUDA_LIB_PATH})


# Define the directories
set(PYBIND11_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../third_party/pybind11")
set(PYBIND11_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/pybind11")
set(PYBIND11_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../third_party/pybind11/include")

# Print the directories as messages
message(STATUS "Pybind11 source directory: ${PYBIND11_SOURCE_DIR}")
message(STATUS "Pybind11 binary directory: ${PYBIND11_BINARY_DIR}")
message(STATUS "Pybind11 include directory: ${PYBIND11_INCLUDE_DIR}")

# Add the third_party/pybind11 directory
add_subdirectory(
    "${PYBIND11_SOURCE_DIR}"  # Source directory
    "${PYBIND11_BINARY_DIR}"  # Binary directory
)

# Include the pybind11 include directory
include_directories(${PYBIND11_INCLUDE_DIR})

pybind11_add_module(image_copy python_bindings.cpp)
target_link_libraries(image_copy PRIVATE ImageCopy ${CUDA_RT_STATIC_LIB_PATH} ${CUDA_LIB_PATH} ${Python3_LIBRARIES})
set_target_properties(image_copy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pymodule
)
# Set properties specific to Windows DLLs
if (WIN32)
    set_target_properties(ImageCopy PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# Install the module and DLL to the virtual environment's site-packages directory
set(VENV_SITE_PACKAGES "${CMAKE_BINARY_DIR}/../.venv/Lib/site-packages/mygputest")

install(TARGETS image_copy
        LIBRARY DESTINATION ${VENV_SITE_PACKAGES}
        RUNTIME DESTINATION ${VENV_SITE_PACKAGES})

install(FILES ${CMAKE_BINARY_DIR}/pymodule/image_copy.cp313-win_amd64.pyd
        DESTINATION ${VENV_SITE_PACKAGES})

# Ensure the ImageCopy DLL is also installed
install(TARGETS ImageCopy
        LIBRARY DESTINATION ${VENV_SITE_PACKAGES}
        RUNTIME DESTINATION ${VENV_SITE_PACKAGES}
)

add_executable(TestExecutable test.cpp)

# Set OpenCV_DIR to the installation directory
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../opencv/opencv/build_debug/install/x64/vc17/lib")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include directories for the test executable
target_include_directories(TestExecutable PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${OpenCV_INCLUDE_DIRS} 
)

# Ensure the runtime library is set correctly for Debug builds
set_target_properties(TestExecutable PROPERTIES 
    MAP_IMPORTED_CONFIG_DEBUG RELEASE
)

target_link_libraries(TestExecutable PRIVATE 
    ImageCopy 
    ${CUDA_RT_STATIC_LIB_PATH} 
    ${CUDA_LIB_PATH}
    ${OpenCV_LIBS}
)

# Copy the OpenCV DLLs to the bin directory after building the executable
set(TEST_DIR "${CMAKE_BINARY_DIR}/../test")

message(STATUS "TEST_DIR: ${TEST_DIR}")

install(TARGETS TestExecutable RUNTIME DESTINATION ${TEST_DIR})

# Ensure the ImageCopy DLL is also installed
install(TARGETS ImageCopy
        RUNTIME DESTINATION ${TEST_DIR}
)

# Set the source files for the library
set(SOURCES_CVLIB
    cvlib.h
    cvlib.cpp
)

# Create the shared library
add_library(cvlib SHARED ${SOURCES_CVLIB})

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# Specify the include directories for the library
target_include_directories(cvlib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS} 
)


# Link the OpenCV and CUDA libraries to the shared library
target_link_libraries(cvlib PRIVATE
    ImageCopy
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBS}
)

# Define the export macro for the library
target_compile_definitions(cvlib PRIVATE CVLIB_EXPORTS)

# Ensure the runtime library is set correctly for Debug builds
set_target_properties(cvlib PROPERTIES 
    MAP_IMPORTED_CONFIG_DEBUG RELEASE
)