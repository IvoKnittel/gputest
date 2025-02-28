cmake_minimum_required(VERSION 3.18)
project(ImageCopy LANGUAGES CXX CUDA)

# Specify the CUDA architecture (optional, adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV_DIR to the installation directory
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../opencv/opencv/build_debug/install/x64/vc17/lib")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Output OpenCV information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")


# Set the source files for the library
set(SOURCES_CVLIB
    cvlib.h
    cvlib.cpp
)

# Create the shared library
add_library(cvlib SHARED ${SOURCES_CVLIB})

# Specify the include directories for the library
target_include_directories(cvlib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS} 
)


# Link the OpenCV and CUDA libraries to the shared library
target_link_libraries(cvlib PRIVATE
    ${OpenCV_LIBS}
)

# Define the export macro for the library
target_compile_definitions(cvlib PRIVATE CVLIB_EXPORTS)

# Ensure the runtime library is set correctly for Debug builds
set_target_properties(cvlib PROPERTIES 
    MAP_IMPORTED_CONFIG_DEBUG RELEASE
)