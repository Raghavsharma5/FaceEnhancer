cmake_minimum_required(VERSION 3.10)
project(FaceEnhancer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenCV package. This must be installed on your system.
find_package(OpenCV REQUIRED)

# Add the include directories for our project and OpenCV
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable from our source files
add_executable(face_enhancer
    src/main.cpp
    src/face_enhancer.cpp
    src/image_processor.cpp
)

# Link the executable against the OpenCV libraries
target_link_libraries(face_enhancer ${OpenCV_LIBS})