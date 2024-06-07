cmake_minimum_required(VERSION 3.22)
project(example VERSION 1.0)


if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Set Release as default")
    set(CMAKE_BUILD_TYPE Release)
endif()


if(NOT CMAKE_CXX_EXTENSIONS)
    message(STATUS "Disabling CMAke extensions")
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()


if (NOT MUJOCO_DIR)
    message(FATAL_ERROR "Specify MUJOCO_DIR!")
endif()


if (NOT GLFW_DIR)
    message(FATAL_ERROR "Specify GLFW_DIR!")
endif()


# Connect GLFW
find_package(glfw3 3.3 REQUIRED PATHS ${GLFW_DIR})
find_package(OpenGL REQUIRED)


# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


# add include directories
include_directories(${MUJOCO_DIR}/include include)
link_directories(${MUJOCO_DIR}/lib)


# add the source directories
# file(GLOB SRCS src/*.cpp)


add_executable(double_pendulum main.cpp)


target_link_libraries(double_pendulum glfw OpenGL::GL mujoco)
