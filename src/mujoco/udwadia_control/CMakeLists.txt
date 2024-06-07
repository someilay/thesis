cmake_minimum_required(VERSION 3.22)
project(udwadia VERSION 1.0)


if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Set Release as default")
    set(CMAKE_BUILD_TYPE Release)
endif()


if(NOT CMAKE_CXX_EXTENSIONS)
    message(STATUS "Disabling CMAke extensions")
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()


if (NOT PINOCCHIO_DIR)
    set(PINOCCHIO_DIR /opt/openrobots)
endif()


if (NOT INICPP_DIR)
    message(WARNING "You should install inicpp! Lib reference https://github.com/SemaiCZE/inicpp/tree/master")
    set(INICPP_DIR /usr/local)
endif()


if (NOT MUJOCO_DIR)
    message(FATAL_ERROR "Specify MUJOCO_DIR!")
endif()


if (NOT EXISTS ${PINOCCHIO_DIR})
    message(FATAL_ERROR "${PINOCCHIO_DIR} not found, specify PINOCCHIO_DIR manually!")
endif()

# Check pkg config
find_package(PkgConfig REQUIRED)

# Update prefix
set(CMAKE_PREFIX_PATH ${PINOCCHIO_DIR})

# Connect GLFW
if (NOT GLFW_DIR)
    pkg_search_module(GLFW REQUIRED glfw3 glfw3>=3.3 glfw3<3.4)
    include_directories(${GLFW_INCLUDE_DIRS})
else()
    find_package(glfw3 3.3 REQUIRED PATHS ${GLFW_DIR})
endif()

# Connect pinocchio
find_package(pinocchio 2.6.20 REQUIRED)
link_directories(${PINOCCHIO_DIR}/lib)

# Connect inicpp
find_package(inicpp REQUIRED PATHS ${INICPP_DIR})
# include_directories(${INICPP_DIR}/include)

# Connect mujoco
include_directories(${MUJOCO_DIR}/include include)
link_directories(${MUJOCO_DIR}/lib)

# Include Eigen headers
pkg_search_module(Eigen3 REQUIRED eigen3 eigen3>=3.4 eigen<3.5)
link_directories(${Eigen3_INCLUDE_DIRS})

# Set standards
set(CMAKE_CXX_FLAGS "-Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(udwadia main.cpp)

target_link_libraries(udwadia glfw mujoco pinocchio::pinocchio inicpp::inicpp)
