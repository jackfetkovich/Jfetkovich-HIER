cmake_minimum_required(VERSION 3.25)
project(mppi)
set(SOURCE_FILES
    Main.cpp
    MPPI.cpp
    Trajectory.cpp
    Dynamics.cpp
    Timer.cpp
    MathFuncs.cpp
    SafetyFilter.cpp
)

set(GUROBI_ROOT "$ENV{GUROBI_HOME}" CACHE PATH "Path to Gurobi installation")

find_path(GUROBI_INCLUDE_DIR
    NAMES gurobi_c++.h
    HINTS "${GUROBI_ROOT}/include"
)

find_library(GUROBI_CXX_LIB
    NAMES gurobi_c++ gurobi_c++mt
    HINTS "${GUROBI_ROOT}/lib"
)

find_library(GUROBI_CORE_LIB
    NAMES gurobi gurobi100 gurobi110 gurobi130
    HINTS "${GUROBI_ROOT}/lib"
)

if(NOT GUROBI_INCLUDE_DIR OR NOT GUROBI_CXX_LIB OR NOT GUROBI_CORE_LIB)
    message(FATAL_ERROR "Could not find Gurobi headers or libraries. Check GUROBI_ROOT/GUROBI_HOME.")
endif()

add_library(gurobi_interface INTERFACE)
target_include_directories(gurobi_interface INTERFACE "${GUROBI_INCLUDE_DIR}")
target_link_libraries(gurobi_interface INTERFACE "${GUROBI_CXX_LIB}" "${GUROBI_CORE_LIB}")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(mppi PRIVATE Eigen3::Eigen)
target_link_libraries(mppi PRIVATE gurobi_interface)

# Download the rerun_sdk
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# Link against rerun_sdk.
target_link_libraries(mppi PRIVATE rerun_sdk)