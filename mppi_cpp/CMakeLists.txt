cmake_minimum_required(VERSION 3.25)
project(mppi)
set(SOURCE_FILES
    Main.cpp
    MPPI.cpp
    Trajectory.cpp
    Dynamics.cpp
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(mppi PRIVATE Eigen3::Eigen)

# Download the rerun_sdk
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# Link against rerun_sdk.
target_link_libraries(mppi PRIVATE rerun_sdk)