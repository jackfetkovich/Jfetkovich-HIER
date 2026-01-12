cmake_minimum_required(VERSION 3.25)
project(mppi)
set(SOURCE_FILES
    Main.cpp
    MPPI.cpp
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(mppi PRIVATE Eigen3::Eigen)