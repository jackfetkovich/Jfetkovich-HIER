cmake_minimum_required(VERSION 3.25)
project(mppi)
set(SOURCE_FILES
    Main.cpp
    MPPI.cpp
)
 
add_executable(${PROJECT_NAME} ${SOURCE_FILES})