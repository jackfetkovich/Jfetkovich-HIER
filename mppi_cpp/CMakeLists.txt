cmake_minimum_required(VERSION 3.25)
project(mppi)

add_subdirectory(external/mpac_go2)

# We can use the 'objects' OBJECT library defined in the submodule.
# Wrap it in a real static library local to this project.
add_library(mpac_go2_local STATIC)

# Pull all object files from the submodule's 'objects' target
target_sources(mpac_go2_local PRIVATE $<TARGET_OBJECTS:objects>)

# Propagate include dirs so you can #include their headers
target_include_directories(mpac_go2_local PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/robot
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/sim
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/ctrl
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/atnmy
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/dynamics
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/kinematics
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/ipc
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/math_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mpac_go2/mpac_core/io
)

# Link the same external deps that the submodule's 'ctrl' executable uses.
# These variables are set in the submodule's CMakeLists via find_package/find_library.
target_link_libraries(mpac_go2_local PUBLIC
    urdfdom_model
    ${PINOCCHIO_LIB}
    m
    boost_system
    ${HDF5_LIBRARIES}
    OsqpEigen::OsqpEigen
    unitree_sdk2
)


set(SOURCE_FILES
    Main.cpp
    MPPI.cpp
    Trajectory.cpp
    Dynamics.cpp
    Timer.cpp
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(mppi PRIVATE /opt/openrobots/include)

target_link_libraries(mppi PRIVATE ${PINOCCHIO_LIB})


# Ensure HDF5 is found in this (top-level) project too:
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Make HDF5 headers visible when compiling mppi (so hdf5.h is found)
target_include_directories(mppi PRIVATE ${HDF5_INCLUDE_DIRS})

# (Optional but nice) also link against HDF5:
target_link_libraries(mppi PRIVATE ${HDF5_LIBRARIES})

target_link_libraries(mppi PRIVATE Eigen3::Eigen)

target_link_libraries(mppi PRIVATE
    Eigen3::Eigen
    rerun_sdk
    mpac_go2_local      # <-- this is your wrapper
)


# Download the rerun_sdk
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# Link against rerun_sdk.
target_link_libraries(mppi PRIVATE rerun_sdk)