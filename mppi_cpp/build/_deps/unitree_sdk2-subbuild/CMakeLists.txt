# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.25.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(unitree_sdk2-populate NONE)


# Pass through things we've already detected in the main project to avoid
# paying the cost of redetecting them again in ExternalProject_Add()
set(GIT_EXECUTABLE [==[/usr/bin/git]==])
set(GIT_VERSION_STRING [==[2.34.1]==])
set_property(GLOBAL PROPERTY _CMAKE_FindGit_GIT_EXECUTABLE_VERSION
  [==[/usr/bin/git;2.34.1]==]
)


include(ExternalProject)
ExternalProject_Add(unitree_sdk2-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/unitreerobotics/unitree_sdk2.git" "GIT_TAG" "eed0898b8d63d83406f7f460a827fa378dd3e631" "CMAKE_ARGS" "-DCMAKE_BUILD_TYPE=Release" "-DBUILD_EXAMPLES=OFF"
                    SOURCE_DIR          "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-src"
                    BINARY_DIR          "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


