# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-src"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-build"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-subbuild/unitree_sdk2-populate-prefix"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-subbuild/unitree_sdk2-populate-prefix/tmp"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-subbuild/unitree_sdk2-populate-prefix/src/unitree_sdk2-populate-stamp"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-subbuild/unitree_sdk2-populate-prefix/src"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-subbuild/unitree_sdk2-populate-prefix/src/unitree_sdk2-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-subbuild/unitree_sdk2-populate-prefix/src/unitree_sdk2-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/unitree_sdk2-subbuild/unitree_sdk2-populate-prefix/src/unitree_sdk2-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
