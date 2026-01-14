# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/tmp"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-stamp"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-stamp${cfgdir}") # cfgdir has leading slash
endif()
