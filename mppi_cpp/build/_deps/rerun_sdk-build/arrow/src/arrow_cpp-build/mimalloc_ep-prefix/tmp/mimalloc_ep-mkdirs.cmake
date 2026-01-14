# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/src/mimalloc_ep"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/src/mimalloc_ep-build"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/tmp"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/src/mimalloc_ep-stamp"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/src"
  "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/src/mimalloc_ep-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/src/mimalloc_ep-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/jfetko/Documents/HIER/mppi_cpp/build/_deps/rerun_sdk-build/arrow/src/arrow_cpp-build/mimalloc_ep-prefix/src/mimalloc_ep-stamp${cfgdir}") # cfgdir has leading slash
endif()
