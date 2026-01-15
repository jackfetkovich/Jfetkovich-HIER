#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "rerun_sdk" for configuration "Release"
set_property(TARGET rerun_sdk APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(rerun_sdk PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librerun_sdk.so"
  IMPORTED_SONAME_RELEASE "librerun_sdk.so"
  )

list(APPEND _cmake_import_check_targets rerun_sdk )
list(APPEND _cmake_import_check_files_for_rerun_sdk "${_IMPORT_PREFIX}/lib/librerun_sdk.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
