# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

## FindNVSHMEM.cmake - Locate NVSHMEM library and headers
# This module provides the following variables:
#  NVSHMEM_FOUND        - True if NVSHMEM was found
#  NVSHMEM_LIBRARIES    - Libraries to link against for NVSHMEM
#  NVSHMEM_INCLUDE_DIRS - Include directories for NVSHMEM
#  NVSHMEM_VERSION      - NVSHMEM version (if detected)
#
# Usage:
#   find_package(NVSHMEM REQUIRED)
#   target_include_directories(MyTarget PUBLIC ${NVSHMEM_INCLUDE_DIRS})
#   target_link_libraries(MyTarget PUBLIC ${NVSHMEM_LIBRARIES})

if(NOT NVSHMEM_FIND_VERSION)
  set(NVSHMEM_FIND_VERSION "")
endif()

# Allow override through environment or CMake variable
set(_nvshmem_root "$ENV{NVSHMEM_ROOT}")
if(NOT _nvshmem_root AND DEFINED NVSHMEM_ROOT)
  set(_nvshmem_root "${NVSHMEM_ROOT}")
endif()

# Find headers
find_path(NVSHMEM_INCLUDE_DIR
  NAMES nvshmem.h
  HINTS
    ${_nvshmem_root}/include
    /usr/local/include
    /usr/include
    $ENV{NVHPC_ROOT}/comm_libs/nvshmem/include
)

# Find library
find_library(NVSHMEM_LIBRARY
  NAMES nvshmem
  HINTS
    ${_nvshmem_root}/lib
    ${_nvshmem_root}/lib64
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
    $ENV{NVHPC_ROOT}/comm_libs/nvshmem/lib
)

# Optionally, try pkg-config
if(NOT NVSHMEM_LIBRARY OR NOT NVSHMEM_INCLUDE_DIR)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PC_NVSHMEM QUIET nvshmem)
    if(PC_NVSHMEM_FOUND)
      set(NVSHMEM_INCLUDE_DIR ${PC_NVSHMEM_INCLUDE_DIRS})
      set(NVSHMEM_LIBRARY ${PC_NVSHMEM_LIBRARIES})
    endif()
  endif()
endif()

# Check results
include(CMakeFindDependencyMacro)
set(NVSHMEM_FOUND FALSE)
if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_LIBRARY)
  set(NVSHMEM_FOUND TRUE)
  set(NVSHMEM_LIBRARIES ${NVSHMEM_LIBRARY})
  set(NVSHMEM_INCLUDE_DIRS ${NVSHMEM_INCLUDE_DIR})
  # Try to extract version from header
  file(READ "${NVSHMEM_INCLUDE_DIR}/nvshmem.h" _nvshmem_header_content)
  string(REGEX MATCH "#define[ \t]+NVSHMEM_VERSION_MAJOR[ \t]+([0-9]+)" _major_match ${_nvshmem_header_content})
  string(REGEX MATCH "#define[ \t]+NVSHMEM_VERSION_MINOR[ \t]+([0-9]+)" _minor_match ${_nvshmem_header_content})
  if(_major_match AND _minor_match)
    string(REGEX REPLACE ".*#define[ \t]+NVSHMEM_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" NVSHMEM_VERSION_MAJOR ${_major_match})
    string(REGEX REPLACE ".*#define[ \t]+NVSHMEM_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" NVSHMEM_VERSION_MINOR ${_minor_match})
    set(NVSHMEM_VERSION "${NVSHMEM_VERSION_MAJOR}.${NVSHMEM_VERSION_MINOR}")
  endif()
endif()

# Provide imported target for modern CMake
if(NVSHMEM_FOUND)
  add_library(NVSHMEM::NVSHMEM UNKNOWN IMPORTED)
  set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${NVSHMEM_LIBRARIES}"
  )
  message(STATUS "Found NVSHMEM: ${NVSHMEM_LIBRARIES} (include: ${NVSHMEM_INCLUDE_DIRS})")
else()
  message(FATAL_ERROR "Could not find NVSHMEM library or headers. Please set NVSHMEM_ROOT to the installation prefix.")
endif()

# Mark as found
mark_as_advanced(NVSHMEM_INCLUDE_DIR NVSHMEM_LIBRARY)
