# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# - Try to find the cuFFTMp multi-process FFT library
# Once done, this will define
#
#  cuFFTMp_FOUND         – Set to TRUE if cuFFTMp was found
#  cuFFTMp_INCLUDE_DIRS  – where to find cufftMp.h
#  cuFFTMp_LIBRARIES     – List of libraries to link against
#  cuFFTMp_VERSION       – cuFFTMp version string if detectable
#
# Usage:
#   find_package(cuFFTMp REQUIRED)
#   target_include_directories(<tgt> PUBLIC ${cuFFTMp_INCLUDE_DIRS})
#   target_link_libraries(<tgt> PUBLIC ${cuFFTMp_LIBRARIES})

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
if (cuFFTMp_FOUND)
  return()
endif()
project(FindCUFFTMP NONE)

# Allow override via cuFFTMp_ROOT
set(_cuFFTMp_ROOT
    $ENV{cuFFTMp_ROOT}
    CACHE PATH "Root directory of cuFFTMp installation")

# Find headers
find_path(cuFFTMp_INCLUDE_DIRS
  NAMES cufftMp.h
  HINTS
    ${_cuFFTMp_ROOT}/include/cufftmp
    ${_cuFFTMp_ROOT}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    $ENV{CUDA_HOME}/include
  PATH_SUFFIXES cufftmp
)

# Find library
find_library(cuFFTMp_LIBRARIES
  NAMES cufftMp libcufftMp
  HINTS
    ${_cuFFTMp_ROOT}/lib
    ${_cuFFTMp_ROOT}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    $ENV{CUDA_HOME}/lib64
)

# Try to extract version from header if found
if (cuFFTMp_INCLUDE_DIRS)
  file(READ "${cuFFTMp_INCLUDE_DIRS}/cufftMp.h" _cufftmp_h_content)
  string(REGEX MATCH "#define[ \t]+CUFFTMP_VERSION_MAJOR[ \t]+([0-9]+)" _major_match "${_cufftmp_h_content}")
  string(REGEX MATCH "#define[ \t]+CUFFTMP_VERSION_MINOR[ \t]+([0-9]+)" _minor_match "${_cufftmp_h_content}")
  if (_major_match AND _minor_match)
    string(REGEX REPLACE ".*#define[ \t]+CUFFTMP_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" cuFFTMp_VERSION_MAJOR "${_major_match}")
    string(REGEX REPLACE ".*#define[ \t]+CUFFTMP_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" cuFFTMp_VERSION_MINOR "${_minor_match}")
    set(cuFFTMp_VERSION "${cuFFTMp_VERSION_MAJOR}.${cuFFTMp_VERSION_MINOR}")
  endif()
endif()

# handle result
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuFFTMp
  REQUIRED_VARS cuFFTMp_LIBRARIES cuFFTMp_INCLUDE_DIRS
  VERSION_VAR    cuFFTMp_VERSION
)

# Provide imported target for modern CMake
if(cuFFTMp_FOUND)
  add_library(cuFFTMp::cuFFTMp UNKNOWN IMPORTED)
  set_target_properties(cuFFTMp::cuFFTMp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${cuFFTMp_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${cuFFTMp_LIBRARIES}"
  )
  message(STATUS "Found cuFFTMp: ${cuFFTMp_LIBRARIES} (include: ${cuFFTMp_INCLUDE_DIRS})")
else()
  message(FATAL_ERROR "Could not find cuFFTMp library or headers. Please set cuFFTMp_ROOT to the installation prefix.")
endif()

mark_as_advanced(cuFFTMp_LIBRARIES cuFFTMp_INCLUDE_DIRS)
