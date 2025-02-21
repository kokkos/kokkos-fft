# SPDX-FileCopyrightText: (C) 2015 Wenzel Jakob
# SPDX-FileCopyrightText: (C) 2015 Patrick Bos
#
# SPDX-License-Identifier: BSD-3-Clause

# - Find the FFTW library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] [COMPONENTS component1 ... componentX] )
#
# It sets the following variables:
#   FFTW_FOUND                  ... true if fftw is found on the system
#   FFTW_[component]_LIB_FOUND  ... true if the component is found on the system (see components below)
#   FFTW_LIBRARIES              ... full paths to all found fftw libraries
#   FFTW_[component]_LIB        ... full path to one of the components (see below)
#   FFTW_INCLUDE_DIRS           ... fftw include directory paths
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS        ... if true, only static libraries are found, otherwise both static and shared.
#   FFTW_ROOT                   ... if set, the libraries are exclusively searched
#                                   under this path
#
# This package supports the following components:
#   FLOAT_LIB
#   DOUBLE_LIB
#   LONGDOUBLE_LIB
#   FLOAT_THREADS_LIB
#   DOUBLE_THREADS_LIB
#   LONGDOUBLE_THREADS_LIB
#   FLOAT_OPENMP_LIB
#   DOUBLE_OPENMP_LIB
#   LONGDOUBLE_OPENMP_LIB
#

# TODO (maybe): extend with ExternalProject download + build option
# TODO: put on conda-forge

# CMake module for finding FFTW 3 using find_package
#
# # Usage
#
# Once added to your project, this module allows you to find FFTW libraries and headers using the CMake `find_package` command:
#
# ```cmake
# find_package(FFTW [REQUIRED] [QUIET] [COMPONENTS component1 ... componentX] )
# ```
#
# This module sets the following variables:
# - `FFTW_FOUND`                  ... true if fftw is found on the system
# - `FFTW_[component]_LIB_FOUND`  ... true if the component is found on the system (see components below)
# - `FFTW_LIBRARIES`              ... full paths to all found fftw libraries
# - `FFTW_[component]_LIB`        ... full path to one of the components (see below)
# - `FFTW_INCLUDE_DIRS`           ... fftw include directory paths
#
# The following variables will be checked by the module:
# - `FFTW_USE_STATIC_LIBS`        ... if true, only static libraries are found, otherwise both static and shared.
# - `FFTW_ROOT`                   ... if set, the libraries are exclusively searched under this path.
#
# This package supports the following components:
# - `FLOAT_LIB`
# - `DOUBLE_LIB`
# - `LONGDOUBLE_LIB`
# - `FLOAT_THREADS_LIB`
# - `DOUBLE_THREADS_LIB`
# - `LONGDOUBLE_THREADS_LIB`
# - `FLOAT_OPENMP_LIB`
# - `DOUBLE_OPENMP_LIB`
# - `LONGDOUBLE_OPENMP_LIB`
# - `FLOAT_MPI_LIB`
# - `DOUBLE_MPI_LIB`
# - `LONGDOUBLE_MPI_LIB`
#
# and the following linking targets
#
# - `FFTW::Float`
# - `FFTW::Double`
# - `FFTW::LongDouble`
# - `FFTW::FloatThreads`
# - `FFTW::DoubleThreads`
# - `FFTW::LongDoubleThreads`
# - `FFTW::FloatOpenMP`
# - `FFTW::DoubleOpenMP`
# - `FFTW::LongDoubleOpenMP`
# - `FFTW::FloatMPI`
# - `FFTW::DoubleMPI`
# - `FFTW::LongDoubleMPI`
#
# # Adding to your project
#
# ## Automatic download from CMake project
#
# Copy the following into the `CMakeLists.txt` file of the project you want to use FindFFTW in:
# ```cmake
# configure_file(downloadFindFFTW.cmake.in findFFTW-download/CMakeLists.txt)
# execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#         RESULT_VARIABLE result
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
# if(result)
#     message(FATAL_ERROR "CMake step for findFFTW failed: ${result}")
#     else()
#     message("CMake step for findFFTW completed (${result}).")
# endif()
# execute_process(COMMAND ${CMAKE_COMMAND} --build .
#         RESULT_VARIABLE result
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
# if(result)
#     message(FATAL_ERROR "Build step for findFFTW failed: ${result}")
# endif()
#
# set(findFFTW_DIR ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src)
#
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${findFFTW_DIR}")
# ```
#
# And add a file called `downloadFindFFTW.cmake.in` to your project containing the following:
# ```cmake
# cmake_minimum_required(VERSION 2.8.2)
#
# project(findFFTW-download NONE)
#
# include(ExternalProject)
#
# ExternalProject_Add(findFFTW_download
#     GIT_REPOSITORY    "https://github.com/egpbos/findfftw.git"
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND     ""
#     INSTALL_COMMAND   ""
#     TEST_COMMAND      ""
#     SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src"
#     BINARY_DIR        ""
#     INSTALL_DIR       ""
# )
# ```
#
# After this, `find_package(FFTW)` can be used in the `CMakeLists.txt` file.
#
# ## Manual
#
# Clone the repository into directory `PREFIX/findFFTW`:
# ```sh
# git clone https://github.com/egpbos/findfftw.git PREFIX/findFFTW
# ```
#
# Then add the following to your `CMakeLists.txt` to allow CMake to find the module:
# ```cmake
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "PREFIX/findFFTW")
# ```

if(NOT FFTW_ROOT AND DEFINED ENV{FFTWDIR})
  set(FFTW_ROOT $ENV{FFTWDIR})
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if(PKG_CONFIG_FOUND AND NOT FFTW_ROOT)
  pkg_check_modules(PKG_FFTW QUIET "fftw3")
endif()

#Check whether to search static or dynamic libs
set(CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(${FFTW_USE_STATIC_LIBS})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV})
endif()
if(FFTW_ROOT)
  # find libs

  find_library(
    FFTW_DOUBLE_LIB
    NAMES "fftw3" libfftw3-3
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_DOUBLE_THREADS_LIB
    NAMES "fftw3_threads"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_DOUBLE_OPENMP_LIB
    NAMES "fftw3_omp"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_DOUBLE_MPI_LIB
    NAMES "fftw3_mpi"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_FLOAT_LIB
    NAMES "fftw3f" libfftw3f-3
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_FLOAT_THREADS_LIB
    NAMES "fftw3f_threads"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_FLOAT_OPENMP_LIB
    NAMES "fftw3f_omp"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_FLOAT_MPI_LIB
    NAMES "fftw3f_mpi"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_LONGDOUBLE_LIB
    NAMES "fftw3l" libfftw3l-3
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_LONGDOUBLE_THREADS_LIB
    NAMES "fftw3l_threads"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_LONGDOUBLE_OPENMP_LIB
    NAMES "fftw3l_omp"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTW_LONGDOUBLE_MPI_LIB
    NAMES "fftw3l_mpi"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
  )

  #find includes
  find_path(
    FFTW_INCLUDE_DIRS
    NAMES "fftw3.h"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
  )
else()
  find_library(FFTW_DOUBLE_LIB NAMES "fftw3" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_DOUBLE_THREADS_LIB NAMES "fftw3_threads" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_DOUBLE_OPENMP_LIB NAMES "fftw3_omp" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_DOUBLE_MPI_LIB NAMES "fftw3_mpi" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_FLOAT_LIB NAMES "fftw3f" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_FLOAT_THREADS_LIB NAMES "fftw3f_threads" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_FLOAT_OPENMP_LIB NAMES "fftw3f_omp" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_FLOAT_MPI_LIB NAMES "fftw3f_mpi" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_LONGDOUBLE_LIB NAMES "fftw3l" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_LONGDOUBLE_THREADS_LIB NAMES "fftw3l_threads" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_LONGDOUBLE_OPENMP_LIB NAMES "fftw3l_omp" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_library(FFTW_LONGDOUBLE_MPI_LIB NAMES "fftw3l_mpi" PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR})
  find_path(FFTW_INCLUDE_DIRS NAMES "fftw3.h" PATHS ${PKG_FFTW_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR})
endif(FFTW_ROOT)

#--------------------------------------- components

if(FFTW_DOUBLE_LIB)
  set(FFTW_DOUBLE_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_DOUBLE_LIB})
  add_library(FFTW::Double INTERFACE IMPORTED)
  set_target_properties(
    FFTW::Double PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                 "${FFTW_DOUBLE_LIB}"
  )
else()
  set(FFTW_DOUBLE_LIB_FOUND FALSE)
endif()

if(FFTW_FLOAT_LIB)
  set(FFTW_FLOAT_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_FLOAT_LIB})
  add_library(FFTW::Float INTERFACE IMPORTED)
  set_target_properties(
    FFTW::Float PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                "${FFTW_FLOAT_LIB}"
  )
else()
  set(FFTW_FLOAT_LIB_FOUND FALSE)
endif()

if(FFTW_LONGDOUBLE_LIB)
  set(FFTW_LONGDOUBLE_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_LONGDOUBLE_LIB})
  add_library(FFTW::LongDouble INTERFACE IMPORTED)
  set_target_properties(
    FFTW::LongDouble PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                     "${FFTW_LONGDOUBLE_LIB}"
  )
else()
  set(FFTW_LONGDOUBLE_LIB_FOUND FALSE)
endif()

if(FFTW_DOUBLE_THREADS_LIB)
  set(FFTW_DOUBLE_THREADS_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_DOUBLE_THREADS_LIB})
  add_library(FFTW::DoubleThreads INTERFACE IMPORTED)
  set_target_properties(
    FFTW::DoubleThreads PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                        "${FFTW_DOUBLE_THREADS_LIB}"
  )
else()
  set(FFTW_DOUBLE_THREADS_LIB_FOUND FALSE)
endif()

if(FFTW_FLOAT_THREADS_LIB)
  set(FFTW_FLOAT_THREADS_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_FLOAT_THREADS_LIB})
  add_library(FFTW::FloatThreads INTERFACE IMPORTED)
  set_target_properties(
    FFTW::FloatThreads PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                       "${FFTW_FLOAT_THREADS_LIB}"
  )
else()
  set(FFTW_FLOAT_THREADS_LIB_FOUND FALSE)
endif()

if(FFTW_LONGDOUBLE_THREADS_LIB)
  set(FFTW_LONGDOUBLE_THREADS_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_LONGDOUBLE_THREADS_LIB})
  add_library(FFTW::LongDoubleThreads INTERFACE IMPORTED)
  set_target_properties(
    FFTW::LongDoubleThreads PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
                                       INTERFACE_LINK_LIBRARIES "${FFTW_LONGDOUBLE_THREADS_LIB}"
  )
else()
  set(FFTW_LONGDOUBLE_THREADS_LIB_FOUND FALSE)
endif()

if(FFTW_DOUBLE_OPENMP_LIB)
  set(FFTW_DOUBLE_OPENMP_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_DOUBLE_OPENMP_LIB})
  add_library(FFTW::DoubleOpenMP INTERFACE IMPORTED)
  set_target_properties(
    FFTW::DoubleOpenMP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                       "${FFTW_DOUBLE_OPENMP_LIB}"
  )
else()
  set(FFTW_DOUBLE_OPENMP_LIB_FOUND FALSE)
endif()

if(FFTW_FLOAT_OPENMP_LIB)
  set(FFTW_FLOAT_OPENMP_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_FLOAT_OPENMP_LIB})
  add_library(FFTW::FloatOpenMP INTERFACE IMPORTED)
  set_target_properties(
    FFTW::FloatOpenMP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                      "${FFTW_FLOAT_OPENMP_LIB}"
  )
else()
  set(FFTW_FLOAT_OPENMP_LIB_FOUND FALSE)
endif()

if(FFTW_LONGDOUBLE_OPENMP_LIB)
  set(FFTW_LONGDOUBLE_OPENMP_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_LONGDOUBLE_OPENMP_LIB})
  add_library(FFTW::LongDoubleOpenMP INTERFACE IMPORTED)
  set_target_properties(
    FFTW::LongDoubleOpenMP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
                                      INTERFACE_LINK_LIBRARIES "${FFTW_LONGDOUBLE_OPENMP_LIB}"
  )
else()
  set(FFTW_LONGDOUBLE_OPENMP_LIB_FOUND FALSE)
endif()

if(FFTW_DOUBLE_MPI_LIB)
  set(FFTW_DOUBLE_MPI_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_DOUBLE_MPI_LIB})
  add_library(FFTW::DoubleMPI INTERFACE IMPORTED)
  set_target_properties(
    FFTW::DoubleMPI PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                    "${FFTW_DOUBLE_MPI_LIB}"
  )
else()
  set(FFTW_DOUBLE_MPI_LIB_FOUND FALSE)
endif()

if(FFTW_FLOAT_MPI_LIB)
  set(FFTW_FLOAT_MPI_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_FLOAT_MPI_LIB})
  add_library(FFTW::FloatMPI INTERFACE IMPORTED)
  set_target_properties(
    FFTW::FloatMPI PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                   "${FFTW_FLOAT_MPI_LIB}"
  )
else()
  set(FFTW_FLOAT_MPI_LIB_FOUND FALSE)
endif()

if(FFTW_LONGDOUBLE_MPI_LIB)
  set(FFTW_LONGDOUBLE_MPI_LIB_FOUND TRUE)
  set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_LONGDOUBLE_MPI_LIB})
  add_library(FFTW::LongDoubleMPI INTERFACE IMPORTED)
  set_target_properties(
    FFTW::LongDoubleMPI PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES
                                                                                        "${FFTW_LONGDOUBLE_MPI_LIB}"
  )
else()
  set(FFTW_LONGDOUBLE_MPI_LIB_FOUND FALSE)
endif()

#--------------------------------------- end components

set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FFTW REQUIRED_VARS FFTW_INCLUDE_DIRS HANDLE_COMPONENTS)

mark_as_advanced(
  FFTW_INCLUDE_DIRS
  FFTW_LIBRARIES
  FFTW_FLOAT_LIB
  FFTW_DOUBLE_LIB
  FFTW_LONGDOUBLE_LIB
  FFTW_FLOAT_THREADS_LIB
  FFTW_DOUBLE_THREADS_LIB
  FFTW_LONGDOUBLE_THREADS_LIB
  FFTW_FLOAT_OPENMP_LIB
  FFTW_DOUBLE_OPENMP_LIB
  FFTW_LONGDOUBLE_OPENMP_LIB
  FFTW_FLOAT_MPI_LIB
  FFTW_DOUBLE_MPI_LIB
  FFTW_LONGDOUBLE_MPI_LIB
)
