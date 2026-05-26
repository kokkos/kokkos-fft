# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.22)

set(CTEST_BINARY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT "${GITHUB_PR_ID}" STREQUAL "")
  set(CTEST_CHANGE_ID ${GITHUB_PR_ID})
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}-PR${GITHUB_PR_ID}")
endif()

set(CTEST_UPDATE_COMMAND git)
set(CTEST_UPDATE_VERSION_ONLY 1)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")

file(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}")

ctest_start(${CDASH_MODEL})
ctest_update()
ctest_configure(OPTIONS "${CMAKE_OPTIONS}" RETURN_VALUE config_result)
ctest_build(RETURN_VALUE build_result)

# Only run tests if the build was successful
if (NOT build_result)
  ctest_test(RETURN_VALUE test_result)
endif()

# Always submit results, even if there were errors
ctest_submit()

if(config_result)
  message(FATAL_ERROR "Error during configuration! Exit code: ${config_result}")
endif()
if(build_result)
  message(FATAL_ERROR "Error during build! Exit code: ${build_result}")
endif()
if(test_result)
  message(FATAL_ERROR "Error during testing! Exit code: ${test_result}")
endif()
