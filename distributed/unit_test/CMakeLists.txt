# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

set(SOURCES
        Test_Main.cpp
        Test_TplComm.cpp
        Test_MPI_Types.cpp
        Test_All2All.cpp
    )

add_executable(unit-tests-mpi ${SOURCES})

target_compile_features(unit-tests-mpi PUBLIC cxx_std_17)
target_link_libraries(unit-tests-mpi PUBLIC GTest::gtest GTest::gmock KokkosFFT::distributed)

# Enable GoogleTest
include(GoogleTest)

# This is the core instruction for CTest.
# NAME: A descriptive name for the test.
# COMMAND: The exact command to run.
#  - ${MPIEXEC_EXECUTABLE}: Variable from find_package(MPI) for mpiexec/mpirun.
#  - -n 2: The flag to specify running with 2 processes.
#  - $<TARGET_FILE:unit-tests-mpi>: A CMake generator expression that correctly
#   resolves to the path of our compiled executable.
add_test(
  NAME UnitTests1MPI
  COMMAND ${MPIEXEC_EXECUTABLE} -n 1 $<TARGET_FILE:unit-tests-mpi>
)
add_test(
  NAME UnitTests2MPI
  COMMAND ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:unit-tests-mpi>
)
add_test(
  NAME UnitTests4MPI
  COMMAND ${MPIEXEC_EXECUTABLE} -n 4 $<TARGET_FILE:unit-tests-mpi>
)

set_property(TEST UnitTests1MPI PROPERTY TIMEOUT 600)
set_property(TEST UnitTests2MPI PROPERTY TIMEOUT 600)
set_property(TEST UnitTests4MPI PROPERTY TIMEOUT 1200)
