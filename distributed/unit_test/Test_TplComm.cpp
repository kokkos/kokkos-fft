// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_All2All.hpp"

namespace {
#if defined(KOKKOS_ENABLE_SERIAL)
using execution_spaces =
    ::testing::Types<Kokkos::Serial, Kokkos::DefaultHostExecutionSpace,
                     Kokkos::DefaultExecutionSpace>;
#else
using execution_spaces = ::testing::Types<Kokkos::DefaultHostExecutionSpace,
                                          Kokkos::DefaultExecutionSpace>;
#endif

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestTplComm : public ::testing::Test {
  using execution_space_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename ExecutionSpace>
void test_is_comm_constructible() {
  using CommType = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;

  // Constructors
  static_assert(
      std::is_constructible_v<CommType, MPI_Comm, const ExecutionSpace&>);
  static_assert(std::is_constructible_v<CommType, MPI_Comm>);

  // Should not be default constructible
  static_assert(!std::is_default_constructible_v<CommType>);

  // Should not be copyable
  static_assert(!std::is_copy_constructible_v<CommType>);
  static_assert(!std::is_copy_assignable_v<CommType>);

  // Should be movable
  static_assert(std::is_move_constructible_v<CommType>);
  static_assert(std::is_move_assignable_v<CommType>);
}

}  // namespace

TYPED_TEST_SUITE(TestTplComm, execution_spaces);

// Tests for constructibility
TYPED_TEST(TestTplComm, is_constructible) {
  using execution_space_type = typename TestFixture::execution_space_type;
  test_is_comm_constructible<execution_space_type>();
}
