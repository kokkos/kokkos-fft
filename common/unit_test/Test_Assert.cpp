// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"

namespace {
// Int like types
using base_int_types = ::testing::Types<int, std::size_t>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestContainerToString : public ::testing::Test {
  using value_type = T;
};

template <typename ContainerType, typename EmptyContainerType>
void test_container_to_string() {
  ContainerType container{1, 2, 3};
  std::string expected = "Container: (1, 2, 3)";
  std::string actual =
      KokkosFFT::Impl::container_to_string("Container: ", container);
  EXPECT_EQ(expected, actual)
      << "Expected: " << expected << ", but got: " << actual;

  EmptyContainerType empty_container{};
  expected = "Container: ()";
  actual = KokkosFFT::Impl::container_to_string("Container: ", empty_container);
  EXPECT_EQ(expected, actual)
      << "Expected: " << expected << ", but got: " << actual;

  ContainerType unnamed_container{4, 5, 6};
  expected = "(4, 5, 6)";
  actual   = KokkosFFT::Impl::container_to_string("", unnamed_container);
  EXPECT_EQ(expected, actual)
      << "Expected: " << expected << ", but got: " << actual;
}

}  // namespace

TYPED_TEST_SUITE(TestContainerToString, base_int_types);

TYPED_TEST(TestContainerToString, array_to_string) {
  using value_type           = typename TestFixture::value_type;
  using container_type       = std::array<value_type, 3>;
  using empty_container_type = std::array<value_type, 0>;
  test_container_to_string<container_type, empty_container_type>();
}

TYPED_TEST(TestContainerToString, vector_to_string) {
  using value_type           = typename TestFixture::value_type;
  using container_type       = std::vector<value_type>;
  using empty_container_type = std::vector<value_type>;
  test_container_to_string<container_type, empty_container_type>();
}
