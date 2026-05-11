// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_utils.hpp"

namespace {
// Int like types
using base_int_types = ::testing::Types<int, std::size_t>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct ContainerTypes : public ::testing::Test {
  static constexpr std::size_t rank = 5;
  using value_type                  = T;
  using vector_type                 = std::vector<T>;
  using array_type                  = std::array<T, rank>;
};

template <typename iType>
void test_array_to_vector(iType nprocs) {
  using vector_type      = std::vector<iType>;
  using array_type       = std::array<iType, 3>;
  using const_array_type = const std::array<iType, 3>;
  array_type arr = {nprocs, 1, 8}, arr_ref = {nprocs, 1, 8};
  const_array_type carr = {nprocs, 1, 8}, carr_ref = {nprocs, 1, 8};
  vector_type ref_vec = {nprocs, 1, 8};

  // Test for Lvalue
  auto vec  = KokkosFFT::Impl::to_vector(arr);
  auto cvec = KokkosFFT::Impl::to_vector(carr);
  EXPECT_EQ(vec, ref_vec);
  EXPECT_EQ(cvec, ref_vec);
  EXPECT_EQ(arr, arr_ref) << "Input container modified in lvalue test";
  EXPECT_EQ(carr, carr_ref) << "Input container modified in lvalue test";

  // Test for Rvalue
  auto vec_tmp  = KokkosFFT::Impl::to_vector(array_type{nprocs, 1, 8});
  auto vec_move = KokkosFFT::Impl::to_vector(std::move(arr));
  EXPECT_EQ(vec_tmp, ref_vec);
  EXPECT_EQ(vec_move, ref_vec);
  EXPECT_EQ(arr, arr_ref) << "Input container modified in rvalue test";
}

}  // namespace

TYPED_TEST_SUITE(ContainerTypes, base_int_types);

TYPED_TEST(ContainerTypes, array_to_vector) {
  using value_type = typename TestFixture::value_type;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_array_to_vector<value_type>(nprocs);
  }
}

TEST(ToArray, lvalue) {
  std::array arr{1, 2, 3};
  ASSERT_EQ(KokkosFFT::Impl::to_array(arr), (Kokkos::Array{1, 2, 3}));
}

TEST(ToArray, rvalue) {
  ASSERT_EQ(KokkosFFT::Impl::to_array(std::array{1, 2}), (Kokkos::Array{1, 2}));
}
