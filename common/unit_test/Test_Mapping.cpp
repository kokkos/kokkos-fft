// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Mapping.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

using int_types    = ::testing::Types<int, std::size_t>;
using layout_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestIsFound : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestGetIndex : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestConvertNegativeAxes : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestMapAxes : public ::testing::Test {
  using layout_type = T;
};

template <typename ContainerType>
void test_is_found() {
  using IntType = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
  ContainerType v{0, 1, 4, 2, 3};

  EXPECT_TRUE(KokkosFFT::Impl::is_found(v, static_cast<IntType>(0)));
  EXPECT_TRUE(KokkosFFT::Impl::is_found(v, static_cast<IntType>(1)));
  EXPECT_TRUE(KokkosFFT::Impl::is_found(v, static_cast<IntType>(2)));
  EXPECT_TRUE(KokkosFFT::Impl::is_found(v, static_cast<IntType>(3)));
  EXPECT_TRUE(KokkosFFT::Impl::is_found(v, static_cast<IntType>(4)));

  if constexpr (std::is_signed_v<IntType>) {
    EXPECT_FALSE(KokkosFFT::Impl::is_found(v, static_cast<IntType>(-1)));
  }
  EXPECT_FALSE(KokkosFFT::Impl::is_found(v, static_cast<IntType>(5)));
}

template <typename ContainerType>
void test_get_index() {
  using IntType = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
  ContainerType v{0, 1, 4, 2, 3};

  EXPECT_EQ(KokkosFFT::Impl::get_index(v, static_cast<IntType>(0)), 0);
  EXPECT_EQ(KokkosFFT::Impl::get_index(v, static_cast<IntType>(1)), 1);
  EXPECT_EQ(KokkosFFT::Impl::get_index(v, static_cast<IntType>(2)), 3);
  EXPECT_EQ(KokkosFFT::Impl::get_index(v, static_cast<IntType>(3)), 4);
  EXPECT_EQ(KokkosFFT::Impl::get_index(v, static_cast<IntType>(4)), 2);

  if constexpr (std::is_signed_v<IntType>) {
    EXPECT_THROW(KokkosFFT::Impl::get_index(v, static_cast<IntType>(-1)),
                 std::runtime_error);
  }
  EXPECT_THROW(KokkosFFT::Impl::get_index(v, static_cast<IntType>(5)),
               std::runtime_error);
}

template <typename IntType>
void test_convert_negative_axes_1d() {
  constexpr std::size_t DIM = 1, Rank = 1;
  using array_type = std::array<IntType, DIM>;

  auto converted_axes_0 =
      KokkosFFT::Impl::convert_negative_axes(array_type{0}, Rank);

  array_type ref_converted_axes_0 = {0};
  EXPECT_EQ(converted_axes_0, ref_converted_axes_0);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-1, 1)
  EXPECT_THROW(
      ({ KokkosFFT::Impl::convert_negative_axes(array_type{1}, Rank); }),
      std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    array_type converted_axes_minus1 =
        KokkosFFT::Impl::convert_negative_axes(array_type{-1}, Rank);
    array_type ref_converted_axes_minus1 = {0};
    EXPECT_EQ(converted_axes_minus1, ref_converted_axes_minus1);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axes(array_type{-2}, Rank); }),
        std::runtime_error);
  }
}

template <typename IntType>
void test_convert_negative_axes_2d() {
  constexpr std::size_t Rank = 2;
  using array_1d_type        = std::array<IntType, 1>;
  using array_2d_type        = std::array<IntType, 2>;

  auto converted_axes_0 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{0}, Rank);
  auto converted_axes_1 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{1}, Rank);
  auto converted_axes_01 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{0, 1}, Rank);

  array_1d_type ref_converted_axis_0  = {0};
  array_1d_type ref_converted_axis_1  = {1};
  array_2d_type ref_converted_axes_01 = {0, 1};

  EXPECT_EQ(converted_axes_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axes_1, ref_converted_axis_1);
  EXPECT_EQ(converted_axes_01, ref_converted_axes_01);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-2, 2)
  EXPECT_THROW(
      ({ KokkosFFT::Impl::convert_negative_axes(array_1d_type{2}, Rank); }),
      std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    array_1d_type ref_converted_axes_minus1  = {1};
    array_2d_type ref_converted_axes_minus21 = {0, 1};
    auto converted_axes_minus1 =
        KokkosFFT::Impl::convert_negative_axes(array_1d_type{-1}, Rank);
    auto converted_axes_minus21 =
        KokkosFFT::Impl::convert_negative_axes(array_2d_type{-2, -1}, Rank);
    EXPECT_EQ(converted_axes_minus1, ref_converted_axes_minus1);
    EXPECT_EQ(converted_axes_minus21, ref_converted_axes_minus21);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axes(array_1d_type{-3}, Rank); }),
        std::runtime_error);
  }
}

template <typename IntType>
void test_convert_negative_axes_3d() {
  constexpr std::size_t Rank = 3;
  using array_1d_type        = std::array<IntType, 1>;
  using array_2d_type        = std::array<IntType, 2>;
  using array_3d_type        = std::array<IntType, 3>;

  auto converted_axes_0 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{0}, Rank);
  auto converted_axes_1 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{1}, Rank);
  auto converted_axes_2 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{2}, Rank);
  auto converted_axes_01 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{0, 1}, Rank);
  auto converted_axes_02 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{0, 2}, Rank);
  auto converted_axes_12 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{1, 2}, Rank);
  auto converted_axes_012 =
      KokkosFFT::Impl::convert_negative_axes(array_3d_type{0, 1, 2}, Rank);

  array_1d_type ref_converted_axes_0   = {0};
  array_1d_type ref_converted_axes_1   = {1};
  array_1d_type ref_converted_axes_2   = {2};
  array_2d_type ref_converted_axes_01  = {0, 1};
  array_2d_type ref_converted_axes_02  = {0, 2};
  array_2d_type ref_converted_axes_12  = {1, 2};
  array_3d_type ref_converted_axes_012 = {0, 1, 2};

  EXPECT_EQ(converted_axes_0, ref_converted_axes_0);
  EXPECT_EQ(converted_axes_1, ref_converted_axes_1);
  EXPECT_EQ(converted_axes_2, ref_converted_axes_2);
  EXPECT_EQ(converted_axes_01, ref_converted_axes_01);
  EXPECT_EQ(converted_axes_02, ref_converted_axes_02);
  EXPECT_EQ(converted_axes_12, ref_converted_axes_12);
  EXPECT_EQ(converted_axes_012, ref_converted_axes_012);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-3, 3)
  EXPECT_THROW(
      ({ KokkosFFT::Impl::convert_negative_axes(array_1d_type{3}, Rank); }),
      std::runtime_error);

  EXPECT_THROW(
      ({
        KokkosFFT::Impl::convert_negative_axes(array_2d_type{3, 1}, Rank);
      }),
      std::runtime_error);

  EXPECT_THROW(
      ({
        KokkosFFT::Impl::convert_negative_axes(array_3d_type{0, 1, 3}, Rank);
      }),
      std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    auto converted_axes_minus1 =
        KokkosFFT::Impl::convert_negative_axes(array_1d_type{-1}, Rank);
    auto converted_axes_minus2 =
        KokkosFFT::Impl::convert_negative_axes(array_1d_type{-2}, Rank);
    auto converted_axes_minus3 =
        KokkosFFT::Impl::convert_negative_axes(array_1d_type{-3}, Rank);

    auto converted_axes_minus21 =
        KokkosFFT::Impl::convert_negative_axes(array_2d_type{-2, -1}, Rank);
    auto converted_axes_minus321 =
        KokkosFFT::Impl::convert_negative_axes(array_3d_type{-3, -2, -1}, Rank);

    array_1d_type ref_converted_axes_minus1   = {2},
                  ref_converted_axes_minus2   = {1},
                  ref_converted_axes_minus3   = {0};
    array_2d_type ref_converted_axes_minus21  = {1, 2};
    array_3d_type ref_converted_axes_minus321 = {0, 1, 2};

    EXPECT_EQ(converted_axes_minus1, ref_converted_axes_minus1);
    EXPECT_EQ(converted_axes_minus2, ref_converted_axes_minus2);
    EXPECT_EQ(converted_axes_minus3, ref_converted_axes_minus3);
    EXPECT_EQ(converted_axes_minus21, ref_converted_axes_minus21);
    EXPECT_EQ(converted_axes_minus321, ref_converted_axes_minus321);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axes(array_1d_type{-4}, Rank); }),
        std::runtime_error);
  }
}

template <typename IntType>
void test_convert_negative_axes_4d() {
  constexpr std::size_t Rank = 4;
  using array_1d_type        = std::array<IntType, 1>;
  using array_2d_type        = std::array<IntType, 2>;
  using array_3d_type        = std::array<IntType, 3>;
  using array_4d_type        = std::array<IntType, 4>;

  auto converted_axes_0 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{0}, Rank);
  auto converted_axes_1 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{1}, Rank);
  auto converted_axes_2 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{2}, Rank);
  auto converted_axes_3 =
      KokkosFFT::Impl::convert_negative_axes(array_1d_type{3}, Rank);
  auto converted_axes_01 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{0, 1}, Rank);
  auto converted_axes_02 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{0, 2}, Rank);
  auto converted_axes_03 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{0, 3}, Rank);
  auto converted_axes_12 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{1, 2}, Rank);
  auto converted_axes_13 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{1, 3}, Rank);
  auto converted_axes_23 =
      KokkosFFT::Impl::convert_negative_axes(array_2d_type{2, 3}, Rank);
  auto converted_axes_012 =
      KokkosFFT::Impl::convert_negative_axes(array_3d_type{0, 1, 2}, Rank);
  auto converted_axes_013 =
      KokkosFFT::Impl::convert_negative_axes(array_3d_type{0, 1, 3}, Rank);
  auto converted_axes_023 =
      KokkosFFT::Impl::convert_negative_axes(array_3d_type{0, 2, 3}, Rank);
  auto converted_axes_123 =
      KokkosFFT::Impl::convert_negative_axes(array_3d_type{1, 2, 3}, Rank);
  auto converted_axes_0123 =
      KokkosFFT::Impl::convert_negative_axes(array_4d_type{0, 1, 2, 3}, Rank);

  array_1d_type ref_converted_axes_0    = {0};
  array_1d_type ref_converted_axes_1    = {1};
  array_1d_type ref_converted_axes_2    = {2};
  array_1d_type ref_converted_axes_3    = {3};
  array_2d_type ref_converted_axes_01   = {0, 1};
  array_2d_type ref_converted_axes_02   = {0, 2};
  array_2d_type ref_converted_axes_03   = {0, 3};
  array_2d_type ref_converted_axes_12   = {1, 2};
  array_2d_type ref_converted_axes_13   = {1, 3};
  array_2d_type ref_converted_axes_23   = {2, 3};
  array_3d_type ref_converted_axes_012  = {0, 1, 2};
  array_3d_type ref_converted_axes_013  = {0, 1, 3};
  array_3d_type ref_converted_axes_023  = {0, 2, 3};
  array_3d_type ref_converted_axes_123  = {1, 2, 3};
  array_4d_type ref_converted_axes_0123 = {0, 1, 2, 3};

  EXPECT_EQ(converted_axes_0, ref_converted_axes_0);
  EXPECT_EQ(converted_axes_1, ref_converted_axes_1);
  EXPECT_EQ(converted_axes_2, ref_converted_axes_2);
  EXPECT_EQ(converted_axes_3, ref_converted_axes_3);
  EXPECT_EQ(converted_axes_01, ref_converted_axes_01);
  EXPECT_EQ(converted_axes_02, ref_converted_axes_02);
  EXPECT_EQ(converted_axes_03, ref_converted_axes_03);
  EXPECT_EQ(converted_axes_12, ref_converted_axes_12);
  EXPECT_EQ(converted_axes_13, ref_converted_axes_13);
  EXPECT_EQ(converted_axes_23, ref_converted_axes_23);
  EXPECT_EQ(converted_axes_012, ref_converted_axes_012);
  EXPECT_EQ(converted_axes_013, ref_converted_axes_013);
  EXPECT_EQ(converted_axes_023, ref_converted_axes_023);
  EXPECT_EQ(converted_axes_123, ref_converted_axes_123);
  EXPECT_EQ(converted_axes_0123, ref_converted_axes_0123);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-4, 4)
  EXPECT_THROW(
      ({ KokkosFFT::Impl::convert_negative_axes(array_1d_type{4}, Rank); }),
      std::runtime_error);

  EXPECT_THROW(
      ({
        KokkosFFT::Impl::convert_negative_axes(array_2d_type{4, 1}, Rank);
      }),
      std::runtime_error);

  EXPECT_THROW(
      ({
        KokkosFFT::Impl::convert_negative_axes(array_3d_type{0, 1, 4}, Rank);
      }),
      std::runtime_error);

  EXPECT_THROW(
      ({
        KokkosFFT::Impl::convert_negative_axes(array_4d_type{0, 1, 2, 4}, Rank);
      }),
      std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    auto converted_axes_minus1 =
        KokkosFFT::Impl::convert_negative_axes(array_1d_type{-1}, Rank);
    auto converted_axes_minus2 =
        KokkosFFT::Impl::convert_negative_axes(array_1d_type{-2}, Rank);
    auto converted_axes_minus3 =
        KokkosFFT::Impl::convert_negative_axes(array_1d_type{-3}, Rank);

    auto converted_axes_minus21 =
        KokkosFFT::Impl::convert_negative_axes(array_2d_type{-2, -1}, Rank);
    auto converted_axes_minus321 =
        KokkosFFT::Impl::convert_negative_axes(array_3d_type{-3, -2, -1}, Rank);
    auto converted_axes_minus4321 = KokkosFFT::Impl::convert_negative_axes(
        array_4d_type{-4, -3, -2, -1}, Rank);

    array_1d_type ref_converted_axis_minus1    = {3},
                  ref_converted_axis_minus2    = {2},
                  ref_converted_axis_minus3    = {1};
    array_2d_type ref_converted_axes_minus21   = {2, 3};
    array_3d_type ref_converted_axes_minus321  = {1, 2, 3};
    array_4d_type ref_converted_axes_minus4321 = {0, 1, 2, 3};

    EXPECT_EQ(converted_axes_minus1, ref_converted_axis_minus1);
    EXPECT_EQ(converted_axes_minus2, ref_converted_axis_minus2);
    EXPECT_EQ(converted_axes_minus3, ref_converted_axis_minus3);
    EXPECT_EQ(converted_axes_minus21, ref_converted_axes_minus21);
    EXPECT_EQ(converted_axes_minus321, ref_converted_axes_minus321);
    EXPECT_EQ(converted_axes_minus4321, ref_converted_axes_minus4321);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axes(array_1d_type{-5}, Rank); }),
        std::runtime_error);
  }
}

// Tests for map axes over ND views
template <typename LayoutType>
void test_map_axes1d() {
  const int len        = 30;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x("x", len);

  auto [map_axes, map_inv_axes] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<1>({0}));

  axes_type<1> ref_map_axes{0};

  EXPECT_TRUE(map_axes == ref_map_axes);
  EXPECT_TRUE(map_inv_axes == ref_map_axes);
}

template <typename LayoutType>
void test_map_axes2d() {
  const int n0 = 3, n1 = 5;
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  RealView2Dtype x("x", n0, n1);

  auto [map_axes_0, map_inv_axes_0] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<1>({0}));
  auto [map_axes_1, map_inv_axes_1] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<1>({1}));
  auto [map_axes_minus1, map_inv_axes_minus1] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<1>({-1}));
  auto [map_axes_0_minus1, map_inv_axes_0_minus1] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<2>({0, -1}));
  auto [map_axes_minus1_0, map_inv_axes_minus1_0] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<2>({-1, 0}));
  auto [map_axes_0_1, map_inv_axes_0_1] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<2>({0, 1}));
  auto [map_axes_1_0, map_inv_axes_1_0] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<2>({1, 0}));

  axes_type<2> ref_map_axes_0, ref_map_inv_axes_0;
  axes_type<2> ref_map_axes_1, ref_map_inv_axes_1;
  axes_type<2> ref_map_axes_minus1, ref_map_inv_axes_minus1;

  axes_type<2> ref_map_axes_0_minus1, ref_map_inv_axes_0_minus1;
  axes_type<2> ref_map_axes_minus1_0, ref_map_inv_axes_minus1_0;
  axes_type<2> ref_map_axes_0_1, ref_map_inv_axes_0_1;
  axes_type<2> ref_map_axes_1_0, ref_map_inv_axes_1_0;

  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    // Layout Left
    ref_map_axes_0 = {0, 1}, ref_map_inv_axes_0 = {0, 1};
    ref_map_axes_1 = {1, 0}, ref_map_inv_axes_1 = {1, 0};
    ref_map_axes_minus1 = {1, 0}, ref_map_inv_axes_minus1 = {1, 0};

    ref_map_axes_0_minus1 = {1, 0}, ref_map_inv_axes_0_minus1 = {1, 0};
    ref_map_axes_minus1_0 = {0, 1}, ref_map_inv_axes_minus1_0 = {0, 1};
    ref_map_axes_0_1 = {1, 0}, ref_map_inv_axes_0_1 = {1, 0};
    ref_map_axes_1_0 = {0, 1}, ref_map_inv_axes_1_0 = {0, 1};
  } else {
    // Layout Right
    ref_map_axes_0 = {1, 0}, ref_map_inv_axes_0 = {1, 0};
    ref_map_axes_1 = {0, 1}, ref_map_inv_axes_1 = {0, 1};
    ref_map_axes_minus1 = {0, 1}, ref_map_inv_axes_minus1 = {0, 1};

    ref_map_axes_0_minus1 = {0, 1}, ref_map_inv_axes_0_minus1 = {0, 1};
    ref_map_axes_minus1_0 = {1, 0}, ref_map_inv_axes_minus1_0 = {1, 0};
    ref_map_axes_0_1 = {0, 1}, ref_map_inv_axes_0_1 = {0, 1};
    ref_map_axes_1_0 = {1, 0}, ref_map_inv_axes_1_0 = {1, 0};
  }

  // Forward mapping
  EXPECT_TRUE(map_axes_0 == ref_map_axes_0);
  EXPECT_TRUE(map_axes_1 == ref_map_axes_1);
  EXPECT_TRUE(map_axes_minus1 == ref_map_axes_minus1);
  EXPECT_TRUE(map_axes_0_minus1 == ref_map_axes_0_minus1);
  EXPECT_TRUE(map_axes_minus1_0 == ref_map_axes_minus1_0);
  EXPECT_TRUE(map_axes_0_1 == ref_map_axes_0_1);
  EXPECT_TRUE(map_axes_1_0 == ref_map_axes_1_0);

  // Inverse mapping
  EXPECT_TRUE(map_inv_axes_0 == ref_map_inv_axes_0);
  EXPECT_TRUE(map_inv_axes_1 == ref_map_inv_axes_1);
  EXPECT_TRUE(map_inv_axes_minus1 == ref_map_inv_axes_minus1);
  EXPECT_TRUE(map_inv_axes_0_minus1 == ref_map_inv_axes_0_minus1);
  EXPECT_TRUE(map_inv_axes_minus1_0 == ref_map_inv_axes_minus1_0);
  EXPECT_TRUE(map_inv_axes_0_1 == ref_map_inv_axes_0_1);
  EXPECT_TRUE(map_inv_axes_1_0 == ref_map_inv_axes_1_0);
}

template <typename LayoutType>
void test_map_axes3d() {
  const int n0 = 3, n1 = 5, n2 = 8;
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  RealView3Dtype x("x", n0, n1, n2);

  auto [map_axes_0, map_inv_axes_0] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<1>({0}));
  auto [map_axes_1, map_inv_axes_1] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<1>({1}));
  auto [map_axes_2, map_inv_axes_2] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<1>({2}));

  auto [map_axes_0_1, map_inv_axes_0_1] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<2>({0, 1}));
  auto [map_axes_0_2, map_inv_axes_0_2] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<2>({0, 2}));
  auto [map_axes_1_0, map_inv_axes_1_0] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<2>({1, 0}));
  auto [map_axes_1_2, map_inv_axes_1_2] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<2>({1, 2}));
  auto [map_axes_2_0, map_inv_axes_2_0] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<2>({2, 0}));
  auto [map_axes_2_1, map_inv_axes_2_1] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<2>({2, 1}));

  auto [map_axes_0_1_2, map_inv_axes_0_1_2] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<3>({0, 1, 2}));
  auto [map_axes_0_2_1, map_inv_axes_0_2_1] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<3>({0, 2, 1}));

  auto [map_axes_1_0_2, map_inv_axes_1_0_2] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<3>({1, 0, 2}));
  auto [map_axes_1_2_0, map_inv_axes_1_2_0] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<3>({1, 2, 0}));
  auto [map_axes_2_0_1, map_inv_axes_2_0_1] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<3>({2, 0, 1}));
  auto [map_axes_2_1_0, map_inv_axes_2_1_0] =
      KokkosFFT::Impl::get_map_axes(x, axes_type<3>({2, 1, 0}));

  axes_type<3> ref_map_axes_0, ref_map_inv_axes_0;
  axes_type<3> ref_map_axes_1, ref_map_inv_axes_1;
  axes_type<3> ref_map_axes_2, ref_map_inv_axes_2;

  axes_type<3> ref_map_axes_0_1, ref_map_inv_axes_0_1;
  axes_type<3> ref_map_axes_0_2, ref_map_inv_axes_0_2;
  axes_type<3> ref_map_axes_1_0, ref_map_inv_axes_1_0;
  axes_type<3> ref_map_axes_1_2, ref_map_inv_axes_1_2;
  axes_type<3> ref_map_axes_2_0, ref_map_inv_axes_2_0;
  axes_type<3> ref_map_axes_2_1, ref_map_inv_axes_2_1;

  axes_type<3> ref_map_axes_0_1_2, ref_map_inv_axes_0_1_2;
  axes_type<3> ref_map_axes_0_2_1, ref_map_inv_axes_0_2_1;
  axes_type<3> ref_map_axes_1_0_2, ref_map_inv_axes_1_0_2;
  axes_type<3> ref_map_axes_1_2_0, ref_map_inv_axes_1_2_0;
  axes_type<3> ref_map_axes_2_0_1, ref_map_inv_axes_2_0_1;
  axes_type<3> ref_map_axes_2_1_0, ref_map_inv_axes_2_1_0;

  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    // Layout Left
    ref_map_axes_0 = {0, 1, 2}, ref_map_inv_axes_0 = {0, 1, 2};
    ref_map_axes_1 = {1, 0, 2}, ref_map_inv_axes_1 = {1, 0, 2};
    ref_map_axes_2 = {2, 0, 1}, ref_map_inv_axes_2 = {1, 2, 0};

    ref_map_axes_0_1 = {1, 0, 2}, ref_map_inv_axes_0_1 = {1, 0, 2};
    ref_map_axes_0_2 = {2, 0, 1}, ref_map_inv_axes_0_2 = {1, 2, 0};
    ref_map_axes_1_0 = {0, 1, 2}, ref_map_inv_axes_1_0 = {0, 1, 2};
    ref_map_axes_1_2 = {2, 1, 0}, ref_map_inv_axes_1_2 = {2, 1, 0};
    ref_map_axes_2_0 = {0, 2, 1}, ref_map_inv_axes_2_0 = {0, 2, 1};
    ref_map_axes_2_1 = {1, 2, 0}, ref_map_inv_axes_2_1 = {2, 0, 1};

    ref_map_axes_0_1_2 = {2, 1, 0}, ref_map_inv_axes_0_1_2 = {2, 1, 0};
    ref_map_axes_0_2_1 = {1, 2, 0}, ref_map_inv_axes_0_2_1 = {2, 0, 1};
    ref_map_axes_1_0_2 = {2, 0, 1}, ref_map_inv_axes_1_0_2 = {1, 2, 0};
    ref_map_axes_1_2_0 = {0, 2, 1}, ref_map_inv_axes_1_2_0 = {0, 2, 1};
    ref_map_axes_2_0_1 = {1, 0, 2}, ref_map_inv_axes_2_0_1 = {1, 0, 2};
    ref_map_axes_2_1_0 = {0, 1, 2}, ref_map_inv_axes_2_1_0 = {0, 1, 2};
  } else {
    // Layout Right
    ref_map_axes_0 = {1, 2, 0}, ref_map_inv_axes_0 = {2, 0, 1};
    ref_map_axes_1 = {0, 2, 1}, ref_map_inv_axes_1 = {0, 2, 1};
    ref_map_axes_2 = {0, 1, 2}, ref_map_inv_axes_2 = {0, 1, 2};

    ref_map_axes_0_1 = {2, 0, 1}, ref_map_inv_axes_0_1 = {1, 2, 0};
    ref_map_axes_0_2 = {1, 0, 2}, ref_map_inv_axes_0_2 = {1, 0, 2};
    ref_map_axes_1_0 = {2, 1, 0}, ref_map_inv_axes_1_0 = {2, 1, 0};
    ref_map_axes_1_2 = {0, 1, 2}, ref_map_inv_axes_1_2 = {0, 1, 2};
    ref_map_axes_2_0 = {1, 2, 0}, ref_map_inv_axes_2_0 = {2, 0, 1};
    ref_map_axes_2_1 = {0, 2, 1}, ref_map_inv_axes_2_1 = {0, 2, 1};

    ref_map_axes_0_1_2 = {0, 1, 2}, ref_map_inv_axes_0_1_2 = {0, 1, 2};
    ref_map_axes_0_2_1 = {0, 2, 1}, ref_map_inv_axes_0_2_1 = {0, 2, 1};
    ref_map_axes_1_0_2 = {1, 0, 2}, ref_map_inv_axes_1_0_2 = {1, 0, 2};
    ref_map_axes_1_2_0 = {1, 2, 0}, ref_map_inv_axes_1_2_0 = {2, 0, 1};
    ref_map_axes_2_0_1 = {2, 0, 1}, ref_map_inv_axes_2_0_1 = {1, 2, 0};
    ref_map_axes_2_1_0 = {2, 1, 0}, ref_map_inv_axes_2_1_0 = {2, 1, 0};
  }

  // Forward mapping
  EXPECT_TRUE(map_axes_0 == ref_map_axes_0);
  EXPECT_TRUE(map_axes_1 == ref_map_axes_1);
  EXPECT_TRUE(map_axes_2 == ref_map_axes_2);

  EXPECT_TRUE(map_axes_0_1 == ref_map_axes_0_1);
  EXPECT_TRUE(map_axes_0_2 == ref_map_axes_0_2);
  EXPECT_TRUE(map_axes_1_0 == ref_map_axes_1_0);
  EXPECT_TRUE(map_axes_1_2 == ref_map_axes_1_2);
  EXPECT_TRUE(map_axes_2_0 == ref_map_axes_2_0);
  EXPECT_TRUE(map_axes_2_1 == ref_map_axes_2_1);

  EXPECT_TRUE(map_axes_0_1_2 == ref_map_axes_0_1_2);
  EXPECT_TRUE(map_axes_0_2_1 == ref_map_axes_0_2_1);
  EXPECT_TRUE(map_axes_1_0_2 == ref_map_axes_1_0_2);
  EXPECT_TRUE(map_axes_1_2_0 == ref_map_axes_1_2_0);
  EXPECT_TRUE(map_axes_2_0_1 == ref_map_axes_2_0_1);
  EXPECT_TRUE(map_axes_2_1_0 == ref_map_axes_2_1_0);

  // Inverse mapping
  EXPECT_TRUE(map_inv_axes_0 == ref_map_inv_axes_0);
  EXPECT_TRUE(map_inv_axes_1 == ref_map_inv_axes_1);
  EXPECT_TRUE(map_inv_axes_2 == ref_map_inv_axes_2);

  EXPECT_TRUE(map_inv_axes_0_1 == ref_map_inv_axes_0_1);
  EXPECT_TRUE(map_inv_axes_0_2 == ref_map_inv_axes_0_2);
  EXPECT_TRUE(map_inv_axes_1_0 == ref_map_inv_axes_1_0);
  EXPECT_TRUE(map_inv_axes_1_2 == ref_map_inv_axes_1_2);
  EXPECT_TRUE(map_inv_axes_2_0 == ref_map_inv_axes_2_0);
  EXPECT_TRUE(map_inv_axes_2_1 == ref_map_inv_axes_2_1);

  EXPECT_TRUE(map_inv_axes_0_1_2 == ref_map_inv_axes_0_1_2);
  EXPECT_TRUE(map_inv_axes_0_2_1 == ref_map_inv_axes_0_2_1);
  EXPECT_TRUE(map_inv_axes_1_0_2 == ref_map_inv_axes_1_0_2);
  EXPECT_TRUE(map_inv_axes_1_2_0 == ref_map_inv_axes_1_2_0);
  EXPECT_TRUE(map_inv_axes_2_0_1 == ref_map_inv_axes_2_0_1);
  EXPECT_TRUE(map_inv_axes_2_1_0 == ref_map_inv_axes_2_1_0);
}
}  // namespace

TYPED_TEST_SUITE(TestIsFound, int_types);
TYPED_TEST_SUITE(TestGetIndex, int_types);
TYPED_TEST_SUITE(TestConvertNegativeAxes, int_types);
TYPED_TEST_SUITE(TestMapAxes, layout_types);

TYPED_TEST(TestIsFound, is_found_from_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  test_is_found<vector_type>();
}

TYPED_TEST(TestIsFound, is_found_from_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 5>;
  test_is_found<array_type>();
}

TYPED_TEST(TestGetIndex, get_index_from_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  test_get_index<vector_type>();
}

TYPED_TEST(TestGetIndex, get_index_from_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 5>;
  test_get_index<array_type>();
}

TYPED_TEST(TestConvertNegativeAxes, 1DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axes_1d<value_type>();
}

TYPED_TEST(TestConvertNegativeAxes, 2DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axes_2d<value_type>();
}

TYPED_TEST(TestConvertNegativeAxes, 3DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axes_3d<value_type>();
}

TYPED_TEST(TestConvertNegativeAxes, 4DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axes_4d<value_type>();
}

// Tests for 1D View
TYPED_TEST(TestMapAxes, 1DView) {
  using layout_type = typename TestFixture::layout_type;
  test_map_axes1d<layout_type>();
}

// Tests for 2D View
TYPED_TEST(TestMapAxes, 2DView) {
  using layout_type = typename TestFixture::layout_type;
  test_map_axes2d<layout_type>();
}

// Tests for 3D View
TYPED_TEST(TestMapAxes, 3DView) {
  using layout_type = typename TestFixture::layout_type;
  test_map_axes3d<layout_type>();
}
