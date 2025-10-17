// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Int like types
using base_int_types   = ::testing::Types<int, std::size_t>;
using signed_int_types = ::testing::Types<int, std::int32_t, std::int64_t>;

// value type combinations that are tested
using paired_scalar_types = ::testing::Types<
    std::pair<float, float>, std::pair<float, Kokkos::complex<float>>,
    std::pair<Kokkos::complex<float>, Kokkos::complex<float>>,
    std::pair<double, double>, std::pair<double, Kokkos::complex<double>>,
    std::pair<Kokkos::complex<double>, Kokkos::complex<double>>>;

using paired_int_types =
    ::testing::Types<std::pair<int, int>, std::pair<int, std::size_t>,
                     std::pair<std::size_t, int>,
                     std::pair<std::size_t, std::size_t>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestConvertNegativeAxis : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestConvertNegativeAxes : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct ContainerTypes : public ::testing::Test {
  static constexpr std::size_t rank = 5;
  using value_type                  = T;
  using vector_type                 = std::vector<T>;
  using array_type                  = std::array<T, rank>;
};

template <typename T>
struct PairedScalarTypes : public ::testing::Test {
  using value_type1 = typename T::first_type;
  using value_type2 = typename T::second_type;
};

template <typename T>
struct TestIndexSequence : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestConvertBaseTypes : public ::testing::Test {
  using value_type1 = typename T::first_type;
  using value_type2 = typename T::second_type;
};

// Tests for convert_negative_axes over ND views
template <typename IntType>
void test_convert_negative_axis_1d() {
  constexpr std::size_t Rank = 1;
  IntType axis0 = 0, axis1 = 1;
  IntType converted_axis_0 =
      KokkosFFT::Impl::convert_negative_axis(axis0, Rank);

  IntType ref_converted_axis_0 = 0;
  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-1, 1)
  EXPECT_THROW(({ KokkosFFT::Impl::convert_negative_axis(axis1, Rank); }),
               std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    IntType axis_minus1 = -1, axis_minus2 = -2;
    IntType converted_axis_minus1 =
        KokkosFFT::Impl::convert_negative_axis(axis_minus1, Rank);
    IntType ref_converted_axis_minus1 = 0;
    EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axis(axis_minus2, Rank); }),
        std::runtime_error);
  }
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
void test_convert_negative_axis_2d() {
  constexpr std::size_t Rank = 2;
  IntType axis0 = 0, axis1 = 1, axis2 = 2;
  IntType converted_axis_0 =
      KokkosFFT::Impl::convert_negative_axis(axis0, Rank);
  IntType converted_axis_1 =
      KokkosFFT::Impl::convert_negative_axis(axis1, Rank);

  IntType ref_converted_axis_0 = 0;
  IntType ref_converted_axis_1 = 1;

  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axis_1, ref_converted_axis_1);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-2, 2)
  EXPECT_THROW(({ KokkosFFT::Impl::convert_negative_axis(axis2, Rank); }),
               std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    IntType axis_minus1 = -1, axis_minus3 = -3;
    IntType converted_axis_minus1 =
        KokkosFFT::Impl::convert_negative_axis(axis_minus1, Rank);
    IntType ref_converted_axis_minus1 = 1;
    EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axis(axis_minus3, Rank); }),
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
void test_convert_negative_axis_3d() {
  constexpr std::size_t Rank = 3;
  IntType axis0 = 0, axis1 = 1, axis2 = 2, axis3 = 3;
  IntType converted_axis_0 =
      KokkosFFT::Impl::convert_negative_axis(axis0, Rank);
  IntType converted_axis_1 =
      KokkosFFT::Impl::convert_negative_axis(axis1, Rank);
  IntType converted_axis_2 =
      KokkosFFT::Impl::convert_negative_axis(axis2, Rank);

  IntType ref_converted_axis_0 = 0;
  IntType ref_converted_axis_1 = 1;
  IntType ref_converted_axis_2 = 2;

  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axis_1, ref_converted_axis_1);
  EXPECT_EQ(converted_axis_2, ref_converted_axis_2);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-3, 3)
  EXPECT_THROW(({ KokkosFFT::Impl::convert_negative_axis(axis3, Rank); }),
               std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    IntType axis_minus1 = -1, axis_minus2 = -2, axis_minus4 = -4;
    IntType converted_axis_minus1 =
        KokkosFFT::Impl::convert_negative_axis(axis_minus1, Rank);
    IntType converted_axis_minus2 =
        KokkosFFT::Impl::convert_negative_axis(axis_minus2, Rank);

    IntType ref_converted_axis_minus1 = 2;
    IntType ref_converted_axis_minus2 = 1;

    EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);
    EXPECT_EQ(converted_axis_minus2, ref_converted_axis_minus2);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axis(axis_minus4, Rank); }),
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
void test_convert_negative_axis_4d() {
  constexpr std::size_t Rank = 4;
  IntType axis0 = 0, axis1 = 1, axis2 = 2, axis3 = 3, axis4 = 4;
  IntType converted_axis_0 =
      KokkosFFT::Impl::convert_negative_axis(axis0, Rank);
  IntType converted_axis_1 =
      KokkosFFT::Impl::convert_negative_axis(axis1, Rank);
  IntType converted_axis_2 =
      KokkosFFT::Impl::convert_negative_axis(axis2, Rank);
  IntType converted_axis_3 =
      KokkosFFT::Impl::convert_negative_axis(axis3, Rank);

  IntType ref_converted_axis_0 = 0;
  IntType ref_converted_axis_1 = 1;
  IntType ref_converted_axis_2 = 2;
  IntType ref_converted_axis_3 = 3;

  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axis_1, ref_converted_axis_1);
  EXPECT_EQ(converted_axis_2, ref_converted_axis_2);
  EXPECT_EQ(converted_axis_3, ref_converted_axis_3);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-4, 4)
  EXPECT_THROW(({ KokkosFFT::Impl::convert_negative_axis(axis4, Rank); }),
               std::runtime_error);

  if constexpr (std::is_signed_v<IntType>) {
    IntType axis_minus1 = -1, axis_minus2 = -2, axis_minus3 = -3,
            axis_minus5 = -5;
    IntType converted_axis_minus1 =
        KokkosFFT::Impl::convert_negative_axis(axis_minus1, Rank);
    IntType converted_axis_minus2 =
        KokkosFFT::Impl::convert_negative_axis(axis_minus2, Rank);
    IntType converted_axis_minus3 =
        KokkosFFT::Impl::convert_negative_axis(axis_minus3, Rank);

    IntType ref_converted_axis_minus1 = 3;
    IntType ref_converted_axis_minus2 = 2;
    IntType ref_converted_axis_minus3 = 1;

    EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);
    EXPECT_EQ(converted_axis_minus2, ref_converted_axis_minus2);
    EXPECT_EQ(converted_axis_minus3, ref_converted_axis_minus3);

    EXPECT_THROW(
        ({ KokkosFFT::Impl::convert_negative_axis(axis_minus5, Rank); }),
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

template <typename ContainerType>
void test_is_found() {
  using IntType   = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  ContainerType v = {0, 1, 4, 2, 3};

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
  using IntType   = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  ContainerType v = {0, 1, 4, 2, 3};

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

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_has_duplicate_values() {
  ContainerType0 v0 = {0, 1, 1};
  ContainerType1 v1 = {0, 1, 1, 1};
  ContainerType1 v2 = {0, 1, 2, 3};
  ContainerType2 v3 = {0};

  EXPECT_TRUE(KokkosFFT::Impl::has_duplicate_values(v0));
  EXPECT_TRUE(KokkosFFT::Impl::has_duplicate_values(v1));
  EXPECT_FALSE(KokkosFFT::Impl::has_duplicate_values(v2));
  EXPECT_FALSE(KokkosFFT::Impl::has_duplicate_values(v3));
}

template <typename ContainerType>
void test_is_out_of_range_value_included() {
  using IntType   = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  ContainerType v = {0, 1, 2, 3, 4}, v2 = {0, 4, 1};

  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(2)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(3)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(4)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(5)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(6)));

  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(2)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(3)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(4)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(5)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(6)));

  if constexpr (std::is_signed_v<IntType>) {
    // Since non-negative value is included, it should be invalid
    ContainerType v3 = {0, 1, -1};
    EXPECT_THROW(
        {
          KokkosFFT::Impl::is_out_of_range_value_included(
              v3, static_cast<IntType>(2));
        },
        std::runtime_error);
  }
}

template <typename IntType>
void test_are_valid_axes() {
  using real_type  = double;
  using View1DType = Kokkos::View<real_type*>;
  using View2DType = Kokkos::View<real_type**>;
  using View3DType = Kokkos::View<real_type***>;
  using View4DType = Kokkos::View<real_type****>;

  std::array<IntType, 1> axes0 = {0};
  std::array<IntType, 2> axes1 = {0, 1};
  std::array<IntType, 3> axes2 = {0, 1, 2};
  std::array<IntType, 3> axes3 = {0, 1, 1};
  std::array<IntType, 3> axes4 = {0, 1, 3};

  View1DType view1;
  View2DType view2;
  View3DType view3;
  View4DType view4;

  // 1D axes on 1D+ Views
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view1, axes0));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view2, axes0));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view3, axes0));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes0));

  // 2D axes on 2D+ Views
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view2, axes1));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view3, axes1));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes1));

  // 3D axes on 3D+ Views
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view3, axes2));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes2));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes4));

  // 3D axes on 3D Views with out of range -> should fail
  EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view3, axes4));

  // axes include overlap -> should fail
  EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view3, axes3));
  EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view4, axes3));

  if constexpr (std::is_signed_v<IntType>) {
    // {0, 1, -2} is converted to {0, 1, 1} for 3D View and {0, 1, 2}
    // for 4D View. Invalid for 3D View but valid for 4D View
    std::array<IntType, 3> axes5 = {0, 1, -2};

    // axes include overlap -> should fail
    EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view3, axes5));

    // axes do not include overlap -> OK
    EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes5));
  }
}

template <typename ValueType1, typename ValueType2>
void test_are_pointers_aliasing() {
  using View1 = Kokkos::View<ValueType1*, execution_space>;
  using View2 = Kokkos::View<ValueType2*, execution_space>;

  const int n1 = 10;
  // sizeof ValueType2 is larger or equal to ValueType1
  const int n2 = sizeof(ValueType1) == sizeof(ValueType2) ? n1 : n1 / 2 + 1;
  View1 view1("view1", n1);
  View2 view2("view2", n1);
  View2 uview2(reinterpret_cast<ValueType2*>(view1.data()), n2);

  EXPECT_TRUE(KokkosFFT::Impl::are_aliasing(view1.data(), uview2.data()));
  EXPECT_FALSE(KokkosFFT::Impl::are_aliasing(view1.data(), view2.data()));
}

template <typename IntType>
void test_index_sequence() {
  // Rank of the index sequence
  constexpr std::size_t DIM0 = 3;
  constexpr std::size_t DIM1 = 4;
  constexpr std::size_t DIM2 = 5;
  if constexpr (std::is_signed_v<IntType>) {
    constexpr IntType start0 = -static_cast<IntType>(DIM0);
    constexpr IntType start1 = -static_cast<IntType>(DIM1);
    constexpr IntType start2 = -static_cast<IntType>(DIM2);

    constexpr auto default_axes0 =
        KokkosFFT::Impl::index_sequence<IntType, DIM0, start0>();
    constexpr auto default_axes1 =
        KokkosFFT::Impl::index_sequence<IntType, DIM1, start1>();
    constexpr auto default_axes2 =
        KokkosFFT::Impl::index_sequence<IntType, DIM2, start2>();

    std::array<IntType, DIM0> ref_axes0 = {-3, -2, -1};
    std::array<IntType, DIM1> ref_axes1 = {-4, -3, -2, -1};
    std::array<IntType, DIM2> ref_axes2 = {-5, -4, -3, -2, -1};

    EXPECT_EQ(default_axes0, ref_axes0);
    EXPECT_EQ(default_axes1, ref_axes1);
    EXPECT_EQ(default_axes2, ref_axes2);
  } else {
    constexpr auto range0 = KokkosFFT::Impl::index_sequence<IntType, DIM0, 0>();
    constexpr auto range1 = KokkosFFT::Impl::index_sequence<IntType, DIM1, 0>();
    constexpr auto range2 = KokkosFFT::Impl::index_sequence<IntType, DIM2, 0>();
    std::array<IntType, DIM0> ref_range0 = {0, 1, 2};
    std::array<IntType, DIM1> ref_range1 = {0, 1, 2, 3};
    std::array<IntType, DIM2> ref_range2 = {0, 1, 2, 3, 4};

    EXPECT_EQ(range0, ref_range0);
    EXPECT_EQ(range1, ref_range1);
    EXPECT_EQ(range2, ref_range2);
  }
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_total_size() {
  using IntType = KokkosFFT::Impl::base_container_value_type<ContainerType0>;

  ContainerType0 v0 = {0, 1, 4, 2, 3};
  ContainerType1 v1 = {2, 3, 5};
  ContainerType2 v2 = {1};
  auto total_size0  = KokkosFFT::Impl::total_size(v0);
  auto total_size1  = KokkosFFT::Impl::total_size(v1);
  auto total_size2  = KokkosFFT::Impl::total_size(v2);

  IntType ref_total_size0 = 0, ref_total_size1 = 30, ref_total_size2 = 1;

  EXPECT_EQ(total_size0, ref_total_size0);
  EXPECT_EQ(total_size1, ref_total_size1);
  EXPECT_EQ(total_size2, ref_total_size2);

  // Failure test with overflow
  ContainerType1 v3 = {2, 3, std::numeric_limits<IntType>::max()},
                 v4 = {1, std::numeric_limits<IntType>::max(),
                       std::numeric_limits<IntType>::max()},
                 v5 = {1, std::numeric_limits<IntType>::min(),
                       std::numeric_limits<IntType>::max()},
                 v6 = {1, std::numeric_limits<IntType>::max(),
                       std::numeric_limits<IntType>::min()},
                 v7 = {1, std::numeric_limits<IntType>::min(),
                       std::numeric_limits<IntType>::min()};

  EXPECT_THROW(
      { [[maybe_unused]] auto total_size3 = KokkosFFT::Impl::total_size(v3); },
      std::overflow_error);

  EXPECT_THROW(
      { [[maybe_unused]] auto total_size4 = KokkosFFT::Impl::total_size(v4); },
      std::overflow_error);

  // We expect overflow
  if constexpr (std::is_signed_v<IntType>) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size5 = KokkosFFT::Impl::total_size(v5);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size6 = KokkosFFT::Impl::total_size(v6);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size7 = KokkosFFT::Impl::total_size(v7);
        },
        std::overflow_error);
  } else {
    // This should be just zero for unsigned case
    auto total_size5 = KokkosFFT::Impl::total_size(v5);
    EXPECT_EQ(total_size5, 0);

    // This should be just zero for unsigned case
    auto total_size6 = KokkosFFT::Impl::total_size(v6);
    EXPECT_EQ(total_size6, 0);

    // This should be just zero for unsigned case
    auto total_size7 = KokkosFFT::Impl::total_size(v7);
    EXPECT_EQ(total_size7, 0);
  }

  // Including max still OK
  ContainerType1 v8       = {1, 2, std::numeric_limits<IntType>::min() / 2};
  auto total_size8        = KokkosFFT::Impl::total_size(v8);
  IntType ref_total_size8 = std::numeric_limits<IntType>::min();
  EXPECT_EQ(total_size8, ref_total_size8);

  if constexpr (std::is_signed_v<IntType>) {
    // Failure test with overflow
    ContainerType1 iv0 = {-1, 2, std::numeric_limits<IntType>::max()},
                   iv1 = {-2, 3, std::numeric_limits<IntType>::min()},
                   iv2 = {1, std::numeric_limits<IntType>::min(), -2},
                   iv3 = {1, std::numeric_limits<IntType>::min(), -1},
                   iv4 = {1, std::numeric_limits<IntType>::max(), -1};
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i0 =
              KokkosFFT::Impl::total_size(iv0);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i1 =
              KokkosFFT::Impl::total_size(iv1);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i2 =
              KokkosFFT::Impl::total_size(iv2);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i3 =
              KokkosFFT::Impl::total_size(iv3);
        },
        std::overflow_error);

    auto total_size_i4        = KokkosFFT::Impl::total_size(iv4);
    IntType ref_total_size_i4 = std::numeric_limits<IntType>::min() + 1;
    EXPECT_EQ(total_size_i4, ref_total_size_i4);
  }
}

template <typename ToContainerType, typename FromContainerType>
void test_convert_base_int_type() {
  using From = KokkosFFT::Impl::base_container_value_type<FromContainerType>;
  using To   = KokkosFFT::Impl::base_container_value_type<ToContainerType>;

  FromContainerType v   = {2, 3, 5};
  ToContainerType v_ref = {2, 3, 5};

  auto v_converted = KokkosFFT::Impl::convert_base_int_type<To>(v);
  EXPECT_EQ(v_converted, v_ref);

  if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
    // Same signedness, should be OK
    FromContainerType v2   = {-2, -3, -5};
    ToContainerType v2_ref = {-2, -3, -5};

    auto v2_converted = KokkosFFT::Impl::convert_base_int_type<To>(v2);
    EXPECT_EQ(v2_converted, v2_ref);
  } else if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
    // From signed to unsigned, negative value should raise error
    FromContainerType v2 = {-2, 3, 5};
    EXPECT_THROW(
        {
          [[maybe_unused]] auto v2_converted =
              KokkosFFT::Impl::convert_base_int_type<To>(v2);
        },
        std::overflow_error);
  } else if constexpr (std::is_unsigned_v<From>) {
    // From std::size_t to int, overflow should raise error
    if (std::numeric_limits<From>::max() > std::numeric_limits<To>::max()) {
      FromContainerType v2 = {2, 3, std::numeric_limits<From>::max()};
      EXPECT_THROW(
          {
            [[maybe_unused]] auto v2_converted =
                KokkosFFT::Impl::convert_base_int_type<To>(v2);
          },
          std::overflow_error);
    }
  }
}
}  // namespace

TYPED_TEST_SUITE(TestConvertNegativeAxis, base_int_types);
TYPED_TEST_SUITE(TestConvertNegativeAxes, base_int_types);
TYPED_TEST_SUITE(ContainerTypes, base_int_types);
TYPED_TEST_SUITE(PairedScalarTypes, paired_scalar_types);
TYPED_TEST_SUITE(TestIndexSequence, base_int_types);
TYPED_TEST_SUITE(TestConvertBaseTypes, paired_int_types);

// Tests for 1D - 4D View
TYPED_TEST(TestConvertNegativeAxis, 1DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axis_1d<value_type>();
}

TYPED_TEST(TestConvertNegativeAxis, 2DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axis_2d<value_type>();
}

TYPED_TEST(TestConvertNegativeAxis, 3DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axis_3d<value_type>();
}

TYPED_TEST(TestConvertNegativeAxis, 4DView) {
  using value_type = typename TestFixture::value_type;
  test_convert_negative_axis_4d<value_type>();
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

TEST(IsTransposeNeeded, 1Dto3D) {
  std::array<int, 1> map1D = {0};
  EXPECT_FALSE(KokkosFFT::Impl::is_transpose_needed(map1D));

  std::array<int, 2> map2D = {0, 1}, map2D_axis0 = {1, 0};
  EXPECT_FALSE(KokkosFFT::Impl::is_transpose_needed(map2D));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map2D_axis0));

  std::array<int, 3> map3D     = {0, 1, 2};
  std::array<int, 3> map3D_021 = {0, 2, 1};
  std::array<int, 3> map3D_102 = {1, 0, 2};
  std::array<int, 3> map3D_120 = {1, 2, 0};
  std::array<int, 3> map3D_201 = {2, 0, 1};
  std::array<int, 3> map3D_210 = {2, 1, 0};

  EXPECT_FALSE(KokkosFFT::Impl::is_transpose_needed(map3D));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_021));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_102));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_120));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_201));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_210));
}

TYPED_TEST(ContainerTypes, is_found_from_vector) {
  using container_type = typename TestFixture::vector_type;
  test_is_found<container_type>();
}

TYPED_TEST(ContainerTypes, is_found_from_array) {
  using container_type = typename TestFixture::array_type;
  test_is_found<container_type>();
}

TYPED_TEST(ContainerTypes, get_index_from_vector) {
  using container_type = typename TestFixture::vector_type;
  test_get_index<container_type>();
}

TYPED_TEST(ContainerTypes, get_index_from_array) {
  using container_type = typename TestFixture::array_type;
  test_get_index<container_type>();
}

TYPED_TEST(ContainerTypes, has_duplicate_values_in_vector) {
  using container_type = typename TestFixture::vector_type;
  test_has_duplicate_values<container_type, container_type, container_type>();
}

TYPED_TEST(ContainerTypes, has_duplicate_values_in_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 3>;
  using container_type1 = std::array<value_type, 4>;
  using container_type2 = std::array<value_type, 1>;
  test_has_duplicate_values<container_type0, container_type1,
                            container_type2>();
}

TYPED_TEST(ContainerTypes, is_out_of_range_value_included_in_vector) {
  using container_type = typename TestFixture::vector_type;
  test_is_out_of_range_value_included<container_type>();
}

TYPED_TEST(ContainerTypes, is_out_of_range_value_included_in_array) {
  using container_type = typename TestFixture::array_type;
  test_is_out_of_range_value_included<container_type>();
}

TYPED_TEST(ContainerTypes, test_total_size_of_vector) {
  using container_type = typename TestFixture::vector_type;
  test_total_size<container_type, container_type, container_type>();
}

TYPED_TEST(ContainerTypes, test_total_size_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 5>;
  using container_type1 = std::array<value_type, 3>;
  using container_type2 = std::array<value_type, 1>;
  test_total_size<container_type0, container_type1, container_type2>();
}

TYPED_TEST(ContainerTypes, are_valid_axes) {
  using value_type = typename TestFixture::value_type;
  test_are_valid_axes<value_type>();
}

TEST(ExtractExtents, 1Dto8D) {
  using View1Dtype = Kokkos::View<double*, execution_space>;
  using View2Dtype = Kokkos::View<double**, execution_space>;
  using View3Dtype = Kokkos::View<double***, execution_space>;
  using View4Dtype = Kokkos::View<double****, execution_space>;
  using View5Dtype = Kokkos::View<double*****, execution_space>;
  using View6Dtype = Kokkos::View<double******, execution_space>;
  using View7Dtype = Kokkos::View<double*******, execution_space>;
  using View8Dtype = Kokkos::View<double********, execution_space>;

  std::size_t n1 = 1, n2 = 1, n3 = 2, n4 = 3, n5 = 5, n6 = 8, n7 = 13, n8 = 21;

  std::array<std::size_t, 1> ref_extents1D = {n1};
  std::array<std::size_t, 2> ref_extents2D = {n1, n2};
  std::array<std::size_t, 3> ref_extents3D = {n1, n2, n3};
  std::array<std::size_t, 4> ref_extents4D = {n1, n2, n3, n4};
  std::array<std::size_t, 5> ref_extents5D = {n1, n2, n3, n4, n5};
  std::array<std::size_t, 6> ref_extents6D = {n1, n2, n3, n4, n5, n6};
  std::array<std::size_t, 7> ref_extents7D = {n1, n2, n3, n4, n5, n6, n7};
  std::array<std::size_t, 8> ref_extents8D = {n1, n2, n3, n4, n5, n6, n7, n8};

  View1Dtype view1D("view1D", n1);
  View2Dtype view2D("view2D", n1, n2);
  View3Dtype view3D("view3D", n1, n2, n3);
  View4Dtype view4D("view4D", n1, n2, n3, n4);
  View5Dtype view5D("view5D", n1, n2, n3, n4, n5);
  View6Dtype view6D("view6D", n1, n2, n3, n4, n5, n6);
  View7Dtype view7D("view7D", n1, n2, n3, n4, n5, n6, n7);
  View8Dtype view8D("view8D", n1, n2, n3, n4, n5, n6, n7, n8);

  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view1D), ref_extents1D);
  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view2D), ref_extents2D);
  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view3D), ref_extents3D);
  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view4D), ref_extents4D);
  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view5D), ref_extents5D);
  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view6D), ref_extents6D);
  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view7D), ref_extents7D);
  EXPECT_EQ(KokkosFFT::Impl::extract_extents(view8D), ref_extents8D);
}

TYPED_TEST(TestIndexSequence, make_sequence_3Dto5D) {
  using value_type = typename TestFixture::value_type;
  test_index_sequence<value_type>();
}

TEST(ToArray, lvalue) {
  std::array arr{1, 2, 3};
  ASSERT_EQ(KokkosFFT::Impl::to_array(arr), (Kokkos::Array{1, 2, 3}));
}

TEST(ToArray, rvalue) {
  ASSERT_EQ(KokkosFFT::Impl::to_array(std::array{1, 2}), (Kokkos::Array{1, 2}));
}

TYPED_TEST(PairedScalarTypes, are_pointers_aliasing) {
  using value_type1 = typename TestFixture::value_type1;
  using value_type2 = typename TestFixture::value_type2;
  test_are_pointers_aliasing<value_type1, value_type2>();
}

TYPED_TEST(TestConvertBaseTypes, convert_arrays) {
  using value_type1     = typename TestFixture::value_type1;
  using value_type2     = typename TestFixture::value_type2;
  using container_type1 = std::array<value_type1, 3>;
  using container_type2 = std::array<value_type2, 3>;
  test_convert_base_int_type<container_type1, container_type2>();
}

TYPED_TEST(TestConvertBaseTypes, convert_vectors) {
  using value_type1     = typename TestFixture::value_type1;
  using value_type2     = typename TestFixture::value_type2;
  using container_type1 = std::vector<value_type1>;
  using container_type2 = std::vector<value_type2>;
  test_convert_base_int_type<container_type1, container_type2>();
}
