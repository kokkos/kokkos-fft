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
