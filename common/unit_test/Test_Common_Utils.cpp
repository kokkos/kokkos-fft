// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Int like types
using base_int_types = ::testing::Types<int, std::size_t>;

// value type combinations that are tested
using paired_scalar_types = ::testing::Types<
    std::pair<float, float>, std::pair<float, Kokkos::complex<float>>,
    std::pair<Kokkos::complex<float>, Kokkos::complex<float>>,
    std::pair<double, double>, std::pair<double, Kokkos::complex<double>>,
    std::pair<Kokkos::complex<double>, Kokkos::complex<double>>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct ConvertNegativeAxis : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct ConvertNegativeShift : public ::testing::Test {
  using layout_type = T;
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

// Tests for convert_negative_axes over ND views
template <typename LayoutType>
void test_convert_negative_axes_1d() {
  const int len        = 30;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x("x", len);

  int converted_axis_0 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_minus1 =
      KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_minus1 = 0;

  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-1, 1)
  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/1); },
               std::runtime_error);

  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-2); },
               std::runtime_error);
}

template <typename LayoutType>
void test_convert_negative_axes_2d() {
  const int n0 = 3, n1 = 5;
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  RealView2Dtype x("x", n0, n1);

  int converted_axis_0 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_minus1 =
      KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_1      = 1;
  int ref_converted_axis_minus1 = 1;

  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axis_1, ref_converted_axis_1);
  EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-2, 2)
  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/2); },
               std::runtime_error);

  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-3); },
               std::runtime_error);
}

template <typename LayoutType>
void test_convert_negative_axes_3d() {
  const int n0 = 3, n1 = 5, n2 = 8;
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  RealView3Dtype x("x", n0, n1, n2);

  int converted_axis_0 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_2 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/2);
  int converted_axis_minus1 =
      KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);
  int converted_axis_minus2 =
      KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-2);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_1      = 1;
  int ref_converted_axis_2      = 2;
  int ref_converted_axis_minus1 = 2;
  int ref_converted_axis_minus2 = 1;

  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axis_1, ref_converted_axis_1);
  EXPECT_EQ(converted_axis_2, ref_converted_axis_2);
  EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);
  EXPECT_EQ(converted_axis_minus2, ref_converted_axis_minus2);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-3, 3)
  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/3); },
               std::runtime_error);

  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-4); },
               std::runtime_error);
}

template <typename LayoutType>
void test_convert_negative_axes_4d() {
  const int n0 = 3, n1 = 5, n2 = 8, n3 = 13;
  using RealView4Dtype = Kokkos::View<double****, LayoutType, execution_space>;
  RealView4Dtype x("x", n0, n1, n2, n3);

  int converted_axis_0 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_2 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/2);
  int converted_axis_3 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/3);
  int converted_axis_minus1 =
      KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);
  int converted_axis_minus2 =
      KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-2);
  int converted_axis_minus3 =
      KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-3);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_1      = 1;
  int ref_converted_axis_2      = 2;
  int ref_converted_axis_3      = 3;
  int ref_converted_axis_minus1 = 3;
  int ref_converted_axis_minus2 = 2;
  int ref_converted_axis_minus3 = 1;

  EXPECT_EQ(converted_axis_0, ref_converted_axis_0);
  EXPECT_EQ(converted_axis_1, ref_converted_axis_1);
  EXPECT_EQ(converted_axis_2, ref_converted_axis_2);
  EXPECT_EQ(converted_axis_3, ref_converted_axis_3);
  EXPECT_EQ(converted_axis_minus1, ref_converted_axis_minus1);
  EXPECT_EQ(converted_axis_minus2, ref_converted_axis_minus2);
  EXPECT_EQ(converted_axis_minus3, ref_converted_axis_minus3);

  // Check if errors are correctly raised against invalid axis
  // axis must be in [-4, 4)
  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/4); },
               std::runtime_error);

  EXPECT_THROW({ KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-5); },
               std::runtime_error);
}

// Tests for convert_negative_shift over ND views
template <typename LayoutType>
void test_convert_negative_shift_1d() {
  const int n0_odd = 29, n0_even = 30;
  const int shift      = 5;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x_odd("x_odd", n0_odd), x_even("x_even", n0_even);

  auto [shift_5_0_odd, shift_5_1_odd, shift_5_2_odd] =
      KokkosFFT::Impl::convert_negative_shift(x_odd, shift, 0);
  auto [shift_5_0_even, shift_5_1_even, shift_5_2_even] =
      KokkosFFT::Impl::convert_negative_shift(x_even, shift, 0);
  auto [shift_0_0_odd, shift_0_1_odd, shift_0_2_odd] =
      KokkosFFT::Impl::convert_negative_shift(x_odd, 0, 0);
  auto [shift_0_0_even, shift_0_1_even, shift_0_2_even] =
      KokkosFFT::Impl::convert_negative_shift(x_even, 0, 0);
  auto [shift_m5_0_odd, shift_m5_1_odd, shift_m5_2_odd] =
      KokkosFFT::Impl::convert_negative_shift(x_odd, -shift, 0);
  auto [shift_m5_0_even, shift_m5_1_even, shift_m5_2_even] =
      KokkosFFT::Impl::convert_negative_shift(x_even, -shift, 0);

  int ref_shift_5_0_odd = shift, ref_shift_5_1_odd = shift + 1,
      ref_shift_5_2_odd  = 0;
  int ref_shift_5_0_even = shift, ref_shift_5_1_even = shift,
      ref_shift_5_2_even = 0;
  int ref_shift_0_0_odd = 0, ref_shift_0_1_odd = 0,
      ref_shift_0_2_odd  = n0_odd / 2;
  int ref_shift_0_0_even = 0, ref_shift_0_1_even = 0,
      ref_shift_0_2_even = n0_even / 2;
  int ref_shift_m5_0_odd = shift + 1, ref_shift_m5_1_odd = shift,
      ref_shift_m5_2_odd  = 0;
  int ref_shift_m5_0_even = shift, ref_shift_m5_1_even = shift,
      ref_shift_m5_2_even = 0;

  EXPECT_EQ(shift_5_0_odd, ref_shift_5_0_odd);
  EXPECT_EQ(shift_5_0_even, ref_shift_5_0_even);
  EXPECT_EQ(shift_0_0_odd, ref_shift_0_0_odd);
  EXPECT_EQ(shift_0_0_even, ref_shift_0_0_even);
  EXPECT_EQ(shift_m5_0_odd, ref_shift_m5_0_odd);
  EXPECT_EQ(shift_m5_0_even, ref_shift_m5_0_even);

  EXPECT_EQ(shift_5_1_odd, ref_shift_5_1_odd);
  EXPECT_EQ(shift_5_1_even, ref_shift_5_1_even);
  EXPECT_EQ(shift_0_1_odd, ref_shift_0_1_odd);
  EXPECT_EQ(shift_0_1_even, ref_shift_0_1_even);
  EXPECT_EQ(shift_m5_1_odd, ref_shift_m5_1_odd);
  EXPECT_EQ(shift_m5_1_even, ref_shift_m5_1_even);

  EXPECT_EQ(shift_5_2_odd, ref_shift_5_2_odd);
  EXPECT_EQ(shift_5_2_even, ref_shift_5_2_even);
  EXPECT_EQ(shift_0_2_odd, ref_shift_0_2_odd);
  EXPECT_EQ(shift_0_2_even, ref_shift_0_2_even);
  EXPECT_EQ(shift_m5_2_odd, ref_shift_m5_2_odd);
  EXPECT_EQ(shift_m5_2_even, ref_shift_m5_2_even);
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
}  // namespace

TYPED_TEST_SUITE(ConvertNegativeAxis, test_types);
TYPED_TEST_SUITE(ConvertNegativeShift, test_types);
TYPED_TEST_SUITE(ContainerTypes, base_int_types);
TYPED_TEST_SUITE(PairedScalarTypes, paired_scalar_types);

// Tests for 1D View
TYPED_TEST(ConvertNegativeAxis, 1DView) {
  using layout_type = typename TestFixture::layout_type;

  test_convert_negative_axes_1d<layout_type>();
}

// Tests for 2D View
TYPED_TEST(ConvertNegativeAxis, 2DView) {
  using layout_type = typename TestFixture::layout_type;

  test_convert_negative_axes_2d<layout_type>();
}

// Tests for 3D View
TYPED_TEST(ConvertNegativeAxis, 3DView) {
  using layout_type = typename TestFixture::layout_type;

  test_convert_negative_axes_3d<layout_type>();
}

// Tests for 4D View
TYPED_TEST(ConvertNegativeAxis, 4DView) {
  using layout_type = typename TestFixture::layout_type;

  test_convert_negative_axes_4d<layout_type>();
}

// Tests for 1D View
TYPED_TEST(ConvertNegativeShift, 1DView) {
  using layout_type = typename TestFixture::layout_type;

  test_convert_negative_shift_1d<layout_type>();
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

TEST(IndexSequence, 3Dto5D) {
  using View3Dtype = Kokkos::View<double***, execution_space>;
  using View4Dtype = Kokkos::View<double****, execution_space>;
  using View5Dtype = Kokkos::View<double*****, execution_space>;

  constexpr std::size_t DIM = 3;
  std::size_t n1 = 1, n2 = 1, n3 = 2, n4 = 3, n5 = 5;
  View3Dtype view3D("view3D", n1, n2, n3);
  View4Dtype view4D("view4D", n1, n2, n3, n4);
  View5Dtype view5D("view5D", n1, n2, n3, n4, n5);
  constexpr int start0 = -static_cast<int>(View3Dtype::rank());
  constexpr int start1 = -static_cast<int>(View4Dtype::rank());
  constexpr int start2 = -static_cast<int>(View5Dtype::rank());

  constexpr auto default_axes0 =
      KokkosFFT::Impl::index_sequence<int, DIM, start0>();
  constexpr auto default_axes1 =
      KokkosFFT::Impl::index_sequence<int, DIM, start1>();
  constexpr auto default_axes2 =
      KokkosFFT::Impl::index_sequence<int, DIM, start2>();

  std::array<int, DIM> ref_axes0 = {-3, -2, -1};
  std::array<int, DIM> ref_axes1 = {-4, -3, -2};
  std::array<int, DIM> ref_axes2 = {-5, -4, -3};

  EXPECT_EQ(default_axes0, ref_axes0);
  EXPECT_EQ(default_axes1, ref_axes1);
  EXPECT_EQ(default_axes2, ref_axes2);
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
