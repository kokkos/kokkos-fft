#include <gtest/gtest.h>
#include "KokkosFFT_utils.hpp"
#include "Test_Types.hpp"

using test_types = ::testing::Types<
  Kokkos::LayoutLeft,
  Kokkos::LayoutRight
>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct ConvertNegativeAxis : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct ConvertNegativeShift : public ::testing::Test {
  using layout_type = T;
};

TYPED_TEST_SUITE(ConvertNegativeAxis, test_types);
TYPED_TEST_SUITE(ConvertNegativeShift, test_types);

// Tests for convert_negative_axes over ND views
template <typename LayoutType>
void test_convert_negative_axes_1d() {
  const int len = 30;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x("x", len);

  int converted_axis_0 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_minus1 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);

  int ref_converted_axis_0 = 0;
  int ref_converted_axis_minus1 = 0;

  EXPECT_EQ( converted_axis_0, ref_converted_axis_0 );
  EXPECT_EQ( converted_axis_minus1, ref_converted_axis_minus1 );
}

template <typename LayoutType>
void test_convert_negative_axes_2d() {
  const int n0 = 3, n1 = 5;
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  RealView2Dtype x("x", n0, n1);

  int converted_axis_0      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_minus1 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_1      = 1;
  int ref_converted_axis_minus1 = 1;

  EXPECT_EQ( converted_axis_0, ref_converted_axis_0 );
  EXPECT_EQ( converted_axis_1, ref_converted_axis_1 );
  EXPECT_EQ( converted_axis_minus1, ref_converted_axis_minus1 );
}

template <typename LayoutType>
void test_convert_negative_axes_3d() {
  const int n0 = 3, n1 = 5, n2 = 8;
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  RealView3Dtype x("x", n0, n1, n2);

  int converted_axis_0      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_2      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/2);
  int converted_axis_minus1 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);
  int converted_axis_minus2 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-2);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_1      = 1;
  int ref_converted_axis_2      = 2;
  int ref_converted_axis_minus1 = 2;
  int ref_converted_axis_minus2 = 1;

  EXPECT_EQ( converted_axis_0, ref_converted_axis_0 );
  EXPECT_EQ( converted_axis_1, ref_converted_axis_1 );
  EXPECT_EQ( converted_axis_2, ref_converted_axis_2 );
  EXPECT_EQ( converted_axis_minus1, ref_converted_axis_minus1 );
  EXPECT_EQ( converted_axis_minus2, ref_converted_axis_minus2 );
}

template <typename LayoutType>
void test_convert_negative_axes_4d() {
  const int n0 = 3, n1 = 5, n2 = 8, n3 = 13;
  using RealView4Dtype = Kokkos::View<double****, LayoutType, execution_space>;
  RealView4Dtype x("x", n0, n1, n2, n3);

  int converted_axis_0      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_2      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/2);
  int converted_axis_3      = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/3);
  int converted_axis_minus1 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-1);
  int converted_axis_minus2 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-2);
  int converted_axis_minus3 = KokkosFFT::Impl::convert_negative_axis(x, /*axis=*/-3);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_1      = 1;
  int ref_converted_axis_2      = 2;
  int ref_converted_axis_3      = 3;
  int ref_converted_axis_minus1 = 3;
  int ref_converted_axis_minus2 = 2;
  int ref_converted_axis_minus3 = 1;

  EXPECT_EQ( converted_axis_0, ref_converted_axis_0 );
  EXPECT_EQ( converted_axis_1, ref_converted_axis_1 );
  EXPECT_EQ( converted_axis_2, ref_converted_axis_2 );
  EXPECT_EQ( converted_axis_3, ref_converted_axis_3 );
  EXPECT_EQ( converted_axis_minus1, ref_converted_axis_minus1 );
  EXPECT_EQ( converted_axis_minus2, ref_converted_axis_minus2 );
  EXPECT_EQ( converted_axis_minus3, ref_converted_axis_minus3 );
}

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

// Tests for convert_negative_shift over ND views
template <typename LayoutType>
void test_convert_negative_shift_1d() {
  const int n0_odd = 29, n0_even = 30;
  const int shift = 5;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x_odd("x_odd", n0_odd), x_even("x_even", n0_even);

  auto [shift_5_0_odd, shift_5_1_odd, shift_5_2_odd] = KokkosFFT::Impl::convert_negative_shift(x_odd, shift, 0);
  auto [shift_5_0_even, shift_5_1_even, shift_5_2_even] = KokkosFFT::Impl::convert_negative_shift(x_even, shift, 0);
  auto [shift_0_0_odd, shift_0_1_odd, shift_0_2_odd] = KokkosFFT::Impl::convert_negative_shift(x_odd, 0, 0);
  auto [shift_0_0_even, shift_0_1_even, shift_0_2_even] = KokkosFFT::Impl::convert_negative_shift(x_even, 0, 0);
  auto [shift_m5_0_odd, shift_m5_1_odd, shift_m5_2_odd] = KokkosFFT::Impl::convert_negative_shift(x_odd, -shift, 0);
  auto [shift_m5_0_even, shift_m5_1_even, shift_m5_2_even] = KokkosFFT::Impl::convert_negative_shift(x_even, -shift, 0);

  int ref_shift_5_0_odd = shift, ref_shift_5_1_odd = shift+1, ref_shift_5_2_odd = 0;
  int ref_shift_5_0_even = shift, ref_shift_5_1_even = shift, ref_shift_5_2_even = 0;
  int ref_shift_0_0_odd = 0, ref_shift_0_1_odd = 0, ref_shift_0_2_odd = n0_odd / 2;
  int ref_shift_0_0_even = 0, ref_shift_0_1_even = 0, ref_shift_0_2_even = n0_even / 2;
  int ref_shift_m5_0_odd = shift+1, ref_shift_m5_1_odd = shift, ref_shift_m5_2_odd = 0;
  int ref_shift_m5_0_even = shift, ref_shift_m5_1_even = shift, ref_shift_m5_2_even = 0;

  EXPECT_EQ( shift_5_0_odd, ref_shift_5_0_odd );
  EXPECT_EQ( shift_5_0_even, ref_shift_5_0_even );
  EXPECT_EQ( shift_0_0_odd, ref_shift_0_0_odd );
  EXPECT_EQ( shift_0_0_even, ref_shift_0_0_even );
  EXPECT_EQ( shift_m5_0_odd, ref_shift_m5_0_odd );
  EXPECT_EQ( shift_m5_0_even, ref_shift_m5_0_even );

  EXPECT_EQ( shift_5_1_odd, ref_shift_5_1_odd );
  EXPECT_EQ( shift_5_1_even, ref_shift_5_1_even );
  EXPECT_EQ( shift_0_1_odd, ref_shift_0_1_odd );
  EXPECT_EQ( shift_0_1_even, ref_shift_0_1_even );
  EXPECT_EQ( shift_m5_1_odd, ref_shift_m5_1_odd );
  EXPECT_EQ( shift_m5_1_even, ref_shift_m5_1_even );

  EXPECT_EQ( shift_5_2_odd, ref_shift_5_2_odd );
  EXPECT_EQ( shift_5_2_even, ref_shift_5_2_even );
  EXPECT_EQ( shift_0_2_odd, ref_shift_0_2_odd );
  EXPECT_EQ( shift_0_2_even, ref_shift_0_2_even );
  EXPECT_EQ( shift_m5_2_odd, ref_shift_m5_2_odd );
  EXPECT_EQ( shift_m5_2_even, ref_shift_m5_2_even );
}

// Tests for 1D View
TYPED_TEST(ConvertNegativeShift, 1DView) {
  using layout_type = typename TestFixture::layout_type;

  test_convert_negative_shift_1d<layout_type>();
}

TEST(IsTransposeNeeded, 1Dto3D) {
  std::array<int, 1> map1D ={0};
  EXPECT_FALSE( KokkosFFT::Impl::is_transpose_needed(map1D) );

  std::array<int, 2> map2D ={0, 1}, map2D_axis0 = {1, 0};
  EXPECT_FALSE( KokkosFFT::Impl::is_transpose_needed(map2D) );
  EXPECT_TRUE( KokkosFFT::Impl::is_transpose_needed(map2D_axis0) );

  std::array<int, 3> map3D =     {0, 1, 2};
  std::array<int, 3> map3D_021 = {0, 2, 1};
  std::array<int, 3> map3D_102 = {1, 0, 2};
  std::array<int, 3> map3D_120 = {1, 2, 0};
  std::array<int, 3> map3D_201 = {2, 0, 1};
  std::array<int, 3> map3D_210 = {2, 1, 0};

  EXPECT_FALSE( KokkosFFT::Impl::is_transpose_needed(map3D) );
  EXPECT_TRUE( KokkosFFT::Impl::is_transpose_needed(map3D_021) );
  EXPECT_TRUE( KokkosFFT::Impl::is_transpose_needed(map3D_102) );
  EXPECT_TRUE( KokkosFFT::Impl::is_transpose_needed(map3D_120) );
  EXPECT_TRUE( KokkosFFT::Impl::is_transpose_needed(map3D_201) );
  EXPECT_TRUE( KokkosFFT::Impl::is_transpose_needed(map3D_210) );
}

TEST(GetIndex, Vectors) {
  std::vector<int> v = {0, 1, 4, 2, 3};

  EXPECT_EQ( KokkosFFT::Impl::get_index(v, 0), 0 );
  EXPECT_EQ( KokkosFFT::Impl::get_index(v, 1), 1 );
  EXPECT_EQ( KokkosFFT::Impl::get_index(v, 2), 3 );
  EXPECT_EQ( KokkosFFT::Impl::get_index(v, 3), 4 );
  EXPECT_EQ( KokkosFFT::Impl::get_index(v, 4), 2 );

  EXPECT_THROW(
    KokkosFFT::Impl::get_index(v, -1),
    std::runtime_error
  );

  EXPECT_THROW(
    KokkosFFT::Impl::get_index(v, 5),
    std::runtime_error
  );
}

TEST(IsOutOfRangeValueIncluded, Array) {
  std::array<int, 4> v = {0, 1, 2, 3};

  EXPECT_TRUE( KokkosFFT::Impl::is_out_of_range_value_included(v, 2) );
  EXPECT_TRUE( KokkosFFT::Impl::is_out_of_range_value_included(v, 3) );
  EXPECT_FALSE( KokkosFFT::Impl::is_out_of_range_value_included(v, 4) );
  EXPECT_FALSE( KokkosFFT::Impl::is_out_of_range_value_included(v, 5) );
}