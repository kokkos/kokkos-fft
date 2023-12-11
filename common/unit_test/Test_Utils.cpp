#include <gtest/gtest.h>
#include "KokkosFFT_utils.hpp"
#include "Test_Types.hpp"

template <typename LayoutType>
void test_convert_negative_axes_1d() {
  const int len = 30;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x("x", len);

  int converted_axis_0 = KokkosFFT::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_minus1 = KokkosFFT::convert_negative_axis(x, /*axis=*/-1);

  int ref_converted_axis_0 = 0;
  int ref_converted_axis_minus1 = 0;

  EXPECT_EQ( converted_axis_0, ref_converted_axis_0 );
  EXPECT_EQ( converted_axis_minus1, ref_converted_axis_minus1 );
}

TEST(ConvertNegativeAxis, 1DLeftView) {
  test_convert_negative_axes_1d<Kokkos::LayoutLeft>();
}

TEST(ConvertNegativeAxis, 1DRightView) {
  test_convert_negative_axes_1d<Kokkos::LayoutRight>();
}

template <typename LayoutType>
void test_convert_negative_axes_2d() {
  const int n0 = 3, n1 = 5;
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  RealView2Dtype x("x", n0, n1);

  int converted_axis_0      = KokkosFFT::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1      = KokkosFFT::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_minus1 = KokkosFFT::convert_negative_axis(x, /*axis=*/-1);

  int ref_converted_axis_0      = 0;
  int ref_converted_axis_1      = 1;
  int ref_converted_axis_minus1 = 1;

  EXPECT_EQ( converted_axis_0, ref_converted_axis_0 );
  EXPECT_EQ( converted_axis_1, ref_converted_axis_1 );
  EXPECT_EQ( converted_axis_minus1, ref_converted_axis_minus1 );
}

TEST(ConvertNegativeAxis, 2DLeftView) {
  test_convert_negative_axes_2d<Kokkos::LayoutLeft>();
}

TEST(ConvertNegativeAxis, 2DRightView) {
  test_convert_negative_axes_2d<Kokkos::LayoutRight>();
}

template <typename LayoutType>
void test_convert_negative_axes_3d() {
  const int n0 = 3, n1 = 5, n2 = 8;
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  RealView3Dtype x("x", n0, n1, n2);

  int converted_axis_0      = KokkosFFT::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1      = KokkosFFT::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_2      = KokkosFFT::convert_negative_axis(x, /*axis=*/2);
  int converted_axis_minus1 = KokkosFFT::convert_negative_axis(x, /*axis=*/-1);
  int converted_axis_minus2 = KokkosFFT::convert_negative_axis(x, /*axis=*/-2);

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

TEST(ConvertNegativeAxis, 3DLeftView) {
  test_convert_negative_axes_3d<Kokkos::LayoutLeft>();
}

TEST(ConvertNegativeAxis, 3DRightView) {
  test_convert_negative_axes_3d<Kokkos::LayoutRight>();
}

template <typename LayoutType>
void test_convert_negative_axes_4d() {
  const int n0 = 3, n1 = 5, n2 = 8, n3 = 13;
  using RealView4Dtype = Kokkos::View<double****, LayoutType, execution_space>;
  RealView4Dtype x("x", n0, n1, n2, n3);

  int converted_axis_0      = KokkosFFT::convert_negative_axis(x, /*axis=*/0);
  int converted_axis_1      = KokkosFFT::convert_negative_axis(x, /*axis=*/1);
  int converted_axis_2      = KokkosFFT::convert_negative_axis(x, /*axis=*/2);
  int converted_axis_3      = KokkosFFT::convert_negative_axis(x, /*axis=*/3);
  int converted_axis_minus1 = KokkosFFT::convert_negative_axis(x, /*axis=*/-1);
  int converted_axis_minus2 = KokkosFFT::convert_negative_axis(x, /*axis=*/-2);
  int converted_axis_minus3 = KokkosFFT::convert_negative_axis(x, /*axis=*/-3);

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

TEST(ConvertNegativeAxis, 4DLeftView) {
  test_convert_negative_axes_4d<Kokkos::LayoutLeft>();
}

TEST(ConvertNegativeAxis, 4DRightView) {
  test_convert_negative_axes_4d<Kokkos::LayoutRight>();
}

TEST(IsTransposeNeeded, 1Dto3D) {
  std::array<int, 1> map1D ={0};
  EXPECT_FALSE( KokkosFFT::is_transpose_needed(map1D) );

  std::array<int, 2> map2D ={0, 1}, map2D_axis0 = {1, 0};
  EXPECT_FALSE( KokkosFFT::is_transpose_needed(map2D) );
  EXPECT_TRUE( KokkosFFT::is_transpose_needed(map2D_axis0) );

  std::array<int, 3> map3D =     {0, 1, 2};
  std::array<int, 3> map3D_021 = {0, 2, 1};
  std::array<int, 3> map3D_102 = {1, 0, 2};
  std::array<int, 3> map3D_120 = {1, 2, 0};
  std::array<int, 3> map3D_201 = {2, 0, 1};
  std::array<int, 3> map3D_210 = {2, 1, 0};

  EXPECT_FALSE( KokkosFFT::is_transpose_needed(map3D) );
  EXPECT_TRUE( KokkosFFT::is_transpose_needed(map3D_021) );
  EXPECT_TRUE( KokkosFFT::is_transpose_needed(map3D_102) );
  EXPECT_TRUE( KokkosFFT::is_transpose_needed(map3D_120) );
  EXPECT_TRUE( KokkosFFT::is_transpose_needed(map3D_201) );
  EXPECT_TRUE( KokkosFFT::is_transpose_needed(map3D_210) );
}

TEST(GetIndex, Vectors) {
  std::vector<int> v = {0, 1, 4, 2, 3};

  EXPECT_EQ( KokkosFFT::get_index(v, 0), 0 );
  EXPECT_EQ( KokkosFFT::get_index(v, 1), 1 );
  EXPECT_EQ( KokkosFFT::get_index(v, 2), 3 );
  EXPECT_EQ( KokkosFFT::get_index(v, 3), 4 );
  EXPECT_EQ( KokkosFFT::get_index(v, 4), 2 );

  EXPECT_THROW(
    KokkosFFT::get_index(v, -1),
    std::runtime_error
  );

  EXPECT_THROW(
    KokkosFFT::get_index(v, 5),
    std::runtime_error
  );
}