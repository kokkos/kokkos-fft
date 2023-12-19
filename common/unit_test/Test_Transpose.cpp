#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_transpose.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

template <typename LayoutType>
void test_map_axes1d() {
  const int len = 30;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x("x", len);

  auto [map_axis, map_inv_axis] = KokkosFFT::get_map_axes(x, /*axis=*/0);
  auto [map_axes, map_inv_axes] = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({0}));

  axes_type<1> ref_map_axis = {0};
  axes_type<1> ref_map_axes = {0};

  EXPECT_TRUE( map_axis == ref_map_axis );
  EXPECT_TRUE( map_axes == ref_map_axes );
  EXPECT_TRUE( map_inv_axis == ref_map_axis );
  EXPECT_TRUE( map_inv_axes == ref_map_axes );
}

TEST(MapAxes, 1DLeftView) {
  test_map_axes1d<Kokkos::LayoutLeft>();
}

TEST(MapAxes, 1DRightView) {
  test_map_axes1d<Kokkos::LayoutRight>();
}

template <typename LayoutType>
void test_map_axes2d() {
  const int n0 = 3, n1 = 5;
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  RealView2Dtype x("x", n0, n1);

  auto [map_axis_0, map_inv_axis_0]               = KokkosFFT::get_map_axes(x, /*axis=*/0);
  auto [map_axis_1, map_inv_axis_1]               = KokkosFFT::get_map_axes(x, /*axis=*/1);
  auto [map_axis_minus1, map_inv_axis_minus1]     = KokkosFFT::get_map_axes(x, /*axis=*/-1);
  auto [map_axes_0, map_inv_axes_0]               = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({0}));
  auto [map_axes_1, map_inv_axes_1]               = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({1}));
  auto [map_axes_minus1, map_inv_axes_minus1]     = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({-1}));
  auto [map_axes_0_minus1, map_inv_axes_0_minus1] = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({0, -1}));
  auto [map_axes_minus1_0, map_inv_axes_minus1_0] = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({-1, 0}));
  auto [map_axes_0_1, map_inv_axes_0_1]           = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({0, 1}));
  auto [map_axes_1_0, map_inv_axes_1_0]           = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({1, 0}));

  axes_type<2> ref_map_axis_0, ref_map_inv_axis_0;
  axes_type<2> ref_map_axis_1, ref_map_inv_axis_1;
  axes_type<2> ref_map_axis_minus1, ref_map_inv_axis_minus1;
  axes_type<2> ref_map_axes_0, ref_map_inv_axes_0;
  axes_type<2> ref_map_axes_1, ref_map_inv_axes_1;
  axes_type<2> ref_map_axes_minus1, ref_map_inv_axes_minus1;

  axes_type<2> ref_map_axes_0_minus1, ref_map_inv_axes_0_minus1;
  axes_type<2> ref_map_axes_minus1_0, ref_map_inv_axes_minus1_0;
  axes_type<2> ref_map_axes_0_1, ref_map_inv_axes_0_1;
  axes_type<2> ref_map_axes_1_0, ref_map_inv_axes_1_0;

  if(std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    // Layout Left
    ref_map_axis_0 = {0, 1}, ref_map_inv_axis_0 = {0, 1};
    ref_map_axis_1 = {1, 0}, ref_map_inv_axis_1 = {1, 0};
    ref_map_axis_minus1 = {1, 0}, ref_map_inv_axis_minus1 = {1, 0};
    ref_map_axes_0 = {0, 1}, ref_map_inv_axes_0 = {0, 1};
    ref_map_axes_1 = {1, 0}, ref_map_inv_axes_1 = {1, 0};
    ref_map_axes_minus1 = {1, 0}, ref_map_inv_axes_minus1 = {1, 0};

    ref_map_axes_0_minus1 = {1, 0}, ref_map_inv_axes_0_minus1 = {1, 0};
    ref_map_axes_minus1_0 = {0, 1}, ref_map_inv_axes_minus1_0 = {0, 1};
    ref_map_axes_0_1 = {1, 0}, ref_map_inv_axes_0_1 = {1, 0};
    ref_map_axes_1_0 = {0, 1}, ref_map_inv_axes_1_0 = {0, 1};
  } else {
    // Layout Right
    ref_map_axis_0 = {1, 0}, ref_map_inv_axis_0 = {1, 0};
    ref_map_axis_1 = {0, 1}, ref_map_inv_axis_1 = {0, 1};
    ref_map_axis_minus1 = {0, 1}, ref_map_inv_axis_minus1 = {0, 1};
    ref_map_axes_0 = {1, 0}, ref_map_inv_axes_0 = {1, 0};
    ref_map_axes_1 = {0, 1}, ref_map_inv_axes_1 = {0, 1};
    ref_map_axes_minus1 = {0, 1}, ref_map_inv_axes_minus1 = {0, 1};

    ref_map_axes_0_minus1 = {0, 1}, ref_map_inv_axes_0_minus1 = {0, 1};
    ref_map_axes_minus1_0 = {1, 0}, ref_map_inv_axes_minus1_0 = {1, 0};
    ref_map_axes_0_1 = {0, 1}, ref_map_inv_axes_0_1 = {0, 1};
    ref_map_axes_1_0 = {1, 0}, ref_map_inv_axes_1_0 = {1, 0};
  }

  // Forward mapping
  EXPECT_TRUE( map_axis_0 == ref_map_axis_0 );
  EXPECT_TRUE( map_axis_1 == ref_map_axis_1 );
  EXPECT_TRUE( map_axis_minus1 == ref_map_axis_minus1 );
  EXPECT_TRUE( map_axes_0 == ref_map_axes_0 );
  EXPECT_TRUE( map_axes_1 == ref_map_axes_1 );
  EXPECT_TRUE( map_axes_minus1 == ref_map_axes_minus1 );
  EXPECT_TRUE( map_axes_0_minus1 == ref_map_axes_0_minus1 );
  EXPECT_TRUE( map_axes_minus1_0 == ref_map_axes_minus1_0 );
  EXPECT_TRUE( map_axes_0_1 == ref_map_axes_0_1 );
  EXPECT_TRUE( map_axes_1_0 == ref_map_axes_1_0 );

  // Inverse mapping
  EXPECT_TRUE( map_inv_axis_0 == ref_map_inv_axis_0 );
  EXPECT_TRUE( map_inv_axis_1 == ref_map_inv_axis_1 );
  EXPECT_TRUE( map_inv_axis_minus1 == ref_map_inv_axis_minus1 );
  EXPECT_TRUE( map_inv_axes_0 == ref_map_inv_axes_0 );
  EXPECT_TRUE( map_inv_axes_1 == ref_map_inv_axes_1 );
  EXPECT_TRUE( map_inv_axes_minus1 == ref_map_inv_axes_minus1 );
  EXPECT_TRUE( map_inv_axes_0_minus1 == ref_map_inv_axes_0_minus1 );
  EXPECT_TRUE( map_inv_axes_minus1_0 == ref_map_inv_axes_minus1_0 );
  EXPECT_TRUE( map_inv_axes_0_1 == ref_map_inv_axes_0_1 );
  EXPECT_TRUE( map_inv_axes_1_0 == ref_map_inv_axes_1_0 );
}

TEST(MapAxes, 2DLeftView) {
  test_map_axes2d<Kokkos::LayoutLeft>();
}

TEST(MapAxes, 2DRightView) {
  test_map_axes2d<Kokkos::LayoutRight>();
}

template <typename LayoutType>
void test_map_axes3d() {
  const int n0 = 3, n1 = 5, n2 = 8;
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  RealView3Dtype x("x", n0, n1, n2);

  auto [map_axis_0, map_inv_axis_0]    = KokkosFFT::get_map_axes(x, 0);
  auto [map_axis_1, map_inv_axis_1]    = KokkosFFT::get_map_axes(x, 1);
  auto [map_axis_2, map_inv_axis_2]    = KokkosFFT::get_map_axes(x, 2);
  auto [map_axes_0, map_inv_axes_0]    = KokkosFFT::get_map_axes(x, axes_type<1>({0}));
  auto [map_axes_1, map_inv_axes_1]    = KokkosFFT::get_map_axes(x, axes_type<1>({1}));
  auto [map_axes_2, map_inv_axes_2]    = KokkosFFT::get_map_axes(x, axes_type<1>({2}));

  auto [map_axes_0_1, map_inv_axes_0_1] = KokkosFFT::get_map_axes(x, axes_type<2>({0, 1}));
  auto [map_axes_0_2, map_inv_axes_0_2] = KokkosFFT::get_map_axes(x, axes_type<2>({0, 2}));
  auto [map_axes_1_0, map_inv_axes_1_0] = KokkosFFT::get_map_axes(x, axes_type<2>({1, 0}));
  auto [map_axes_1_2, map_inv_axes_1_2] = KokkosFFT::get_map_axes(x, axes_type<2>({1, 2}));
  auto [map_axes_2_0, map_inv_axes_2_0] = KokkosFFT::get_map_axes(x, axes_type<2>({2, 0}));
  auto [map_axes_2_1, map_inv_axes_2_1] = KokkosFFT::get_map_axes(x, axes_type<2>({2, 1}));

  auto [map_axes_0_1_2, map_inv_axes_0_1_2] = KokkosFFT::get_map_axes(x, axes_type<3>({0, 1, 2}));
  auto [map_axes_0_2_1, map_inv_axes_0_2_1] = KokkosFFT::get_map_axes(x, axes_type<3>({0, 2, 1}));

  auto [map_axes_1_0_2, map_inv_axes_1_0_2] = KokkosFFT::get_map_axes(x, axes_type<3>({1, 0, 2}));
  auto [map_axes_1_2_0, map_inv_axes_1_2_0] = KokkosFFT::get_map_axes(x, axes_type<3>({1, 2, 0}));
  auto [map_axes_2_0_1, map_inv_axes_2_0_1] = KokkosFFT::get_map_axes(x, axes_type<3>({2, 0, 1}));
  auto [map_axes_2_1_0, map_inv_axes_2_1_0] = KokkosFFT::get_map_axes(x, axes_type<3>({2, 1, 0}));

  axes_type<3> ref_map_axis_0, ref_map_inv_axis_0;
  axes_type<3> ref_map_axis_1, ref_map_inv_axis_1;
  axes_type<3> ref_map_axis_2, ref_map_inv_axis_2;

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

  if(std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    // Layout Left
    ref_map_axis_0 = {0, 1, 2}, ref_map_inv_axis_0 = {0, 1, 2};
    ref_map_axis_1 = {1, 0, 2}, ref_map_inv_axis_1 = {1, 0, 2};
    ref_map_axis_2 = {2, 0, 1}, ref_map_inv_axis_2 = {1, 2, 0};

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
    ref_map_axis_0 = {1, 2, 0}, ref_map_inv_axis_0 = {2, 0, 1};
    ref_map_axis_1 = {0, 2, 1}, ref_map_inv_axis_1 = {0, 2, 1};
    ref_map_axis_2 = {0, 1, 2}, ref_map_inv_axis_2 = {0, 1, 2};

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
  EXPECT_TRUE( map_axis_0 == ref_map_axis_0 );
  EXPECT_TRUE( map_axis_1 == ref_map_axis_1 );
  EXPECT_TRUE( map_axis_2 == ref_map_axis_2 );
  EXPECT_TRUE( map_axes_0 == ref_map_axes_0 );
  EXPECT_TRUE( map_axes_1 == ref_map_axes_1 );
  EXPECT_TRUE( map_axes_2 == ref_map_axes_2 );

  EXPECT_TRUE( map_axes_0_1 == ref_map_axes_0_1 );
  EXPECT_TRUE( map_axes_0_2 == ref_map_axes_0_2 );
  EXPECT_TRUE( map_axes_1_0 == ref_map_axes_1_0 );
  EXPECT_TRUE( map_axes_1_2 == ref_map_axes_1_2 );
  EXPECT_TRUE( map_axes_2_0 == ref_map_axes_2_0 );
  EXPECT_TRUE( map_axes_2_1 == ref_map_axes_2_1 );

  EXPECT_TRUE( map_axes_0_1_2 == ref_map_axes_0_1_2 );
  EXPECT_TRUE( map_axes_0_2_1 == ref_map_axes_0_2_1 );
  EXPECT_TRUE( map_axes_1_0_2 == ref_map_axes_1_0_2 );
  EXPECT_TRUE( map_axes_1_2_0 == ref_map_axes_1_2_0 );
  EXPECT_TRUE( map_axes_2_0_1 == ref_map_axes_2_0_1 );
  EXPECT_TRUE( map_axes_2_1_0 == ref_map_axes_2_1_0 );

  // Inverse mapping
  EXPECT_TRUE( map_inv_axis_0 == ref_map_inv_axis_0 );
  EXPECT_TRUE( map_inv_axis_1 == ref_map_inv_axis_1 );
  EXPECT_TRUE( map_inv_axis_2 == ref_map_inv_axis_2 );
  EXPECT_TRUE( map_inv_axes_0 == ref_map_inv_axes_0 );
  EXPECT_TRUE( map_inv_axes_1 == ref_map_inv_axes_1 );
  EXPECT_TRUE( map_inv_axes_2 == ref_map_inv_axes_2 );

  EXPECT_TRUE( map_inv_axes_0_1 == ref_map_inv_axes_0_1 );
  EXPECT_TRUE( map_inv_axes_0_2 == ref_map_inv_axes_0_2 );
  EXPECT_TRUE( map_inv_axes_1_0 == ref_map_inv_axes_1_0 );
  EXPECT_TRUE( map_inv_axes_1_2 == ref_map_inv_axes_1_2 );
  EXPECT_TRUE( map_inv_axes_2_0 == ref_map_inv_axes_2_0 );
  EXPECT_TRUE( map_inv_axes_2_1 == ref_map_inv_axes_2_1 );

  EXPECT_TRUE( map_inv_axes_0_1_2 == ref_map_inv_axes_0_1_2 );
  EXPECT_TRUE( map_inv_axes_0_2_1 == ref_map_inv_axes_0_2_1 );
  EXPECT_TRUE( map_inv_axes_1_0_2 == ref_map_inv_axes_1_0_2 );
  EXPECT_TRUE( map_inv_axes_1_2_0 == ref_map_inv_axes_1_2_0 );
  EXPECT_TRUE( map_inv_axes_2_0_1 == ref_map_inv_axes_2_0_1 );
  EXPECT_TRUE( map_inv_axes_2_1_0 == ref_map_inv_axes_2_1_0 );
}

TEST(MapAxes, 3DLeftView) {
  test_map_axes3d<Kokkos::LayoutLeft>();
}

TEST(MapAxes, 3DRightView) {
  test_map_axes3d<Kokkos::LayoutRight>();
}

template <typename LayoutType>
void test_transpose_1d() {
  // When transpose is not necessary, we should not call transpose method
  const int len = 30;
  View1D<double> x("x", len), ref("ref", len);
  View1D<double> xt;

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref, x);

  Kokkos::fence();

  EXPECT_THROW(
    KokkosFFT::transpose(execution_space(), x, xt, axes_type<1>({0})),
    std::runtime_error
  );
}

TEST(Transpose, 1DLeftView) {
  test_transpose_1d<Kokkos::LayoutLeft>();
}

TEST(Transpose, 1DRightView) {
  test_transpose_1d<Kokkos::LayoutRight>();
}

TEST(Transpose, 2DLeftView) {
  const int n0 = 3, n1 = 5;
  LeftView2D<double> x("x", n0, n1);
  LeftView2D<double> xt_axis0, xt_axis1; // views are allocated internally
  LeftView2D<double> ref_axis1("ref_axis1", n1, n0);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  // Transposed views
  auto h_x = Kokkos::create_mirror_view(x);
  auto h_ref_axis1 = Kokkos::create_mirror_view(ref_axis1);
  Kokkos::deep_copy(h_x, x);

  for(int i0=0; i0<h_x.extent(0); i0++) {
    for(int i1=0; i1<h_x.extent(1); i1++) {
      h_ref_axis1(i1, i0) = h_x(i0, i1);
    }
  }
  Kokkos::deep_copy(ref_axis1, h_ref_axis1);
  Kokkos::fence();

  EXPECT_THROW(
    KokkosFFT::transpose(execution_space(), x, xt_axis0, axes_type<2>({0, 1})), // xt is identical to x
    std::runtime_error
  );

  KokkosFFT::transpose(execution_space(), x, xt_axis1, axes_type<2>({1, 0})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis1, ref_axis1, 1.e-5, 1.e-12) );
}

TEST(Transpose, 2DRightView) {
  const int n0 = 3, n1 = 5;
  RightView2D<double> x("x", n0, n1), ref_axis0("ref_axis0", n1, n0);
  RightView2D<double> xt_axis0, xt_axis1; // views are allocated internally

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  // Transposed views
  auto h_x = Kokkos::create_mirror_view(x);
  auto h_ref_axis0 = Kokkos::create_mirror_view(ref_axis0);
  Kokkos::deep_copy(h_x, x);

  for(int i0=0; i0<h_x.extent(0); i0++) {
    for(int i1=0; i1<h_x.extent(1); i1++) {
      h_ref_axis0(i1, i0) = h_x(i0, i1);
    }
  }
  Kokkos::deep_copy(ref_axis0, h_ref_axis0);
  Kokkos::fence();

  KokkosFFT::transpose(execution_space(), x, xt_axis0, axes_type<2>({1, 0})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis0, ref_axis0, 1.e-5, 1.e-12) );

  EXPECT_THROW(
    KokkosFFT::transpose(execution_space(), x, xt_axis1, axes_type<2>({0, 1})), // xt is identical to x
    std::runtime_error
  );
}

TEST(Transpose, 3DLeftView) {
  const int n0 = 3, n1 = 5, n2 = 8;
  LeftView3D<double> x("x", n0, n1, n2);
  LeftView3D<double> xt_axis012, xt_axis021, xt_axis102, xt_axis120, xt_axis201, xt_axis210; // views are allocated internally
  LeftView3D<double> ref_axis021("ref_axis021", n0, n2, n1), ref_axis102("ref_axis102", n1, n0, n2);
  LeftView3D<double> ref_axis120("ref_axis120", n1, n2, n0), ref_axis201("ref_axis201", n2, n0, n1), ref_axis210("ref_axis210", n2, n1, n0);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  // Transposed views
  auto h_x = Kokkos::create_mirror_view(x);
  auto h_ref_axis021 = Kokkos::create_mirror_view(ref_axis021);
  auto h_ref_axis102 = Kokkos::create_mirror_view(ref_axis102);
  auto h_ref_axis120 = Kokkos::create_mirror_view(ref_axis120);
  auto h_ref_axis201 = Kokkos::create_mirror_view(ref_axis201);
  auto h_ref_axis210 = Kokkos::create_mirror_view(ref_axis210);

  Kokkos::deep_copy(h_x, x);

  for(int i0=0; i0<h_x.extent(0); i0++) {
    for(int i1=0; i1<h_x.extent(1); i1++) {
      for(int i2=0; i2<h_x.extent(2); i2++) {
        h_ref_axis021(i0, i2, i1) = h_x(i0, i1, i2);
        h_ref_axis102(i1, i0, i2) = h_x(i0, i1, i2);
        h_ref_axis120(i1, i2, i0) = h_x(i0, i1, i2);
        h_ref_axis201(i2, i0, i1) = h_x(i0, i1, i2);
        h_ref_axis210(i2, i1, i0) = h_x(i0, i1, i2);
      }
    }
  }

  Kokkos::deep_copy(ref_axis021, h_ref_axis021);
  Kokkos::deep_copy(ref_axis102, h_ref_axis102);
  Kokkos::deep_copy(ref_axis120, h_ref_axis120);
  Kokkos::deep_copy(ref_axis201, h_ref_axis201);
  Kokkos::deep_copy(ref_axis210, h_ref_axis210);

  Kokkos::fence();

  EXPECT_THROW(
    KokkosFFT::transpose(execution_space(), x, xt_axis012, axes_type<3>({0, 1, 2})), // xt is identical to x
    std::runtime_error
  );

  KokkosFFT::transpose(execution_space(), x, xt_axis021, axes_type<3>({0, 2, 1})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis021, ref_axis021, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis102, axes_type<3>({1, 0, 2})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis102, ref_axis102, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis120, axes_type<3>({1, 2, 0})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis120, ref_axis120, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis201, axes_type<3>({2, 0, 1})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis201, ref_axis201, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis210, axes_type<3>({2, 1, 0})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis210, ref_axis210, 1.e-5, 1.e-12) );
}

TEST(Transpose, 3DRightView) {
  const int n0 = 3, n1 = 5, n2 = 8;
  RightView3D<double> x("x", n0, n1, n2);
  RightView3D<double> xt_axis012, xt_axis021, xt_axis102, xt_axis120, xt_axis201, xt_axis210; // views are allocated internally
  RightView3D<double> ref_axis021("ref_axis021", n0, n2, n1), ref_axis102("ref_axis102", n1, n0, n2);
  RightView3D<double> ref_axis120("ref_axis120", n1, n2, n0), ref_axis201("ref_axis201", n2, n0, n1), ref_axis210("ref_axis210", n2, n1, n0);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  // Transposed views
  auto h_x = Kokkos::create_mirror_view(x);
  auto h_ref_axis021 = Kokkos::create_mirror_view(ref_axis021);
  auto h_ref_axis102 = Kokkos::create_mirror_view(ref_axis102);
  auto h_ref_axis120 = Kokkos::create_mirror_view(ref_axis120);
  auto h_ref_axis201 = Kokkos::create_mirror_view(ref_axis201);
  auto h_ref_axis210 = Kokkos::create_mirror_view(ref_axis210);

  Kokkos::deep_copy(h_x, x);

  for(int i0=0; i0<h_x.extent(0); i0++) {
    for(int i1=0; i1<h_x.extent(1); i1++) {
      for(int i2=0; i2<h_x.extent(2); i2++) {
        h_ref_axis021(i0, i2, i1) = h_x(i0, i1, i2);
        h_ref_axis102(i1, i0, i2) = h_x(i0, i1, i2);
        h_ref_axis120(i1, i2, i0) = h_x(i0, i1, i2);
        h_ref_axis201(i2, i0, i1) = h_x(i0, i1, i2);
        h_ref_axis210(i2, i1, i0) = h_x(i0, i1, i2);
      }
    }
  }

  Kokkos::deep_copy(ref_axis021, h_ref_axis021);
  Kokkos::deep_copy(ref_axis102, h_ref_axis102);
  Kokkos::deep_copy(ref_axis120, h_ref_axis120);
  Kokkos::deep_copy(ref_axis201, h_ref_axis201);
  Kokkos::deep_copy(ref_axis210, h_ref_axis210);

  Kokkos::fence();

  EXPECT_THROW(
    KokkosFFT::transpose(execution_space(), x, xt_axis012, axes_type<3>({0, 1, 2})), // xt is identical to x
    std::runtime_error
  );

  KokkosFFT::transpose(execution_space(), x, xt_axis021, axes_type<3>({0, 2, 1})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis021, ref_axis021, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis102, axes_type<3>({1, 0, 2})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis102, ref_axis102, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis120, axes_type<3>({1, 2, 0})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis120, ref_axis120, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis201, axes_type<3>({2, 0, 1})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis201, ref_axis201, 1.e-5, 1.e-12) );

  KokkosFFT::transpose(execution_space(), x, xt_axis210, axes_type<3>({2, 1, 0})); // xt is the transpose of x
  EXPECT_TRUE( allclose(xt_axis210, ref_axis210, 1.e-5, 1.e-12) );
}