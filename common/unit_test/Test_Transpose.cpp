#include <algorithm>
#include <random>
#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_transpose.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct MapAxes : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct Transpose : public ::testing::Test {
  using layout_type = T;
};

TYPED_TEST_SUITE(MapAxes, test_types);
TYPED_TEST_SUITE(Transpose, test_types);

// Tests for map axes over ND views
template <typename LayoutType>
void test_map_axes1d() {
  const int len        = 30;
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  RealView1Dtype x("x", len);

  auto [map_axis, map_inv_axis] = KokkosFFT::Impl::get_map_axes(x, /*axis=*/0);
  auto [map_axes, map_inv_axes] =
      KokkosFFT::Impl::get_map_axes(x, /*axes=*/axes_type<1>({0}));

  axes_type<1> ref_map_axis = {0};
  axes_type<1> ref_map_axes = {0};

  EXPECT_TRUE(map_axis == ref_map_axis);
  EXPECT_TRUE(map_axes == ref_map_axes);
  EXPECT_TRUE(map_inv_axis == ref_map_axis);
  EXPECT_TRUE(map_inv_axes == ref_map_axes);
}

template <typename LayoutType>
void test_map_axes2d() {
  const int n0 = 3, n1 = 5;
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  RealView2Dtype x("x", n0, n1);

  auto [map_axis_0, map_inv_axis_0] =
      KokkosFFT::Impl::get_map_axes(x, /*axis=*/0);
  auto [map_axis_1, map_inv_axis_1] =
      KokkosFFT::Impl::get_map_axes(x, /*axis=*/1);
  auto [map_axis_minus1, map_inv_axis_minus1] =
      KokkosFFT::Impl::get_map_axes(x, /*axis=*/-1);
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

  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
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
  EXPECT_TRUE(map_axis_0 == ref_map_axis_0);
  EXPECT_TRUE(map_axis_1 == ref_map_axis_1);
  EXPECT_TRUE(map_axis_minus1 == ref_map_axis_minus1);
  EXPECT_TRUE(map_axes_0 == ref_map_axes_0);
  EXPECT_TRUE(map_axes_1 == ref_map_axes_1);
  EXPECT_TRUE(map_axes_minus1 == ref_map_axes_minus1);
  EXPECT_TRUE(map_axes_0_minus1 == ref_map_axes_0_minus1);
  EXPECT_TRUE(map_axes_minus1_0 == ref_map_axes_minus1_0);
  EXPECT_TRUE(map_axes_0_1 == ref_map_axes_0_1);
  EXPECT_TRUE(map_axes_1_0 == ref_map_axes_1_0);

  // Inverse mapping
  EXPECT_TRUE(map_inv_axis_0 == ref_map_inv_axis_0);
  EXPECT_TRUE(map_inv_axis_1 == ref_map_inv_axis_1);
  EXPECT_TRUE(map_inv_axis_minus1 == ref_map_inv_axis_minus1);
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

  auto [map_axis_0, map_inv_axis_0] = KokkosFFT::Impl::get_map_axes(x, 0);
  auto [map_axis_1, map_inv_axis_1] = KokkosFFT::Impl::get_map_axes(x, 1);
  auto [map_axis_2, map_inv_axis_2] = KokkosFFT::Impl::get_map_axes(x, 2);
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

  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
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
  EXPECT_TRUE(map_axis_0 == ref_map_axis_0);
  EXPECT_TRUE(map_axis_1 == ref_map_axis_1);
  EXPECT_TRUE(map_axis_2 == ref_map_axis_2);
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
  EXPECT_TRUE(map_inv_axis_0 == ref_map_inv_axis_0);
  EXPECT_TRUE(map_inv_axis_1 == ref_map_inv_axis_1);
  EXPECT_TRUE(map_inv_axis_2 == ref_map_inv_axis_2);
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

// Tests for 1D View
TYPED_TEST(MapAxes, 1DView) {
  using layout_type = typename TestFixture::layout_type;

  test_map_axes1d<layout_type>();
}

// Tests for 2D View
TYPED_TEST(MapAxes, 2DView) {
  using layout_type = typename TestFixture::layout_type;

  test_map_axes2d<layout_type>();
}

// Tests for 3D View
TYPED_TEST(MapAxes, 3DView) {
  using layout_type = typename TestFixture::layout_type;

  test_map_axes3d<layout_type>();
}

// Tests for transpose
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
      KokkosFFT::Impl::transpose(execution_space(), x, xt, axes_type<1>({0})),
      std::runtime_error);
}

template <typename LayoutType>
void test_transpose_2d() {
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  const int n0 = 3, n1 = 5;
  RealView2Dtype x("x", n0, n1), ref("ref", n1, n0);
  RealView2Dtype xt_axis01, xt_axis10;  // views are allocated internally

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  // Transposed views
  auto h_x   = Kokkos::create_mirror_view(x);
  auto h_ref = Kokkos::create_mirror_view(ref);
  Kokkos::deep_copy(h_x, x);

  for (int i0 = 0; i0 < h_x.extent(0); i0++) {
    for (int i1 = 0; i1 < h_x.extent(1); i1++) {
      h_ref(i1, i0) = h_x(i0, i1);
    }
  }
  Kokkos::deep_copy(ref, h_ref);
  Kokkos::fence();

  EXPECT_THROW(
      KokkosFFT::Impl::transpose(execution_space(), x, xt_axis01,
                                 axes_type<2>({0, 1})),  // xt is identical to x
      std::runtime_error);

  KokkosFFT::Impl::transpose(execution_space(), x, xt_axis10,
                             axes_type<2>({1, 0}));  // xt is the transpose of x
  EXPECT_TRUE(allclose(xt_axis10, ref, 1.e-5, 1.e-12));
}

template <typename LayoutType>
void test_transpose_3d() {
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  const int n0 = 3, n1 = 5, n2 = 8;
  RealView3Dtype x("x", n0, n1, n2);
  RealView3Dtype xt_axis012, xt_axis021, xt_axis102, xt_axis120,
      xt_axis201, xt_axis210;  // views are allocated internally
  RealView3Dtype ref_axis021("ref_axis021", n0, n2, n1),
      ref_axis102("ref_axis102", n1, n0, n2);
  RealView3Dtype ref_axis120("ref_axis120", n1, n2, n0),
      ref_axis201("ref_axis201", n2, n0, n1),
      ref_axis210("ref_axis210", n2, n1, n0);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  // Transposed views
  auto h_x           = Kokkos::create_mirror_view(x);
  auto h_ref_axis021 = Kokkos::create_mirror_view(ref_axis021);
  auto h_ref_axis102 = Kokkos::create_mirror_view(ref_axis102);
  auto h_ref_axis120 = Kokkos::create_mirror_view(ref_axis120);
  auto h_ref_axis201 = Kokkos::create_mirror_view(ref_axis201);
  auto h_ref_axis210 = Kokkos::create_mirror_view(ref_axis210);

  Kokkos::deep_copy(h_x, x);

  for (int i0 = 0; i0 < h_x.extent(0); i0++) {
    for (int i1 = 0; i1 < h_x.extent(1); i1++) {
      for (int i2 = 0; i2 < h_x.extent(2); i2++) {
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

  EXPECT_THROW(KokkosFFT::Impl::transpose(
                   execution_space(), x, xt_axis012,
                   axes_type<3>({0, 1, 2})),  // xt is identical to x
               std::runtime_error);

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis021,
      axes_type<3>({0, 2, 1}));  // xt is the transpose of x
  EXPECT_TRUE(allclose(xt_axis021, ref_axis021, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis102,
      axes_type<3>({1, 0, 2}));  // xt is the transpose of x
  EXPECT_TRUE(allclose(xt_axis102, ref_axis102, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis120,
      axes_type<3>({1, 2, 0}));  // xt is the transpose of x
  EXPECT_TRUE(allclose(xt_axis120, ref_axis120, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis201,
      axes_type<3>({2, 0, 1}));  // xt is the transpose of x
  EXPECT_TRUE(allclose(xt_axis201, ref_axis201, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis210,
      axes_type<3>({2, 1, 0}));  // xt is the transpose of x
  EXPECT_TRUE(allclose(xt_axis210, ref_axis210, 1.e-5, 1.e-12));
}

template <typename LayoutType>
void test_transpose_4d() {
  using RealView4DType = Kokkos::View<double****, LayoutType, execution_space>;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5;
  constexpr std::size_t DIM = 4;
  RealView4DType x("x", n0, n1, n2, n3);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3});

  std::vector< axes_type<DIM> > list_of_tested_axes = {
    axes_type<DIM>({0, 1, 2, 3}),
    axes_type<DIM>({0, 1, 3, 2}),
    axes_type<DIM>({0, 2, 1, 3}),
    axes_type<DIM>({0, 2, 3, 1}),
    axes_type<DIM>({0, 3, 1, 2}),
    axes_type<DIM>({0, 3, 2, 1}),
    axes_type<DIM>({1, 0, 2, 3}),
    axes_type<DIM>({1, 0, 3, 2}),
    axes_type<DIM>({1, 2, 0, 3}),
    axes_type<DIM>({1, 2, 3, 0}),
    axes_type<DIM>({1, 3, 0, 2}),
    axes_type<DIM>({1, 3, 2, 0}),
    axes_type<DIM>({2, 0, 1, 3}),
    axes_type<DIM>({2, 0, 3, 1}),
    axes_type<DIM>({2, 1, 0, 3}),
    axes_type<DIM>({2, 1, 3, 0}),
    axes_type<DIM>({2, 3, 0, 1}),
    axes_type<DIM>({2, 3, 1, 0}),
    axes_type<DIM>({3, 0, 1, 2}),
    axes_type<DIM>({3, 0, 2, 1}),
    axes_type<DIM>({3, 1, 0, 2}),
    axes_type<DIM>({3, 1, 2, 0}),
    axes_type<DIM>({3, 2, 0, 1}),
    axes_type<DIM>({3, 2, 1, 0})
  };

  for(auto& tested_axes: list_of_tested_axes) {
    axes_type<DIM> out_extents;
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, tested_axes);

    // Convert to vector, need to reverse the order for LayoutLeft
    std::vector<int> _map(map.begin(), map.end());
    if(std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
      std::reverse(_map.begin(), _map.end());
    }

    for(int i=0; i<DIM; i++) {
      out_extents.at(i) = x.extent(_map.at(i));
    }

    auto [_n0, _n1, _n2, _n3] = out_extents;
    RealView4DType xt;
    RealView4DType ref("ref", _n0, _n1, _n2, _n3);

    // Transposed Views
    auto h_ref = Kokkos::create_mirror_view(ref);

    // Filling the transposed View
    for (int i0 = 0; i0 < h_x.extent(0); i0++) {
      for (int i1 = 0; i1 < h_x.extent(1); i1++) {
        for (int i2 = 0; i2 < h_x.extent(2); i2++) {
          for (int i3 = 0; i3 < h_x.extent(3); i3++) {
            int _i0 = i0, _i1 = i1, _i2 = i2, _i3 = i3;
            if (_map[0] == 1) {
              _i0 = i1;
            } else if (_map[0] == 2) {
              _i0 = i2;
            } else if (_map[0] == 3) {
              _i0 = i3;
            }

            if (_map[1] == 0) {
              _i1 = i0;
            } else if (_map[1] == 2) {
              _i1 = i2;
            } else if (_map[1] == 3) {
              _i1 = i3;
            }

            if (_map[2] == 0) {
              _i2 = i0;
            } else if (_map[2] == 1) {
              _i2 = i1;
            } else if (_map[2] == 3) {
              _i2 = i3;
            }

            if (_map[3] == 0) {
              _i3 = i0;
            } else if (_map[3] == 1) {
              _i3 = i1;
            } else if (_map[3] == 2) {
              _i3 = i2;
            }

            h_ref(_i0, _i1, _i2, _i3) = h_x(i0, i1, i2, i3);
          }
        }
      }
    }

    Kokkos::deep_copy(ref, h_ref);
    Kokkos::fence();

    if(tested_axes == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(
                       execution_space(), x, xt,
                       tested_axes),  // xt is identical to x
                   std::runtime_error);
    } else {
      KokkosFFT::Impl::transpose(
          execution_space(), x, xt,
          tested_axes);  // xt is the transpose of x
      EXPECT_TRUE(allclose(xt, ref, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_5d() {
  using RealView5DType = Kokkos::View<double*****, LayoutType, execution_space>;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6;
  constexpr std::size_t DIM = 5;
  RealView5DType x("x", n0, n1, n2, n3, n4);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4});

  // Randomly choosen axes
  std::vector< axes_type<DIM> > list_of_tested_axes = {
    axes_type<DIM>({0, 1, 2, 3, 4}),
    axes_type<DIM>({0, 1, 3, 2, 4}),
    axes_type<DIM>({0, 2, 1, 3, 4}),
    axes_type<DIM>({0, 2, 3, 4, 1}),
    axes_type<DIM>({0, 3, 4, 1, 2}),
    axes_type<DIM>({0, 3, 2, 1, 4}),
    axes_type<DIM>({0, 4, 3, 1, 2}),
    axes_type<DIM>({0, 4, 2, 1, 3}),
    axes_type<DIM>({1, 0, 2, 3, 4}),
    axes_type<DIM>({1, 0, 4, 3, 2}),
    axes_type<DIM>({1, 2, 0, 3, 4}),
    axes_type<DIM>({1, 2, 3, 4, 0}),
    axes_type<DIM>({1, 3, 0, 4, 2}),
    axes_type<DIM>({1, 3, 4, 2, 0}),
    axes_type<DIM>({1, 4, 0, 2, 3}),
    axes_type<DIM>({1, 4, 3, 2, 0}),
    axes_type<DIM>({2, 0, 1, 3, 4}),
    axes_type<DIM>({2, 0, 4, 3, 1}),
    axes_type<DIM>({2, 1, 0, 3, 4}),
    axes_type<DIM>({2, 1, 3, 4, 0}),
    axes_type<DIM>({2, 3, 4, 0, 1}),
    axes_type<DIM>({2, 3, 4, 1, 0}),
    axes_type<DIM>({2, 4, 3, 0, 1}),
    axes_type<DIM>({2, 4, 1, 0, 3}),
    axes_type<DIM>({3, 0, 1, 2, 4}),
    axes_type<DIM>({3, 0, 2, 4, 1}),
    axes_type<DIM>({3, 1, 0, 2, 4}),
    axes_type<DIM>({3, 1, 4, 2, 0}),
    axes_type<DIM>({3, 2, 0, 1, 4}),
    axes_type<DIM>({3, 2, 1, 4, 0}),
    axes_type<DIM>({3, 4, 0, 1, 2}),
    axes_type<DIM>({3, 4, 1, 2, 0}),
    axes_type<DIM>({4, 0, 1, 2, 3}),
    axes_type<DIM>({4, 0, 1, 3, 2}),
    axes_type<DIM>({4, 1, 2, 0, 3}),
    axes_type<DIM>({4, 1, 2, 3, 0}),
    axes_type<DIM>({4, 2, 3, 1, 0}),
    axes_type<DIM>({4, 2, 3, 0, 1}),
    axes_type<DIM>({4, 3, 1, 0, 2}),
    axes_type<DIM>({4, 3, 2, 0, 1})
  };

  for(auto& tested_axes: list_of_tested_axes) {
    axes_type<DIM> out_extents;
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, tested_axes);

    // Convert to vector, need to reverse the order for LayoutLeft
    std::vector<int> _map(map.begin(), map.end());
    if(std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
      std::reverse(_map.begin(), _map.end());
    }

    for(int i=0; i<DIM; i++) {
      out_extents.at(i) = x.extent(_map.at(i));
    }

    auto [_n0, _n1, _n2, _n3, _n4] = out_extents;
    RealView5DType xt;
    RealView5DType ref("ref", _n0, _n1, _n2, _n3, _n4);

    // Transposed Views
    auto h_ref = Kokkos::create_mirror_view(ref);

    // Filling the transposed View
    for (int i0 = 0; i0 < h_x.extent(0); i0++) {
      for (int i1 = 0; i1 < h_x.extent(1); i1++) {
        for (int i2 = 0; i2 < h_x.extent(2); i2++) {
          for (int i3 = 0; i3 < h_x.extent(3); i3++) {
            for (int i4 = 0; i4 < h_x.extent(4); i4++) {
              int _i0 = i0, _i1 = i1, _i2 = i2, _i3 = i3, _i4 = i4;
              if (_map[0] == 1) {
                _i0 = i1;
              } else if (_map[0] == 2) {
                _i0 = i2;
              } else if (_map[0] == 3) {
                _i0 = i3;
              } else if (_map[0] == 4) {
                _i0 = i4;
              }
 
              if (_map[1] == 0) {
                _i1 = i0;
              } else if (_map[1] == 2) {
                _i1 = i2;
              } else if (_map[1] == 3) {
                _i1 = i3;
              } else if (_map[1] == 4) {
                _i1 = i4;
              }

              if (_map[2] == 0) {
                _i2 = i0;
              } else if (_map[2] == 1) {
                _i2 = i1;
              } else if (_map[2] == 3) {
                _i2 = i3;
              } else if (_map[2] == 4) {
                _i2 = i4;
              }

              if (_map[3] == 0) {
                _i3 = i0;
              } else if (_map[3] == 1) {
                _i3 = i1;
              } else if (_map[3] == 2) {
                _i3 = i2;
              } else if (_map[3] == 4) {
                _i3 = i4;
              }

              if (_map[4] == 0) {
                _i4 = i0;
              } else if (_map[4] == 1) {
                _i4 = i1;
              } else if (_map[4] == 2) {
                _i4 = i2;
              } else if (_map[4] == 3) {
                _i4 = i3;
              }

              h_ref(_i0, _i1, _i2, _i3, _i4) = h_x(i0, i1, i2, i3, i4);
            }
          }
        }
      }
    }

    Kokkos::deep_copy(ref, h_ref);
    Kokkos::fence();

    if(tested_axes == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(
                       execution_space(), x, xt,
                       tested_axes),  // xt is identical to x
                   std::runtime_error);
    } else {
      KokkosFFT::Impl::transpose(
          execution_space(), x, xt,
          tested_axes);  // xt is the transpose of x
      EXPECT_TRUE(allclose(xt, ref, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_6d() {
  using RealView6DType = Kokkos::View<double******, LayoutType, execution_space>;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7;
  constexpr std::size_t DIM = 6;
  RealView6DType x("x", n0, n1, n2, n3, n4, n5);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5});

  // Too much combinations, choose axes randomly
  std::vector< axes_type<DIM> > list_of_tested_axes;

  constexpr int nb_trials = 100;
  auto rng = std::default_random_engine {};

  for(int i=0; i<nb_trials; i++) {
    axes_type<DIM> tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);
    list_of_tested_axes.push_back(tmp_axes);
  }
  
  for(auto& tested_axes: list_of_tested_axes) {
    axes_type<DIM> out_extents;
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, tested_axes);

    // Convert to vector, need to reverse the order for LayoutLeft
    std::vector<int> _map(map.begin(), map.end());
    if(std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
      std::reverse(_map.begin(), _map.end());
    }

    for(int i=0; i<DIM; i++) {
      out_extents.at(i) = x.extent(_map.at(i));
    }

    auto [_n0, _n1, _n2, _n3, _n4, _n5] = out_extents;
    RealView6DType xt;
    RealView6DType ref("ref", _n0, _n1, _n2, _n3, _n4, _n5);

    // Transposed Views
    auto h_ref = Kokkos::create_mirror_view(ref);

    // Filling the transposed View
    for (int i0 = 0; i0 < h_x.extent(0); i0++) {
      for (int i1 = 0; i1 < h_x.extent(1); i1++) {
        for (int i2 = 0; i2 < h_x.extent(2); i2++) {
          for (int i3 = 0; i3 < h_x.extent(3); i3++) {
            for (int i4 = 0; i4 < h_x.extent(4); i4++) {
              for (int i5 = 0; i5 < h_x.extent(5); i5++) {
                int _i0 = i0, _i1 = i1, _i2 = i2, _i3 = i3, _i4 = i4, _i5 = i5;
                if (_map[0] == 1) {
                  _i0 = i1;
                } else if (_map[0] == 2) {
                  _i0 = i2;
                } else if (_map[0] == 3) {
                  _i0 = i3;
                } else if (_map[0] == 4) {
                  _i0 = i4;
                } else if (_map[0] == 5) {
                  _i0 = i5;
                }
  
                if (_map[1] == 0) {
                  _i1 = i0;
                } else if (_map[1] == 2) {
                  _i1 = i2;
                } else if (_map[1] == 3) {
                  _i1 = i3;
                } else if (_map[1] == 4) {
                  _i1 = i4;
                } else if (_map[1] == 5) {
                  _i1 = i5;
                }

                if (_map[2] == 0) {
                  _i2 = i0;
                } else if (_map[2] == 1) {
                  _i2 = i1;
                } else if (_map[2] == 3) {
                  _i2 = i3;
                } else if (_map[2] == 4) {
                  _i2 = i4;
                } else if (_map[2] == 5) {
                  _i2 = i5;
                }

                if (_map[3] == 0) {
                  _i3 = i0;
                } else if (_map[3] == 1) {
                  _i3 = i1;
                } else if (_map[3] == 2) {
                  _i3 = i2;
                } else if (_map[3] == 4) {
                  _i3 = i4;
                } else if (_map[3] == 5) {
                  _i3 = i5;
                }

                if (_map[4] == 0) {
                  _i4 = i0;
                } else if (_map[4] == 1) {
                  _i4 = i1;
                } else if (_map[4] == 2) {
                  _i4 = i2;
                } else if (_map[4] == 3) {
                  _i4 = i3;
                } else if (_map[4] == 5) {
                  _i4 = i5;
                }

                if (_map[5] == 0) {
                  _i5 = i0;
                } else if (_map[5] == 1) {
                  _i5 = i1;
                } else if (_map[5] == 2) {
                  _i5 = i2;
                } else if (_map[5] == 3) {
                  _i5 = i3;
                } else if (_map[5] == 4) {
                  _i5 = i4;
                }

                h_ref(_i0, _i1, _i2, _i3, _i4, _i5) = h_x(i0, i1, i2, i3, i4, i5);
              }
            }
          }
        }
      }
    }

    Kokkos::deep_copy(ref, h_ref);
    Kokkos::fence();

    if(tested_axes == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(
                       execution_space(), x, xt,
                       tested_axes),  // xt is identical to x
                   std::runtime_error);
    } else {
      KokkosFFT::Impl::transpose(
          execution_space(), x, xt,
          tested_axes);  // xt is the transpose of x
      EXPECT_TRUE(allclose(xt, ref, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_7d() {
  using RealView7DType = Kokkos::View<double*******, LayoutType, execution_space>;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8;
  constexpr std::size_t DIM = 7;
  RealView7DType x("x", n0, n1, n2, n3, n4, n5, n6);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6});

  // Too much combinations, choose axes randomly
  std::vector< axes_type<DIM> > list_of_tested_axes;

  constexpr int nb_trials = 100;
  auto rng = std::default_random_engine {};

  for(int i=0; i<nb_trials; i++) {
    axes_type<DIM> tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);
    list_of_tested_axes.push_back(tmp_axes);
  }
  
  for(auto& tested_axes: list_of_tested_axes) {
    axes_type<DIM> out_extents;
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, tested_axes);

    // Convert to vector, need to reverse the order for LayoutLeft
    std::vector<int> _map(map.begin(), map.end());
    if(std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
      std::reverse(_map.begin(), _map.end());
    }

    for(int i=0; i<DIM; i++) {
      out_extents.at(i) = x.extent(_map.at(i));
    }

    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6] = out_extents;
    RealView7DType xt;
    RealView7DType ref("ref", _n0, _n1, _n2, _n3, _n4, _n5, _n6);

    // Transposed Views
    auto h_ref = Kokkos::create_mirror_view(ref);

    // Filling the transposed View
    for (int i0 = 0; i0 < h_x.extent(0); i0++) {
      for (int i1 = 0; i1 < h_x.extent(1); i1++) {
        for (int i2 = 0; i2 < h_x.extent(2); i2++) {
          for (int i3 = 0; i3 < h_x.extent(3); i3++) {
            for (int i4 = 0; i4 < h_x.extent(4); i4++) {
              for (int i5 = 0; i5 < h_x.extent(5); i5++) {
                for (int i6 = 0; i6 < h_x.extent(6); i6++) {
                  int _i0 = i0, _i1 = i1, _i2 = i2, _i3 = i3, _i4 = i4, _i5 = i5, _i6 = i6;
                  if (_map[0] == 1) {
                    _i0 = i1;
                  } else if (_map[0] == 2) {
                    _i0 = i2;
                  } else if (_map[0] == 3) {
                    _i0 = i3;
                  } else if (_map[0] == 4) {
                    _i0 = i4;
                  } else if (_map[0] == 5) {
                    _i0 = i5;
                  } else if (_map[0] == 6) {
                    _i0 = i6;
                  }
    
                  if (_map[1] == 0) {
                    _i1 = i0;
                  } else if (_map[1] == 2) {
                    _i1 = i2;
                  } else if (_map[1] == 3) {
                    _i1 = i3;
                  } else if (_map[1] == 4) {
                    _i1 = i4;
                  } else if (_map[1] == 5) {
                    _i1 = i5;
                  } else if (_map[1] == 6) {
                    _i1 = i6;
                  }

                  if (_map[2] == 0) {
                    _i2 = i0;
                  } else if (_map[2] == 1) {
                    _i2 = i1;
                  } else if (_map[2] == 3) {
                    _i2 = i3;
                  } else if (_map[2] == 4) {
                    _i2 = i4;
                  } else if (_map[2] == 5) {
                    _i2 = i5;
                  } else if (_map[2] == 6) {
                    _i2 = i6;
                  }

                  if (_map[3] == 0) {
                    _i3 = i0;
                  } else if (_map[3] == 1) {
                    _i3 = i1;
                  } else if (_map[3] == 2) {
                    _i3 = i2;
                  } else if (_map[3] == 4) {
                    _i3 = i4;
                  } else if (_map[3] == 5) {
                    _i3 = i5;
                  } else if (_map[3] == 6) {
                    _i3 = i6;
                  }

                  if (_map[4] == 0) {
                    _i4 = i0;
                  } else if (_map[4] == 1) {
                    _i4 = i1;
                  } else if (_map[4] == 2) {
                    _i4 = i2;
                  } else if (_map[4] == 3) {
                    _i4 = i3;
                  } else if (_map[4] == 5) {
                    _i4 = i5;
                  } else if (_map[4] == 6) {
                    _i4 = i6;
                  }

                  if (_map[5] == 0) {
                    _i5 = i0;
                  } else if (_map[5] == 1) {
                    _i5 = i1;
                  } else if (_map[5] == 2) {
                    _i5 = i2;
                  } else if (_map[5] == 3) {
                    _i5 = i3;
                  } else if (_map[5] == 4) {
                    _i5 = i4;
                  } else if (_map[5] == 6) {
                    _i5 = i6;
                  }

                  if (_map[6] == 0) {
                    _i6 = i0;
                  } else if (_map[6] == 1) {
                    _i6 = i1;
                  } else if (_map[6] == 2) {
                    _i6 = i2;
                  } else if (_map[6] == 3) {
                    _i6 = i3;
                  } else if (_map[6] == 4) {
                    _i6 = i4;
                  } else if (_map[6] == 5) {
                    _i6 = i5;
                  }

                  h_ref(_i0, _i1, _i2, _i3, _i4, _i5, _i6) = h_x(i0, i1, i2, i3, i4, i5, i6);
                }
              }
            }
          }
        }
      }
    }

    Kokkos::deep_copy(ref, h_ref);
    Kokkos::fence();

    if(tested_axes == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(
                       execution_space(), x, xt,
                       tested_axes),  // xt is identical to x
                   std::runtime_error);
    } else {
      KokkosFFT::Impl::transpose(
          execution_space(), x, xt,
          tested_axes);  // xt is the transpose of x
      EXPECT_TRUE(allclose(xt, ref, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_8d() {
  using RealView8DType = Kokkos::View<double********, LayoutType, execution_space>;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8, n7 = 9;
  constexpr std::size_t DIM = 8;
  RealView8DType x("x", n0, n1, n2, n3, n4, n5, n6, n7);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6, 7});

  // Too much combinations, choose axes randomly
  std::vector< axes_type<DIM> > list_of_tested_axes;

  constexpr int nb_trials = 100;
  auto rng = std::default_random_engine {};

  for(int i=0; i<nb_trials; i++) {
    axes_type<DIM> tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);
    list_of_tested_axes.push_back(tmp_axes);
  }
  
  for(auto& tested_axes: list_of_tested_axes) {
    axes_type<DIM> out_extents;
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, tested_axes);

    // Convert to vector, need to reverse the order for LayoutLeft
    std::vector<int> _map(map.begin(), map.end());
    if(std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
      std::reverse(_map.begin(), _map.end());
    }

    for(int i=0; i<DIM; i++) {
      out_extents.at(i) = x.extent(_map.at(i));
    }

    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7] = out_extents;
    RealView8DType xt;
    RealView8DType ref("ref", _n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7);

    // Transposed Views
    auto h_ref = Kokkos::create_mirror_view(ref);

    // Filling the transposed View
    for (int i0 = 0; i0 < h_x.extent(0); i0++) {
      for (int i1 = 0; i1 < h_x.extent(1); i1++) {
        for (int i2 = 0; i2 < h_x.extent(2); i2++) {
          for (int i3 = 0; i3 < h_x.extent(3); i3++) {
            for (int i4 = 0; i4 < h_x.extent(4); i4++) {
              for (int i5 = 0; i5 < h_x.extent(5); i5++) {
                for (int i6 = 0; i6 < h_x.extent(6); i6++) {
                  for (int i7 = 0; i7 < h_x.extent(7); i7++) {
                    int _i0 = i0, _i1 = i1, _i2 = i2, _i3 = i3, _i4 = i4, _i5 = i5, _i6 = i6, _i7 = i7;
                    if (_map[0] == 1) {
                      _i0 = i1;
                    } else if (_map[0] == 2) {
                      _i0 = i2;
                    } else if (_map[0] == 3) {
                      _i0 = i3;
                    } else if (_map[0] == 4) {
                      _i0 = i4;
                    } else if (_map[0] == 5) {
                      _i0 = i5;
                    } else if (_map[0] == 6) {
                      _i0 = i6;
                    } else if (_map[0] == 7) {
                      _i0 = i7;
                    }
      
                    if (_map[1] == 0) {
                      _i1 = i0;
                    } else if (_map[1] == 2) {
                      _i1 = i2;
                    } else if (_map[1] == 3) {
                      _i1 = i3;
                    } else if (_map[1] == 4) {
                      _i1 = i4;
                    } else if (_map[1] == 5) {
                      _i1 = i5;
                    } else if (_map[1] == 6) {
                      _i1 = i6;
                    } else if (_map[1] == 7) {
                      _i1 = i7;
                    }

                    if (_map[2] == 0) {
                      _i2 = i0;
                    } else if (_map[2] == 1) {
                      _i2 = i1;
                    } else if (_map[2] == 3) {
                      _i2 = i3;
                    } else if (_map[2] == 4) {
                      _i2 = i4;
                    } else if (_map[2] == 5) {
                      _i2 = i5;
                    } else if (_map[2] == 6) {
                      _i2 = i6;
                    } else if (_map[2] == 7) {
                      _i2 = i7;
                    }

                    if (_map[3] == 0) {
                      _i3 = i0;
                    } else if (_map[3] == 1) {
                      _i3 = i1;
                    } else if (_map[3] == 2) {
                      _i3 = i2;
                    } else if (_map[3] == 4) {
                      _i3 = i4;
                    } else if (_map[3] == 5) {
                      _i3 = i5;
                    } else if (_map[3] == 6) {
                      _i3 = i6;
                    } else if (_map[3] == 7) {
                      _i3 = i7;
                    }

                    if (_map[4] == 0) {
                      _i4 = i0;
                    } else if (_map[4] == 1) {
                      _i4 = i1;
                    } else if (_map[4] == 2) {
                      _i4 = i2;
                    } else if (_map[4] == 3) {
                      _i4 = i3;
                    } else if (_map[4] == 5) {
                      _i4 = i5;
                    } else if (_map[4] == 6) {
                      _i4 = i6;
                    } else if (_map[4] == 7) {
                      _i4 = i7;
                    }

                    if (_map[5] == 0) {
                      _i5 = i0;
                    } else if (_map[5] == 1) {
                      _i5 = i1;
                    } else if (_map[5] == 2) {
                      _i5 = i2;
                    } else if (_map[5] == 3) {
                      _i5 = i3;
                    } else if (_map[5] == 4) {
                      _i5 = i4;
                    } else if (_map[5] == 6) {
                      _i5 = i6;
                    } else if (_map[5] == 7) {
                      _i5 = i7;
                    }

                    if (_map[6] == 0) {
                      _i6 = i0;
                    } else if (_map[6] == 1) {
                      _i6 = i1;
                    } else if (_map[6] == 2) {
                      _i6 = i2;
                    } else if (_map[6] == 3) {
                      _i6 = i3;
                    } else if (_map[6] == 4) {
                      _i6 = i4;
                    } else if (_map[6] == 5) {
                      _i6 = i5;
                    } else if (_map[6] == 7) {
                      _i6 = i7;
                    }

                    if (_map[7] == 0) {
                      _i7 = i0;
                    } else if (_map[7] == 1) {
                      _i7 = i1;
                    } else if (_map[7] == 2) {
                      _i7 = i2;
                    } else if (_map[7] == 3) {
                      _i7 = i3;
                    } else if (_map[7] == 4) {
                      _i7 = i4;
                    } else if (_map[7] == 5) {
                      _i7 = i5;
                    } else if (_map[7] == 6) {
                      _i7 = i6;
                    }

                    h_ref(_i0, _i1, _i2, _i3, _i4, _i5, _i6, _i7) = h_x(i0, i1, i2, i3, i4, i5, i6, i7);
                  }
                }
              }
            }
          }
        }
      }
    }

    Kokkos::deep_copy(ref, h_ref);
    Kokkos::fence();

    if(tested_axes == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(
                       execution_space(), x, xt,
                       tested_axes),  // xt is identical to x
                   std::runtime_error);
    } else {
      KokkosFFT::Impl::transpose(
          execution_space(), x, xt,
          tested_axes);  // xt is the transpose of x
      EXPECT_TRUE(allclose(xt, ref, 1.e-5, 1.e-12));
    }
  }
}

TYPED_TEST(Transpose, 1DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d<layout_type>();
}

TYPED_TEST(Transpose, 2DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d<layout_type>();
}

TYPED_TEST(Transpose, 3DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_3d<layout_type>();
}

TYPED_TEST(Transpose, 4DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_4d<layout_type>();
}

TYPED_TEST(Transpose, 5DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_5d<layout_type>();
}

TYPED_TEST(Transpose, 6DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_6d<layout_type>();
}

TYPED_TEST(Transpose, 7DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_7d<layout_type>();
}

TYPED_TEST(Transpose, 8DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_8d<layout_type>();
}