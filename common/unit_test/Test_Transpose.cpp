// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <random>
#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_transpose.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct MapAxes : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct Transpose1D : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct Transpose2D : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct Transpose3D : public ::testing::Test {
  using layout_type = T;
};

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

// Tests for transpose
// 1D Transpose
template <typename LayoutType>
void test_transpose_1d_1dview() {
  // When transpose is not necessary, we should not call transpose method
  using RealView1Dtype = Kokkos::View<double*, LayoutType, execution_space>;
  const int len        = 30;
  RealView1Dtype x("x", len), ref("ref", len);
  RealView1Dtype xt("xt", len);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref, x);

  Kokkos::fence();

  EXPECT_THROW(
      KokkosFFT::Impl::transpose(execution_space(), x, xt, axes_type<1>({0})),
      std::runtime_error);
}

template <typename LayoutType>
void test_transpose_1d_2dview() {
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  constexpr int DIM    = 2;
  const int n0 = 3, n1 = 5;
  RealView2Dtype x("x", n0, n1);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axis0);
    axes_type<DIM> out_extents;
    for (int i = 0; i < DIM; i++) {
      out_extents.at(i) = x.extent(map.at(i));
    }
    auto [nt0, nt1] = out_extents;

    RealView2Dtype xt("xt", nt0, nt1);
    if (map == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                              map),  // xt is identical to x
                   std::runtime_error);
    } else {
      // Transposed Views
      RealView2Dtype ref("ref", nt0, nt1);
      auto h_ref = Kokkos::create_mirror_view(ref);
      // Filling the transposed View
      for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
        for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
          h_ref(i1, i0) = h_x(i0, i1);
        }
      }

      Kokkos::deep_copy(ref, h_ref);
      Kokkos::fence();

      KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                 map);  // xt is the transpose of x
      EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      RealView2Dtype x_inv("x_inv", n0, n1);
      KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
      EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_1d_3dview() {
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  constexpr int DIM    = 3;
  const int n0 = 3, n1 = 5, n2 = 8;
  RealView3Dtype x("x", n0, n1, n2);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axis0);
    axes_type<DIM> out_extents;
    for (int i = 0; i < DIM; i++) {
      out_extents.at(i) = x.extent(map.at(i));
    }
    auto [nt0, nt1, nt2] = out_extents;

    RealView3Dtype xt("xt", nt0, nt1, nt2);
    if (map == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                              map),  // xt is identical to x
                   std::runtime_error);
    } else {
      // Transposed Views
      RealView3Dtype ref("ref", nt0, nt1, nt2);
      auto h_ref = Kokkos::create_mirror_view(ref);
      // Filling the transposed View
      for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
        for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
          for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2); i2++) {
            int dst_i0 = (map[0] == 1) ? i1 : (map[0] == 2) ? i2 : i0;
            int dst_i1 = (map[1] == 0) ? i0 : (map[1] == 2) ? i2 : i1;
            int dst_i2 = (map[2] == 0) ? i0 : (map[2] == 1) ? i1 : i2;

            h_ref(dst_i0, dst_i1, dst_i2) = h_x(i0, i1, i2);
          }
        }
      }

      Kokkos::deep_copy(ref, h_ref);
      Kokkos::fence();

      KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                 map);  // xt is the transpose of x
      EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      RealView3Dtype x_inv("x_invx", n0, n1, n2);
      KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
      EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_1d_4dview() {
  using RealView4Dtype = Kokkos::View<double****, LayoutType, execution_space>;
  constexpr int DIM    = 4;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5;
  RealView4Dtype x("x", n0, n1, n2, n3);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axis0);
    axes_type<DIM> out_extents;
    for (int i = 0; i < DIM; i++) {
      out_extents.at(i) = x.extent(map.at(i));
    }
    auto [nt0, nt1, nt2, nt3] = out_extents;

    RealView4Dtype xt("xt", nt0, nt1, nt2, nt3);
    if (map == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                              map),  // xt is identical to x
                   std::runtime_error);
    } else {
      // Transposed Views
      RealView4Dtype ref("ref", nt0, nt1, nt2, nt3);
      auto h_ref = Kokkos::create_mirror_view(ref);
      // Filling the transposed View
      for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
        for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
          for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2); i2++) {
            for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                 i3++) {
              int dst_i0 = (map[0] == 1)   ? i1
                           : (map[0] == 2) ? i2
                           : (map[0] == 3) ? i3
                                           : i0;
              int dst_i1 = (map[1] == 0)   ? i0
                           : (map[1] == 2) ? i2
                           : (map[1] == 3) ? i3
                                           : i1;
              int dst_i2 = (map[2] == 0)   ? i0
                           : (map[2] == 1) ? i1
                           : (map[2] == 3) ? i3
                                           : i2;
              int dst_i3 = (map[3] == 0)   ? i0
                           : (map[3] == 1) ? i1
                           : (map[3] == 2) ? i2
                                           : i3;

              h_ref(dst_i0, dst_i1, dst_i2, dst_i3) = h_x(i0, i1, i2, i3);
            }
          }
        }
      }

      Kokkos::deep_copy(ref, h_ref);
      Kokkos::fence();

      KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                 map);  // xt is the transpose of x
      EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      RealView4Dtype x_inv("x_inv", n0, n1, n2, n3);
      KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
      EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_1d_5dview() {
  using RealView5Dtype = Kokkos::View<double*****, LayoutType, execution_space>;
  constexpr int DIM    = 5;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6;
  RealView5Dtype x("x", n0, n1, n2, n3, n4);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axis0);
    axes_type<DIM> out_extents;
    for (int i = 0; i < DIM; i++) {
      out_extents.at(i) = x.extent(map.at(i));
    }
    auto [nt0, nt1, nt2, nt3, nt4] = out_extents;

    RealView5Dtype xt("xt", nt0, nt1, nt2, nt3, nt4);
    if (map == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                              map),  // xt is identical to x
                   std::runtime_error);
    } else {
      // Transposed Views
      RealView5Dtype ref("ref", nt0, nt1, nt2, nt3, nt4);
      auto h_ref = Kokkos::create_mirror_view(ref);
      // Filling the transposed View
      for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
        for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
          for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2); i2++) {
            for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                 i3++) {
              for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                   i4++) {
                int dst_i0 = (map[0] == 1)   ? i1
                             : (map[0] == 2) ? i2
                             : (map[0] == 3) ? i3
                             : (map[0] == 4) ? i4
                                             : i0;
                int dst_i1 = (map[1] == 0)   ? i0
                             : (map[1] == 2) ? i2
                             : (map[1] == 3) ? i3
                             : (map[1] == 4) ? i4
                                             : i1;
                int dst_i2 = (map[2] == 0)   ? i0
                             : (map[2] == 1) ? i1
                             : (map[2] == 3) ? i3
                             : (map[2] == 4) ? i4
                                             : i2;
                int dst_i3 = (map[3] == 0)   ? i0
                             : (map[3] == 1) ? i1
                             : (map[3] == 2) ? i2
                             : (map[3] == 4) ? i4
                                             : i3;
                int dst_i4 = (map[4] == 0)   ? i0
                             : (map[4] == 1) ? i1
                             : (map[4] == 2) ? i2
                             : (map[4] == 3) ? i3
                                             : i4;
                h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4) =
                    h_x(i0, i1, i2, i3, i4);
              }
            }
          }
        }
      }

      Kokkos::deep_copy(ref, h_ref);
      Kokkos::fence();

      KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                 map);  // xt is the transpose of x
      EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      RealView5Dtype x_inv("x_inv", n0, n1, n2, n3, n4);
      KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
      EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_1d_6dview() {
  using RealView6Dtype =
      Kokkos::View<double******, LayoutType, execution_space>;
  constexpr int DIM = 6;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7;
  RealView6Dtype x("x", n0, n1, n2, n3, n4, n5);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axis0);
    axes_type<DIM> out_extents;
    for (int i = 0; i < DIM; i++) {
      out_extents.at(i) = x.extent(map.at(i));
    }
    auto [nt0, nt1, nt2, nt3, nt4, nt5] = out_extents;

    RealView6Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5);
    if (map == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                              map),  // xt is identical to x
                   std::runtime_error);
    } else {
      // Transposed Views
      RealView6Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5);
      auto h_ref = Kokkos::create_mirror_view(ref);
      // Filling the transposed View
      for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
        for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
          for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2); i2++) {
            for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                 i3++) {
              for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                   i4++) {
                for (int i5 = 0; static_cast<std::size_t>(i5) < h_x.extent(5);
                     i5++) {
                  int dst_i0 = (map[0] == 1)   ? i1
                               : (map[0] == 2) ? i2
                               : (map[0] == 3) ? i3
                               : (map[0] == 4) ? i4
                               : (map[0] == 5) ? i5
                                               : i0;
                  int dst_i1 = (map[1] == 0)   ? i0
                               : (map[1] == 2) ? i2
                               : (map[1] == 3) ? i3
                               : (map[1] == 4) ? i4
                               : (map[1] == 5) ? i5
                                               : i1;
                  int dst_i2 = (map[2] == 0)   ? i0
                               : (map[2] == 1) ? i1
                               : (map[2] == 3) ? i3
                               : (map[2] == 4) ? i4
                               : (map[2] == 5) ? i5
                                               : i2;
                  int dst_i3 = (map[3] == 0)   ? i0
                               : (map[3] == 1) ? i1
                               : (map[3] == 2) ? i2
                               : (map[3] == 4) ? i4
                               : (map[3] == 5) ? i5
                                               : i3;
                  int dst_i4 = (map[4] == 0)   ? i0
                               : (map[4] == 1) ? i1
                               : (map[4] == 2) ? i2
                               : (map[4] == 3) ? i3
                               : (map[4] == 5) ? i5
                                               : i4;
                  int dst_i5 = (map[5] == 0)   ? i0
                               : (map[5] == 1) ? i1
                               : (map[5] == 2) ? i2
                               : (map[5] == 3) ? i3
                               : (map[5] == 4) ? i4
                                               : i5;
                  h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5) =
                      h_x(i0, i1, i2, i3, i4, i5);
                }
              }
            }
          }
        }
      }

      Kokkos::deep_copy(ref, h_ref);
      Kokkos::fence();

      KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                 map);  // xt is the transpose of x
      EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      RealView6Dtype x_inv("x_inv_x", n0, n1, n2, n3, n4, n5);
      KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
      EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_1d_7dview() {
  using RealView7Dtype =
      Kokkos::View<double*******, LayoutType, execution_space>;
  constexpr int DIM = 7;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8;
  RealView7Dtype x("x", n0, n1, n2, n3, n4, n5, n6);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axis0);
    axes_type<DIM> out_extents;
    for (int i = 0; i < DIM; i++) {
      out_extents.at(i) = x.extent(map.at(i));
    }
    auto [nt0, nt1, nt2, nt3, nt4, nt5, nt6] = out_extents;

    RealView7Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5, nt6);
    if (map == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                              map),  // xt is identical to x
                   std::runtime_error);
    } else {
      // Transposed Views
      RealView7Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5, nt6);
      auto h_ref = Kokkos::create_mirror_view(ref);
      // Filling the transposed View
      for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
        for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
          for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2); i2++) {
            for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                 i3++) {
              for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                   i4++) {
                for (int i5 = 0; static_cast<std::size_t>(i5) < h_x.extent(5);
                     i5++) {
                  for (int i6 = 0; static_cast<std::size_t>(i6) < h_x.extent(6);
                       i6++) {
                    int dst_i0    = (map[0] == 1)   ? i1
                                    : (map[0] == 2) ? i2
                                    : (map[0] == 3) ? i3
                                    : (map[0] == 4) ? i4
                                    : (map[0] == 5) ? i5
                                    : (map[0] == 6) ? i6
                                                    : i0;
                    int dst_i1    = (map[1] == 0)   ? i0
                                    : (map[1] == 2) ? i2
                                    : (map[1] == 3) ? i3
                                    : (map[1] == 4) ? i4
                                    : (map[1] == 5) ? i5
                                    : (map[1] == 6) ? i6
                                                    : i1;
                    int dst_i2    = (map[2] == 0)   ? i0
                                    : (map[2] == 1) ? i1
                                    : (map[2] == 3) ? i3
                                    : (map[2] == 4) ? i4
                                    : (map[2] == 5) ? i5
                                    : (map[2] == 6) ? i6
                                                    : i2;
                    int dst_i3    = (map[3] == 0)   ? i0
                                    : (map[3] == 1) ? i1
                                    : (map[3] == 2) ? i2
                                    : (map[3] == 4) ? i4
                                    : (map[3] == 5) ? i5
                                    : (map[3] == 6) ? i6
                                                    : i3;
                    int dst_i4    = (map[4] == 0)   ? i0
                                    : (map[4] == 1) ? i1
                                    : (map[4] == 2) ? i2
                                    : (map[4] == 3) ? i3
                                    : (map[4] == 5) ? i5
                                    : (map[4] == 6) ? i6
                                                    : i4;
                    int dst_i5    = (map[5] == 0)   ? i0
                                    : (map[5] == 1) ? i1
                                    : (map[5] == 2) ? i2
                                    : (map[5] == 3) ? i3
                                    : (map[5] == 4) ? i4
                                    : (map[5] == 6) ? i6
                                                    : i5;
                    int dst_i6    = (map[6] == 0)   ? i0
                                    : (map[6] == 1) ? i1
                                    : (map[6] == 2) ? i2
                                    : (map[6] == 3) ? i3
                                    : (map[6] == 4) ? i4
                                    : (map[6] == 5) ? i5
                                                    : i6;
                    h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5,
                          dst_i6) = h_x(i0, i1, i2, i3, i4, i5, i6);
                  }
                }
              }
            }
          }
        }
      }

      Kokkos::deep_copy(ref, h_ref);
      Kokkos::fence();

      KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                 map);  // xt is the transpose of x
      EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      RealView7Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5, n6);
      KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
      EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_1d_8dview() {
  using RealView8Dtype =
      Kokkos::View<double********, LayoutType, execution_space>;
  constexpr int DIM = 8;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8, n7 = 9;
  RealView8Dtype x("x", n0, n1, n2, n3, n4, n5, n6, n7);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6, 7});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axis0);
    axes_type<DIM> out_extents;
    for (int i = 0; i < DIM; i++) {
      out_extents.at(i) = x.extent(map.at(i));
    }
    auto [nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7] = out_extents;

    RealView8Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7);
    if (map == default_axes) {
      EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                              map),  // xt is identical to x
                   std::runtime_error);
    } else {
      // Transposed Views
      RealView8Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7);
      auto h_ref = Kokkos::create_mirror_view(ref);
      // Filling the transposed View
      for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
        for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
          for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2); i2++) {
            for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                 i3++) {
              for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                   i4++) {
                for (int i5 = 0; static_cast<std::size_t>(i5) < h_x.extent(5);
                     i5++) {
                  for (int i6 = 0; static_cast<std::size_t>(i6) < h_x.extent(6);
                       i6++) {
                    for (int i7 = 0;
                         static_cast<std::size_t>(i7) < h_x.extent(7); i7++) {
                      int dst_i0 = (map[0] == 1)   ? i1
                                   : (map[0] == 2) ? i2
                                   : (map[0] == 3) ? i3
                                   : (map[0] == 4) ? i4
                                   : (map[0] == 5) ? i5
                                   : (map[0] == 6) ? i6
                                   : (map[0] == 7) ? i7
                                                   : i0;
                      int dst_i1 = (map[1] == 0)   ? i0
                                   : (map[1] == 2) ? i2
                                   : (map[1] == 3) ? i3
                                   : (map[1] == 4) ? i4
                                   : (map[1] == 5) ? i5
                                   : (map[1] == 6) ? i6
                                   : (map[1] == 7) ? i7
                                                   : i1;
                      int dst_i2 = (map[2] == 0)   ? i0
                                   : (map[2] == 1) ? i1
                                   : (map[2] == 3) ? i3
                                   : (map[2] == 4) ? i4
                                   : (map[2] == 5) ? i5
                                   : (map[2] == 6) ? i6
                                   : (map[2] == 7) ? i7
                                                   : i2;
                      int dst_i3 = (map[3] == 0)   ? i0
                                   : (map[3] == 1) ? i1
                                   : (map[3] == 2) ? i2
                                   : (map[3] == 4) ? i4
                                   : (map[3] == 5) ? i5
                                   : (map[3] == 6) ? i6
                                   : (map[3] == 7) ? i7
                                                   : i3;
                      int dst_i4 = (map[4] == 0)   ? i0
                                   : (map[4] == 1) ? i1
                                   : (map[4] == 2) ? i2
                                   : (map[4] == 3) ? i3
                                   : (map[4] == 5) ? i5
                                   : (map[4] == 6) ? i6
                                   : (map[4] == 7) ? i7
                                                   : i4;
                      int dst_i5 = (map[5] == 0)   ? i0
                                   : (map[5] == 1) ? i1
                                   : (map[5] == 2) ? i2
                                   : (map[5] == 3) ? i3
                                   : (map[5] == 4) ? i4
                                   : (map[5] == 6) ? i6
                                   : (map[5] == 7) ? i7
                                                   : i5;
                      int dst_i6 = (map[6] == 0)   ? i0
                                   : (map[6] == 1) ? i1
                                   : (map[6] == 2) ? i2
                                   : (map[6] == 3) ? i3
                                   : (map[6] == 4) ? i4
                                   : (map[6] == 5) ? i5
                                   : (map[6] == 7) ? i7
                                                   : i6;
                      int dst_i7 = (map[7] == 0)   ? i0
                                   : (map[7] == 1) ? i1
                                   : (map[7] == 2) ? i2
                                   : (map[7] == 3) ? i3
                                   : (map[7] == 4) ? i4
                                   : (map[7] == 5) ? i5
                                   : (map[7] == 6) ? i6
                                                   : i7;
                      h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5,
                            dst_i6, dst_i7) =
                          h_x(i0, i1, i2, i3, i4, i5, i6, i7);
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

      KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                 map);  // xt is the transpose of x
      EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      RealView8Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5, n6, n7);
      KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
      EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
    }
  }
}

template <typename LayoutType>
void test_transpose_2d_2dview() {
  using RealView2Dtype = Kokkos::View<double**, LayoutType, execution_space>;
  const int n0 = 3, n1 = 5;
  RealView2Dtype x("x", n0, n1), x_inv("x_inv", n0, n1), ref("ref", n1, n0);
  RealView2Dtype xt_axis01("xt_axis01", n0, n1), xt_axis10("xt_axis10", n1, n0);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  // Transposed views
  auto h_x   = Kokkos::create_mirror_view(x);
  auto h_ref = Kokkos::create_mirror_view(ref);
  Kokkos::deep_copy(h_x, x);

  for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
    for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
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
  EXPECT_TRUE(allclose(execution_space(), xt_axis10, ref, 1.e-5, 1.e-12));

  // Inverse (transpose of transpose is identical to the original)
  KokkosFFT::Impl::transpose(execution_space(), xt_axis10, x_inv,
                             axes_type<2>({1, 0}));  // xt is the transpose of x
  EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
}

template <typename LayoutType>
void test_transpose_2d_3dview() {
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  constexpr int DIM    = 3;
  const int n0 = 3, n1 = 5, n2 = 8;
  RealView3Dtype x("x", n0, n1, n2);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      KokkosFFT::axis_type<2> axes = {axis0, axis1};

      auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
      axes_type<DIM> out_extents;
      for (int i = 0; i < DIM; i++) {
        out_extents.at(i) = x.extent(map.at(i));
      }
      auto [nt0, nt1, nt2] = out_extents;

      RealView3Dtype xt("xt", nt0, nt1, nt2);
      if (map == default_axes) {
        EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                map),  // xt is identical to x
                     std::runtime_error);
      } else {
        // Transposed Views
        RealView3Dtype ref("ref", nt0, nt1, nt2);
        auto h_ref = Kokkos::create_mirror_view(ref);
        // Filling the transposed View
        for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
          for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
            for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                 i2++) {
              int dst_i0 = (map[0] == 1) ? i1 : (map[0] == 2) ? i2 : i0;
              int dst_i1 = (map[1] == 0) ? i0 : (map[1] == 2) ? i2 : i1;
              int dst_i2 = (map[2] == 0) ? i0 : (map[2] == 1) ? i1 : i2;

              h_ref(dst_i0, dst_i1, dst_i2) = h_x(i0, i1, i2);
            }
          }
        }

        Kokkos::deep_copy(ref, h_ref);
        Kokkos::fence();

        KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                   map);  // xt is the transpose of x
        EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

        // Inverse (transpose of transpose is identical to the original)
        RealView3Dtype x_inv("x_inv", n0, n1, n2);
        KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
        EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_2d_4dview() {
  using RealView4Dtype = Kokkos::View<double****, LayoutType, execution_space>;
  constexpr int DIM    = 4;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5;
  RealView4Dtype x("x", n0, n1, n2, n3);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      KokkosFFT::axis_type<2> axes = {axis0, axis1};

      auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
      axes_type<DIM> out_extents;
      for (int i = 0; i < DIM; i++) {
        out_extents.at(i) = x.extent(map.at(i));
      }
      auto [nt0, nt1, nt2, nt3] = out_extents;

      RealView4Dtype xt("xt", nt0, nt1, nt2, nt3);
      if (map == default_axes) {
        EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                map),  // xt is identical to x
                     std::runtime_error);
      } else {
        // Transposed Views
        RealView4Dtype ref("ref", nt0, nt1, nt2, nt3);
        auto h_ref = Kokkos::create_mirror_view(ref);
        // Filling the transposed View
        for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
          for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
            for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                 i2++) {
              for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                   i3++) {
                int dst_i0 = (map[0] == 1)   ? i1
                             : (map[0] == 2) ? i2
                             : (map[0] == 3) ? i3
                                             : i0;
                int dst_i1 = (map[1] == 0)   ? i0
                             : (map[1] == 2) ? i2
                             : (map[1] == 3) ? i3
                                             : i1;
                int dst_i2 = (map[2] == 0)   ? i0
                             : (map[2] == 1) ? i1
                             : (map[2] == 3) ? i3
                                             : i2;
                int dst_i3 = (map[3] == 0)   ? i0
                             : (map[3] == 1) ? i1
                             : (map[3] == 2) ? i2
                                             : i3;

                h_ref(dst_i0, dst_i1, dst_i2, dst_i3) = h_x(i0, i1, i2, i3);
              }
            }
          }
        }

        Kokkos::deep_copy(ref, h_ref);
        Kokkos::fence();

        KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                   map);  // xt is the transpose of x
        EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

        // Inverse (transpose of transpose is identical to the original)
        RealView4Dtype x_inv("x_inv", n0, n1, n2, n3);
        KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
        EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_2d_5dview() {
  using RealView5Dtype = Kokkos::View<double*****, LayoutType, execution_space>;
  constexpr int DIM    = 5;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6;
  RealView5Dtype x("x", n0, n1, n2, n3, n4);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      KokkosFFT::axis_type<2> axes = {axis0, axis1};

      auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
      axes_type<DIM> out_extents;
      for (int i = 0; i < DIM; i++) {
        out_extents.at(i) = x.extent(map.at(i));
      }
      auto [nt0, nt1, nt2, nt3, nt4] = out_extents;

      RealView5Dtype xt("xt", nt0, nt1, nt2, nt3, nt4);
      if (map == default_axes) {
        EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                map),  // xt is identical to x
                     std::runtime_error);
      } else {
        // Transposed Views
        RealView5Dtype ref("ref", nt0, nt1, nt2, nt3, nt4);
        auto h_ref = Kokkos::create_mirror_view(ref);
        // Filling the transposed View
        for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
          for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
            for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                 i2++) {
              for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                   i3++) {
                for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                     i4++) {
                  int dst_i0 = (map[0] == 1)   ? i1
                               : (map[0] == 2) ? i2
                               : (map[0] == 3) ? i3
                               : (map[0] == 4) ? i4
                                               : i0;
                  int dst_i1 = (map[1] == 0)   ? i0
                               : (map[1] == 2) ? i2
                               : (map[1] == 3) ? i3
                               : (map[1] == 4) ? i4
                                               : i1;
                  int dst_i2 = (map[2] == 0)   ? i0
                               : (map[2] == 1) ? i1
                               : (map[2] == 3) ? i3
                               : (map[2] == 4) ? i4
                                               : i2;
                  int dst_i3 = (map[3] == 0)   ? i0
                               : (map[3] == 1) ? i1
                               : (map[3] == 2) ? i2
                               : (map[3] == 4) ? i4
                                               : i3;
                  int dst_i4 = (map[4] == 0)   ? i0
                               : (map[4] == 1) ? i1
                               : (map[4] == 2) ? i2
                               : (map[4] == 3) ? i3
                                               : i4;
                  h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4) =
                      h_x(i0, i1, i2, i3, i4);
                }
              }
            }
          }
        }

        Kokkos::deep_copy(ref, h_ref);
        Kokkos::fence();

        KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                   map);  // xt is the transpose of x
        EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

        // Inverse (transpose of transpose is identical to the original)
        RealView5Dtype x_inv("x_inv", n0, n1, n2, n3, n4);
        KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
        EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_2d_6dview() {
  using RealView6Dtype =
      Kokkos::View<double******, LayoutType, execution_space>;
  constexpr int DIM = 6;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7;
  RealView6Dtype x("x", n0, n1, n2, n3, n4, n5);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      KokkosFFT::axis_type<2> axes = {axis0, axis1};

      auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
      axes_type<DIM> out_extents;
      for (int i = 0; i < DIM; i++) {
        out_extents.at(i) = x.extent(map.at(i));
      }
      auto [nt0, nt1, nt2, nt3, nt4, nt5] = out_extents;

      RealView6Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5);
      if (map == default_axes) {
        EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                map),  // xt is identical to x
                     std::runtime_error);
      } else {
        // Transposed Views
        RealView6Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5);
        auto h_ref = Kokkos::create_mirror_view(ref);
        // Filling the transposed View
        for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
          for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
            for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                 i2++) {
              for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                   i3++) {
                for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                     i4++) {
                  for (int i5 = 0; static_cast<std::size_t>(i5) < h_x.extent(5);
                       i5++) {
                    int dst_i0 = (map[0] == 1)   ? i1
                                 : (map[0] == 2) ? i2
                                 : (map[0] == 3) ? i3
                                 : (map[0] == 4) ? i4
                                 : (map[0] == 5) ? i5
                                                 : i0;
                    int dst_i1 = (map[1] == 0)   ? i0
                                 : (map[1] == 2) ? i2
                                 : (map[1] == 3) ? i3
                                 : (map[1] == 4) ? i4
                                 : (map[1] == 5) ? i5
                                                 : i1;
                    int dst_i2 = (map[2] == 0)   ? i0
                                 : (map[2] == 1) ? i1
                                 : (map[2] == 3) ? i3
                                 : (map[2] == 4) ? i4
                                 : (map[2] == 5) ? i5
                                                 : i2;
                    int dst_i3 = (map[3] == 0)   ? i0
                                 : (map[3] == 1) ? i1
                                 : (map[3] == 2) ? i2
                                 : (map[3] == 4) ? i4
                                 : (map[3] == 5) ? i5
                                                 : i3;
                    int dst_i4 = (map[4] == 0)   ? i0
                                 : (map[4] == 1) ? i1
                                 : (map[4] == 2) ? i2
                                 : (map[4] == 3) ? i3
                                 : (map[4] == 5) ? i5
                                                 : i4;
                    int dst_i5 = (map[5] == 0)   ? i0
                                 : (map[5] == 1) ? i1
                                 : (map[5] == 2) ? i2
                                 : (map[5] == 3) ? i3
                                 : (map[5] == 4) ? i4
                                                 : i5;
                    h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5) =
                        h_x(i0, i1, i2, i3, i4, i5);
                  }
                }
              }
            }
          }
        }

        Kokkos::deep_copy(ref, h_ref);
        Kokkos::fence();

        KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                   map);  // xt is the transpose of x
        EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

        // Inverse (transpose of transpose is identical to the original)
        RealView6Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5);
        KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
        EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_2d_7dview() {
  using RealView7Dtype =
      Kokkos::View<double*******, LayoutType, execution_space>;
  constexpr int DIM = 7;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8;
  RealView7Dtype x("x", n0, n1, n2, n3, n4, n5, n6);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      KokkosFFT::axis_type<2> axes = {axis0, axis1};

      auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
      axes_type<DIM> out_extents;
      for (int i = 0; i < DIM; i++) {
        out_extents.at(i) = x.extent(map.at(i));
      }
      auto [nt0, nt1, nt2, nt3, nt4, nt5, nt6] = out_extents;

      RealView7Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5, nt6);
      if (map == default_axes) {
        EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                map),  // xt is identical to x
                     std::runtime_error);
      } else {
        // Transposed Views
        RealView7Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5, nt6);
        auto h_ref = Kokkos::create_mirror_view(ref);
        // Filling the transposed View
        for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
          for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
            for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                 i2++) {
              for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                   i3++) {
                for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                     i4++) {
                  for (int i5 = 0; static_cast<std::size_t>(i5) < h_x.extent(5);
                       i5++) {
                    for (int i6 = 0;
                         static_cast<std::size_t>(i6) < h_x.extent(6); i6++) {
                      int dst_i0    = (map[0] == 1)   ? i1
                                      : (map[0] == 2) ? i2
                                      : (map[0] == 3) ? i3
                                      : (map[0] == 4) ? i4
                                      : (map[0] == 5) ? i5
                                      : (map[0] == 6) ? i6
                                                      : i0;
                      int dst_i1    = (map[1] == 0)   ? i0
                                      : (map[1] == 2) ? i2
                                      : (map[1] == 3) ? i3
                                      : (map[1] == 4) ? i4
                                      : (map[1] == 5) ? i5
                                      : (map[1] == 6) ? i6
                                                      : i1;
                      int dst_i2    = (map[2] == 0)   ? i0
                                      : (map[2] == 1) ? i1
                                      : (map[2] == 3) ? i3
                                      : (map[2] == 4) ? i4
                                      : (map[2] == 5) ? i5
                                      : (map[2] == 6) ? i6
                                                      : i2;
                      int dst_i3    = (map[3] == 0)   ? i0
                                      : (map[3] == 1) ? i1
                                      : (map[3] == 2) ? i2
                                      : (map[3] == 4) ? i4
                                      : (map[3] == 5) ? i5
                                      : (map[3] == 6) ? i6
                                                      : i3;
                      int dst_i4    = (map[4] == 0)   ? i0
                                      : (map[4] == 1) ? i1
                                      : (map[4] == 2) ? i2
                                      : (map[4] == 3) ? i3
                                      : (map[4] == 5) ? i5
                                      : (map[4] == 6) ? i6
                                                      : i4;
                      int dst_i5    = (map[5] == 0)   ? i0
                                      : (map[5] == 1) ? i1
                                      : (map[5] == 2) ? i2
                                      : (map[5] == 3) ? i3
                                      : (map[5] == 4) ? i4
                                      : (map[5] == 6) ? i6
                                                      : i5;
                      int dst_i6    = (map[6] == 0)   ? i0
                                      : (map[6] == 1) ? i1
                                      : (map[6] == 2) ? i2
                                      : (map[6] == 3) ? i3
                                      : (map[6] == 4) ? i4
                                      : (map[6] == 5) ? i5
                                                      : i6;
                      h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5,
                            dst_i6) = h_x(i0, i1, i2, i3, i4, i5, i6);
                    }
                  }
                }
              }
            }
          }
        }

        Kokkos::deep_copy(ref, h_ref);
        Kokkos::fence();

        KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                   map);  // xt is the transpose of x
        EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

        // Inverse (transpose of transpose is identical to the original)
        RealView7Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5, n6);
        KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
        EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_2d_8dview() {
  using RealView8Dtype =
      Kokkos::View<double********, LayoutType, execution_space>;
  constexpr int DIM = 8;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8, n7 = 9;
  RealView8Dtype x("x", n0, n1, n2, n3, n4, n5, n6, n7);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6, 7});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      KokkosFFT::axis_type<2> axes = {axis0, axis1};

      auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
      axes_type<DIM> out_extents;
      for (int i = 0; i < DIM; i++) {
        out_extents.at(i) = x.extent(map.at(i));
      }
      auto [nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7] = out_extents;

      RealView8Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7);
      if (map == default_axes) {
        EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                map),  // xt is identical to x
                     std::runtime_error);
      } else {
        // Transposed Views
        RealView8Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7);
        auto h_ref = Kokkos::create_mirror_view(ref);
        // Filling the transposed View
        for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
          for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
            for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                 i2++) {
              for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                   i3++) {
                for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                     i4++) {
                  for (int i5 = 0; static_cast<std::size_t>(i5) < h_x.extent(5);
                       i5++) {
                    for (int i6 = 0;
                         static_cast<std::size_t>(i6) < h_x.extent(6); i6++) {
                      for (int i7 = 0;
                           static_cast<std::size_t>(i7) < h_x.extent(7); i7++) {
                        int dst_i0 = (map[0] == 1)   ? i1
                                     : (map[0] == 2) ? i2
                                     : (map[0] == 3) ? i3
                                     : (map[0] == 4) ? i4
                                     : (map[0] == 5) ? i5
                                     : (map[0] == 6) ? i6
                                     : (map[0] == 7) ? i7
                                                     : i0;
                        int dst_i1 = (map[1] == 0)   ? i0
                                     : (map[1] == 2) ? i2
                                     : (map[1] == 3) ? i3
                                     : (map[1] == 4) ? i4
                                     : (map[1] == 5) ? i5
                                     : (map[1] == 6) ? i6
                                     : (map[1] == 7) ? i7
                                                     : i1;
                        int dst_i2 = (map[2] == 0)   ? i0
                                     : (map[2] == 1) ? i1
                                     : (map[2] == 3) ? i3
                                     : (map[2] == 4) ? i4
                                     : (map[2] == 5) ? i5
                                     : (map[2] == 6) ? i6
                                     : (map[2] == 7) ? i7
                                                     : i2;
                        int dst_i3 = (map[3] == 0)   ? i0
                                     : (map[3] == 1) ? i1
                                     : (map[3] == 2) ? i2
                                     : (map[3] == 4) ? i4
                                     : (map[3] == 5) ? i5
                                     : (map[3] == 6) ? i6
                                     : (map[3] == 7) ? i7
                                                     : i3;
                        int dst_i4 = (map[4] == 0)   ? i0
                                     : (map[4] == 1) ? i1
                                     : (map[4] == 2) ? i2
                                     : (map[4] == 3) ? i3
                                     : (map[4] == 5) ? i5
                                     : (map[4] == 6) ? i6
                                     : (map[4] == 7) ? i7
                                                     : i4;
                        int dst_i5 = (map[5] == 0)   ? i0
                                     : (map[5] == 1) ? i1
                                     : (map[5] == 2) ? i2
                                     : (map[5] == 3) ? i3
                                     : (map[5] == 4) ? i4
                                     : (map[5] == 6) ? i6
                                     : (map[5] == 7) ? i7
                                                     : i5;
                        int dst_i6 = (map[6] == 0)   ? i0
                                     : (map[6] == 1) ? i1
                                     : (map[6] == 2) ? i2
                                     : (map[6] == 3) ? i3
                                     : (map[6] == 4) ? i4
                                     : (map[6] == 5) ? i5
                                     : (map[6] == 7) ? i7
                                                     : i6;
                        int dst_i7 = (map[7] == 0)   ? i0
                                     : (map[7] == 1) ? i1
                                     : (map[7] == 2) ? i2
                                     : (map[7] == 3) ? i3
                                     : (map[7] == 4) ? i4
                                     : (map[7] == 5) ? i5
                                     : (map[7] == 6) ? i6
                                                     : i7;
                        h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5,
                              dst_i6, dst_i7) =
                            h_x(i0, i1, i2, i3, i4, i5, i6, i7);
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

        KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                   map);  // xt is the transpose of x
        EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

        // Inverse (transpose of transpose is identical to the original)
        RealView8Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5, n6, n7);
        KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
        EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_3d_3dview() {
  using RealView3Dtype = Kokkos::View<double***, LayoutType, execution_space>;
  const int n0 = 3, n1 = 5, n2 = 8;
  RealView3Dtype x("x", n0, n1, n2);
  RealView3Dtype xt_axis012("xt_axis012", n0, n1, n2),
      xt_axis021("xt_axis021", n0, n2, n1),
      xt_axis102("xt_axis102", n1, n0, n2),
      xt_axis120("xt_axis120", n1, n2, n0),
      xt_axis201("xt_axis201", n2, n0, n1),
      xt_axis210("xt_axis210", n2, n1, n0);  // views are allocated internally
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

  for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
    for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1); i1++) {
      for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2); i2++) {
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
  EXPECT_TRUE(
      allclose(execution_space(), xt_axis021, ref_axis021, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis102,
      axes_type<3>({1, 0, 2}));  // xt is the transpose of x
  EXPECT_TRUE(
      allclose(execution_space(), xt_axis102, ref_axis102, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis120,
      axes_type<3>({1, 2, 0}));  // xt is the transpose of x
  EXPECT_TRUE(
      allclose(execution_space(), xt_axis120, ref_axis120, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis201,
      axes_type<3>({2, 0, 1}));  // xt is the transpose of x
  EXPECT_TRUE(
      allclose(execution_space(), xt_axis201, ref_axis201, 1.e-5, 1.e-12));

  KokkosFFT::Impl::transpose(
      execution_space(), x, xt_axis210,
      axes_type<3>({2, 1, 0}));  // xt is the transpose of x
  EXPECT_TRUE(
      allclose(execution_space(), xt_axis210, ref_axis210, 1.e-5, 1.e-12));
}

template <typename LayoutType>
void test_transpose_3d_4dview() {
  using RealView4Dtype = Kokkos::View<double****, LayoutType, execution_space>;
  constexpr int DIM    = 4;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5;
  RealView4Dtype x("x", n0, n1, n2, n3);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        KokkosFFT::axis_type<3> axes = {axis0, axis1, axis2};

        auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
        axes_type<DIM> out_extents;
        for (int i = 0; i < DIM; i++) {
          out_extents.at(i) = x.extent(map.at(i));
        }
        auto [nt0, nt1, nt2, nt3] = out_extents;

        RealView4Dtype xt("xt", nt0, nt1, nt2, nt3);
        if (map == default_axes) {
          EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                  map),  // xt is identical to x
                       std::runtime_error);
        } else {
          // Transposed Views
          RealView4Dtype ref("ref", nt0, nt1, nt2, nt3);
          auto h_ref = Kokkos::create_mirror_view(ref);
          // Filling the transposed View
          for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
            for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1);
                 i1++) {
              for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                   i2++) {
                for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                     i3++) {
                  int dst_i0 = (map[0] == 1)   ? i1
                               : (map[0] == 2) ? i2
                               : (map[0] == 3) ? i3
                                               : i0;
                  int dst_i1 = (map[1] == 0)   ? i0
                               : (map[1] == 2) ? i2
                               : (map[1] == 3) ? i3
                                               : i1;
                  int dst_i2 = (map[2] == 0)   ? i0
                               : (map[2] == 1) ? i1
                               : (map[2] == 3) ? i3
                                               : i2;
                  int dst_i3 = (map[3] == 0)   ? i0
                               : (map[3] == 1) ? i1
                               : (map[3] == 2) ? i2
                                               : i3;

                  h_ref(dst_i0, dst_i1, dst_i2, dst_i3) = h_x(i0, i1, i2, i3);
                }
              }
            }
          }

          Kokkos::deep_copy(ref, h_ref);
          Kokkos::fence();

          KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                     map);  // xt is the transpose of x
          EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

          // Inverse (transpose of transpose is identical to the original)
          RealView4Dtype x_inv("x_inv", n0, n1, n2, n3);
          KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
          EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
        }
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_3d_5dview() {
  using RealView5Dtype = Kokkos::View<double*****, LayoutType, execution_space>;
  constexpr int DIM    = 5;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6;
  RealView5Dtype x("x", n0, n1, n2, n3, n4);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;

        KokkosFFT::axis_type<3> axes = {axis0, axis1, axis2};

        auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
        axes_type<DIM> out_extents;
        for (int i = 0; i < DIM; i++) {
          out_extents.at(i) = x.extent(map.at(i));
        }
        auto [nt0, nt1, nt2, nt3, nt4] = out_extents;

        RealView5Dtype xt("xt", nt0, nt1, nt2, nt3, nt4);
        if (map == default_axes) {
          EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                  map),  // xt is identical to x
                       std::runtime_error);
        } else {
          // Transposed Views
          RealView5Dtype ref("ref", nt0, nt1, nt2, nt3, nt4);
          auto h_ref = Kokkos::create_mirror_view(ref);
          // Filling the transposed View
          for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
            for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1);
                 i1++) {
              for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                   i2++) {
                for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                     i3++) {
                  for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                       i4++) {
                    int dst_i0 = (map[0] == 1)   ? i1
                                 : (map[0] == 2) ? i2
                                 : (map[0] == 3) ? i3
                                 : (map[0] == 4) ? i4
                                                 : i0;
                    int dst_i1 = (map[1] == 0)   ? i0
                                 : (map[1] == 2) ? i2
                                 : (map[1] == 3) ? i3
                                 : (map[1] == 4) ? i4
                                                 : i1;
                    int dst_i2 = (map[2] == 0)   ? i0
                                 : (map[2] == 1) ? i1
                                 : (map[2] == 3) ? i3
                                 : (map[2] == 4) ? i4
                                                 : i2;
                    int dst_i3 = (map[3] == 0)   ? i0
                                 : (map[3] == 1) ? i1
                                 : (map[3] == 2) ? i2
                                 : (map[3] == 4) ? i4
                                                 : i3;
                    int dst_i4 = (map[4] == 0)   ? i0
                                 : (map[4] == 1) ? i1
                                 : (map[4] == 2) ? i2
                                 : (map[4] == 3) ? i3
                                                 : i4;
                    h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4) =
                        h_x(i0, i1, i2, i3, i4);
                  }
                }
              }
            }
          }

          Kokkos::deep_copy(ref, h_ref);
          Kokkos::fence();

          KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                     map);  // xt is the transpose of x
          EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

          // Inverse (transpose of transpose is identical to the original)
          RealView5Dtype x_inv("x_inv", n0, n1, n2, n3, n4);
          KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
          EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
        }
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_3d_6dview() {
  using RealView6Dtype =
      Kokkos::View<double******, LayoutType, execution_space>;
  constexpr int DIM = 6;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7;
  RealView6Dtype x("x", n0, n1, n2, n3, n4, n5);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;

        KokkosFFT::axis_type<3> axes = {axis0, axis1, axis2};

        auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
        axes_type<DIM> out_extents;
        for (int i = 0; i < DIM; i++) {
          out_extents.at(i) = x.extent(map.at(i));
        }
        auto [nt0, nt1, nt2, nt3, nt4, nt5] = out_extents;

        RealView6Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5);
        if (map == default_axes) {
          EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                  map),  // xt is identical to x
                       std::runtime_error);
        } else {
          // Transposed Views
          RealView6Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5);
          auto h_ref = Kokkos::create_mirror_view(ref);
          // Filling the transposed View
          for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
            for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1);
                 i1++) {
              for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                   i2++) {
                for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                     i3++) {
                  for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                       i4++) {
                    for (int i5 = 0;
                         static_cast<std::size_t>(i5) < h_x.extent(5); i5++) {
                      int dst_i0 = (map[0] == 1)   ? i1
                                   : (map[0] == 2) ? i2
                                   : (map[0] == 3) ? i3
                                   : (map[0] == 4) ? i4
                                   : (map[0] == 5) ? i5
                                                   : i0;
                      int dst_i1 = (map[1] == 0)   ? i0
                                   : (map[1] == 2) ? i2
                                   : (map[1] == 3) ? i3
                                   : (map[1] == 4) ? i4
                                   : (map[1] == 5) ? i5
                                                   : i1;
                      int dst_i2 = (map[2] == 0)   ? i0
                                   : (map[2] == 1) ? i1
                                   : (map[2] == 3) ? i3
                                   : (map[2] == 4) ? i4
                                   : (map[2] == 5) ? i5
                                                   : i2;
                      int dst_i3 = (map[3] == 0)   ? i0
                                   : (map[3] == 1) ? i1
                                   : (map[3] == 2) ? i2
                                   : (map[3] == 4) ? i4
                                   : (map[3] == 5) ? i5
                                                   : i3;
                      int dst_i4 = (map[4] == 0)   ? i0
                                   : (map[4] == 1) ? i1
                                   : (map[4] == 2) ? i2
                                   : (map[4] == 3) ? i3
                                   : (map[4] == 5) ? i5
                                                   : i4;
                      int dst_i5 = (map[5] == 0)   ? i0
                                   : (map[5] == 1) ? i1
                                   : (map[5] == 2) ? i2
                                   : (map[5] == 3) ? i3
                                   : (map[5] == 4) ? i4
                                                   : i5;
                      h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5) =
                          h_x(i0, i1, i2, i3, i4, i5);
                    }
                  }
                }
              }
            }
          }

          Kokkos::deep_copy(ref, h_ref);
          Kokkos::fence();

          KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                     map);  // xt is the transpose of x
          EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

          // Inverse (transpose of transpose is identical to the original)
          RealView6Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5);
          KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
          EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
        }
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_3d_7dview() {
  using RealView7Dtype =
      Kokkos::View<double*******, LayoutType, execution_space>;
  constexpr int DIM = 7;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8;
  RealView7Dtype x("x", n0, n1, n2, n3, n4, n5, n6);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;

        KokkosFFT::axis_type<3> axes = {axis0, axis1, axis2};

        auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
        axes_type<DIM> out_extents;
        for (int i = 0; i < DIM; i++) {
          out_extents.at(i) = x.extent(map.at(i));
        }
        auto [nt0, nt1, nt2, nt3, nt4, nt5, nt6] = out_extents;

        RealView7Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5, nt6);
        if (map == default_axes) {
          EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                  map),  // xt is identical to x
                       std::runtime_error);
        } else {
          // Transposed Views
          RealView7Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5, nt6);
          auto h_ref = Kokkos::create_mirror_view(ref);
          // Filling the transposed View
          for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
            for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1);
                 i1++) {
              for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                   i2++) {
                for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                     i3++) {
                  for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                       i4++) {
                    for (int i5 = 0;
                         static_cast<std::size_t>(i5) < h_x.extent(5); i5++) {
                      for (int i6 = 0;
                           static_cast<std::size_t>(i6) < h_x.extent(6); i6++) {
                        int dst_i0    = (map[0] == 1)   ? i1
                                        : (map[0] == 2) ? i2
                                        : (map[0] == 3) ? i3
                                        : (map[0] == 4) ? i4
                                        : (map[0] == 5) ? i5
                                        : (map[0] == 6) ? i6
                                                        : i0;
                        int dst_i1    = (map[1] == 0)   ? i0
                                        : (map[1] == 2) ? i2
                                        : (map[1] == 3) ? i3
                                        : (map[1] == 4) ? i4
                                        : (map[1] == 5) ? i5
                                        : (map[1] == 6) ? i6
                                                        : i1;
                        int dst_i2    = (map[2] == 0)   ? i0
                                        : (map[2] == 1) ? i1
                                        : (map[2] == 3) ? i3
                                        : (map[2] == 4) ? i4
                                        : (map[2] == 5) ? i5
                                        : (map[2] == 6) ? i6
                                                        : i2;
                        int dst_i3    = (map[3] == 0)   ? i0
                                        : (map[3] == 1) ? i1
                                        : (map[3] == 2) ? i2
                                        : (map[3] == 4) ? i4
                                        : (map[3] == 5) ? i5
                                        : (map[3] == 6) ? i6
                                                        : i3;
                        int dst_i4    = (map[4] == 0)   ? i0
                                        : (map[4] == 1) ? i1
                                        : (map[4] == 2) ? i2
                                        : (map[4] == 3) ? i3
                                        : (map[4] == 5) ? i5
                                        : (map[4] == 6) ? i6
                                                        : i4;
                        int dst_i5    = (map[5] == 0)   ? i0
                                        : (map[5] == 1) ? i1
                                        : (map[5] == 2) ? i2
                                        : (map[5] == 3) ? i3
                                        : (map[5] == 4) ? i4
                                        : (map[5] == 6) ? i6
                                                        : i5;
                        int dst_i6    = (map[6] == 0)   ? i0
                                        : (map[6] == 1) ? i1
                                        : (map[6] == 2) ? i2
                                        : (map[6] == 3) ? i3
                                        : (map[6] == 4) ? i4
                                        : (map[6] == 5) ? i5
                                                        : i6;
                        h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5,
                              dst_i6) = h_x(i0, i1, i2, i3, i4, i5, i6);
                      }
                    }
                  }
                }
              }
            }
          }

          Kokkos::deep_copy(ref, h_ref);
          Kokkos::fence();

          KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                     map);  // xt is the transpose of x
          EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

          // Inverse (transpose of transpose is identical to the original)
          RealView7Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5, n6);
          KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
          EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
        }
      }
    }
  }
}

template <typename LayoutType>
void test_transpose_3d_8dview() {
  using RealView8Dtype =
      Kokkos::View<double********, LayoutType, execution_space>;
  constexpr int DIM = 8;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8, n7 = 9;
  RealView8Dtype x("x", n0, n1, n2, n3, n4, n5, n6, n7);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);

  // Transposed views
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6, 7});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;

        KokkosFFT::axis_type<3> axes = {axis0, axis1, axis2};

        auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
        axes_type<DIM> out_extents;
        for (int i = 0; i < DIM; i++) {
          out_extents.at(i) = x.extent(map.at(i));
        }
        auto [nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7] = out_extents;

        RealView8Dtype xt("xt", nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7);
        if (map == default_axes) {
          EXPECT_THROW(KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                                  map),  // xt is identical to x
                       std::runtime_error);
        } else {
          // Transposed Views
          RealView8Dtype ref("ref", nt0, nt1, nt2, nt3, nt4, nt5, nt6, nt7);
          auto h_ref = Kokkos::create_mirror_view(ref);
          // Filling the transposed View
          for (int i0 = 0; static_cast<std::size_t>(i0) < h_x.extent(0); i0++) {
            for (int i1 = 0; static_cast<std::size_t>(i1) < h_x.extent(1);
                 i1++) {
              for (int i2 = 0; static_cast<std::size_t>(i2) < h_x.extent(2);
                   i2++) {
                for (int i3 = 0; static_cast<std::size_t>(i3) < h_x.extent(3);
                     i3++) {
                  for (int i4 = 0; static_cast<std::size_t>(i4) < h_x.extent(4);
                       i4++) {
                    for (int i5 = 0;
                         static_cast<std::size_t>(i5) < h_x.extent(5); i5++) {
                      for (int i6 = 0;
                           static_cast<std::size_t>(i6) < h_x.extent(6); i6++) {
                        for (int i7 = 0;
                             static_cast<std::size_t>(i7) < h_x.extent(7);
                             i7++) {
                          int dst_i0 = (map[0] == 1)   ? i1
                                       : (map[0] == 2) ? i2
                                       : (map[0] == 3) ? i3
                                       : (map[0] == 4) ? i4
                                       : (map[0] == 5) ? i5
                                       : (map[0] == 6) ? i6
                                       : (map[0] == 7) ? i7
                                                       : i0;
                          int dst_i1 = (map[1] == 0)   ? i0
                                       : (map[1] == 2) ? i2
                                       : (map[1] == 3) ? i3
                                       : (map[1] == 4) ? i4
                                       : (map[1] == 5) ? i5
                                       : (map[1] == 6) ? i6
                                       : (map[1] == 7) ? i7
                                                       : i1;
                          int dst_i2 = (map[2] == 0)   ? i0
                                       : (map[2] == 1) ? i1
                                       : (map[2] == 3) ? i3
                                       : (map[2] == 4) ? i4
                                       : (map[2] == 5) ? i5
                                       : (map[2] == 6) ? i6
                                       : (map[2] == 7) ? i7
                                                       : i2;
                          int dst_i3 = (map[3] == 0)   ? i0
                                       : (map[3] == 1) ? i1
                                       : (map[3] == 2) ? i2
                                       : (map[3] == 4) ? i4
                                       : (map[3] == 5) ? i5
                                       : (map[3] == 6) ? i6
                                       : (map[3] == 7) ? i7
                                                       : i3;
                          int dst_i4 = (map[4] == 0)   ? i0
                                       : (map[4] == 1) ? i1
                                       : (map[4] == 2) ? i2
                                       : (map[4] == 3) ? i3
                                       : (map[4] == 5) ? i5
                                       : (map[4] == 6) ? i6
                                       : (map[4] == 7) ? i7
                                                       : i4;
                          int dst_i5 = (map[5] == 0)   ? i0
                                       : (map[5] == 1) ? i1
                                       : (map[5] == 2) ? i2
                                       : (map[5] == 3) ? i3
                                       : (map[5] == 4) ? i4
                                       : (map[5] == 6) ? i6
                                       : (map[5] == 7) ? i7
                                                       : i5;
                          int dst_i6 = (map[6] == 0)   ? i0
                                       : (map[6] == 1) ? i1
                                       : (map[6] == 2) ? i2
                                       : (map[6] == 3) ? i3
                                       : (map[6] == 4) ? i4
                                       : (map[6] == 5) ? i5
                                       : (map[6] == 7) ? i7
                                                       : i6;
                          int dst_i7 = (map[7] == 0)   ? i0
                                       : (map[7] == 1) ? i1
                                       : (map[7] == 2) ? i2
                                       : (map[7] == 3) ? i3
                                       : (map[7] == 4) ? i4
                                       : (map[7] == 5) ? i5
                                       : (map[7] == 6) ? i6
                                                       : i7;
                          h_ref(dst_i0, dst_i1, dst_i2, dst_i3, dst_i4, dst_i5,
                                dst_i6, dst_i7) =
                              h_x(i0, i1, i2, i3, i4, i5, i6, i7);
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

          KokkosFFT::Impl::transpose(execution_space(), x, xt,
                                     map);  // xt is the transpose of x
          EXPECT_TRUE(allclose(execution_space(), xt, ref, 1.e-5, 1.e-12));

          // Inverse (transpose of transpose is identical to the original)
          RealView8Dtype x_inv("x_inv", n0, n1, n2, n3, n4, n5, n6, n7);
          KokkosFFT::Impl::transpose(execution_space(), xt, x_inv, map_inv);
          EXPECT_TRUE(allclose(execution_space(), x_inv, x, 1.e-5, 1.e-12));
        }
      }
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(MapAxes, test_types);
TYPED_TEST_SUITE(Transpose1D, test_types);
TYPED_TEST_SUITE(Transpose2D, test_types);
TYPED_TEST_SUITE(Transpose3D, test_types);

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

TYPED_TEST(Transpose1D, 1DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_1dview<layout_type>();
}

TYPED_TEST(Transpose1D, 2DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_2dview<layout_type>();
}

TYPED_TEST(Transpose1D, 3DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_3dview<layout_type>();
}

TYPED_TEST(Transpose1D, 4DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_4dview<layout_type>();
}

TYPED_TEST(Transpose1D, 5DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_5dview<layout_type>();
}

TYPED_TEST(Transpose1D, 6DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_6dview<layout_type>();
}

TYPED_TEST(Transpose1D, 7DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_7dview<layout_type>();
}

TYPED_TEST(Transpose1D, 8DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_1d_8dview<layout_type>();
}

TYPED_TEST(Transpose2D, 2DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d_2dview<layout_type>();
}

TYPED_TEST(Transpose2D, 3DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d_3dview<layout_type>();
}

TYPED_TEST(Transpose2D, 4DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d_4dview<layout_type>();
}

TYPED_TEST(Transpose2D, 5DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d_5dview<layout_type>();
}

TYPED_TEST(Transpose2D, 6DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d_6dview<layout_type>();
}

TYPED_TEST(Transpose2D, 7DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d_7dview<layout_type>();
}

TYPED_TEST(Transpose2D, 8DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_2d_8dview<layout_type>();
}

TYPED_TEST(Transpose3D, 3DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_3d_3dview<layout_type>();
}

TYPED_TEST(Transpose3D, 4DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_3d_4dview<layout_type>();
}

TYPED_TEST(Transpose3D, 5DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_3d_5dview<layout_type>();
}

TYPED_TEST(Transpose3D, 6DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_3d_6dview<layout_type>();
}

TYPED_TEST(Transpose3D, 7DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_3d_7dview<layout_type>();
}

TYPED_TEST(Transpose3D, 8DView) {
  using layout_type = typename TestFixture::layout_type;

  test_transpose_3d_8dview<layout_type>();
}
