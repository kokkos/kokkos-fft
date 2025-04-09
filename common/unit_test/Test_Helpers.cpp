// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <vector>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Helpers.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight> >;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct FFTHelper : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

class FFTShiftParamTests : public ::testing::TestWithParam<int> {};

// Tests for FFT Freq
template <typename T, typename LayoutType>
void test_fft_freq(T atol = 1.0e-12) {
  constexpr std::size_t n_odd = 9, n_even = 10;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  RealView1DType x_odd_ref("x_odd_ref", n_odd),
      x_even_ref("x_even_ref", n_even);

  auto h_x_odd_ref  = Kokkos::create_mirror_view(x_odd_ref);
  auto h_x_even_ref = Kokkos::create_mirror_view(x_even_ref);

  std::vector<int> tmp_x_odd_ref  = {0, 1, 2, 3, 4, -4, -3, -2, -1};
  std::vector<int> tmp_x_even_ref = {0, 1, 2, 3, 4, -5, -4, -3, -2, -1};

  for (std::size_t i = 0; i < tmp_x_odd_ref.size(); i++) {
    h_x_odd_ref(i) = static_cast<T>(tmp_x_odd_ref.at(i));
  }

  for (std::size_t i = 0; i < tmp_x_even_ref.size(); i++) {
    h_x_even_ref(i) = static_cast<T>(tmp_x_even_ref.at(i));
  }

  Kokkos::deep_copy(x_odd_ref, h_x_odd_ref);
  Kokkos::deep_copy(x_even_ref, h_x_even_ref);
  T pi       = static_cast<T>(M_PI);
  auto x_odd = KokkosFFT::fftfreq<execution_space, T>(execution_space(), n_odd);
  auto x_odd_pi =
      KokkosFFT::fftfreq<execution_space, T>(execution_space(), n_odd, pi);
  multiply(execution_space(), x_odd, static_cast<T>(n_odd));
  multiply(execution_space(), x_odd_pi, static_cast<T>(n_odd) * pi);

  EXPECT_TRUE(allclose(execution_space(), x_odd, x_odd_ref, 1.e-5, atol));
  EXPECT_TRUE(allclose(execution_space(), x_odd_pi, x_odd_ref, 1.e-5, atol));

  auto x_even =
      KokkosFFT::fftfreq<execution_space, T>(execution_space(), n_even);
  auto x_even_pi =
      KokkosFFT::fftfreq<execution_space, T>(execution_space(), n_even, pi);
  multiply(execution_space(), x_even, static_cast<T>(n_even));
  multiply(execution_space(), x_even_pi, static_cast<T>(n_even) * pi);

  EXPECT_TRUE(allclose(execution_space(), x_even, x_even_ref, 1.e-5, atol));
  EXPECT_TRUE(allclose(execution_space(), x_even_pi, x_even_ref, 1.e-5, atol));
}

// Tests for RFFT Freq
template <typename T, typename LayoutType>
void test_rfft_freq(T atol = 1.0e-12) {
  constexpr std::size_t n_odd = 9, n_even = 10;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  RealView1DType x_odd_ref("x_odd_ref", n_odd / 2 + 1),
      x_even_ref("x_even_ref", n_even / 2 + 1);

  auto h_x_odd_ref  = Kokkos::create_mirror_view(x_odd_ref);
  auto h_x_even_ref = Kokkos::create_mirror_view(x_even_ref);

  std::vector<int> tmp_x_odd_ref  = {0, 1, 2, 3, 4};
  std::vector<int> tmp_x_even_ref = {0, 1, 2, 3, 4, 5};

  for (std::size_t i = 0; i < tmp_x_odd_ref.size(); i++) {
    h_x_odd_ref(i) = static_cast<T>(tmp_x_odd_ref.at(i));
  }

  for (std::size_t i = 0; i < tmp_x_even_ref.size(); i++) {
    h_x_even_ref(i) = static_cast<T>(tmp_x_even_ref.at(i));
  }

  Kokkos::deep_copy(x_odd_ref, h_x_odd_ref);
  Kokkos::deep_copy(x_even_ref, h_x_even_ref);
  T pi = static_cast<T>(M_PI);
  auto x_odd =
      KokkosFFT::rfftfreq<execution_space, T>(execution_space(), n_odd);
  auto x_odd_pi =
      KokkosFFT::rfftfreq<execution_space, T>(execution_space(), n_odd, pi);
  multiply(execution_space(), x_odd, static_cast<T>(n_odd));
  multiply(execution_space(), x_odd_pi, static_cast<T>(n_odd) * pi);

  EXPECT_TRUE(allclose(execution_space(), x_odd, x_odd_ref, 1.e-5, atol));
  EXPECT_TRUE(allclose(execution_space(), x_odd_pi, x_odd_ref, 1.e-5, atol));

  auto x_even =
      KokkosFFT::rfftfreq<execution_space, T>(execution_space(), n_even);
  auto x_even_pi =
      KokkosFFT::rfftfreq<execution_space, T>(execution_space(), n_even, pi);
  multiply(execution_space(), x_even, static_cast<T>(n_even));
  multiply(execution_space(), x_even_pi, static_cast<T>(n_even) * pi);

  EXPECT_TRUE(allclose(execution_space(), x_even, x_even_ref, 1.e-5, atol));
  EXPECT_TRUE(allclose(execution_space(), x_even_pi, x_even_ref, 1.e-5, atol));
}

// Tests for get shifts
void test_get_shift1D_1DView(int n0, int direction) {
  using RealView1DType = Kokkos::View<double*, execution_space>;
  RealView1DType x("x", n0);

  Kokkos::Array<std::size_t, 1> shifts1_ref = {};
  int shift0                                = direction * n0 / 2;
  if (shift0 < 0) shift0 += n0;
  shifts1_ref[0] = static_cast<std::size_t>(shift0);
  auto shifts1 =
      KokkosFFT::Impl::get_shifts(x, KokkosFFT::axis_type<1>({0}), direction);
  EXPECT_TRUE(shifts1 == shifts1_ref);
}

void test_get_shift1D_2DView(int n0, int direction) {
  using RealView2DType = Kokkos::View<double**, execution_space>;
  const int n1         = 5;
  RealView2DType x("x", n0, n1);

  Kokkos::Array<std::size_t, 2> shifts1_axis0_ref = {}, shifts1_axis1_ref = {};
  int shift0 = direction * n0 / 2;
  if (shift0 < 0) shift0 += n0;
  shifts1_axis0_ref[0] = static_cast<std::size_t>(shift0);

  int shift1 = direction * n1 / 2;
  if (shift1 < 0) shift1 += n1;
  shifts1_axis1_ref[1] = static_cast<std::size_t>(shift1);

  auto shifts1_axis0 =
      KokkosFFT::Impl::get_shifts(x, KokkosFFT::axis_type<1>({0}), direction);
  auto shifts1_axis1 =
      KokkosFFT::Impl::get_shifts(x, KokkosFFT::axis_type<1>({1}), direction);
  EXPECT_TRUE(shifts1_axis0 == shifts1_axis0_ref);
  EXPECT_TRUE(shifts1_axis1 == shifts1_axis1_ref);
}

void test_get_shift2D_2DView(int n0, int direction) {
  using RealView2DType = Kokkos::View<double**, execution_space>;
  const int n1         = 5;
  RealView2DType x("x", n0, n1);

  Kokkos::Array<std::size_t, 2> shifts2_ref = {};
  int shift0                                = direction * n0 / 2;
  if (shift0 < 0) shift0 += n0;
  shifts2_ref[0] = static_cast<std::size_t>(shift0);

  int shift1 = direction * n1 / 2;
  if (shift1 < 0) shift1 += n1;
  shifts2_ref[1] = static_cast<std::size_t>(shift1);

  auto shifts2 = KokkosFFT::Impl::get_shifts(x, KokkosFFT::axis_type<2>({0, 1}),
                                             direction);
  EXPECT_TRUE(shifts2 == shifts2_ref);
}

// Identity Tests for fftshift1D on 1D View
void test_fftshift1D_1DView_identity(int n0) {
  using RealView1DType = Kokkos::View<double*, execution_space>;

  RealView1DType x("x", n0), x_ref("x_ref", n0);

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(x, random_pool, 1.0);
  Kokkos::deep_copy(x_ref, x);

  Kokkos::fence();

  KokkosFFT::fftshift(execution_space(), x);
  KokkosFFT::ifftshift(execution_space(), x);

  EXPECT_TRUE(allclose(execution_space(), x, x_ref, 1.e-5, 1.e-12));
}

// Tests for fftshift1D on 1D View
void test_fftshift1D_1DView(int n0) {
  using RealView1DType = Kokkos::View<double*, execution_space>;
  RealView1DType x("x", n0), y("y", n0);
  RealView1DType x_ref("x_ref", n0), y_ref("y_ref", n0);

  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  auto h_y_ref = Kokkos::create_mirror_view(y_ref);

  std::vector<int> tmp_x_ref;
  std::vector<int> tmp_y_ref;

  if (n0 % 2 == 0) {
    tmp_x_ref = {0, 1, 2, 3, 4, -5, -4, -3, -2, -1};
    tmp_y_ref = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
  } else {
    tmp_x_ref = {0, 1, 2, 3, 4, -4, -3, -2, -1};
    tmp_y_ref = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
  }

  for (int i = 0; i < n0; i++) {
    h_x_ref(i) = static_cast<double>(tmp_x_ref.at(i));
    h_y_ref(i) = static_cast<double>(tmp_y_ref.at(i));
  }

  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(y_ref, h_y_ref);
  Kokkos::deep_copy(x, h_x_ref);
  Kokkos::deep_copy(y, h_y_ref);

  KokkosFFT::fftshift(execution_space(), x);
  KokkosFFT::ifftshift(execution_space(), y);

  EXPECT_TRUE(allclose(execution_space(), x, y_ref));
  EXPECT_TRUE(allclose(execution_space(), y, x_ref));
}

// Tests for fftshift1D on 2D View
void test_fftshift1D_2DView(int n0) {
  using RealView2DType =
      Kokkos::View<double**, Kokkos::LayoutLeft, execution_space>;
  constexpr int n1 = 3;
  RealView2DType x("x", n0, n1), y_axis0("y_axis0", n0, n1),
      y_axis1("y_axis1", n0, n1);
  RealView2DType x_ref("x_ref", n0, n1);
  RealView2DType y_axis0_ref("y_axis0_ref", n0, n1),
      y_axis1_ref("y_axis1_ref", n0, n1);

  auto h_x_ref       = Kokkos::create_mirror_view(x_ref);
  auto h_y_axis0_ref = Kokkos::create_mirror_view(y_axis0_ref);
  auto h_y_axis1_ref = Kokkos::create_mirror_view(y_axis1_ref);

  std::vector<int> tmp_x_ref;
  std::vector<int> tmp_y0_ref, tmp_y1_ref;

  if (n0 % 2 == 0) {
    tmp_x_ref  = {0,   1,  2,  3,  4,  5,   6,   7,   8,   9,
                  10,  11, 12, 13, 14, -15, -14, -13, -12, -11,
                  -10, -9, -8, -7, -6, -5,  -4,  -3,  -2,  -1};
    tmp_y0_ref = {
        5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  -15, -14, -13, -12, -11,
        10, 11, 12, 13, 14, -5, -4, -3, -2, -1, -10, -9,  -8,  -7,  -6,
    };
    tmp_y1_ref = {-10, -9, -8, -7, -6, -5,  -4,  -3,  -2,  -1,
                  0,   1,  2,  3,  4,  5,   6,   7,   8,   9,
                  10,  11, 12, 13, 14, -15, -14, -13, -12, -11};
  } else {
    tmp_x_ref  = {0,   1,   2,   3,   4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                  -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1};
    tmp_y0_ref = {5,  6,  7,  8,  0,  1,  2,  3,  4,  -13, -12, -11, -10, 9,
                  10, 11, 12, 13, -4, -3, -2, -1, -9, -8,  -7,  -6,  -5};
    tmp_y1_ref = {-9, -8, -7, -6, -5, -4, -3, -2, -1, 0,   1,   2,   3,  4,
                  5,  6,  7,  8,  9,  10, 11, 12, 13, -13, -12, -11, -10};
  }

  for (int i1 = 0; i1 < n1; i1++) {
    for (int i0 = 0; i0 < n0; i0++) {
      std::size_t i         = i0 + i1 * n0;
      h_x_ref(i0, i1)       = static_cast<double>(tmp_x_ref.at(i));
      h_y_axis0_ref(i0, i1) = static_cast<double>(tmp_y0_ref.at(i));
      h_y_axis1_ref(i0, i1) = static_cast<double>(tmp_y1_ref.at(i));
    }
  }

  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(y_axis0_ref, h_y_axis0_ref);
  Kokkos::deep_copy(y_axis1_ref, h_y_axis1_ref);
  Kokkos::deep_copy(x, h_x_ref);
  Kokkos::deep_copy(y_axis0, h_y_axis0_ref);
  Kokkos::deep_copy(y_axis1, h_y_axis1_ref);

  KokkosFFT::fftshift(execution_space(), x, axes_type<1>({0}));
  KokkosFFT::ifftshift(execution_space(), y_axis0, axes_type<1>({0}));

  EXPECT_TRUE(allclose(execution_space(), x, y_axis0_ref));
  EXPECT_TRUE(allclose(execution_space(), y_axis0, x_ref));

  Kokkos::deep_copy(x, h_x_ref);

  KokkosFFT::fftshift(execution_space(), x, axes_type<1>({1}));
  KokkosFFT::ifftshift(execution_space(), y_axis1, axes_type<1>({1}));

  EXPECT_TRUE(allclose(execution_space(), x, y_axis1_ref));
  EXPECT_TRUE(allclose(execution_space(), y_axis1, x_ref));
}

// Tests for fftshift2D on 2D View
void test_fftshift2D_2DView(int n0) {
  using RealView2DType =
      Kokkos::View<double**, Kokkos::LayoutLeft, execution_space>;
  constexpr int n1 = 3;
  RealView2DType x("x", n0, n1), y("y", n0, n1);
  RealView2DType x_ref("x_ref", n0, n1), y_ref("y_ref", n0, n1);

  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  auto h_y_ref = Kokkos::create_mirror_view(y_ref);

  std::vector<int> tmp_x_ref;
  std::vector<int> tmp_y_ref;

  if (n0 % 2 == 0) {
    tmp_x_ref = {0,   1,  2,  3,  4,  5,   6,   7,   8,   9,
                 10,  11, 12, 13, 14, -15, -14, -13, -12, -11,
                 -10, -9, -8, -7, -6, -5,  -4,  -3,  -2,  -1};
    tmp_y_ref = {-5,  -4,  -3,  -2,  -1,  -10, -9, -8, -7, -6,
                 5,   6,   7,   8,   9,   0,   1,  2,  3,  4,
                 -15, -14, -13, -12, -11, 10,  11, 12, 13, 14};
  } else {
    tmp_x_ref = {0,   1,   2,   3,   4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                 -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1};
    tmp_y_ref = {-4, -3, -2, -1, -9,  -8,  -7,  -6,  -5, 5,  6,  7,  8, 0,
                 1,  2,  3,  4,  -13, -12, -11, -10, 9,  10, 11, 12, 13};
  }

  for (int i1 = 0; i1 < n1; i1++) {
    for (int i0 = 0; i0 < n0; i0++) {
      std::size_t i   = i0 + i1 * n0;
      h_x_ref(i0, i1) = static_cast<double>(tmp_x_ref.at(i));
      h_y_ref(i0, i1) = static_cast<double>(tmp_y_ref.at(i));
    }
  }

  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(y_ref, h_y_ref);
  Kokkos::deep_copy(x, h_x_ref);
  Kokkos::deep_copy(y, h_y_ref);

  KokkosFFT::fftshift(execution_space(), x, axes_type<2>({0, 1}));
  KokkosFFT::ifftshift(execution_space(), y, axes_type<2>({0, 1}));

  EXPECT_TRUE(allclose(execution_space(), x, y_ref));
  EXPECT_TRUE(allclose(execution_space(), y, x_ref));
}
}  // namespace

TYPED_TEST_SUITE(FFTHelper, test_types);

// Tests for fftfreq
TYPED_TEST(FFTHelper, fftfreq) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft_freq<float_type, layout_type>(atol);
}

// Tests for rfftfreq
TYPED_TEST(FFTHelper, rfftfreq) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_rfft_freq<float_type, layout_type>(atol);
}

// Tests for get shift
TEST_P(FFTShiftParamTests, GetForwardShift1D1DView) {
  int n0 = GetParam();
  test_get_shift1D_1DView(n0, /* direction= */ 1);
}

TEST_P(FFTShiftParamTests, GetBackwardShift1D1DView) {
  int n0 = GetParam();
  test_get_shift1D_1DView(n0, /* direction= */ -1);
}

TEST_P(FFTShiftParamTests, GetForwardShift1D2DView) {
  int n0 = GetParam();
  test_get_shift1D_2DView(n0, /* direction= */ 1);
}

TEST_P(FFTShiftParamTests, GetBackwardShift1D2DView) {
  int n0 = GetParam();
  test_get_shift1D_2DView(n0, /* direction= */ -1);
}

TEST_P(FFTShiftParamTests, GetForwardShift2D2DView) {
  int n0 = GetParam();
  test_get_shift2D_2DView(n0, /* direction= */ 1);
}

TEST_P(FFTShiftParamTests, GetBackwardShift2D2DView) {
  int n0 = GetParam();
  test_get_shift2D_2DView(n0, /* direction= */ -1);
}

// Identity Tests for fftshift1D on 1D View
TEST_P(FFTShiftParamTests, Identity) {
  int n0 = GetParam();
  test_fftshift1D_1DView_identity(n0);
}

// Tests for fftshift1D on 1D View
TEST_P(FFTShiftParamTests, 1DShift1DView) {
  int n0 = GetParam();
  test_fftshift1D_1DView(n0);
}

// Tests for fftshift1D on 2D View
TEST_P(FFTShiftParamTests, 1DShift2DView) {
  int n0 = GetParam();
  test_fftshift1D_2DView(n0);
}

// Tests for fftshift2D on 2D View
TEST_P(FFTShiftParamTests, 2DShift2DView) {
  int n0 = GetParam();
  test_fftshift2D_2DView(n0);
}

INSTANTIATE_TEST_SUITE_P(FFTShift, FFTShiftParamTests,
                         ::testing::Values(9, 10));
