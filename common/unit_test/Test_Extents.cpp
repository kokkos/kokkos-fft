// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include <vector>
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_Traits.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_int_types  = ::testing::Types<int, std::size_t>;
using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestExtents : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestGetExtents1D : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct TestGetExtents2D : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct TestGetDynExtents1D : public ::testing::Test {
  using layout_type = T;
};

template <typename T>
struct TestGetDynExtents2D : public ::testing::Test {
  using layout_type = T;
};

// Tests for extent_after_transform
template <typename T>
void test_extent_after_transform() {
  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;
  const int n0 = 6, n1 = 7;

  if constexpr (is_R2C) {
    auto n0h         = KokkosFFT::Impl::extent_after_transform(n0, is_R2C);
    auto n1h         = KokkosFFT::Impl::extent_after_transform(n1, is_R2C);
    const int ref_n0 = n0 / 2 + 1;
    const int ref_n1 = n1 / 2 + 1;
    EXPECT_EQ(n0h, ref_n0);
    EXPECT_EQ(n1h, ref_n1);
  } else {
    auto n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C);
    auto n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C);
    EXPECT_EQ(n0h, n0);
    EXPECT_EQ(n1h, n1);
  }
}

// Tests for padded_extents
template <typename T, typename iType>
void test_padded_extents() {
  using extents1D_type = std::array<iType, 1>;
  using extents2D_type = std::array<iType, 2>;
  using extents3D_type = std::array<iType, 3>;
  using axes1D_type    = std::array<iType, 1>;
  using axes2D_type    = std::array<iType, 2>;
  using axes3D_type    = std::array<iType, 3>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;
  const iType n0 = 5, n1 = 6, n2 = 7;
  const iType n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
              n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C),
              n2h = KokkosFFT::Impl::extent_after_transform(n2, is_R2C);

  extents1D_type out_extents1{n0h};
  extents2D_type out_extents2_ax0{n0h, n1}, out_extents2_ax1{n0, n1h};
  extents3D_type out_extents3_ax0{n0h, n1, n2}, out_extents3_ax1{n0, n1h, n2},
      out_extents3_ax2{n0, n1, n2h};

  axes1D_type ax0{0};
  axes2D_type ax01{0, 1}, ax10{1, 0};
  axes3D_type ax012{0, 1, 2}, ax021{0, 2, 1}, ax102{1, 0, 2}, ax120{1, 2, 0},
      ax201{2, 0, 1}, ax210{2, 1, 0};

  if constexpr (is_R2C) {
    axes1D_type ax1{1}, ax2{2};
    axes2D_type ax02{0, 2}, ax12{1, 2}, ax20{2, 0}, ax21{2, 1};
    auto ref_padded_extents1     = out_extents1;
    auto ref_padded_extents2_ax0 = out_extents2_ax0;
    auto ref_padded_extents2_ax1 = out_extents2_ax1;
    auto ref_padded_extents3_ax0 = out_extents3_ax0;
    auto ref_padded_extents3_ax1 = out_extents3_ax1;
    auto ref_padded_extents3_ax2 = out_extents3_ax2;

    ref_padded_extents1.at(0) *= 2;
    ref_padded_extents2_ax0.at(0) *= 2;
    ref_padded_extents2_ax1.at(1) *= 2;
    ref_padded_extents3_ax0.at(0) *= 2;
    ref_padded_extents3_ax1.at(1) *= 2;
    ref_padded_extents3_ax2.at(2) *= 2;

    auto padded_extents1 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents1, ax0);
    EXPECT_EQ(padded_extents1, ref_padded_extents1);

    auto padded_extents2_ax0 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents2_ax0, ax0);
    auto padded_extents2_ax1 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents2_ax1, ax1);
    auto padded_extents2_ax01 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents2_ax1, ax01);
    auto padded_extents2_ax10 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents2_ax0, ax10);
    EXPECT_EQ(padded_extents2_ax0, ref_padded_extents2_ax0);
    EXPECT_EQ(padded_extents2_ax1, ref_padded_extents2_ax1);
    EXPECT_EQ(padded_extents2_ax01, ref_padded_extents2_ax1);
    EXPECT_EQ(padded_extents2_ax10, ref_padded_extents2_ax0);

    auto padded_extents3_ax0 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax0, ax0);
    auto padded_extents3_ax1 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax1, ax1);
    auto padded_extents3_ax2 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax2, ax2);
    auto padded_extents3_ax01 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax1, ax01);
    auto padded_extents3_ax02 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax2, ax02);
    auto padded_extents3_ax10 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax0, ax10);
    auto padded_extents3_ax12 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax2, ax12);
    auto padded_extents3_ax20 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax0, ax20);
    auto padded_extents3_ax21 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax1, ax21);
    auto padded_extents3_ax012 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax2, ax012);
    auto padded_extents3_ax021 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax1, ax021);
    auto padded_extents3_ax102 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax2, ax102);
    auto padded_extents3_ax120 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax0, ax120);
    auto padded_extents3_ax201 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax1, ax201);
    auto padded_extents3_ax210 =
        KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax0, ax210);
    EXPECT_EQ(padded_extents3_ax0, ref_padded_extents3_ax0);
    EXPECT_EQ(padded_extents3_ax1, ref_padded_extents3_ax1);
    EXPECT_EQ(padded_extents3_ax2, ref_padded_extents3_ax2);
    EXPECT_EQ(padded_extents3_ax01, ref_padded_extents3_ax1);
    EXPECT_EQ(padded_extents3_ax02, ref_padded_extents3_ax2);
    EXPECT_EQ(padded_extents3_ax10, ref_padded_extents3_ax0);
    EXPECT_EQ(padded_extents3_ax12, ref_padded_extents3_ax2);
    EXPECT_EQ(padded_extents3_ax20, ref_padded_extents3_ax0);
    EXPECT_EQ(padded_extents3_ax21, ref_padded_extents3_ax1);
    EXPECT_EQ(padded_extents3_ax012, ref_padded_extents3_ax2);
    EXPECT_EQ(padded_extents3_ax021, ref_padded_extents3_ax1);
    EXPECT_EQ(padded_extents3_ax102, ref_padded_extents3_ax2);
    EXPECT_EQ(padded_extents3_ax120, ref_padded_extents3_ax0);
    EXPECT_EQ(padded_extents3_ax201, ref_padded_extents3_ax1);
    EXPECT_EQ(padded_extents3_ax210, ref_padded_extents3_ax0);
  } else {
    // For complex data, just return the input
    std::vector<axes1D_type> all_axes1D{ax0};
    std::vector<axes2D_type> all_axes2D{ax01, ax10};
    std::vector<axes3D_type> all_axes3D{ax012, ax021, ax102,
                                        ax120, ax201, ax210};

    for (auto axes1D : all_axes1D) {
      auto padded_extents1 =
          KokkosFFT::Impl::compute_padded_extents<T>(out_extents1, axes1D);
      EXPECT_EQ(padded_extents1, out_extents1);
    }

    for (auto axes2D : all_axes2D) {
      auto padded_extents2_ax0 =
          KokkosFFT::Impl::compute_padded_extents<T>(out_extents2_ax0, axes2D);
      auto padded_extents2_ax1 =
          KokkosFFT::Impl::compute_padded_extents<T>(out_extents2_ax1, axes2D);
      EXPECT_EQ(padded_extents2_ax0, out_extents2_ax0);
      EXPECT_EQ(padded_extents2_ax1, out_extents2_ax1);
    }

    for (auto axes3D : all_axes3D) {
      auto padded_extents3_ax0 =
          KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax0, axes3D);
      auto padded_extents3_ax1 =
          KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax1, axes3D);
      auto padded_extents3_ax2 =
          KokkosFFT::Impl::compute_padded_extents<T>(out_extents3_ax2, axes3D);
      EXPECT_EQ(padded_extents3_ax0, out_extents3_ax0);
      EXPECT_EQ(padded_extents3_ax1, out_extents3_ax1);
      EXPECT_EQ(padded_extents3_ax2, out_extents3_ax2);
    }
  }
}

// Tests for mapped extents
template <typename ContainerType, typename iType>
void test_compute_mapped_extents(iType n) {
  using extents_type = std::array<iType, 3>;
  extents_type extents{n, 3, 8};
  ContainerType map012{0, 1, 2}, map021{0, 2, 1}, map102{1, 0, 2},
      map120{1, 2, 0}, map201{2, 0, 1}, map210{2, 1, 0};
  auto mapped_extents012 =
      KokkosFFT::Impl::compute_mapped_extents(extents, map012);
  auto mapped_extents021 =
      KokkosFFT::Impl::compute_mapped_extents(extents, map021);
  auto mapped_extents102 =
      KokkosFFT::Impl::compute_mapped_extents(extents, map102);
  auto mapped_extents120 =
      KokkosFFT::Impl::compute_mapped_extents(extents, map120);
  auto mapped_extents201 =
      KokkosFFT::Impl::compute_mapped_extents(extents, map201);
  auto mapped_extents210 =
      KokkosFFT::Impl::compute_mapped_extents(extents, map210);

  extents_type ref_mapped_extents012{n, 3, 8}, ref_mapped_extents021{n, 8, 3},
      ref_mapped_extents102{3, n, 8}, ref_mapped_extents120{3, 8, n},
      ref_mapped_extents201{8, n, 3}, ref_mapped_extents210{8, 3, n};

  EXPECT_EQ(mapped_extents012, ref_mapped_extents012);
  EXPECT_EQ(mapped_extents021, ref_mapped_extents021);
  EXPECT_EQ(mapped_extents102, ref_mapped_extents102);
  EXPECT_EQ(mapped_extents120, ref_mapped_extents120);
  EXPECT_EQ(mapped_extents201, ref_mapped_extents201);
  EXPECT_EQ(mapped_extents210, ref_mapped_extents210);

  // Check if errors are correctly raised against invalid map
  ContainerType overlapped_map{0, 1, 1};
  EXPECT_THROW(
      {
        [[maybe_unused]] auto extents011 =
            KokkosFFT::Impl::compute_mapped_extents(extents, overlapped_map);
      },
      std::runtime_error);

  // Check if errors are correctly raised against out-of-range map
  ContainerType invalid_map{0, 1, 3};
  EXPECT_THROW(
      {
        [[maybe_unused]] auto extents013 =
            KokkosFFT::Impl::compute_mapped_extents(extents, invalid_map);
      },
      std::runtime_error);

  if constexpr (std::is_signed_v<iType>) {
    ContainerType negative_map{-1, 0, 1};
    EXPECT_THROW(
        {
          [[maybe_unused]] auto extents_neg =
              KokkosFFT::Impl::compute_mapped_extents(extents, negative_map);
        },
        std::runtime_error);
  }
}

// Tests for 1D FFT
template <typename T, typename LayoutType>
void test_extents_1d_view_1d(bool is_static = true) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using axes_type    = KokkosFFT::axis_type<1>;
  using ViewType     = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexViewtype =
      Kokkos::View<complex_type*, LayoutType, execution_space>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  const std::size_t n0  = 6;
  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C);

  ViewType xr("xr", n0);
  ComplexViewtype xc("xc", n0h);
  ComplexViewtype xcin("xcin", n0), xcout("xcout", n0);

  std::vector<std::size_t> ref_in_extents = {n0}, ref_out_extents = {n0h},
                           ref_fft_extents = {n0};
  std::size_t ref_howmany                  = 1;

  std::vector<std::size_t> in_extents(1), out_extents(1), fft_extents(1);
  std::size_t howmany;

  // R2C or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc, axes_type({0}));

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, axes_type({0})); },
                   std::runtime_error);
    }
  } else {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc, 1);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, 1); },
                   std::runtime_error);
    }
  }

  EXPECT_TRUE(in_extents == ref_in_extents);
  EXPECT_TRUE(out_extents == ref_out_extents);
  EXPECT_TRUE(fft_extents == ref_fft_extents);
  EXPECT_EQ(howmany, ref_howmany);

  // C2R or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc, xr, axes_type({0}));

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, axes_type({0})); },
                   std::runtime_error);
    }
  } else {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc, xr, 1);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, 1); },
                   std::runtime_error);
    }
  }

  EXPECT_TRUE(in_extents == ref_out_extents);
  EXPECT_TRUE(out_extents == ref_in_extents);
  EXPECT_TRUE(fft_extents == ref_fft_extents);
  EXPECT_EQ(howmany, ref_howmany);
}

template <typename T, typename LayoutType>
void test_extents_1d_view_2d(bool is_static = true) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using axes_type    = KokkosFFT::axis_type<1>;
  using ViewType     = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexViewtype =
      Kokkos::View<complex_type**, LayoutType, execution_space>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  const std::size_t n0 = 6, n1 = 10;
  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
                    n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C);

  ViewType xr("xr", n0, n1);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1), xc_axis1("xc_axis1", n0, n1h);
  ComplexViewtype xcin("xcin", n0, n1), xcout("xcout", n0, n1);

  [[maybe_unused]] std::vector<std::size_t> ref_in_extents_axis0  = {n0},
                                            ref_out_extents_axis0 = {n0h},
                                            ref_fft_extents_axis0 = {n0},
                                            ref_in_extents_axis1  = {n1},
                                            ref_out_extents_axis1 = {n1h},
                                            ref_fft_extents_axis1 = {n1};
  [[maybe_unused]] std::size_t ref_howmany_axis0 = n1, ref_howmany_axis1 = n0;

  std::vector<std::size_t> in_extents(1), out_extents(1), fft_extents(1);
  std::size_t howmany;

  // R2C or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis0, axes_type({0}));

    EXPECT_TRUE(in_extents == ref_in_extents_axis0);
    EXPECT_TRUE(out_extents == ref_out_extents_axis0);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
    EXPECT_EQ(howmany, ref_howmany_axis0);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis1, axes_type({1}));

    EXPECT_TRUE(in_extents == ref_in_extents_axis1);
    EXPECT_TRUE(out_extents == ref_out_extents_axis1);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis1);
    EXPECT_EQ(howmany, ref_howmany_axis1);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, axes_type({0})); },
                   std::runtime_error);
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, axes_type({1})); },
                   std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis0, 1);

      EXPECT_TRUE(in_extents == ref_in_extents_axis0);
      EXPECT_TRUE(out_extents == ref_out_extents_axis0);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
      EXPECT_EQ(howmany, ref_howmany_axis0);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis1, 1);
      EXPECT_TRUE(in_extents == ref_in_extents_axis1);
      EXPECT_TRUE(out_extents == ref_out_extents_axis1);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis1);
      EXPECT_EQ(howmany, ref_howmany_axis1);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, 1); },
                   std::runtime_error);
    }
  }

  // C2R or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis0, xr, axes_type({0}));

    EXPECT_TRUE(in_extents == ref_out_extents_axis0);
    EXPECT_TRUE(out_extents == ref_in_extents_axis0);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
    EXPECT_EQ(howmany, ref_howmany_axis0);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis1, xr, axes_type({1}));

    EXPECT_TRUE(in_extents == ref_out_extents_axis1);
    EXPECT_TRUE(out_extents == ref_in_extents_axis1);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis1);
    EXPECT_EQ(howmany, ref_howmany_axis1);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, axes_type({0})); },
                   std::runtime_error);
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, axes_type({1})); },
                   std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis0, xr, 1);

      EXPECT_TRUE(in_extents == ref_out_extents_axis0);
      EXPECT_TRUE(out_extents == ref_in_extents_axis0);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
      EXPECT_EQ(howmany, ref_howmany_axis0);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis1, xr, 1);
      EXPECT_TRUE(in_extents == ref_out_extents_axis1);
      EXPECT_TRUE(out_extents == ref_in_extents_axis1);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis1);
      EXPECT_EQ(howmany, ref_howmany_axis1);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, 1); },
                   std::runtime_error);
    }
  }
}

template <typename T, typename LayoutType>
void test_extents_1d_view_3d(bool is_static = true) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using axes_type    = KokkosFFT::axis_type<1>;
  using ViewType     = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexViewtype =
      Kokkos::View<complex_type***, LayoutType, execution_space>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  const std::size_t n0 = 6, n1 = 10, n2 = 8;
  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
                    n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C),
                    n2h = KokkosFFT::Impl::extent_after_transform(n2, is_R2C);

  ViewType xr("xr", n0, n1, n2);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1, n2),
      xc_axis1("xc_axis1", n0, n1h, n2), xc_axis2("xc_axis2", n0, n1, n2h);
  ComplexViewtype xcin("xcin", n0, n1, n2), xcout("xcout", n0, n1, n2);

  [[maybe_unused]] std::vector<std::size_t> ref_in_extents_axis0  = {n0},
                                            ref_out_extents_axis0 = {n0h},
                                            ref_fft_extents_axis0 = {n0},
                                            ref_in_extents_axis1  = {n1},
                                            ref_out_extents_axis1 = {n1h},
                                            ref_fft_extents_axis1 = {n1},
                                            ref_in_extents_axis2  = {n2},
                                            ref_out_extents_axis2 = {n2h},
                                            ref_fft_extents_axis2 = {n2};
  [[maybe_unused]] std::size_t ref_howmany_axis0                  = n1 * n2,
                               ref_howmany_axis1                  = n0 * n2,
                               ref_howmany_axis2                  = n0 * n1;

  std::vector<std::size_t> in_extents(1), out_extents(1), fft_extents(1);
  std::size_t howmany;

  // R2C or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis0, axes_type({0}));

    EXPECT_TRUE(in_extents == ref_in_extents_axis0);
    EXPECT_TRUE(out_extents == ref_out_extents_axis0);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
    EXPECT_EQ(howmany, ref_howmany_axis0);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis1, axes_type({1}));

    EXPECT_TRUE(in_extents == ref_in_extents_axis1);
    EXPECT_TRUE(out_extents == ref_out_extents_axis1);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis1);
    EXPECT_EQ(howmany, ref_howmany_axis1);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis2, axes_type({2}));

    EXPECT_TRUE(in_extents == ref_in_extents_axis2);
    EXPECT_TRUE(out_extents == ref_out_extents_axis2);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis2);
    EXPECT_EQ(howmany, ref_howmany_axis2);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, axes_type({0})); },
                   std::runtime_error);
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, axes_type({1})); },
                   std::runtime_error);
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, axes_type({2})); },
                   std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis0, 1);

      EXPECT_TRUE(in_extents == ref_in_extents_axis0);
      EXPECT_TRUE(out_extents == ref_out_extents_axis0);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
      EXPECT_EQ(howmany, ref_howmany_axis0);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis2, 1);
      EXPECT_TRUE(in_extents == ref_in_extents_axis2);
      EXPECT_TRUE(out_extents == ref_out_extents_axis2);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis2);
      EXPECT_EQ(howmany, ref_howmany_axis2);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, 1); },
                   std::runtime_error);
    }
  }

  // C2R or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis0, xr, axes_type({0}));

    EXPECT_TRUE(in_extents == ref_out_extents_axis0);
    EXPECT_TRUE(out_extents == ref_in_extents_axis0);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
    EXPECT_EQ(howmany, ref_howmany_axis0);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis1, xr, axes_type({1}));

    EXPECT_TRUE(in_extents == ref_out_extents_axis1);
    EXPECT_TRUE(out_extents == ref_in_extents_axis1);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis1);
    EXPECT_EQ(howmany, ref_howmany_axis1);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis2, xr, axes_type({2}));

    EXPECT_TRUE(in_extents == ref_out_extents_axis2);
    EXPECT_TRUE(out_extents == ref_in_extents_axis2);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axis2);
    EXPECT_EQ(howmany, ref_howmany_axis2);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, axes_type({0})); },
                   std::runtime_error);
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, axes_type({1})); },
                   std::runtime_error);
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, axes_type({2})); },
                   std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis0, xr, 1);

      EXPECT_TRUE(in_extents == ref_out_extents_axis0);
      EXPECT_TRUE(out_extents == ref_in_extents_axis0);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis0);
      EXPECT_EQ(howmany, ref_howmany_axis0);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis2, xr, 1);
      EXPECT_TRUE(in_extents == ref_out_extents_axis2);
      EXPECT_TRUE(out_extents == ref_in_extents_axis2);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axis2);
      EXPECT_EQ(howmany, ref_howmany_axis2);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, 1); },
                   std::runtime_error);
    }
  }
}

template <typename T, typename LayoutType>
void test_extents_2d_view_2d(bool is_static = true) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using axes_type    = KokkosFFT::axis_type<2>;
  using ViewType     = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexViewtype =
      Kokkos::View<complex_type**, LayoutType, execution_space>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  const std::size_t n0 = 6, n1 = 10;
  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
                    n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C);

  ViewType xr("xr", n0, n1);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1), xc_axis1("xc_axis1", n0, n1h);
  ComplexViewtype xcin("xcin", n0, n1), xcout("xcout", n0, n1);

  [[maybe_unused]] std::vector<std::size_t> ref_in_extents_axes01  = {n0, n1},
                                            ref_out_extents_axes01 = {n0, n1h},
                                            ref_fft_extents_axes01 = {n0, n1},
                                            ref_in_extents_axes10  = {n1, n0},
                                            ref_out_extents_axes10 = {n1, n0h},
                                            ref_fft_extents_axes10 = {n1, n0};
  std::size_t ref_howmany                                          = 1;

  std::vector<std::size_t> in_extents(2), out_extents(2), fft_extents(2);
  std::size_t howmany;

  // R2C or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis1, axes_type({0, 1}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes01);
    EXPECT_TRUE(out_extents == ref_out_extents_axes01);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes01);
    EXPECT_EQ(howmany, ref_howmany);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis0, axes_type({1, 0}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes10);
    EXPECT_TRUE(out_extents == ref_out_extents_axes10);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
    EXPECT_EQ(howmany, ref_howmany);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({0, 1}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({1, 0}));
          },
          std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis0, 2);

      // Order reversed for LayoutLeft
      EXPECT_TRUE(in_extents == ref_in_extents_axes10);
      EXPECT_TRUE(out_extents == ref_out_extents_axes10);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
      EXPECT_EQ(howmany, ref_howmany);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis1, 2);
      EXPECT_TRUE(in_extents == ref_in_extents_axes01);
      EXPECT_TRUE(out_extents == ref_out_extents_axes01);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes01);
      EXPECT_EQ(howmany, ref_howmany);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, 2); },
                   std::runtime_error);
    }
  }

  // C2R or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis1, xr, axes_type({0, 1}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes01);
    EXPECT_TRUE(out_extents == ref_in_extents_axes01);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes01);
    EXPECT_EQ(howmany, ref_howmany);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis0, xr, axes_type({1, 0}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes10);
    EXPECT_TRUE(out_extents == ref_in_extents_axes10);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
    EXPECT_EQ(howmany, ref_howmany);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({0, 1}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({1, 0}));
          },
          std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis0, xr, 2);

      // Order reversed for LayoutLeft
      EXPECT_TRUE(in_extents == ref_out_extents_axes10);
      EXPECT_TRUE(out_extents == ref_in_extents_axes10);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
      EXPECT_EQ(howmany, ref_howmany);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis1, xr, 2);
      EXPECT_TRUE(in_extents == ref_out_extents_axes01);
      EXPECT_TRUE(out_extents == ref_in_extents_axes01);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes01);
      EXPECT_EQ(howmany, ref_howmany);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, 2); },
                   std::runtime_error);
    }
  }
}

template <typename T, typename LayoutType>
void test_extents_2d_view_3d(bool is_static = true) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using axes_type    = KokkosFFT::axis_type<2>;
  using ViewType     = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexViewtype =
      Kokkos::View<complex_type***, LayoutType, execution_space>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  const std::size_t n0 = 6, n1 = 10, n2 = 8;
  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
                    n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C),
                    n2h = KokkosFFT::Impl::extent_after_transform(n2, is_R2C);

  ViewType xr("xr", n0, n1, n2);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1, n2),
      xc_axis1("xc_axis1", n0, n1h, n2), xc_axis2("xc_axis2", n0, n1, n2h);
  ComplexViewtype xcin("xcin", n0, n1, n2), xcout("xcout", n0, n1, n2);

  [[maybe_unused]] std::vector<std::size_t> ref_in_extents_axes01  = {n0, n1},
                                            ref_out_extents_axes01 = {n0, n1h},
                                            ref_fft_extents_axes01 = {n0, n1},
                                            ref_in_extents_axes02  = {n0, n2},
                                            ref_out_extents_axes02 = {n0, n2h},
                                            ref_fft_extents_axes02 = {n0, n2},
                                            ref_in_extents_axes10  = {n1, n0},
                                            ref_out_extents_axes10 = {n1, n0h},
                                            ref_fft_extents_axes10 = {n1, n0},
                                            ref_in_extents_axes12  = {n1, n2},
                                            ref_out_extents_axes12 = {n1, n2h},
                                            ref_fft_extents_axes12 = {n1, n2},
                                            ref_in_extents_axes20  = {n2, n0},
                                            ref_out_extents_axes20 = {n2, n0h},
                                            ref_fft_extents_axes20 = {n2, n0},
                                            ref_in_extents_axes21  = {n2, n1},
                                            ref_out_extents_axes21 = {n2, n1h},
                                            ref_fft_extents_axes21 = {n2, n1};
  [[maybe_unused]] std::size_t ref_howmany_axes01 = n2, ref_howmany_axes02 = n1,
                               ref_howmany_axes10 = n2, ref_howmany_axes12 = n0,
                               ref_howmany_axes20 = n1, ref_howmany_axes21 = n0;

  std::vector<std::size_t> in_extents(2), out_extents(2), fft_extents(2);
  std::size_t howmany;

  // R2C or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis1, axes_type({0, 1}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes01);
    EXPECT_TRUE(out_extents == ref_out_extents_axes01);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes01);
    EXPECT_EQ(howmany, ref_howmany_axes01);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis2, axes_type({0, 2}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes02);
    EXPECT_TRUE(out_extents == ref_out_extents_axes02);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes02);
    EXPECT_EQ(howmany, ref_howmany_axes02);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis0, axes_type({1, 0}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes10);
    EXPECT_TRUE(out_extents == ref_out_extents_axes10);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
    EXPECT_EQ(howmany, ref_howmany_axes10);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis2, axes_type({1, 2}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes12);
    EXPECT_TRUE(out_extents == ref_out_extents_axes12);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes12);
    EXPECT_EQ(howmany, ref_howmany_axes12);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis0, axes_type({2, 0}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes20);
    EXPECT_TRUE(out_extents == ref_out_extents_axes20);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes20);
    EXPECT_EQ(howmany, ref_howmany_axes20);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xr, xc_axis1, axes_type({2, 1}));

    EXPECT_TRUE(in_extents == ref_in_extents_axes21);
    EXPECT_TRUE(out_extents == ref_out_extents_axes21);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes21);
    EXPECT_EQ(howmany, ref_howmany_axes21);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({0, 1}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({0, 2}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({1, 0}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({1, 2}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({2, 0}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xr, xcout, axes_type({2, 1}));
          },
          std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis0, 2);

      // Order reversed for LayoutLeft
      EXPECT_TRUE(in_extents == ref_in_extents_axes10);
      EXPECT_TRUE(out_extents == ref_out_extents_axes10);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
      EXPECT_EQ(howmany, ref_howmany_axes10);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xr, xc_axis2, 2);
      EXPECT_TRUE(in_extents == ref_in_extents_axes12);
      EXPECT_TRUE(out_extents == ref_out_extents_axes12);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes12);
      EXPECT_EQ(howmany, ref_howmany_axes12);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xr, xcout, 2); },
                   std::runtime_error);
    }
  }

  // C2R or C2C
  if (is_static) {
    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis1, xr, axes_type({0, 1}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes01);
    EXPECT_TRUE(out_extents == ref_in_extents_axes01);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes01);
    EXPECT_EQ(howmany, ref_howmany_axes01);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis2, xr, axes_type({0, 2}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes02);
    EXPECT_TRUE(out_extents == ref_in_extents_axes02);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes02);
    EXPECT_EQ(howmany, ref_howmany_axes02);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis0, xr, axes_type({1, 0}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes10);
    EXPECT_TRUE(out_extents == ref_in_extents_axes10);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
    EXPECT_EQ(howmany, ref_howmany_axes10);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis2, xr, axes_type({1, 2}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes12);
    EXPECT_TRUE(out_extents == ref_in_extents_axes12);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes12);
    EXPECT_EQ(howmany, ref_howmany_axes12);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis0, xr, axes_type({2, 0}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes20);
    EXPECT_TRUE(out_extents == ref_in_extents_axes20);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes20);
    EXPECT_EQ(howmany, ref_howmany_axes20);

    std::tie(in_extents, out_extents, fft_extents, howmany) =
        KokkosFFT::Impl::get_extents(xc_axis1, xr, axes_type({2, 1}));

    EXPECT_TRUE(in_extents == ref_out_extents_axes21);
    EXPECT_TRUE(out_extents == ref_in_extents_axes21);
    EXPECT_TRUE(fft_extents == ref_fft_extents_axes21);
    EXPECT_EQ(howmany, ref_howmany_axes21);

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({0, 1}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({0, 2}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({1, 0}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({1, 2}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({2, 0}));
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            KokkosFFT::Impl::get_extents(xcin, xr, axes_type({2, 1}));
          },
          std::runtime_error);
    }
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis0, xr, 2);

      // Order reversed for LayoutLeft
      EXPECT_TRUE(in_extents == ref_out_extents_axes10);
      EXPECT_TRUE(out_extents == ref_in_extents_axes10);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes10);
      EXPECT_EQ(howmany, ref_howmany_axes10);
    } else {
      std::tie(in_extents, out_extents, fft_extents, howmany) =
          KokkosFFT::Impl::get_extents(xc_axis2, xr, 2);
      EXPECT_TRUE(in_extents == ref_out_extents_axes12);
      EXPECT_TRUE(out_extents == ref_in_extents_axes12);
      EXPECT_TRUE(fft_extents == ref_fft_extents_axes12);
      EXPECT_EQ(howmany, ref_howmany_axes12);
    }

    if constexpr (is_R2C) {
      // Check if errors are correctly raised against invalid extents
      EXPECT_THROW({ KokkosFFT::Impl::get_extents(xcin, xr, 2); },
                   std::runtime_error);
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(TestExtents, test_int_types);
TYPED_TEST_SUITE(TestGetExtents1D, test_types);
TYPED_TEST_SUITE(TestGetExtents2D, test_types);
TYPED_TEST_SUITE(TestGetDynExtents1D, test_types);
TYPED_TEST_SUITE(TestGetDynExtents2D, test_types);

// Get output extent
TEST(TestGetOutputExtent, R2C) {
  using float_type = double;
  test_extent_after_transform<float_type>();
}

TEST(TestGetOutputExtent, C2C) {
  using float_type = Kokkos::complex<double>;
  test_extent_after_transform<float_type>();
}

// Padded extents
TYPED_TEST(TestExtents, R2C) {
  using float_type = double;
  using int_type   = typename TestFixture::value_type;
  test_padded_extents<float_type, int_type>();
}

TYPED_TEST(TestExtents, C2C) {
  using float_type = Kokkos::complex<double>;
  using int_type   = typename TestFixture::value_type;
  test_padded_extents<float_type, int_type>();
}

// Mapped extents
TYPED_TEST(TestExtents, mapped_extents_of_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  for (value_type n = 1; n <= 6; ++n) {
    test_compute_mapped_extents<vector_type, value_type>(n);
  }
}

TYPED_TEST(TestExtents, mapped_extents_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type n = 1; n <= 6; ++n) {
    test_compute_mapped_extents<array_type, value_type>(n);
  }
}

// get_extents with static dimension
TYPED_TEST(TestGetExtents1D, 1DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_1d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents1D, 1DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_1d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents1D, 2DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_2d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents1D, 2DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_2d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents1D, 3DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_3d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents1D, 3DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_3d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents2D, 2DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_2d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents2D, 2DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_2d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents2D, 3DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_3d<float_type, layout_type>();
}

TYPED_TEST(TestGetExtents2D, 3DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_3d<float_type, layout_type>();
}

// get_extents with dynamic dimension
TYPED_TEST(TestGetDynExtents1D, 1DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_1d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents1D, 1DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_1d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents1D, 2DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_2d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents1D, 2DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_2d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents1D, 3DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_3d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents1D, 3DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_1d_view_3d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents2D, 2DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_2d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents2D, 2DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_2d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents2D, 3DView_R2C) {
  using float_type  = double;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_3d<float_type, layout_type>(false);
}

TYPED_TEST(TestGetDynExtents2D, 3DView_C2C) {
  using float_type  = Kokkos::complex<double>;
  using layout_type = typename TestFixture::layout_type;
  test_extents_2d_view_3d<float_type, layout_type>(false);
}
