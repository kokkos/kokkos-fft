// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include <vector>
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_traits.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Basically the same fixtures, used for labeling tests
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
  using float_type      = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type    = Kokkos::complex<float_type>;
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

  const int n0  = 6;
  const int n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C);

  ViewType xr("xr", n0);
  ComplexViewtype xc("xc", n0h);
  ComplexViewtype xcin("xcin", n0), xcout("xcout", n0);

  std::vector<int> ref_in_extents = {n0}, ref_out_extents = {n0h},
                   ref_fft_extents = {n0};
  int ref_howmany                  = 1;

  std::vector<int> in_extents(1), out_extents(1), fft_extents(1);
  int howmany;

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

  const int n0 = 6, n1 = 10;
  const int n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
            n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C);

  ViewType xr("xr", n0, n1);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1), xc_axis1("xc_axis1", n0, n1h);
  ComplexViewtype xcin("xcin", n0, n1), xcout("xcout", n0, n1);

  [[maybe_unused]] std::vector<int> ref_in_extents_axis0  = {n0},
                                    ref_out_extents_axis0 = {n0h},
                                    ref_fft_extents_axis0 = {n0},
                                    ref_in_extents_axis1  = {n1},
                                    ref_out_extents_axis1 = {n1h},
                                    ref_fft_extents_axis1 = {n1};
  [[maybe_unused]] int ref_howmany_axis0 = n1, ref_howmany_axis1 = n0;

  std::vector<int> in_extents(1), out_extents(1), fft_extents(1);
  int howmany;

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

  const int n0 = 6, n1 = 10, n2 = 8;
  const int n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
            n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C),
            n2h = KokkosFFT::Impl::extent_after_transform(n2, is_R2C);

  ViewType xr("xr", n0, n1, n2);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1, n2),
      xc_axis1("xc_axis1", n0, n1h, n2), xc_axis2("xc_axis2", n0, n1, n2h);
  ComplexViewtype xcin("xcin", n0, n1, n2), xcout("xcout", n0, n1, n2);

  [[maybe_unused]] std::vector<int> ref_in_extents_axis0  = {n0},
                                    ref_out_extents_axis0 = {n0h},
                                    ref_fft_extents_axis0 = {n0},
                                    ref_in_extents_axis1  = {n1},
                                    ref_out_extents_axis1 = {n1h},
                                    ref_fft_extents_axis1 = {n1},
                                    ref_in_extents_axis2  = {n2},
                                    ref_out_extents_axis2 = {n2h},
                                    ref_fft_extents_axis2 = {n2};
  [[maybe_unused]] int ref_howmany_axis0 = n1 * n2, ref_howmany_axis1 = n0 * n2,
                       ref_howmany_axis2 = n0 * n1;

  std::vector<int> in_extents(1), out_extents(1), fft_extents(1);
  int howmany;

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

  const int n0 = 6, n1 = 10;
  const int n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
            n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C);

  ViewType xr("xr", n0, n1);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1), xc_axis1("xc_axis1", n0, n1h);
  ComplexViewtype xcin("xcin", n0, n1), xcout("xcout", n0, n1);

  [[maybe_unused]] std::vector<int> ref_in_extents_axes01  = {n0, n1},
                                    ref_out_extents_axes01 = {n0, n1h},
                                    ref_fft_extents_axes01 = {n0, n1},
                                    ref_in_extents_axes10  = {n1, n0},
                                    ref_out_extents_axes10 = {n1, n0h},
                                    ref_fft_extents_axes10 = {n1, n0};
  int ref_howmany                                          = 1;

  std::vector<int> in_extents(2), out_extents(2), fft_extents(2);
  int howmany;

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

  const int n0 = 6, n1 = 10, n2 = 8;
  const int n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
            n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C),
            n2h = KokkosFFT::Impl::extent_after_transform(n2, is_R2C);

  ViewType xr("xr", n0, n1, n2);
  ComplexViewtype xc_axis0("xc_axis0", n0h, n1, n2),
      xc_axis1("xc_axis1", n0, n1h, n2), xc_axis2("xc_axis2", n0, n1, n2h);
  ComplexViewtype xcin("xcin", n0, n1, n2), xcout("xcout", n0, n1, n2);

  [[maybe_unused]] std::vector<int> ref_in_extents_axes01  = {n0, n1},
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
  [[maybe_unused]] int ref_howmany_axes01 = n2, ref_howmany_axes02 = n1,
                       ref_howmany_axes10 = n2, ref_howmany_axes12 = n0,
                       ref_howmany_axes20 = n1, ref_howmany_axes21 = n0;

  std::vector<int> in_extents(2), out_extents(2), fft_extents(2);
  int howmany;

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
