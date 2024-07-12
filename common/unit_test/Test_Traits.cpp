// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_traits.hpp"
#include "Test_Utils.hpp"

using real_types = ::testing::Types<float, double, long double>;
using view_types =
    ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                     std::pair<float, Kokkos::LayoutRight>,
                     std::pair<float, Kokkos::LayoutStride>,
                     std::pair<double, Kokkos::LayoutLeft>,
                     std::pair<double, Kokkos::LayoutRight>,
                     std::pair<double, Kokkos::LayoutStride>,
                     std::pair<long double, Kokkos::LayoutLeft>,
                     std::pair<long double, Kokkos::LayoutRight>,
                     std::pair<long double, Kokkos::LayoutStride>>;

template <typename T>
struct RealAndComplexTypes : public ::testing::Test {
  using real_type    = T;
  using complex_type = Kokkos::complex<T>;
};

template <typename T>
struct RealAndComplexViewTypes : public ::testing::Test {
  using real_type    = typename T::first_type;
  using complex_type = Kokkos::complex<real_type>;
  using layout_type  = typename T::second_type;
};

TYPED_TEST_SUITE(RealAndComplexTypes, real_types);
TYPED_TEST_SUITE(RealAndComplexViewTypes, view_types);

// Tests for real type deduction
template <typename RealType, typename ComplexType>
void test_get_real_type() {
  using real_type_from_RealType =
      KokkosFFT::Impl::base_floating_point_type<RealType>;
  using real_type_from_ComplexType =
      KokkosFFT::Impl::base_floating_point_type<ComplexType>;

  static_assert(std::is_same_v<real_type_from_RealType, RealType>,
                "Real type not deduced correctly from real type");
  static_assert(std::is_same_v<real_type_from_ComplexType, RealType>,
                "Real type not deduced correctly from complex type");
}

// Tests for admissible real types (float or double)
template <typename T>
void test_admissible_real_type() {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    static_assert(KokkosFFT::Impl::is_real_v<T>,
                  "Real type must be float or double");
  } else {
    static_assert(!KokkosFFT::Impl::is_real_v<T>,
                  "Real type must be float or double");
  }
}

template <typename T>
void test_admissible_complex_type() {
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  if constexpr (std::is_same_v<real_type, float> ||
                std::is_same_v<real_type, double>) {
    static_assert(KokkosFFT::Impl::is_complex_v<T>,
                  "Complex type must be Kokkos::complex<float> or "
                  "Kokkos::complex<double>");
  } else {
    static_assert(!KokkosFFT::Impl::is_complex_v<T>,
                  "Complex type must be Kokkos::complex<float> or "
                  "Kokkos::complex<double>");
  }
}

TYPED_TEST(RealAndComplexTypes, get_real_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;

  test_get_real_type<real_type, complex_type>();
}

TYPED_TEST(RealAndComplexTypes, admissible_real_type) {
  using real_type = typename TestFixture::real_type;

  test_admissible_real_type<real_type>();
}

TYPED_TEST(RealAndComplexTypes, admissible_complex_type) {
  using complex_type = typename TestFixture::complex_type;

  test_admissible_complex_type<complex_type>();
}

// Tests for admissible view types
template <typename T, typename LayoutType>
void test_admissible_value_type() {
  using ViewType  = Kokkos::View<T*, LayoutType>;
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  if constexpr (std::is_same_v<real_type, float> ||
                std::is_same_v<real_type, double>) {
    static_assert(KokkosFFT::Impl::is_admissible_value_type_v<ViewType>,
                  "Real type must be float or double");
  } else {
    static_assert(!KokkosFFT::Impl::is_admissible_value_type_v<ViewType>,
                  "Real type must be float or double");
  }
}

template <typename T, typename LayoutType>
void test_admissible_layout_type() {
  using ViewType = Kokkos::View<T*, LayoutType>;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<ViewType>,
                  "View Layout must be either LayoutLeft or LayoutRight.");
  } else {
    static_assert(!KokkosFFT::Impl::is_layout_left_or_right_v<ViewType>,
                  "View Layout must be either LayoutLeft or LayoutRight.");
  }
}

template <typename T, typename LayoutType>
void test_admissible_view_type() {
  using ViewType  = Kokkos::View<T*, LayoutType>;
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  if constexpr (
      (std::is_same_v<real_type, float> || std::is_same_v<real_type, double>)&&(
          std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
          std::is_same_v<LayoutType, Kokkos::LayoutRight>)) {
    static_assert(KokkosFFT::Impl::is_admissible_view_v<ViewType>,
                  "View value type must be float, double, "
                  "Kokkos::Complex<float>, Kokkos::Complex<double>. Layout "
                  "must be either LayoutLeft or LayoutRight.");
  } else {
    static_assert(!KokkosFFT::Impl::is_admissible_view_v<ViewType>,
                  "View value type must be float, double, "
                  "Kokkos::Complex<float>, Kokkos::Complex<double>. Layout "
                  "must be either LayoutLeft or LayoutRight.");
  }
}

TYPED_TEST(RealAndComplexViewTypes, admissible_value_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_value_type<real_type, layout_type>();
  test_admissible_value_type<complex_type, layout_type>();
}

TYPED_TEST(RealAndComplexViewTypes, admissible_layout_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_layout_type<real_type, layout_type>();
  test_admissible_layout_type<complex_type, layout_type>();
}

TYPED_TEST(RealAndComplexViewTypes, admissible_view_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_view_type<real_type, layout_type>();
  test_admissible_view_type<complex_type, layout_type>();
}
