// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include "KokkosFFT_Ulps.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using float_types =
    ::testing::Types<Kokkos::Experimental::half_t,
                     Kokkos::Experimental::bhalf_t, float, double>;

template <typename T>
struct TestAlmostEqualUlps : public ::testing::Test {
  using float_type   = T;
  using BoolViewType = Kokkos::View<bool, execution_space>;
};

template <typename T, typename BoolViewType>
void test_almost_equal_ulps_positive() {
  T a(1.0);
  T b = a;
  T c = nextafter_wrapper(a, T(2.0));
  T d = nextafter_wrapper(c, T(2.0));

  BoolViewType a_b_are_almost_equal("a_b_are_almost_equal"),
      a_c_are_almost_equal("a_c_are_almost_equal"),
      a_d_are_almost_equal_ulp1("a_d_are_almost_equal_ulp1"),
      a_d_are_almost_equal_ulp2("a_d_are_almost_equal_ulp2");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        // ULP diff 0 -> true
        a_b_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, b, 0);

        // ULP diff 1 -> true
        a_c_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, c, 1);

        // ULP diff 2 -> false if ulp_diff is 1 and true if ulp_diff is 2
        a_d_are_almost_equal_ulp1() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, d, 1);
        a_d_are_almost_equal_ulp2() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, d, 2);
      });

  auto h_a_b_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_b_are_almost_equal);
  auto h_a_c_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_c_are_almost_equal);
  auto h_a_d_are_almost_equal_ulp1 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_d_are_almost_equal_ulp1);
  auto h_a_d_are_almost_equal_ulp2 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_d_are_almost_equal_ulp2);

  ASSERT_TRUE(h_a_b_are_almost_equal());
  ASSERT_TRUE(h_a_c_are_almost_equal());
  ASSERT_FALSE(h_a_d_are_almost_equal_ulp1());
  ASSERT_TRUE(h_a_d_are_almost_equal_ulp2());
}

template <typename T, typename BoolViewType>
void test_almost_equal_ulps_negative() {
  T a(-1.0);
  T b = a;
  T c = nextafter_wrapper(a, T(-2.0));
  T d = nextafter_wrapper(c, T(-2.0));

  BoolViewType a_b_are_almost_equal("a_b_are_almost_equal"),
      a_c_are_almost_equal("a_c_are_almost_equal"),
      a_d_are_almost_equal_ulp1("a_d_are_almost_equal_ulp1"),
      a_d_are_almost_equal_ulp2("a_d_are_almost_equal_ulp2");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        // ULP diff 0 -> true
        a_b_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, b, 0);

        // ULP diff 1 -> true
        a_c_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, c, 1);

        // ULP diff 2 -> false if ulp_diff is 1 and true if ulp_diff is 2
        a_d_are_almost_equal_ulp1() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, d, 1);
        a_d_are_almost_equal_ulp2() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, d, 2);
      });

  auto h_a_b_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_b_are_almost_equal);
  auto h_a_c_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_c_are_almost_equal);
  auto h_a_d_are_almost_equal_ulp1 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_d_are_almost_equal_ulp1);
  auto h_a_d_are_almost_equal_ulp2 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_d_are_almost_equal_ulp2);

  ASSERT_TRUE(h_a_b_are_almost_equal());
  ASSERT_TRUE(h_a_c_are_almost_equal());
  ASSERT_FALSE(h_a_d_are_almost_equal_ulp1());
  ASSERT_TRUE(h_a_d_are_almost_equal_ulp2());
}

template <typename T, typename BoolViewType>
void test_almost_equal_ulps_near_zero() {
  T c(0.0);
  T d(-0.0);
  T a = nextafter_wrapper(c, T(1.0));
  T b = nextafter_wrapper(d, T(-1.0));

  BoolViewType a_b_are_almost_equal("a_b_are_almost_equal"),
      a_c_are_almost_equal("a_c_are_almost_equal"),
      b_d_are_almost_equal("b_d_are_almost_equal"),
      c_d_are_almost_equal("c_d_are_almost_equal");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        // ULP diff should be big -> false
        a_b_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, b, 1);
        // ULP diff should be 1 -> true
        a_c_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, c, 1);
        // ULP diff should be 1 -> true
        b_d_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(b, d, 1);
        // ULP diff should be 0 -> true (not sure this is a good behaviour)
        c_d_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(c, d, 0);
      });

  auto h_a_b_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_b_are_almost_equal);
  auto h_a_c_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_c_are_almost_equal);
  auto h_b_d_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, b_d_are_almost_equal);
  auto h_c_d_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, c_d_are_almost_equal);

  ASSERT_FALSE(h_a_b_are_almost_equal());
  ASSERT_TRUE(h_a_c_are_almost_equal());
  ASSERT_TRUE(h_b_d_are_almost_equal());
  ASSERT_TRUE(h_c_d_are_almost_equal());
}

template <typename T, typename BoolViewType>
void test_almost_equal_ulps_inf() {
  T pos_inf = Kokkos::Experimental::infinity<T>::value;
  T neg_inf = -static_cast<T>(Kokkos::Experimental::infinity<T>::value);

  BoolViewType a_b_are_almost_equal("a_b_are_almost_equal"),
      a_c_are_almost_equal("a_c_are_almost_equal");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        // ULP diff should be 0 -> true
        a_b_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(pos_inf, pos_inf, 0);
        // They are always different -> false
        a_c_are_almost_equal() = KokkosFFT::Testing::Impl::almost_equal_ulps(
            pos_inf, neg_inf, 1000000);
      });

  auto h_a_b_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_b_are_almost_equal);
  auto h_a_c_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_c_are_almost_equal);

  ASSERT_TRUE(h_a_b_are_almost_equal());
  ASSERT_FALSE(h_a_c_are_almost_equal());
}

template <typename T, typename BoolViewType>
void test_almost_equal_ulps_nan() {
  T a = Kokkos::Experimental::quiet_NaN<T>::value;
  T b = a;
  T c(1.0);

  BoolViewType a_b_are_almost_equal("a_b_are_almost_equal"),
      a_c_are_almost_equal("a_c_are_almost_equal");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        // Nans are considered as different -> false
        a_b_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, b, 1000000);
        // Value and Nan are always different -> false
        a_c_are_almost_equal() =
            KokkosFFT::Testing::Impl::almost_equal_ulps(a, c, 1000000);
      });

  auto h_a_b_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_b_are_almost_equal);
  auto h_a_c_are_almost_equal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, a_c_are_almost_equal);

  ASSERT_FALSE(h_a_b_are_almost_equal());
  ASSERT_FALSE(h_a_c_are_almost_equal());
}

}  // namespace

TYPED_TEST_SUITE(TestAlmostEqualUlps, float_types);

TYPED_TEST(TestAlmostEqualUlps, PositiveNumbers) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_almost_equal_ulps_positive<float_type, BoolViewType>();
}

TYPED_TEST(TestAlmostEqualUlps, NegativeNumbers) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_almost_equal_ulps_negative<float_type, BoolViewType>();
}

TYPED_TEST(TestAlmostEqualUlps, NearZero) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_almost_equal_ulps_near_zero<float_type, BoolViewType>();
}

TYPED_TEST(TestAlmostEqualUlps, Infinity) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_almost_equal_ulps_inf<float_type, BoolViewType>();
}

TYPED_TEST(TestAlmostEqualUlps, Nan) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_almost_equal_ulps_nan<float_type, BoolViewType>();
}
