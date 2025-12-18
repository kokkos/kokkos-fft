// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include "KokkosFFT_AlmostEqual.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using float_types =
    ::testing::Types<Kokkos::Experimental::half_t,
                     Kokkos::Experimental::bhalf_t, float, double>;

template <typename T>
struct TestAreClose : public ::testing::Test {
  using float_type    = T;
  using BoolViewType  = Kokkos::View<bool, execution_space>;
  const double m_rtol = 1.0e-5;
  const double m_atol = 1.0e-8;
};

template <typename T, typename BoolViewType>
void test_are_close_rtol(double rtol) {
  T a(1.0);
  T b               = a;
  T c               = a + 1.0;
  T d               = a + a * 0.01 * rtol;
  const double atol = 0.0;
  BoolViewType a_b_are_close("a_b_are_close"), a_c_are_close("a_c_are_close"),
      a_d_are_close("a_d_are_close");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        a_b_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, b, rtol, atol);
        a_c_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, c, rtol, atol);
        a_d_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, d, rtol, atol);
      });

  auto h_a_b_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_b_are_close);
  auto h_a_c_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_c_are_close);
  auto h_a_d_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_d_are_close);

  ASSERT_TRUE(h_a_b_are_close());
  ASSERT_FALSE(h_a_c_are_close());
  ASSERT_TRUE(h_a_d_are_close());
}

template <typename T, typename BoolViewType>
void test_are_close_atol(double atol) {
  T a(1.0);
  T b               = a;
  T c               = a + 1.0;
  T d               = a + 0.01 * atol;
  const double rtol = 0.0;
  BoolViewType a_b_are_close("a_b_are_close"), a_c_are_close("a_c_are_close"),
      a_d_are_close("a_d_are_close");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        a_b_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, b, rtol, atol);
        a_c_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, c, rtol, atol);
        a_d_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, d, rtol, atol);
      });

  auto h_a_b_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_b_are_close);
  auto h_a_c_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_c_are_close);
  auto h_a_d_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_d_are_close);

  ASSERT_TRUE(h_a_b_are_close());
  ASSERT_FALSE(h_a_c_are_close());
  ASSERT_TRUE(h_a_d_are_close());
}

template <typename T, typename BoolViewType>
void test_are_close_rtol_and_atol(double rtol, double atol) {
  T a(1.0);
  T b = a;
  T c = a + 1.0;
  T d = a + a * 0.01 * rtol + 0.01 * atol;
  BoolViewType a_b_are_close("a_b_are_close"), a_c_are_close("a_c_are_close"),
      a_d_are_close("a_d_are_close");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        a_b_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, b, rtol, atol);
        a_c_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, c, rtol, atol);
        a_d_are_close() =
            KokkosFFT::Testing::Impl::are_almost_equal(a, d, rtol, atol);
      });

  auto h_a_b_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_b_are_close);
  auto h_a_c_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_c_are_close);
  auto h_a_d_are_close =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_d_are_close);

  ASSERT_TRUE(h_a_b_are_close());
  ASSERT_FALSE(h_a_c_are_close());
  ASSERT_TRUE(h_a_d_are_close());
}

}  // namespace

TYPED_TEST_SUITE(TestAreClose, float_types);

TYPED_TEST(TestAreClose, RelativeTolerance) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_are_close_rtol<float_type, BoolViewType>(this->m_rtol);
}

TYPED_TEST(TestAreClose, AbsoluteTolerance) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_are_close_atol<float_type, BoolViewType>(this->m_rtol);
}

TYPED_TEST(TestAreClose, RelativeAndAbsoluteTolerance) {
  using float_type   = typename TestFixture::float_type;
  using BoolViewType = typename TestFixture::BoolViewType;
  test_are_close_rtol_and_atol<float_type, BoolViewType>(this->m_rtol,
                                                         this->m_atol);
}
