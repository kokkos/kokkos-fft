// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Allclose.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestAllClose : public ::testing::Test {
  using float_type = T;

  const double m_rtol = 1.0e-5;
  const double m_atol = 1.0e-8;
};

TYPED_TEST_SUITE(TestAllClose, float_types);

template <typename T>
void test_allclose_1D_analytical(double rtol, double atol) {
  using View1DType = Kokkos::View<T*>;
  const int n      = 3;
  View1DType a("a", n), b("b", n), c("c", n), d("d", n), e("e", n);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);
  Kokkos::deep_copy(e, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  auto h_e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), e);

  // b(0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(0) = h_b(0) + 2.0 * (h_b(0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1) = h_c(1) + 2.0 * atol;

  // d includes both the relative and absolute errors -> error count 1
  h_d(0) = h_b(0);
  h_d(1) = h_c(1);

  // e includes small relative and absolute errors -> error count 0
  h_e(0) = h_e(0) + 0.1 * (h_e(0) * rtol);
  h_e(1) = h_e(1) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_THAT(b, ::testing::Not(KokkosFFT::Testing::allclose(a)));
  EXPECT_THAT(c, KokkosFFT::Testing::allclose(a, rtol));
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::allclose(a, rtol, atol)));
  EXPECT_THAT(e, KokkosFFT::Testing::allclose(a, rtol, atol, 3));
}

template <typename T>
void test_allclose_2D_analytical(double rtol, double atol) {
  using View2DType = Kokkos::View<T**>;
  const int n0 = 3, n1 = 2;
  View2DType a("a", n0, n1), b("b", n0, n1), c("c", n0, n1), d("d", n0, n1),
      e("e", n0, n1);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);
  Kokkos::deep_copy(e, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  auto h_e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), e);

  // b(0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(0, 0) = h_b(0, 0) + 2.0 * (h_b(0, 0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1, 0) = h_c(1, 0) + 2.0 * atol;

  // d includes both the relative and absolute errors -> error count 1
  h_d(0, 0) = h_b(0, 0);
  h_d(1, 0) = h_c(1, 0);

  // e includes small relative and absolute errors -> error count 0
  h_e(0, 0) = h_e(0, 0) + 0.1 * (h_e(0, 0) * rtol);
  h_e(1, 0) = h_e(1, 0) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_THAT(b, ::testing::Not(KokkosFFT::Testing::allclose(a)));
  EXPECT_THAT(c, KokkosFFT::Testing::allclose(a, rtol));
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::allclose(a, rtol, atol)));
  EXPECT_THAT(e, KokkosFFT::Testing::allclose(a, rtol, atol, 3));
}

TYPED_TEST(TestAllClose, View1D) {
  using float_type = typename TestFixture::float_type;
  test_allclose_1D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestAllClose, View2D) {
  using float_type = typename TestFixture::float_type;
  test_allclose_2D_analytical<float_type>(this->m_rtol, this->m_atol);
}

}  // namespace
