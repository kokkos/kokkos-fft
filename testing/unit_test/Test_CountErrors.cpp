// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_CountErrors.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestCountErrors : public ::testing::Test {
  using float_type = T;

  const double m_rtol = 1.0e-5;
  const double m_atol = 1.0e-8;
};

template <typename T>
void test_view_errors_1D_analytical(double rtol, double atol) {
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

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}

template <typename T>
void test_view_errors_2D_analytical(double rtol, double atol) {
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

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}

template <typename T>
void test_view_errors_3D_analytical(double rtol, double atol) {
  using View3DType = Kokkos::View<T***>;
  const int n0 = 3, n1 = 2, n2 = 4;
  View3DType a("a", n0, n1, n2), b("b", n0, n1, n2), c("c", n0, n1, n2),
      d("d", n0, n1, n2), e("e", n0, n1, n2);

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
  h_b(0, 0, 0) = h_b(0, 0, 0) + 2.0 * (h_b(0, 0, 0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1, 0, 0) = h_c(1, 0, 0) + 2.0 * atol;

  // d includes two values with bigger errors than acceptance -> error count 2
  h_d(0, 0, 0) = h_b(0, 0, 0);
  h_d(1, 0, 0) = h_c(1, 0, 0);
  h_d(1, 0, 2) = h_b(1, 0, 2) + 3.0 * (h_b(1, 0, 2) * rtol);

  // e includes small relative and absolute errors -> error count 0
  h_e(0, 0, 0) = h_e(0, 0, 0) + 0.1 * (h_e(0, 0, 0) * rtol);
  h_e(1, 0, 0) = h_e(1, 0, 0) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            2);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}

template <typename T>
void test_view_errors_4D_analytical(double rtol, double atol) {
  using View4DType = Kokkos::View<T****>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5;
  View4DType a("a", n0, n1, n2, n3), b("b", n0, n1, n2, n3),
      c("c", n0, n1, n2, n3), d("d", n0, n1, n2, n3), e("e", n0, n1, n2, n3);

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
  h_b(0, 0, 0, 0) = h_b(0, 0, 0, 0) + 2.0 * (h_b(0, 0, 0, 0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1, 0, 0, 0) = h_c(1, 0, 0, 0) + 2.0 * atol;

  // d includes three values with bigger errors than acceptance -> error count 3
  h_d(0, 0, 0, 0) = h_b(0, 0, 0, 0);
  h_d(1, 0, 0, 0) = h_c(1, 0, 0, 0);
  h_d(1, 0, 2, 0) = h_b(1, 0, 2, 0) + 3.0 * (h_b(1, 0, 2, 0) * rtol);
  h_d(1, 0, 1, 3) = h_b(1, 0, 1, 3) + 2.5 * (h_b(1, 0, 1, 3) * rtol);

  // e includes small relative and absolute errors -> error count 0
  h_e(0, 0, 0, 0) = h_e(0, 0, 0, 0) + 0.1 * (h_e(0, 0, 0, 0) * rtol);
  h_e(1, 0, 0, 0) = h_e(1, 0, 0, 0) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            3);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}

template <typename T>
void test_view_errors_5D_analytical(double rtol, double atol) {
  using View5DType = Kokkos::View<T*****>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6;
  View5DType a("a", n0, n1, n2, n3, n4), b("b", n0, n1, n2, n3, n4),
      c("c", n0, n1, n2, n3, n4), d("d", n0, n1, n2, n3, n4),
      e("e", n0, n1, n2, n3, n4);

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
  h_b(0, 0, 0, 0, 0) = h_b(0, 0, 0, 0, 0) + 2.0 * (h_b(0, 0, 0, 0, 0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0) + 2.0 * atol;

  // d includes four values with bigger errors than acceptance -> error count 4
  h_d(0, 0, 0, 0, 0) = h_b(0, 0, 0, 0, 0);
  h_d(1, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0);
  h_d(1, 0, 2, 0, 0) = h_b(1, 0, 2, 0, 0) + 3.0 * (h_b(1, 0, 2, 0, 0) * rtol);
  h_d(1, 0, 1, 3, 0) = h_b(1, 0, 1, 3, 0) + 2.5 * (h_b(1, 0, 1, 3, 0) * rtol);
  h_d(1, 0, 2, 1, 4) = h_b(1, 0, 2, 1, 4) + 1.5 * (h_b(1, 0, 2, 1, 4) * rtol);

  // e includes small relative and absolute errors -> error count 0
  h_e(0, 0, 0, 0, 0) = h_e(0, 0, 0, 0, 0) + 0.1 * (h_e(0, 0, 0, 0, 0) * rtol);
  h_e(1, 0, 0, 0, 0) = h_e(1, 0, 0, 0, 0) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            4);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}

template <typename T>
void test_view_errors_6D_analytical(double rtol, double atol) {
  using View6DType = Kokkos::View<T******>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6, n5 = 3;
  View6DType a("a", n0, n1, n2, n3, n4, n5), b("b", n0, n1, n2, n3, n4, n5),
      c("c", n0, n1, n2, n3, n4, n5), d("d", n0, n1, n2, n3, n4, n5),
      e("e", n0, n1, n2, n3, n4, n5);

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
  h_b(0, 0, 0, 0, 0, 0) =
      h_b(0, 0, 0, 0, 0, 0) + 2.0 * (h_b(0, 0, 0, 0, 0, 0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1, 0, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0, 0) + 2.0 * atol;

  // d includes five values with bigger errors than acceptance -> error count 5
  h_d(0, 0, 0, 0, 0, 0) = h_b(0, 0, 0, 0, 0, 0);
  h_d(1, 0, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0, 0);
  h_d(1, 0, 2, 0, 0, 0) =
      h_b(1, 0, 2, 0, 0, 0) + 3.0 * (h_b(1, 0, 2, 0, 0, 0) * rtol);
  h_d(1, 0, 1, 3, 0, 0) =
      h_b(1, 0, 1, 3, 0, 0) + 2.5 * (h_b(1, 0, 1, 3, 0, 0) * rtol);
  h_d(1, 0, 2, 1, 4, 0) =
      h_b(1, 0, 2, 1, 4, 0) + 1.5 * (h_b(1, 0, 2, 1, 4, 0) * rtol);
  h_d(0, 0, 2, 1, 3, 1) =
      h_b(0, 0, 2, 1, 3, 1) + 1.8 * (h_b(0, 0, 2, 1, 3, 1) * rtol);

  // e includes small relative and absolute errors -> error count 0
  h_e(0, 0, 0, 0, 0, 0) =
      h_e(0, 0, 0, 0, 0, 0) + 0.1 * (h_e(0, 0, 0, 0, 0, 0) * rtol);
  h_e(1, 0, 0, 0, 0, 0) = h_e(1, 0, 0, 0, 0, 0) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            5);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}

template <typename T>
void test_view_errors_7D_analytical(double rtol, double atol) {
  using View7DType = Kokkos::View<T*******>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6, n5 = 3, n6 = 3;
  View7DType a("a", n0, n1, n2, n3, n4, n5, n6),
      b("b", n0, n1, n2, n3, n4, n5, n6), c("c", n0, n1, n2, n3, n4, n5, n6),
      d("d", n0, n1, n2, n3, n4, n5, n6), e("e", n0, n1, n2, n3, n4, n5, n6);

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
  h_b(0, 0, 0, 0, 0, 0, 0) =
      h_b(0, 0, 0, 0, 0, 0, 0) + 2.0 * (h_b(0, 0, 0, 0, 0, 0, 0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1, 0, 0, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0, 0, 0) + 2.0 * atol;

  // d includes six values with bigger errors than acceptance -> error count 6
  h_d(0, 0, 0, 0, 0, 0, 0) = h_b(0, 0, 0, 0, 0, 0, 0);
  h_d(1, 0, 0, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0, 0, 0);
  h_d(1, 0, 2, 0, 0, 0, 0) =
      h_b(1, 0, 2, 0, 0, 0, 0) + 3.0 * (h_b(1, 0, 2, 0, 0, 0, 0) * rtol);
  h_d(1, 0, 1, 3, 0, 0, 0) =
      h_b(1, 0, 1, 3, 0, 0, 0) + 2.5 * (h_b(1, 0, 1, 3, 0, 0, 0) * rtol);
  h_d(1, 0, 2, 1, 4, 0, 0) =
      h_b(1, 0, 2, 1, 4, 0, 0) + 1.5 * (h_b(1, 0, 2, 1, 4, 0, 0) * rtol);
  h_d(0, 0, 2, 1, 3, 1, 0) =
      h_b(0, 0, 2, 1, 3, 1, 0) + 1.8 * (h_b(0, 0, 2, 1, 3, 1, 0) * rtol);
  h_d(0, 0, 2, 1, 4, 1, 2) =
      h_b(0, 0, 2, 1, 4, 1, 2) + 2.1 * (h_b(0, 0, 2, 1, 4, 1, 2) * rtol);

  // e includes small relative and absolute errors -> error count 0
  h_e(0, 0, 0, 0, 0, 0, 0) =
      h_e(0, 0, 0, 0, 0, 0, 0) + 0.1 * (h_e(0, 0, 0, 0, 0, 0, 0) * rtol);
  h_e(1, 0, 0, 0, 0, 0, 0) = h_e(1, 0, 0, 0, 0, 0, 0) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            6);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}

template <typename T>
void test_view_errors_8D_analytical(double rtol, double atol) {
  using View8DType = Kokkos::View<T********>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6, n5 = 3, n6 = 3, n7 = 7;
  View8DType a("a", n0, n1, n2, n3, n4, n5, n6, n7),
      b("b", n0, n1, n2, n3, n4, n5, n6, n7),
      c("c", n0, n1, n2, n3, n4, n5, n6, n7),
      d("d", n0, n1, n2, n3, n4, n5, n6, n7),
      e("e", n0, n1, n2, n3, n4, n5, n6, n7);

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
  h_b(0, 0, 0, 0, 0, 0, 0, 0) =
      h_b(0, 0, 0, 0, 0, 0, 0, 0) + 2.0 * (h_b(0, 0, 0, 0, 0, 0, 0, 0) * rtol);

  // c(a) includes 3.00000001 which is acceptable by rtol -> error count 0
  h_c(1, 0, 0, 0, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0, 0, 0, 0) + 2.0 * atol;

  // d includes seven values with bigger errors than acceptance -> error count 7
  h_d(0, 0, 0, 0, 0, 0, 0, 0) = h_b(0, 0, 0, 0, 0, 0, 0, 0);
  h_d(1, 0, 0, 0, 0, 0, 0, 0) = h_c(1, 0, 0, 0, 0, 0, 0, 0);
  h_d(1, 0, 2, 0, 0, 0, 0, 0) =
      h_b(1, 0, 2, 0, 0, 0, 0, 0) + 3.0 * (h_b(1, 0, 2, 0, 0, 0, 0, 0) * rtol);
  h_d(1, 0, 1, 3, 0, 0, 0, 0) =
      h_b(1, 0, 1, 3, 0, 0, 0, 0) + 2.5 * (h_b(1, 0, 1, 3, 0, 0, 0, 0) * rtol);
  h_d(1, 0, 2, 1, 4, 0, 0, 0) =
      h_b(1, 0, 2, 1, 4, 0, 0, 0) + 1.5 * (h_b(1, 0, 2, 1, 4, 0, 0, 0) * rtol);
  h_d(0, 0, 2, 1, 3, 1, 0, 0) =
      h_b(0, 0, 2, 1, 3, 1, 0, 0) + 1.8 * (h_b(0, 0, 2, 1, 3, 1, 0, 0) * rtol);
  h_d(0, 0, 2, 1, 4, 1, 2, 0) =
      h_b(0, 0, 2, 1, 4, 1, 2, 0) + 2.1 * (h_b(0, 0, 2, 1, 4, 1, 2, 0) * rtol);
  h_d(0, 0, 2, 1, 4, 1, 2, 5) =
      h_b(0, 0, 2, 1, 4, 1, 2, 5) + 1.7 * (h_b(0, 0, 2, 1, 4, 1, 2, 5) * rtol);

  // e includes small relative and absolute errors -> error count 0
  h_e(0, 0, 0, 0, 0, 0, 0, 0) =
      h_e(0, 0, 0, 0, 0, 0, 0, 0) + 0.1 * (h_e(0, 0, 0, 0, 0, 0, 0, 0) * rtol);
  h_e(1, 0, 0, 0, 0, 0, 0, 0) = h_e(1, 0, 0, 0, 0, 0, 0, 0) + 0.1 * atol;

  Kokkos::deep_copy(b, h_b);
  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), b, a,
                                                   rtol, atol),
            1);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), c, a,
                                                   rtol, atol),
            0);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), d, a,
                                                   rtol, atol),
            7);
  EXPECT_EQ(KokkosFFT::Testing::Impl::count_errors(execution_space(), e, a,
                                                   rtol, atol),
            0);
}
}  // namespace

TYPED_TEST_SUITE(TestCountErrors, float_types);

TYPED_TEST(TestCountErrors, View1D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_1D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestCountErrors, View2D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_2D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestCountErrors, View3D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_3D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestCountErrors, View4D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_4D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestCountErrors, View5D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_5D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestCountErrors, View6D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_6D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestCountErrors, View7D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_7D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestCountErrors, View8D) {
  using float_type = typename TestFixture::float_type;
  test_view_errors_8D_analytical<float_type>(this->m_rtol, this->m_atol);
}
