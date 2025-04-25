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
struct TestFindErrors : public ::testing::Test {
  using float_type = T;

  const double m_rtol = 1.0e-5;
  const double m_atol = 1.0e-8;
};

template <typename T>
void test_find_errors_1D_analytical(double rtol, double atol) {
  using View1DType    = Kokkos::View<T*>;
  using CountViewType = Kokkos::View<std::size_t**>;
  const std::size_t n = 3, nb_errors = 1;
  View1DType a("a", n), b("b", n);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 2);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(0) = h_b(0) + 2.0 * (h_b(0) * rtol);

  h_ref_a_error(0)      = T(3.0);
  h_ref_b_error(0)      = h_b(0);
  h_ref_loc_error(0, 0) = 0;  // global idx
  h_ref_loc_error(0, 1) = 0;  // idx of dimension 0
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
  }
}

template <typename T>
void test_find_errors_2D_analytical(double rtol, double atol) {
  using View1DType     = Kokkos::View<T*>;
  using View2DType     = Kokkos::View<T**>;
  using CountViewType  = Kokkos::View<std::size_t**>;
  const std::size_t n0 = 3, n1 = 2, nb_errors = 1;
  View2DType a("a", n0, n1), b("b", n0, n1);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 3);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0, 0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(2, 1) = h_b(2, 1) + 2.0 * (h_b(2, 1) * rtol);

  h_ref_a_error(0)      = T(3.0);
  h_ref_b_error(0)      = h_b(2, 1);
  h_ref_loc_error(0, 0) = 2 + 1 * b.extent(0);  // global idx
  h_ref_loc_error(0, 1) = 2;                    // idx of dimension 0
  h_ref_loc_error(0, 2) = 1;                    // idx of dimension 1
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
    EXPECT_EQ(h_loc_error(i, 2), h_ref_loc_error(i, 2));
  }
}

template <typename T>
void test_find_errors_3D_analytical(double rtol, double atol) {
  using View1DType     = Kokkos::View<T*>;
  using View3DType     = Kokkos::View<T***>;
  using CountViewType  = Kokkos::View<std::size_t**>;
  const std::size_t n0 = 3, n1 = 2, n2 = 4, nb_errors = 1;
  View3DType a("a", n0, n1, n2), b("b", n0, n1, n2);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 4);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0, 0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(2, 1, 3) = h_b(2, 1, 3) + 2.0 * (h_b(2, 1, 3) * rtol);

  h_ref_a_error(0) = T(3.0);
  h_ref_b_error(0) = h_b(2, 1, 3);
  h_ref_loc_error(0, 0) =
      2 + 1 * b.extent(0) + 3 * b.extent(0) * b.extent(1);  // global idx
  h_ref_loc_error(0, 1) = 2;  // idx of dimension 0
  h_ref_loc_error(0, 2) = 1;  // idx of dimension 1
  h_ref_loc_error(0, 3) = 3;  // idx of dimension 2
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
    EXPECT_EQ(h_loc_error(i, 2), h_ref_loc_error(i, 2));
    EXPECT_EQ(h_loc_error(i, 3), h_ref_loc_error(i, 3));
  }
}

template <typename T>
void test_find_errors_4D_analytical(double rtol, double atol) {
  using View1DType     = Kokkos::View<T*>;
  using View4DType     = Kokkos::View<T****>;
  using CountViewType  = Kokkos::View<std::size_t**>;
  const std::size_t n0 = 3, n1 = 2, n2 = 4, n3 = 5, nb_errors = 1;
  View4DType a("a", n0, n1, n2, n3), b("b", n0, n1, n2, n3);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 5);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0, 0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(2, 1, 3, 1) = h_b(2, 1, 3, 1) + 2.0 * (h_b(2, 1, 3, 1) * rtol);

  h_ref_a_error(0) = T(3.0);
  h_ref_b_error(0) = h_b(2, 1, 3, 1);
  h_ref_loc_error(0, 0) =
      2 + 1 * b.extent(0) + 3 * b.extent(0) * b.extent(1) +
      1 * b.extent(0) * b.extent(1) * b.extent(2);  // global idx
  h_ref_loc_error(0, 1) = 2;                        // idx of dimension 0
  h_ref_loc_error(0, 2) = 1;                        // idx of dimension 1
  h_ref_loc_error(0, 3) = 3;                        // idx of dimension 2
  h_ref_loc_error(0, 4) = 1;                        // idx of dimension 3
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
    EXPECT_EQ(h_loc_error(i, 2), h_ref_loc_error(i, 2));
    EXPECT_EQ(h_loc_error(i, 3), h_ref_loc_error(i, 3));
    EXPECT_EQ(h_loc_error(i, 4), h_ref_loc_error(i, 4));
  }
}

template <typename T>
void test_find_errors_5D_analytical(double rtol, double atol) {
  using View1DType     = Kokkos::View<T*>;
  using View5DType     = Kokkos::View<T*****>;
  using CountViewType  = Kokkos::View<std::size_t**>;
  const std::size_t n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 3, nb_errors = 1;
  View5DType a("a", n0, n1, n2, n3, n4), b("b", n0, n1, n2, n3, n4);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 6);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0, 0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(2, 1, 3, 1, 2) = h_b(2, 1, 3, 1, 2) + 2.0 * (h_b(2, 1, 3, 1, 2) * rtol);

  h_ref_a_error(0) = T(3.0);
  h_ref_b_error(0) = h_b(2, 1, 3, 1, 2);
  h_ref_loc_error(0, 0) =
      2 + 1 * b.extent(0) + 3 * b.extent(0) * b.extent(1) +
      1 * b.extent(0) * b.extent(1) * b.extent(2) +
      2 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3);  // global idx
  h_ref_loc_error(0, 1) = 2;  // idx of dimension 0
  h_ref_loc_error(0, 2) = 1;  // idx of dimension 1
  h_ref_loc_error(0, 3) = 3;  // idx of dimension 2
  h_ref_loc_error(0, 4) = 1;  // idx of dimension 3
  h_ref_loc_error(0, 5) = 2;  // idx of dimension 3
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
    EXPECT_EQ(h_loc_error(i, 2), h_ref_loc_error(i, 2));
    EXPECT_EQ(h_loc_error(i, 3), h_ref_loc_error(i, 3));
    EXPECT_EQ(h_loc_error(i, 4), h_ref_loc_error(i, 4));
    EXPECT_EQ(h_loc_error(i, 5), h_ref_loc_error(i, 5));
  }
}

template <typename T>
void test_find_errors_6D_analytical(double rtol, double atol) {
  using View1DType     = Kokkos::View<T*>;
  using View6DType     = Kokkos::View<T******>;
  using CountViewType  = Kokkos::View<std::size_t**>;
  const std::size_t n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 3, n5 = 4,
                    nb_errors = 1;
  View6DType a("a", n0, n1, n2, n3, n4, n5), b("b", n0, n1, n2, n3, n4, n5);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 7);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0, 0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(2, 1, 3, 1, 2, 0) =
      h_b(2, 1, 3, 1, 2, 0) + 2.0 * (h_b(2, 1, 3, 1, 2, 0) * rtol);

  h_ref_a_error(0) = T(3.0);
  h_ref_b_error(0) = h_b(2, 1, 3, 1, 2, 0);
  h_ref_loc_error(0, 0) =
      2 + 1 * b.extent(0) + 3 * b.extent(0) * b.extent(1) +
      1 * b.extent(0) * b.extent(1) * b.extent(2) +
      2 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) +
      0 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) *
          b.extent(4);        // global idx
  h_ref_loc_error(0, 1) = 2;  // idx of dimension 0
  h_ref_loc_error(0, 2) = 1;  // idx of dimension 1
  h_ref_loc_error(0, 3) = 3;  // idx of dimension 2
  h_ref_loc_error(0, 4) = 1;  // idx of dimension 3
  h_ref_loc_error(0, 5) = 2;  // idx of dimension 4
  h_ref_loc_error(0, 6) = 0;  // idx of dimension 5
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
    EXPECT_EQ(h_loc_error(i, 2), h_ref_loc_error(i, 2));
    EXPECT_EQ(h_loc_error(i, 3), h_ref_loc_error(i, 3));
    EXPECT_EQ(h_loc_error(i, 4), h_ref_loc_error(i, 4));
    EXPECT_EQ(h_loc_error(i, 5), h_ref_loc_error(i, 5));
    EXPECT_EQ(h_loc_error(i, 6), h_ref_loc_error(i, 6));
  }
}

template <typename T>
void test_find_errors_7D_analytical(double rtol, double atol) {
  using View1DType     = Kokkos::View<T*>;
  using View7DType     = Kokkos::View<T*******>;
  using CountViewType  = Kokkos::View<std::size_t**>;
  const std::size_t n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 3, n5 = 4, n6 = 2,
                    nb_errors = 1;
  View7DType a("a", n0, n1, n2, n3, n4, n5, n6),
      b("b", n0, n1, n2, n3, n4, n5, n6);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 8);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0, 0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(2, 1, 3, 1, 2, 0, 1) =
      h_b(2, 1, 3, 1, 2, 0, 1) + 2.0 * (h_b(2, 1, 3, 1, 2, 0, 1) * rtol);

  h_ref_a_error(0) = T(3.0);
  h_ref_b_error(0) = h_b(2, 1, 3, 1, 2, 0, 1);
  h_ref_loc_error(0, 0) =
      2 + 1 * b.extent(0) + 3 * b.extent(0) * b.extent(1) +
      1 * b.extent(0) * b.extent(1) * b.extent(2) +
      2 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) +
      0 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) * b.extent(4) +
      1 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) * b.extent(4) *
          b.extent(5);        // global idx
  h_ref_loc_error(0, 1) = 2;  // idx of dimension 0
  h_ref_loc_error(0, 2) = 1;  // idx of dimension 1
  h_ref_loc_error(0, 3) = 3;  // idx of dimension 2
  h_ref_loc_error(0, 4) = 1;  // idx of dimension 3
  h_ref_loc_error(0, 5) = 2;  // idx of dimension 4
  h_ref_loc_error(0, 6) = 0;  // idx of dimension 5
  h_ref_loc_error(0, 7) = 1;  // idx of dimension 6
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
    EXPECT_EQ(h_loc_error(i, 2), h_ref_loc_error(i, 2));
    EXPECT_EQ(h_loc_error(i, 3), h_ref_loc_error(i, 3));
    EXPECT_EQ(h_loc_error(i, 4), h_ref_loc_error(i, 4));
    EXPECT_EQ(h_loc_error(i, 5), h_ref_loc_error(i, 5));
    EXPECT_EQ(h_loc_error(i, 6), h_ref_loc_error(i, 6));
    EXPECT_EQ(h_loc_error(i, 7), h_ref_loc_error(i, 7));
  }
}

template <typename T>
void test_find_errors_8D_analytical(double rtol, double atol) {
  using View1DType     = Kokkos::View<T*>;
  using View8DType     = Kokkos::View<T********>;
  using CountViewType  = Kokkos::View<std::size_t**>;
  const std::size_t n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 3, n5 = 4, n6 = 2,
                    n7 = 1, nb_errors = 1;
  View8DType a("a", n0, n1, n2, n3, n4, n5, n6, n7),
      b("b", n0, n1, n2, n3, n4, n5, n6, n7);
  View1DType ref_a_error("ref_a_error", nb_errors),
      ref_b_error("ref_b_error", nb_errors);
  CountViewType ref_loc_error("ref_loc_error", nb_errors, 9);

  Kokkos::deep_copy(a, T(3.0));
  Kokkos::deep_copy(b, a);

  auto h_ref_a_error   = Kokkos::create_mirror_view(ref_a_error);
  auto h_ref_b_error   = Kokkos::create_mirror_view(ref_b_error);
  auto h_ref_loc_error = Kokkos::create_mirror_view(ref_loc_error);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

  // Initialization and prepare reference at host
  // b(0, 0) includes 3.00006 with bigger error than acceptance -> error count 1
  h_b(2, 1, 3, 1, 2, 0, 1, 0) =
      h_b(2, 1, 3, 1, 2, 0, 1, 0) + 2.0 * (h_b(2, 1, 3, 1, 2, 0, 1, 0) * rtol);

  h_ref_a_error(0) = T(3.0);
  h_ref_b_error(0) = h_b(2, 1, 3, 1, 2, 0, 1, 0);
  h_ref_loc_error(0, 0) =
      2 + 1 * b.extent(0) + 3 * b.extent(0) * b.extent(1) +
      1 * b.extent(0) * b.extent(1) * b.extent(2) +
      2 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) +
      0 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) * b.extent(4) +
      1 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) * b.extent(4) *
          b.extent(5) +
      0 * b.extent(0) * b.extent(1) * b.extent(2) * b.extent(3) * b.extent(4) *
          b.extent(5) * b.extent(6);  // global idx
  h_ref_loc_error(0, 1) = 2;          // idx of dimension 0
  h_ref_loc_error(0, 2) = 1;          // idx of dimension 1
  h_ref_loc_error(0, 3) = 3;          // idx of dimension 2
  h_ref_loc_error(0, 4) = 1;          // idx of dimension 3
  h_ref_loc_error(0, 5) = 2;          // idx of dimension 4
  h_ref_loc_error(0, 6) = 0;          // idx of dimension 5
  h_ref_loc_error(0, 7) = 1;          // idx of dimension 6
  h_ref_loc_error(0, 8) = 0;          // idx of dimension 7
  Kokkos::deep_copy(b, h_b);

  auto [b_error, a_error, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      execution_space(), b, a, nb_errors, rtol, atol);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < nb_errors; i++) {
    EXPECT_LT(Kokkos::abs(h_a_error(i) - h_ref_a_error(i)), epsilon);
    EXPECT_LT(Kokkos::abs(h_b_error(i) - h_ref_b_error(i)), epsilon);
    EXPECT_EQ(h_loc_error(i, 0), h_ref_loc_error(i, 0));
    EXPECT_EQ(h_loc_error(i, 1), h_ref_loc_error(i, 1));
    EXPECT_EQ(h_loc_error(i, 2), h_ref_loc_error(i, 2));
    EXPECT_EQ(h_loc_error(i, 3), h_ref_loc_error(i, 3));
    EXPECT_EQ(h_loc_error(i, 4), h_ref_loc_error(i, 4));
    EXPECT_EQ(h_loc_error(i, 5), h_ref_loc_error(i, 5));
    EXPECT_EQ(h_loc_error(i, 6), h_ref_loc_error(i, 6));
    EXPECT_EQ(h_loc_error(i, 7), h_ref_loc_error(i, 7));
    EXPECT_EQ(h_loc_error(i, 8), h_ref_loc_error(i, 8));
  }
}
}  // namespace

TYPED_TEST_SUITE(TestFindErrors, float_types);

TYPED_TEST(TestFindErrors, View1D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_1D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestFindErrors, View2D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_2D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestFindErrors, View3D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_3D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestFindErrors, View4D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_4D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestFindErrors, View5D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_5D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestFindErrors, View6D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_6D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestFindErrors, View7D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_7D_analytical<float_type>(this->m_rtol, this->m_atol);
}

TYPED_TEST(TestFindErrors, View8D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_8D_analytical<float_type>(this->m_rtol, this->m_atol);
}
