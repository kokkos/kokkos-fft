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

TYPED_TEST_SUITE(TestFindErrors, float_types);

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

TYPED_TEST(TestFindErrors, View1D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_1D_analytical<float_type>(this->m_rtol, this->m_atol);
}

/*
TYPED_TEST(TestFindErrors, View2D) {
  using float_type = typename TestFixture::float_type;
  test_find_errors_2D_analytical<float_type>(this->m_rtol, this->m_atol);
}
*/

}  // namespace
