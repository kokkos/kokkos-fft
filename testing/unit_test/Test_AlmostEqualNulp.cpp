// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_AlmostEqualNulp.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using float_types =
    ::testing::Types<Kokkos::Experimental::half_t,
                     Kokkos::Experimental::bhalf_t, float, double>;

template <typename T>
struct TestAlmostEqualNulp : public ::testing::Test {
  using float_type = T;
};

template <typename T>
void test_almost_equal_nulp_1D_analytical() {
  using View1DType = Kokkos::View<T*>;
  const int n      = 3;
  View1DType a("a", n), b("b", n), c("c", n), d("d", n);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);

  h_c(1) = nextafter_wrapper(h_b(1), T(2.0));
  h_d(1) = nextafter_wrapper(h_c(1), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

template <typename T>
void test_almost_equal_nulp_2D_analytical() {
  using View2DType = Kokkos::View<T**>;
  const int n0 = 3, n1 = 2;
  View2DType a("a", n0, n1), b("b", n0, n1), c("c", n0, n1), d("d", n0, n1);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);

  h_c(1, 0) = nextafter_wrapper(h_b(1, 0), T(2.0));
  h_d(0, 0) = nextafter_wrapper(h_b(0, 0), T(2.0));
  h_d(1, 0) = nextafter_wrapper(h_c(1, 0), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

template <typename T>
void test_almost_equal_nulp_3D_analytical() {
  using View3DType = Kokkos::View<T***>;
  const int n0 = 3, n1 = 2, n2 = 4;
  View3DType a("a", n0, n1, n2), b("b", n0, n1, n2), c("c", n0, n1, n2),
      d("d", n0, n1, n2);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);

  h_c(1, 0, 0) = nextafter_wrapper(h_b(1, 0, 0), T(2.0));
  h_d(0, 0, 0) = nextafter_wrapper(h_b(0, 0, 0), T(2.0));
  h_d(1, 0, 0) = nextafter_wrapper(h_c(1, 0, 0), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

template <typename T>
void test_almost_equal_nulp_4D_analytical() {
  using View4DType = Kokkos::View<T****>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5;
  View4DType a("a", n0, n1, n2, n3), b("b", n0, n1, n2, n3),
      c("c", n0, n1, n2, n3), d("d", n0, n1, n2, n3);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  ;

  h_c(1, 0, 0, 0) = nextafter_wrapper(h_b(1, 0, 0, 0), T(2.0));
  h_d(0, 0, 0, 0) = nextafter_wrapper(h_b(0, 0, 0, 0), T(2.0));
  h_d(1, 0, 2, 0) = nextafter_wrapper(h_c(1, 0, 0, 0), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

template <typename T>
void test_almost_equal_nulp_5D_analytical() {
  using View5DType = Kokkos::View<T*****>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6;
  View5DType a("a", n0, n1, n2, n3, n4), b("b", n0, n1, n2, n3, n4),
      c("c", n0, n1, n2, n3, n4), d("d", n0, n1, n2, n3, n4);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);

  h_c(1, 0, 0, 0, 0) = nextafter_wrapper(h_b(1, 0, 0, 0, 0), T(2.0));
  h_d(0, 0, 0, 0, 0) = nextafter_wrapper(h_b(0, 0, 0, 0, 0), T(2.0));
  h_d(1, 0, 2, 0, 0) = nextafter_wrapper(h_c(1, 0, 0, 0, 0), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

template <typename T>
void test_almost_equal_nulp_6D_analytical() {
  using View6DType = Kokkos::View<T******>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6, n5 = 3;
  View6DType a("a", n0, n1, n2, n3, n4, n5), b("b", n0, n1, n2, n3, n4, n5),
      c("c", n0, n1, n2, n3, n4, n5), d("d", n0, n1, n2, n3, n4, n5);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);

  h_c(1, 0, 0, 0, 0, 0) = nextafter_wrapper(h_b(1, 0, 0, 0, 0, 0), T(2.0));
  h_d(0, 0, 0, 0, 0, 0) = nextafter_wrapper(h_b(0, 0, 0, 0, 0, 0), T(2.0));
  h_d(1, 0, 2, 0, 0, 1) = nextafter_wrapper(h_c(1, 0, 0, 0, 0, 0), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

template <typename T>
void test_almost_equal_nulp_7D_analytical() {
  using View7DType = Kokkos::View<T*******>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6, n5 = 3, n6 = 3;
  View7DType a("a", n0, n1, n2, n3, n4, n5, n6),
      b("b", n0, n1, n2, n3, n4, n5, n6), c("c", n0, n1, n2, n3, n4, n5, n6),
      d("d", n0, n1, n2, n3, n4, n5, n6);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);

  h_c(1, 0, 0, 0, 0, 0, 0) =
      nextafter_wrapper(h_b(1, 0, 0, 0, 0, 0, 0), T(2.0));
  h_d(0, 0, 0, 0, 0, 0, 0) =
      nextafter_wrapper(h_b(0, 0, 0, 0, 0, 0, 0), T(2.0));
  h_d(1, 0, 2, 0, 0, 1, 2) =
      nextafter_wrapper(h_c(1, 0, 0, 0, 0, 0, 0), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

template <typename T>
void test_almost_equal_nulp_8D_analytical() {
  using View8DType = Kokkos::View<T********>;
  const int n0 = 3, n1 = 2, n2 = 4, n3 = 5, n4 = 6, n5 = 3, n6 = 3, n7 = 7;
  View8DType a("a", n0, n1, n2, n3, n4, n5, n6, n7),
      b("b", n0, n1, n2, n3, n4, n5, n6, n7),
      c("c", n0, n1, n2, n3, n4, n5, n6, n7),
      d("d", n0, n1, n2, n3, n4, n5, n6, n7);

  Kokkos::deep_copy(a, T(1.0));
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a);
  Kokkos::deep_copy(d, a);

  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);

  h_c(1, 0, 0, 0, 0, 0, 0, 0) =
      nextafter_wrapper(h_b(1, 0, 0, 0, 0, 0, 0, 0), T(2.0));
  h_d(0, 0, 0, 0, 0, 0, 0, 0) =
      nextafter_wrapper(h_b(0, 0, 0, 0, 0, 0, 0, 0), T(2.0));
  h_d(1, 0, 2, 0, 0, 1, 2, 0) =
      nextafter_wrapper(h_c(1, 0, 0, 0, 0, 0, 0, 0), T(2.0));

  Kokkos::deep_copy(c, h_c);
  Kokkos::deep_copy(d, h_d);

  // 0 ULP diff -> error count 0
  EXPECT_THAT(b, KokkosFFT::Testing::almost_equal_nulp(a, 0));

  // 1 ULP diff -> error count 0
  EXPECT_THAT(c, KokkosFFT::Testing::almost_equal_nulp(a));

  // 1 ULP diff + 2 ULP diff -> error count 1
  EXPECT_THAT(d, ::testing::Not(KokkosFFT::Testing::almost_equal_nulp(a, 1)));

  // 1 ULP diff + 2 ULP diff -> error count 0
  EXPECT_THAT(d, KokkosFFT::Testing::almost_equal_nulp(a, 2, 3));
}

}  // namespace

TYPED_TEST_SUITE(TestAlmostEqualNulp, float_types);

TYPED_TEST(TestAlmostEqualNulp, View1D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_1D_analytical<float_type>();
}

TYPED_TEST(TestAlmostEqualNulp, View2D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_2D_analytical<float_type>();
}

TYPED_TEST(TestAlmostEqualNulp, View3D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_3D_analytical<float_type>();
}

TYPED_TEST(TestAlmostEqualNulp, View4D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_4D_analytical<float_type>();
}

TYPED_TEST(TestAlmostEqualNulp, View5D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_5D_analytical<float_type>();
}

TYPED_TEST(TestAlmostEqualNulp, View6D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_6D_analytical<float_type>();
}

TYPED_TEST(TestAlmostEqualNulp, View7D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_7D_analytical<float_type>();
}

TYPED_TEST(TestAlmostEqualNulp, View8D) {
  using float_type = typename TestFixture::float_type;
  test_almost_equal_nulp_8D_analytical<float_type>();
}
