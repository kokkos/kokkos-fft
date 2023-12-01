#include <gtest/gtest.h>
#include "KokkosFFT_Plans.hpp"
#include "Test_Types.hpp"

TEST(Plans, 1DDefaultAxis) {
  const int n = 30;
  View1D<double> x("x", n);
  View1D<Kokkos::complex<double>> x_c("x_c", n/2+1);
  View1D<Kokkos::complex<double>> x_cin("x_cin", n), x_cout("x_cout", n);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c);

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD);
}

TEST(Plans, 2DLeftDefaultAxis) {
  const int n0 = 10, n1 = 6;
  LeftView2D<double> x("x", n0, n1);
  LeftView2D<Kokkos::complex<double>> x_c("x_c", n0/2+1, n1);
  LeftView2D<Kokkos::complex<double>> x_cin("x_cin", n0, n1), x_cout("x_cout", n0, n1);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c);

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD);
}

TEST(Plans, 2DRightDefaultAxis) {
  const int n0 = 10, n1 = 6;
  RightView2D<double> x("x", n0, n1);
  RightView2D<Kokkos::complex<double>> x_c("x_c", n0/2+1, n1);
  RightView2D<Kokkos::complex<double>> x_cin("x_cin", n0, n1), x_cout("x_cout", n0, n1);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c);

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD);
}

TEST(Plans, 3DLeftDefaultAxis) {
  const int n0 = 10, n1 = 6, n2 = 8;
  LeftView3D<double> x("x", n0, n1, n2);
  LeftView3D<Kokkos::complex<double>> x_c("x_c", n0/2+1, n1, n2);
  LeftView3D<Kokkos::complex<double>> x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c);

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD);
}

TEST(Plans, 3DRightDefaultAxis) {
  const int n0 = 10, n1 = 6, n2 = 8;
  RightView3D<double> x("x", n0, n1, n2);
  RightView3D<Kokkos::complex<double>> x_c("x_c", n0/2+1, n1, n2);
  RightView3D<Kokkos::complex<double>> x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c);

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD);
}

TEST(Plans, 1DFFT_2DRightDefaultAxis) {
  const int n0 = 10, n1 = 6;
  LeftView2D<double> x("x", n0, n1);
  LeftView2D<Kokkos::complex<double>> x_c("x_c", n0, n1/2+1);
  LeftView2D<Kokkos::complex<double>> x_cin("x_cin", n0, n1), x_cout("x_cout", n0, n1);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c, -1);

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x, -1);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD, -1);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD, -1);
}

TEST(Plans, 1DFFT_3DRightDefaultAxis) {
  const int n0 = 10, n1 = 6, n2 = 8;
  LeftView3D<double> x("x", n0, n1, n2);
  LeftView3D<Kokkos::complex<double>> x_c("x_c", n0, n1, n2/2+1);
  LeftView3D<Kokkos::complex<double>> x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c, -1);

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x, -1);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD, -1);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD, -1);
}

TEST(Plans, 2DFFT_3DRightDefaultAxis) {
  using axes_type = KokkosFFT::axis_type<2>;

  const int n0 = 10, n1 = 6, n2 = 8;
  LeftView3D<double> x("x", n0, n1, n2);
  LeftView3D<Kokkos::complex<double>> x_c("x_c", n0, n1, n2/2+1);
  LeftView3D<Kokkos::complex<double>> x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Plan plan_r2c(x, x_c, axes_type({-2, -1}));

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x, axes_type({-2, -1}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD, axes_type({-2, -1}));
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD, axes_type({-2, -1}));
}