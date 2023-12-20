#include <gtest/gtest.h>
#include "KokkosFFT_Plans.hpp"
#include "Test_Types.hpp"

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

template <typename T, typename LayoutType>
void test_plan_1dfft_1dview() {
  const int n = 30;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType = Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  RealView1DType x("x", n);
  ComplexView1DType x_c("x_c", n/2+1);
  ComplexView1DType x_cin("x_cin", n), x_cout("x_cout", n);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axis_0(execution_space(), x, x_c, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_r2c_axes_0(execution_space(), x, x_c, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<1>({0}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axis0(execution_space(), x_c, x, KOKKOS_FFT_BACKWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2r_axes0(execution_space(), x_c, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<1>({0}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axis0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_f_axes0(execution_space(), x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<1>({0}));
}

TEST(Plans, 1DFFT_1DLeftViewFloat) {
  test_plan_1dfft_1dview<float, Kokkos::LayoutLeft>();
}

TEST(Plans, 1DFFT_1DLeftViewDouble) {
  test_plan_1dfft_1dview<double, Kokkos::LayoutLeft>();
}

TEST(Plans, 1DFFT_1DRightViewFloat) {
  test_plan_1dfft_1dview<float, Kokkos::LayoutRight>();
}

TEST(Plans, 1DFFT_1DRightViewDouble) {
  test_plan_1dfft_1dview<double, Kokkos::LayoutRight>();
}

template <typename T, typename LayoutType>
void test_plan_2dfft_2dview() {
  const int n0 = 10, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  RealView2DType x("x", n0, n1);
  ComplexView2DType x_c_axis_0("x_c_axis_0", n0/2+1, n1), x_c_axis_1("x_c_axis_1", n0, n1/2+1);
  ComplexView2DType x_cin("x_cin", n0, n1), x_cout("x_cout", n0, n1);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axes_0_1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_0(execution_space(), x, x_c_axis_0, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 0}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axes_0_1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_0(execution_space(), x_c_axis_0, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({1, 0}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 0}));
}

TEST(Plans, 2DFFT_2DLeftViewFloat) {
  test_plan_2dfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(Plans, 2DFFT_2DLeftViewDouble) {
  test_plan_2dfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(Plans, 2DFFT_2DRightViewFloat) {
  test_plan_2dfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(Plans, 2DFFT_2DRightViewDouble) {
  test_plan_2dfft_2dview<double, Kokkos::LayoutRight>();
}

template <typename T, typename LayoutType>
void test_plan_3dfft_3dview() {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2);
  ComplexView3DType x_c_axis_0("x_c_axis_0", n0/2+1, n1, n2), x_c_axis_1("x_c_axis_1", n0, n1/2+1, n2), x_c_axis_2("x_c_axis_2", n0, n1, n2/2+1);
  ComplexView3DType x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axes_0_1_2(execution_space(), x, x_c_axis_2, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_0_2_1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_0_2(execution_space(), x, x_c_axis_2, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_2_0(execution_space(), x, x_c_axis_0, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_0_1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_1_0(execution_space(), x, x_c_axis_0, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({2, 1, 0}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axes_0_1_2(execution_space(), x_c_axis_2, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_0_2_1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_0_2(execution_space(), x_c_axis_2, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_2_0(execution_space(), x_c_axis_0, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_0_1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_1_0(execution_space(), x_c_axis_0, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<3>({2, 1, 0}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_1_2(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_2_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_0_2(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_2_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_0_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_1_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({2, 1, 0}));
}

TEST(Plans, 3DFFT_3DLeftViewFloat) {
  test_plan_3dfft_3dview<float, Kokkos::LayoutLeft>();
}

TEST(Plans, 3DFFT_3DLeftViewDouble) {
  test_plan_3dfft_3dview<double, Kokkos::LayoutLeft>();
}

TEST(Plans, 3DFFT_3DRightViewFloat) {
  test_plan_3dfft_3dview<float, Kokkos::LayoutRight>();
}

TEST(Plans, 3DFFT_3DRightViewDouble) {
  test_plan_3dfft_3dview<double, Kokkos::LayoutRight>();
}

template <typename T, typename LayoutType>
void test_plan_1dfft_2dview() {
  const int n0 = 10, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  RealView2DType x("x", n0, n1);
  ComplexView2DType x_c_axis_0("x_c_axis_0", n0/2+1, n1), x_c_axis_1("x_c_axis_1", n0, n1/2+1);
  ComplexView2DType x_cin("x_cin", n0, n1), x_cout("x_cout", n0, n1);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axis_0(execution_space(), x, x_c_axis_0, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_r2c_axis_1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axis=*/1);
  KokkosFFT::Impl::Plan plan_r2c_axis_minus1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axis=*/-1);

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axis_0(execution_space(), x_c_axis_0, x, KOKKOS_FFT_BACKWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2r_axis_1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2r_axis_minus1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axis=*/-1);

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axis_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_f_axis_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/1);
}

TEST(Plans, 1DBatchedFFT_2DLeftViewFloat) {
  test_plan_1dfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(Plans, 1DBatchedFFT_2DLeftViewDouble) {
  test_plan_1dfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(Plans, 1DBatchedFFT_2DRightViewFloat) {
  test_plan_1dfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(Plans, 1DBatchedFFT_2DRightViewDouble) {
  test_plan_1dfft_2dview<double, Kokkos::LayoutRight>();
}

template <typename T, typename LayoutType>
void test_plan_1dfft_3dview() {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2);
  ComplexView3DType x_c_axis_0("x_c_axis_0", n0/2+1, n1, n2), x_c_axis_1("x_c_axis_1", n0, n1/2+1, n2), x_c_axis_2("x_c_axis_2", n0, n1, n2/2+1);
  ComplexView3DType x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axis_0(execution_space(), x, x_c_axis_0, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_r2c_axis_1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axis=*/1);
  KokkosFFT::Impl::Plan plan_r2c_axis_2(execution_space(), x, x_c_axis_2, KOKKOS_FFT_FORWARD, /*axis=*/2);

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axis_0(execution_space(), x_c_axis_0, x, KOKKOS_FFT_BACKWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2r_axis_1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2r_axis_2(execution_space(), x_c_axis_2, x, KOKKOS_FFT_BACKWARD, /*axis=*/2);

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axis_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_f_axis_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2c_f_axis_2(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/2);
  KokkosFFT::Impl::Plan plan_c2c_b_axis_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_b_axis_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2c_b_axis_2(execution_space(), x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axis=*/2);
}

TEST(Plans, 1DBatchedFFT_3DLeftViewFloat) {
  test_plan_1dfft_3dview<float, Kokkos::LayoutLeft>();
}

TEST(Plans, 1DBatchedFFT_3DLeftViewDouble) {
  test_plan_1dfft_3dview<double, Kokkos::LayoutLeft>();
}

TEST(Plans, 1DBatchedFFT_3DRightViewFloat) {
  test_plan_1dfft_3dview<float, Kokkos::LayoutRight>();
}

TEST(Plans, 1DBatchedFFT_3DRightViewDouble) {
  test_plan_1dfft_3dview<double, Kokkos::LayoutRight>();
}

template <typename T, typename LayoutType>
void test_plan_2dfft_3dview() {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2);
  ComplexView3DType x_c_axis_0("x_c_axis_0", n0/2+1, n1, n2), x_c_axis_1("x_c_axis_1", n0, n1/2+1, n2), x_c_axis_2("x_c_axis_2", n0, n1, n2/2+1);
  ComplexView3DType x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axes_0_1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_0_2(execution_space(), x, x_c_axis_2, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_0(execution_space(), x, x_c_axis_0, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_2(execution_space(), x, x_c_axis_2, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_0(execution_space(), x, x_c_axis_0, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_1(execution_space(), x, x_c_axis_1, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({2, 1}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axes_0_1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_0_2(execution_space(), x_c_axis_2, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_0(execution_space(), x_c_axis_0, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_2(execution_space(), x_c_axis_2, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_0(execution_space(), x_c_axis_0, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_1(execution_space(), x_c_axis_1, x, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<2>({2, 1}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_2(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_2(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_0(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_1(execution_space(), x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({2, 1}));
}

TEST(Plans, 2DBatchedFFT_3DLeftViewFloat) {
  test_plan_2dfft_3dview<float, Kokkos::LayoutLeft>();
}

TEST(Plans, 2DBatchedFFT_3DLeftViewDouble) {
  test_plan_2dfft_3dview<double, Kokkos::LayoutLeft>();
}

TEST(Plans, 2DBatchedFFT_3DRightViewFloat) {
  test_plan_2dfft_3dview<float, Kokkos::LayoutRight>();
}

TEST(Plans, 2DBatchedFFT_3DRightViewDouble) {
  test_plan_2dfft_3dview<double, Kokkos::LayoutRight>();
}