#include <algorithm>
#include <random>
#include <gtest/gtest.h>
#include "KokkosFFT_Plans.hpp"
#include "Test_Types.hpp"

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight> >;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct Plans1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct Plans2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct Plans3D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct Plans4D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct Plans5D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct Plans6D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct Plans7D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct Plans8D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

TYPED_TEST_SUITE(Plans1D, test_types);
TYPED_TEST_SUITE(Plans2D, test_types);
TYPED_TEST_SUITE(Plans3D, test_types);
TYPED_TEST_SUITE(Plans4D, test_types);
TYPED_TEST_SUITE(Plans5D, test_types);
TYPED_TEST_SUITE(Plans6D, test_types);
TYPED_TEST_SUITE(Plans7D, test_types);
TYPED_TEST_SUITE(Plans8D, test_types);

// Tests for 1D FFT Plans
template <typename T, typename LayoutType>
void test_plan_1dfft_1dview() {
  const int n          = 30;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  RealView1DType x("x", n);
  ComplexView1DType x_c("x_c", n / 2 + 1);
  ComplexView1DType x_cin("x_cin", n), x_cout("x_cout", n);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axis_0(execution_space(), x, x_c,
                                        KokkosFFT::Impl::Direction::Forward,
                                        /*axis=*/0);
  KokkosFFT::Impl::Plan plan_r2c_axes_0(execution_space(), x, x_c,
                                        KokkosFFT::Impl::Direction::Forward,
                                        /*axes=*/axes_type<1>({0}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axis0(execution_space(), x_c, x,
                                       KokkosFFT::Impl::Direction::Backward,
                                       /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2r_axes0(execution_space(), x_c, x,
                                       KokkosFFT::Impl::Direction::Backward,
                                       /*axes=*/axes_type<1>({0}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axis0(execution_space(), x_cin, x_cout,
                                         KokkosFFT::Impl::Direction::Forward,
                                         /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_f_axes0(execution_space(), x_cin, x_cout,
                                         KokkosFFT::Impl::Direction::Backward,
                                         /*axes=*/axes_type<1>({0}));
}

template <typename T, typename LayoutType>
void test_plan_1dfft_2dview() {
  const int n0 = 10, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  RealView2DType x("x", n0, n1);
  ComplexView2DType x_c_axis_0("x_c_axis_0", n0 / 2 + 1, n1),
      x_c_axis_1("x_c_axis_1", n0, n1 / 2 + 1);
  ComplexView2DType x_cin("x_cin", n0, n1), x_cout("x_cout", n0, n1);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axis_0(execution_space(), x, x_c_axis_0,
                                        KokkosFFT::Impl::Direction::Forward,
                                        /*axis=*/0);
  KokkosFFT::Impl::Plan plan_r2c_axis_1(execution_space(), x, x_c_axis_1,
                                        KokkosFFT::Impl::Direction::Forward,
                                        /*axis=*/1);
  KokkosFFT::Impl::Plan plan_r2c_axis_minus1(
      execution_space(), x, x_c_axis_1, KokkosFFT::Impl::Direction::Forward,
      /*axis=*/-1);

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axis_0(execution_space(), x_c_axis_0, x,
                                        KokkosFFT::Impl::Direction::Backward,
                                        /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2r_axis_1(execution_space(), x_c_axis_1, x,
                                        KokkosFFT::Impl::Direction::Backward,
                                        /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2r_axis_minus1(
      execution_space(), x_c_axis_1, x, KokkosFFT::Impl::Direction::Backward,
      /*axis=*/-1);

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axis_0(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_f_axis_1(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axis=*/1);
}

template <typename T, typename LayoutType>
void test_plan_1dfft_3dview() {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2);
  ComplexView3DType x_c_axis_0("x_c_axis_0", n0 / 2 + 1, n1, n2),
      x_c_axis_1("x_c_axis_1", n0, n1 / 2 + 1, n2),
      x_c_axis_2("x_c_axis_2", n0, n1, n2 / 2 + 1);
  ComplexView3DType x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axis_0(execution_space(), x, x_c_axis_0,
                                        KokkosFFT::Impl::Direction::Forward,
                                        /*axis=*/0);
  KokkosFFT::Impl::Plan plan_r2c_axis_1(execution_space(), x, x_c_axis_1,
                                        KokkosFFT::Impl::Direction::Forward,
                                        /*axis=*/1);
  KokkosFFT::Impl::Plan plan_r2c_axis_2(execution_space(), x, x_c_axis_2,
                                        KokkosFFT::Impl::Direction::Forward,
                                        /*axis=*/2);

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axis_0(execution_space(), x_c_axis_0, x,
                                        KokkosFFT::Impl::Direction::Backward,
                                        /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2r_axis_1(execution_space(), x_c_axis_1, x,
                                        KokkosFFT::Impl::Direction::Backward,
                                        /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2r_axis_2(execution_space(), x_c_axis_2, x,
                                        KokkosFFT::Impl::Direction::Backward,
                                        /*axis=*/2);

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axis_0(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_f_axis_1(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2c_f_axis_2(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axis=*/2);
  KokkosFFT::Impl::Plan plan_c2c_b_axis_0(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axis=*/0);
  KokkosFFT::Impl::Plan plan_c2c_b_axis_1(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axis=*/1);
  KokkosFFT::Impl::Plan plan_c2c_b_axis_2(execution_space(), x_cin, x_cout,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axis=*/2);
}

// Tests for 1D FFT plan on 1D View
TYPED_TEST(Plans1D, FFT1D_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_1dfft_1dview<float_type, layout_type>();
}

// Tests for 1D batched FFT plan on 2D View
TYPED_TEST(Plans1D, FFT1D_batched_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_1dfft_2dview<float_type, layout_type>();
}

// Tests for 1D batched FFT plan on 3D View
TYPED_TEST(Plans1D, FFT1D_batched_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_1dfft_3dview<float_type, layout_type>();
}

// Tests for 2D FFT Plans
template <typename T, typename LayoutType>
void test_plan_2dfft_2dview() {
  const int n0 = 10, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  RealView2DType x("x", n0, n1);
  ComplexView2DType x_c_axis_0("x_c_axis_0", n0 / 2 + 1, n1),
      x_c_axis_1("x_c_axis_1", n0, n1 / 2 + 1);
  ComplexView2DType x_cin("x_cin", n0, n1), x_cout("x_cout", n0, n1);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axes_0_1(execution_space(), x, x_c_axis_1,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_0(execution_space(), x, x_c_axis_0,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({1, 0}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axes_0_1(execution_space(), x_c_axis_1, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_0(execution_space(), x_c_axis_0, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({1, 0}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_1(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_0(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({1, 0}));
}

template <typename T, typename LayoutType>
void test_plan_2dfft_3dview() {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2);
  ComplexView3DType x_c_axis_0("x_c_axis_0", n0 / 2 + 1, n1, n2),
      x_c_axis_1("x_c_axis_1", n0, n1 / 2 + 1, n2),
      x_c_axis_2("x_c_axis_2", n0, n1, n2 / 2 + 1);
  ComplexView3DType x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axes_0_1(execution_space(), x, x_c_axis_1,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_0_2(execution_space(), x, x_c_axis_2,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_0(execution_space(), x, x_c_axis_0,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_2(execution_space(), x, x_c_axis_2,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_0(execution_space(), x, x_c_axis_0,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_1(execution_space(), x, x_c_axis_1,
                                          KokkosFFT::Impl::Direction::Forward,
                                          /*axes=*/axes_type<2>({2, 1}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axes_0_1(execution_space(), x_c_axis_1, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_0_2(execution_space(), x_c_axis_2, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_0(execution_space(), x_c_axis_0, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_2(execution_space(), x_c_axis_2, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_0(execution_space(), x_c_axis_0, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_1(execution_space(), x_c_axis_1, x,
                                          KokkosFFT::Impl::Direction::Backward,
                                          /*axes=*/axes_type<2>({2, 1}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_1(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_2(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_0(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_2(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_0(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_1(execution_space(), x_cin, x_cout,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<2>({2, 1}));
}

// Tests for 2D FFT plan on 2D View
TYPED_TEST(Plans2D, FFT2D_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_2dfft_2dview<float_type, layout_type>();
}

// Tests for 2D batched FFT plan on 3D View
TYPED_TEST(Plans2D, FFT2D_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_2dfft_3dview<float_type, layout_type>();
}

// Tests for 3D FFT Plans
template <typename T, typename LayoutType>
void test_plan_3dfft_3dview() {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2);
  ComplexView3DType x_c_axis_0("x_c_axis_0", n0 / 2 + 1, n1, n2),
      x_c_axis_1("x_c_axis_1", n0, n1 / 2 + 1, n2),
      x_c_axis_2("x_c_axis_2", n0, n1, n2 / 2 + 1);
  ComplexView3DType x_cin("x_cin", n0, n1, n2), x_cout("x_cout", n0, n1, n2);

  // R2C plan
  KokkosFFT::Impl::Plan plan_r2c_axes_0_1_2(execution_space(), x, x_c_axis_2,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_0_2_1(execution_space(), x, x_c_axis_1,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_0_2(execution_space(), x, x_c_axis_2,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Impl::Plan plan_r2c_axes_1_2_0(execution_space(), x, x_c_axis_0,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_0_1(execution_space(), x, x_c_axis_1,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Impl::Plan plan_r2c_axes_2_1_0(execution_space(), x, x_c_axis_0,
                                            KokkosFFT::Impl::Direction::Forward,
                                            /*axes=*/axes_type<3>({2, 1, 0}));

  // C2R plan
  KokkosFFT::Impl::Plan plan_c2r_axes_0_1_2(
      execution_space(), x_c_axis_2, x, KokkosFFT::Impl::Direction::Backward,
      /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_0_2_1(
      execution_space(), x_c_axis_1, x, KokkosFFT::Impl::Direction::Backward,
      /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_0_2(
      execution_space(), x_c_axis_2, x, KokkosFFT::Impl::Direction::Backward,
      /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Impl::Plan plan_c2r_axes_1_2_0(
      execution_space(), x_c_axis_0, x, KokkosFFT::Impl::Direction::Backward,
      /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_0_1(
      execution_space(), x_c_axis_1, x, KokkosFFT::Impl::Direction::Backward,
      /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Impl::Plan plan_c2r_axes_2_1_0(
      execution_space(), x_c_axis_0, x, KokkosFFT::Impl::Direction::Backward,
      /*axes=*/axes_type<3>({2, 1, 0}));

  // C2C plan
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_1_2(
      execution_space(), x_cin, x_cout, KokkosFFT::Impl::Direction::Forward,
      /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_0_2_1(
      execution_space(), x_cin, x_cout, KokkosFFT::Impl::Direction::Forward,
      /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_0_2(
      execution_space(), x_cin, x_cout, KokkosFFT::Impl::Direction::Forward,
      /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_1_2_0(
      execution_space(), x_cin, x_cout, KokkosFFT::Impl::Direction::Forward,
      /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_0_1(
      execution_space(), x_cin, x_cout, KokkosFFT::Impl::Direction::Forward,
      /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Impl::Plan plan_c2c_f_axes_2_1_0(
      execution_space(), x_cin, x_cout, KokkosFFT::Impl::Direction::Forward,
      /*axes=*/axes_type<3>({2, 1, 0}));
}

// Tests for 3D FFT plan on 3D View
TYPED_TEST(Plans3D, FFT3D_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_3dfft_3dview<float_type, layout_type>();
}

// Tests for 4D FFT Plans
template <typename T, typename LayoutType>
void test_plan_4dfft_4dview() {
  constexpr std::size_t DIM = 4;
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5;
  using RealView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<T>****, LayoutType, execution_space>;

  RealView4DType x("x", n0, n1, n2, n3);
  ComplexView4DType x_cin("x_cin", n0, n1, n2, n3),
      x_cout("x_cout", n0, n1, n2, n3);

  std::vector<axes_type<DIM> > list_of_tested_axes = {
      axes_type<DIM>({0, 1, 2, 3}), axes_type<DIM>({0, 1, 3, 2}),
      axes_type<DIM>({0, 2, 1, 3}), axes_type<DIM>({0, 2, 3, 1}),
      axes_type<DIM>({0, 3, 1, 2}), axes_type<DIM>({0, 3, 2, 1}),
      axes_type<DIM>({1, 0, 2, 3}), axes_type<DIM>({1, 0, 3, 2}),
      axes_type<DIM>({1, 2, 0, 3}), axes_type<DIM>({1, 2, 3, 0}),
      axes_type<DIM>({1, 3, 0, 2}), axes_type<DIM>({1, 3, 2, 0}),
      axes_type<DIM>({2, 0, 1, 3}), axes_type<DIM>({2, 0, 3, 1}),
      axes_type<DIM>({2, 1, 0, 3}), axes_type<DIM>({2, 1, 3, 0}),
      axes_type<DIM>({2, 3, 0, 1}), axes_type<DIM>({2, 3, 1, 0}),
      axes_type<DIM>({3, 0, 1, 2}), axes_type<DIM>({3, 0, 2, 1}),
      axes_type<DIM>({3, 1, 0, 2}), axes_type<DIM>({3, 1, 2, 0}),
      axes_type<DIM>({3, 2, 0, 1}), axes_type<DIM>({3, 2, 1, 0})};

  std::array<int, DIM> in_shape = {n0, n1, n2, n3};
  for (auto& tested_axes : list_of_tested_axes) {
    int fft_axis                   = tested_axes[DIM - 1];
    std::array<int, DIM> out_shape = in_shape;
    out_shape.at(fft_axis)         = out_shape.at(fft_axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3]      = out_shape;

    ComplexView4DType x_c("x_c", _n0, _n1, _n2, _n3);

    // R2C plan
    KokkosFFT::Impl::Plan plan_r2c(execution_space(), x, x_c,
                                   KokkosFFT::Impl::Direction::Forward,
                                   /*axes=*/tested_axes);

    // C2R plan
    KokkosFFT::Impl::Plan plan_c2r(execution_space(), x_c, x,
                                   KokkosFFT::Impl::Direction::Backward,
                                   /*axes=*/tested_axes);

    // C2C plan
    KokkosFFT::Impl::Plan plan_c2c_f(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Forward,
                                     /*axes=*/tested_axes);

    KokkosFFT::Impl::Plan plan_c2c_b(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Backward,
                                     /*axes=*/tested_axes);
  }
}

// Tests for 4D FFT plan on 4D View
TYPED_TEST(Plans4D, FFT4D_4DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_4dfft_4dview<float_type, layout_type>();
}

// Tests for 5D FFT Plans
template <typename T, typename LayoutType>
void test_plan_5dfft_5dview() {
  constexpr std::size_t DIM = 5;
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 3;
  using RealView5DType = Kokkos::View<T*****, LayoutType, execution_space>;
  using ComplexView5DType =
      Kokkos::View<Kokkos::complex<T>*****, LayoutType, execution_space>;

  RealView5DType x("x", n0, n1, n2, n3, n4);
  ComplexView5DType x_cin("x_cin", n0, n1, n2, n3, n4),
      x_cout("x_cout", n0, n1, n2, n3, n4);

  // Too much combinations, choose axes randomly
  axes_type<DIM> default_axes({0, 1, 2, 3, 4});
  std::vector<axes_type<DIM> > list_of_tested_axes;

  constexpr int nb_trials = 100;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    axes_type<DIM> tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);
    list_of_tested_axes.push_back(tmp_axes);
  }

  std::array<int, DIM> in_shape = {n0, n1, n2, n3, n4};
  for (auto& tested_axes : list_of_tested_axes) {
    int fft_axis                   = tested_axes[DIM - 1];
    std::array<int, DIM> out_shape = in_shape;
    out_shape.at(fft_axis)         = out_shape.at(fft_axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4] = out_shape;

    ComplexView5DType x_c("x_c", _n0, _n1, _n2, _n3, _n4);

    // R2C plan
    KokkosFFT::Impl::Plan plan_r2c(execution_space(), x, x_c,
                                   KokkosFFT::Impl::Direction::Forward,
                                   /*axes=*/tested_axes);

    // C2R plan
    KokkosFFT::Impl::Plan plan_c2r(execution_space(), x_c, x,
                                   KokkosFFT::Impl::Direction::Backward,
                                   /*axes=*/tested_axes);

    // C2C plan
    KokkosFFT::Impl::Plan plan_c2c_f(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Forward,
                                     /*axes=*/tested_axes);

    KokkosFFT::Impl::Plan plan_c2c_b(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Backward,
                                     /*axes=*/tested_axes);
  }
}

// Tests for 5D FFT plan on 5D View
TYPED_TEST(Plans5D, FFT5D_5DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_5dfft_5dview<float_type, layout_type>();
}

// Tests for 6D FFT Plans
template <typename T, typename LayoutType>
void test_plan_6dfft_6dview() {
  constexpr std::size_t DIM = 6;
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 3, n5 = 2;
  using RealView6DType = Kokkos::View<T******, LayoutType, execution_space>;
  using ComplexView6DType =
      Kokkos::View<Kokkos::complex<T>******, LayoutType, execution_space>;

  RealView6DType x("x", n0, n1, n2, n3, n4, n5);
  ComplexView6DType x_cin("x_cin", n0, n1, n2, n3, n4, n5),
      x_cout("x_cout", n0, n1, n2, n3, n4, n5);

  // Too much combinations, choose axes randomly
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5});
  std::vector<axes_type<DIM> > list_of_tested_axes;

  constexpr int nb_trials = 32;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    axes_type<DIM> tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);
    list_of_tested_axes.push_back(tmp_axes);
  }

  std::array<int, DIM> in_shape = {n0, n1, n2, n3, n4, n5};
  for (auto& tested_axes : list_of_tested_axes) {
    int fft_axis                        = tested_axes[DIM - 1];
    std::array<int, DIM> out_shape      = in_shape;
    out_shape.at(fft_axis)              = out_shape.at(fft_axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4, _n5] = out_shape;

    ComplexView6DType x_c("x_c", _n0, _n1, _n2, _n3, _n4, _n5);

    // R2C plan
    KokkosFFT::Impl::Plan plan_r2c(execution_space(), x, x_c,
                                   KokkosFFT::Impl::Direction::Forward,
                                   /*axes=*/tested_axes);

    // C2R plan
    KokkosFFT::Impl::Plan plan_c2r(execution_space(), x_c, x,
                                   KokkosFFT::Impl::Direction::Backward,
                                   /*axes=*/tested_axes);

    // C2C plan
    KokkosFFT::Impl::Plan plan_c2c_f(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Forward,
                                     /*axes=*/tested_axes);

    KokkosFFT::Impl::Plan plan_c2c_b(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Backward,
                                     /*axes=*/tested_axes);
  }
}

// Tests for 6D FFT plan on 6D View
TYPED_TEST(Plans6D, FFT6D_6DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_6dfft_6dview<float_type, layout_type>();
}

// Tests for 7D FFT Plans
template <typename T, typename LayoutType>
void test_plan_7dfft_7dview() {
  constexpr std::size_t DIM = 7;
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 3, n5 = 2, n6 = 4;
  using RealView7DType = Kokkos::View<T*******, LayoutType, execution_space>;
  using ComplexView7DType =
      Kokkos::View<Kokkos::complex<T>*******, LayoutType, execution_space>;

  RealView7DType x("x", n0, n1, n2, n3, n4, n5, n6);
  ComplexView7DType x_cin("x_cin", n0, n1, n2, n3, n4, n5, n6),
      x_cout("x_cout", n0, n1, n2, n3, n4, n5, n6);

  // Too much combinations, choose axes randomly
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6});
  std::vector<axes_type<DIM> > list_of_tested_axes;

  constexpr int nb_trials = 32;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    axes_type<DIM> tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);
    list_of_tested_axes.push_back(tmp_axes);
  }

  std::array<int, DIM> in_shape = {n0, n1, n2, n3, n4, n5, n6};
  for (auto& tested_axes : list_of_tested_axes) {
    int fft_axis                             = tested_axes[DIM - 1];
    std::array<int, DIM> out_shape           = in_shape;
    out_shape.at(fft_axis)                   = out_shape.at(fft_axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6] = out_shape;

    ComplexView7DType x_c("x_c", _n0, _n1, _n2, _n3, _n4, _n5, _n6);

    // R2C plan
    KokkosFFT::Impl::Plan plan_r2c(execution_space(), x, x_c,
                                   KokkosFFT::Impl::Direction::Forward,
                                   /*axes=*/tested_axes);

    // C2R plan
    KokkosFFT::Impl::Plan plan_c2r(execution_space(), x_c, x,
                                   KokkosFFT::Impl::Direction::Backward,
                                   /*axes=*/tested_axes);

    // C2C plan
    KokkosFFT::Impl::Plan plan_c2c_f(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Forward,
                                     /*axes=*/tested_axes);

    KokkosFFT::Impl::Plan plan_c2c_b(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Backward,
                                     /*axes=*/tested_axes);
  }
}

// Tests for 7D FFT plan on 7D View
TYPED_TEST(Plans7D, FFT7D_7DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_7dfft_7dview<float_type, layout_type>();
}

// Tests for 8D FFT Plans
template <typename T, typename LayoutType>
void test_plan_8dfft_8dview() {
  constexpr std::size_t DIM = 8;
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 3, n5 = 2, n6 = 4, n7 = 7;
  using RealView8DType = Kokkos::View<T********, LayoutType, execution_space>;
  using ComplexView8DType =
      Kokkos::View<Kokkos::complex<T>********, LayoutType, execution_space>;

  RealView8DType x("x", n0, n1, n2, n3, n4, n5, n6, n7);
  ComplexView8DType x_cin("x_cin", n0, n1, n2, n3, n4, n5, n6, n7),
      x_cout("x_cout", n0, n1, n2, n3, n4, n5, n6, n7);

  // Too much combinations, choose axes randomly
  axes_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6, 7});
  std::vector<axes_type<DIM> > list_of_tested_axes;

  constexpr int nb_trials = 32;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    axes_type<DIM> tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);
    list_of_tested_axes.push_back(tmp_axes);
  }

  std::array<int, DIM> in_shape = {n0, n1, n2, n3, n4, n5, n6, n7};
  for (auto& tested_axes : list_of_tested_axes) {
    int fft_axis                   = tested_axes[DIM - 1];
    std::array<int, DIM> out_shape = in_shape;
    out_shape.at(fft_axis)         = out_shape.at(fft_axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7] = out_shape;

    ComplexView8DType x_c("x_c", _n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7);

    // R2C plan
    KokkosFFT::Impl::Plan plan_r2c(execution_space(), x, x_c,
                                   KokkosFFT::Impl::Direction::Forward,
                                   /*axes=*/tested_axes);

    // C2R plan
    KokkosFFT::Impl::Plan plan_c2r(execution_space(), x_c, x,
                                   KokkosFFT::Impl::Direction::Backward,
                                   /*axes=*/tested_axes);

    // C2C plan
    KokkosFFT::Impl::Plan plan_c2c_f(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Forward,
                                     /*axes=*/tested_axes);

    KokkosFFT::Impl::Plan plan_c2c_b(execution_space(), x_cin, x_cout,
                                     KokkosFFT::Impl::Direction::Backward,
                                     /*axes=*/tested_axes);
  }
}

// Tests for 8D FFT plan on 8D View
TYPED_TEST(Plans8D, FFT8D_8DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_8dfft_8dview<float_type, layout_type>();
}