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
  KokkosFFT::Plan plan_r2c(x, x_c);
  KokkosFFT::Plan plan_r2c_axis_0(x, x_c, /*axis=*/0);
  KokkosFFT::Plan plan_r2c_axes_0(x, x_c, /*axes=*/axes_type<1>({0}));

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c, x);
  KokkosFFT::Plan plan_c2r_axis0(x_c, x, /*axis=*/0);
  KokkosFFT::Plan plan_c2r_axes0(x_c, x, /*axes=*/axes_type<1>({0}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD);
  KokkosFFT::Plan plan_c2c_f_axis0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Plan plan_c2c_f_axes0(x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axes=*/axes_type<1>({0}));

  EXPECT_THROW(
    {
      KokkosFFT::Plan plan_c2c(x_cin, x_cout);
    },
    std::runtime_error
  );
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
  KokkosFFT::Plan plan_r2c(x, x_c_axis_1);
  KokkosFFT::Plan plan_r2c_axes_0_1(x, x_c_axis_1, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_r2c_axes_1_0(x, x_c_axis_0, /*axes=*/axes_type<2>({1, 0}));

  // C2R plan
  KokkosFFT::Plan plan_c2r(x_c_axis_1, x);
  KokkosFFT::Plan plan_c2r_axes_0_1(x_c_axis_1, x, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2r_axes_1_0(x_c_axis_0, x, /*axes=*/axes_type<2>({1, 0}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f(x_cin, x_cout, KOKKOS_FFT_FORWARD);
  KokkosFFT::Plan plan_c2c_b(x_cin, x_cout, KOKKOS_FFT_BACKWARD);
  KokkosFFT::Plan plan_c2c_f_axes_0_1(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_1_0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 0}));

  EXPECT_THROW(
    {
      KokkosFFT::Plan plan_c2c(x_cin, x_cout);
    },
    std::runtime_error
  );

  EXPECT_THROW(
    {
      KokkosFFT::Plan plan_c2c_axes_0_1(x_cin, x_cout, /*axes=*/axes_type<2>({0, 1}));
    },
    std::runtime_error
  );

  EXPECT_THROW(
    {
      KokkosFFT::Plan plan_c2c_axes_1_0(x_cin, x_cout, /*axes=*/axes_type<2>({1, 0}));
    },
    std::runtime_error
  );
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
  KokkosFFT::Plan plan_r2c_axes_0_1_2(x, x_c_axis_2, /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Plan plan_r2c_axes_0_2_1(x, x_c_axis_1, /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Plan plan_r2c_axes_1_0_2(x, x_c_axis_2, /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Plan plan_r2c_axes_1_2_0(x, x_c_axis_0, /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Plan plan_r2c_axes_2_0_1(x, x_c_axis_1, /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Plan plan_r2c_axes_2_1_0(x, x_c_axis_0, /*axes=*/axes_type<3>({2, 1, 0}));

  // C2R plan
  KokkosFFT::Plan plan_c2r_axes_0_1_2(x_c_axis_2, x, /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Plan plan_c2r_axes_0_2_1(x_c_axis_1, x, /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Plan plan_c2r_axes_1_0_2(x_c_axis_2, x, /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Plan plan_c2r_axes_1_2_0(x_c_axis_0, x, /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Plan plan_c2r_axes_2_0_1(x_c_axis_1, x, /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Plan plan_c2r_axes_2_1_0(x_c_axis_0, x, /*axes=*/axes_type<3>({2, 1, 0}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axes_0_1_2(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_0_2_1(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_1_0_2(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_1_2_0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Plan plan_c2c_f_axes_2_0_1(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_2_1_0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<3>({2, 1, 0}));

  EXPECT_THROW(
    {
      KokkosFFT::Plan plan_c2c(x_cin, x_cout);
    },
    std::runtime_error
  );

  EXPECT_THROW(
    {
      KokkosFFT::Plan plan_c2c_axes_0_1_2(x_cin, x_cout, /*axes=*/axes_type<3>({0, 1, 2}));
    },
    std::runtime_error
  );
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
  KokkosFFT::Plan plan_r2c_axis_0(x, x_c_axis_0, /*axis=*/0);
  KokkosFFT::Plan plan_r2c_axis_1(x, x_c_axis_1, /*axis=*/1);
  KokkosFFT::Plan plan_r2c_axis_minus1(x, x_c_axis_1, /*axis=*/-1);

  // C2R plan
  KokkosFFT::Plan plan_c2r_axis_0(x_c_axis_0, x, /*axis=*/0);
  KokkosFFT::Plan plan_c2r_axis_1(x_c_axis_1, x, /*axis=*/1);
  KokkosFFT::Plan plan_c2r_axis_minus1(x_c_axis_1, x, /*axis=*/-1);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axis_0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Plan plan_c2c_f_axis_1(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/1);

  // [TO DO] Fix this, this can be instanized with explicit Plan(InViewType& in, OutViewType& out, FFTDirectionType direction)
  // Because FFTDirectionType is int for most libraries
  //EXPECT_THROW(
  //  {
  //    KokkosFFT::Plan plan_c2c_axis_0(x_cin, x_cout, /*axis=*/0);
  //  },
  //  std::runtime_error
  //);

  //EXPECT_THROW(
  //  {
  //    KokkosFFT::Plan plan_c2c_axis_1(x_cin, x_cout, /*axis=*/1);
  //  },
  //  std::runtime_error
  //);
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
  KokkosFFT::Plan plan_r2c_axis_0(x, x_c_axis_0, /*axis=*/0);
  KokkosFFT::Plan plan_r2c_axis_1(x, x_c_axis_1, /*axis=*/1);
  KokkosFFT::Plan plan_r2c_axis_2(x, x_c_axis_2, /*axis=*/2);

  // C2R plan
  KokkosFFT::Plan plan_c2r_axis_0(x_c_axis_0, x, /*axis=*/0);
  KokkosFFT::Plan plan_c2r_axis_1(x_c_axis_1, x, /*axis=*/1);
  KokkosFFT::Plan plan_c2r_axis_2(x_c_axis_2, x, /*axis=*/2);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axis_0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/0);
  KokkosFFT::Plan plan_c2c_f_axis_1(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/1);
  KokkosFFT::Plan plan_c2c_f_axis_2(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axis=*/2);
  KokkosFFT::Plan plan_c2c_b_axis_0(x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axis=*/0);
  KokkosFFT::Plan plan_c2c_b_axis_1(x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axis=*/1);
  KokkosFFT::Plan plan_c2c_b_axis_2(x_cin, x_cout, KOKKOS_FFT_BACKWARD, /*axis=*/2);

  // [TO DO] Fix this, this can be instanized with explicit Plan(InViewType& in, OutViewType& out, FFTDirectionType direction)
  // Because FFTDirectionType is int for most libraries
  //EXPECT_THROW(
  //  {
  //    KokkosFFT::Plan plan_c2c_axis_0(x_cin, x_cout, /*axis=*/0);
  //  },
  //  std::runtime_error
  //);

  //EXPECT_THROW(
  //  {
  //    KokkosFFT::Plan plan_c2c_axis_1(x_cin, x_cout, /*axis=*/1);
  //  },
  //  std::runtime_error
  //);

  //EXPECT_THROW(
  //  {
  //    KokkosFFT::Plan plan_c2c_axis_1(x_cin, x_cout, /*axis=*/2);
  //  },
  //  std::runtime_error
  //);
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
  KokkosFFT::Plan plan_r2c_axes_0_1(x, x_c_axis_1, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_r2c_axes_0_2(x, x_c_axis_2, /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Plan plan_r2c_axes_1_0(x, x_c_axis_0, /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Plan plan_r2c_axes_1_2(x, x_c_axis_2, /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Plan plan_r2c_axes_2_0(x, x_c_axis_0, /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Plan plan_r2c_axes_2_1(x, x_c_axis_1, /*axes=*/axes_type<2>({2, 1}));

  // C2R plan
  KokkosFFT::Plan plan_c2r_axes_0_1(x_c_axis_1, x, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2r_axes_0_2(x_c_axis_2, x, /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Plan plan_c2r_axes_1_0(x_c_axis_0, x, /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Plan plan_c2r_axes_1_2(x_c_axis_2, x, /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Plan plan_c2r_axes_2_0(x_c_axis_0, x, /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Plan plan_c2r_axes_2_1(x_c_axis_1, x, /*axes=*/axes_type<2>({2, 1}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axes_0_1(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_0_2(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_1_0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Plan plan_c2c_f_axes_1_2(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_2_0(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Plan plan_c2c_f_axes_2_1(x_cin, x_cout, KOKKOS_FFT_FORWARD, /*axes=*/axes_type<2>({2, 1}));

  EXPECT_THROW(
    {
      KokkosFFT::Plan plan_c2c_axes_0_1(x_cin, x_cout, /*axes=*/axes_type<2>({0, 1}));
    },
    std::runtime_error
  );
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