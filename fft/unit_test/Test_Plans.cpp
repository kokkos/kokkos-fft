// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_Plans.hpp"

namespace {
#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
using execution_space = Kokkos::DefaultExecutionSpace;
#else
using execution_space = Kokkos::DefaultHostExecutionSpace;
#endif

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight> >;

#if defined(KOKKOS_ENABLE_SERIAL)
using execution_spaces =
    ::testing::Types<Kokkos::Serial, Kokkos::DefaultHostExecutionSpace,
                     Kokkos::DefaultExecutionSpace>;
#else
using execution_spaces = ::testing::Types<Kokkos::DefaultHostExecutionSpace,
                                          Kokkos::DefaultExecutionSpace>;
#endif

template <typename T>
struct ExecutionSpaceType : public ::testing::Test {
  using execution_space_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

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

// Tests for execution space
template <typename ExecutionSpace>
void test_allowed_exec_space() {
#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
  // A plan can be constructible from Kokkos::DefaultExecutionSpace,
  // Kokkos::DefaultHostExecutionSpace or Kokkos::Serial (if enabled)
  static_assert(KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>);
#else
  // Only device backend library is available
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    static_assert(KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>);
  } else {
    static_assert(!KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>);
  }
#endif
#else
  // Only host backend library is available
  // If device libraries are not enabled, at least FFTW is enabled
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
  // A plan can only be constructible from HostSpace
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    static_assert(!KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>);
  } else {
    static_assert(KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>);
  }
#else
  // A plan can be constructible from HostSpace
  static_assert(KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>);
#endif
#endif
}

// Tests for execution space
template <typename ExecutionSpace>
void test_plan_constructible() {
  using ValueType      = double;
  using RealView1DType = Kokkos::View<ValueType*, ExecutionSpace>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<ValueType>*, ExecutionSpace>;
  using PlanType =
      KokkosFFT::Plan<ExecutionSpace, RealView1DType, ComplexView1DType>;

#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
  // A plan can be constructible from Kokkos::DefaultExecutionSpace,
  // Kokkos::DefaultHostExecutionSpace or Kokkos::Serial (if enabled)
  static_assert(
      std::is_constructible_v<PlanType, const ExecutionSpace&, RealView1DType&,
                              ComplexView1DType&, KokkosFFT::Direction, int>);
#else
  // Only device backend library is available
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    static_assert(std::is_constructible_v<PlanType, const ExecutionSpace&,
                                          RealView1DType&, ComplexView1DType&,
                                          KokkosFFT::Direction, int>);
  }
#endif
#else
  // Only host backend library is available
  // If device libraries are not enabled, at least FFTW is enabled
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
  // A plan can only be constructible from HostSpace
  if constexpr (!std::is_same_v<ExecutionSpace,
                                Kokkos::DefaultExecutionSpace>) {
    static_assert(std::is_constructible_v<PlanType, const ExecutionSpace&,
                                          RealView1DType&, ComplexView1DType&,
                                          KokkosFFT::Direction, int>);
  }
#else
  // A plan can be constructible from HostSpace
  static_assert(
      std::is_constructible_v<PlanType, const ExecutionSpace&, RealView1DType&,
                              ComplexView1DType&, KokkosFFT::Direction, int>);
#endif
#endif
}

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
  KokkosFFT::Plan plan_r2c_axis_0(execution_space(), x, x_c,
                                  KokkosFFT::Direction::forward,
                                  /*axis=*/0);
  KokkosFFT::Plan plan_r2c_axes_0(execution_space(), x, x_c,
                                  KokkosFFT::Direction::forward,
                                  /*axes=*/axes_type<1>({0}));

  // C2R plan
  KokkosFFT::Plan plan_c2r_axis0(execution_space(), x_c, x,
                                 KokkosFFT::Direction::backward,
                                 /*axis=*/0);
  KokkosFFT::Plan plan_c2r_axes0(execution_space(), x_c, x,
                                 KokkosFFT::Direction::backward,
                                 /*axes=*/axes_type<1>({0}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axis0(execution_space(), x_cin, x_cout,
                                   KokkosFFT::Direction::forward,
                                   /*axis=*/0);
  KokkosFFT::Plan plan_c2c_f_axes0(execution_space(), x_cin, x_cout,
                                   KokkosFFT::Direction::backward,
                                   /*axes=*/axes_type<1>({0}));

  // Check if errors are correctly raised against wrong direction
  // Input Real, Output Complex -> must be forward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axis0(execution_space(), x, x_c,
                                             KokkosFFT::Direction::backward,
                                             /*axis=*/0);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes0(execution_space(), x, x_c,
                                             KokkosFFT::Direction::backward,
                                             /*axes=*/axes_type<1>({0}));
      },
      std::runtime_error);

  // Input Complex, Output Real -> must be backward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axis0(execution_space(), x_c, x,
                                             KokkosFFT::Direction::forward,
                                             /*axis=*/0);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes0(execution_space(), x_c, x,
                                             KokkosFFT::Direction::forward,
                                             /*axes=*/axes_type<1>({0}));
      },
      std::runtime_error);
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
  KokkosFFT::Plan plan_r2c_axis_0(execution_space(), x, x_c_axis_0,
                                  KokkosFFT::Direction::forward,
                                  /*axis=*/0);
  KokkosFFT::Plan plan_r2c_axis_1(execution_space(), x, x_c_axis_1,
                                  KokkosFFT::Direction::forward,
                                  /*axis=*/1);
  KokkosFFT::Plan plan_r2c_axis_minus1(execution_space(), x, x_c_axis_1,
                                       KokkosFFT::Direction::forward,
                                       /*axis=*/-1);

  // C2R plan
  KokkosFFT::Plan plan_c2r_axis_0(execution_space(), x_c_axis_0, x,
                                  KokkosFFT::Direction::backward,
                                  /*axis=*/0);
  KokkosFFT::Plan plan_c2r_axis_1(execution_space(), x_c_axis_1, x,
                                  KokkosFFT::Direction::backward,
                                  /*axis=*/1);
  KokkosFFT::Plan plan_c2r_axis_minus1(execution_space(), x_c_axis_1, x,
                                       KokkosFFT::Direction::backward,
                                       /*axis=*/-1);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axis_0(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::forward,
                                    /*axis=*/0);
  KokkosFFT::Plan plan_c2c_f_axis_1(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::forward,
                                    /*axis=*/1);

  // Check if errors are correctly raised against wrong direction
  // Input Real, Output Complex -> must be forward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axis_0(execution_space(), x, x_c_axis_0,
                                              KokkosFFT::Direction::backward,
                                              /*axis=*/0);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axis_1(execution_space(), x, x_c_axis_1,
                                              KokkosFFT::Direction::backward,
                                              /*axis=*/1);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axis_minus1(
            execution_space(), x, x_c_axis_1, KokkosFFT::Direction::backward,
            /*axis=*/-1);
      },
      std::runtime_error);

  // Input Complex, Output Real -> must be backward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axis_0(execution_space(), x_c_axis_0, x,
                                              KokkosFFT::Direction::forward,
                                              /*axis=*/0);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axis_1(execution_space(), x_c_axis_1, x,
                                              KokkosFFT::Direction::forward,
                                              /*axis=*/1);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axis_minus1(
            execution_space(), x_c_axis_1, x, KokkosFFT::Direction::forward,
            /*axis=*/-1);
      },
      std::runtime_error);
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
  KokkosFFT::Plan plan_r2c_axis_0(execution_space(), x, x_c_axis_0,
                                  KokkosFFT::Direction::forward,
                                  /*axis=*/0);
  KokkosFFT::Plan plan_r2c_axis_1(execution_space(), x, x_c_axis_1,
                                  KokkosFFT::Direction::forward,
                                  /*axis=*/1);
  KokkosFFT::Plan plan_r2c_axis_2(execution_space(), x, x_c_axis_2,
                                  KokkosFFT::Direction::forward,
                                  /*axis=*/2);

  // C2R plan
  KokkosFFT::Plan plan_c2r_axis_0(execution_space(), x_c_axis_0, x,
                                  KokkosFFT::Direction::backward,
                                  /*axis=*/0);
  KokkosFFT::Plan plan_c2r_axis_1(execution_space(), x_c_axis_1, x,
                                  KokkosFFT::Direction::backward,
                                  /*axis=*/1);
  KokkosFFT::Plan plan_c2r_axis_2(execution_space(), x_c_axis_2, x,
                                  KokkosFFT::Direction::backward,
                                  /*axis=*/2);

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axis_0(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::forward,
                                    /*axis=*/0);
  KokkosFFT::Plan plan_c2c_f_axis_1(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::forward,
                                    /*axis=*/1);
  KokkosFFT::Plan plan_c2c_f_axis_2(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::forward,
                                    /*axis=*/2);
  KokkosFFT::Plan plan_c2c_b_axis_0(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::backward,
                                    /*axis=*/0);
  KokkosFFT::Plan plan_c2c_b_axis_1(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::backward,
                                    /*axis=*/1);
  KokkosFFT::Plan plan_c2c_b_axis_2(execution_space(), x_cin, x_cout,
                                    KokkosFFT::Direction::backward,
                                    /*axis=*/2);

  // Check if errors are correctly raised against wrong direction
  // Input Real, Output Complex -> must be forward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axis_0(execution_space(), x, x_c_axis_0,
                                              KokkosFFT::Direction::backward,
                                              /*axis=*/0);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axis_1(execution_space(), x, x_c_axis_1,
                                              KokkosFFT::Direction::backward,
                                              /*axis=*/1);
      },
      std::runtime_error);

  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axis_2(execution_space(), x, x_c_axis_2,
                                              KokkosFFT::Direction::backward,
                                              /*axis=*/2);
      },
      std::runtime_error);

  // Input Complex, Output Real -> must be backward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axis_0(execution_space(), x_c_axis_0, x,
                                              KokkosFFT::Direction::forward,
                                              /*axis=*/0);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axis_1(execution_space(), x_c_axis_1, x,
                                              KokkosFFT::Direction::forward,
                                              /*axis=*/1);
      },
      std::runtime_error);

  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axis_2(execution_space(), x_c_axis_2, x,
                                              KokkosFFT::Direction::forward,
                                              /*axis=*/2);
      },
      std::runtime_error);
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
  KokkosFFT::Plan plan_r2c_axes_0_1(execution_space(), x, x_c_axis_1,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_r2c_axes_1_0(execution_space(), x, x_c_axis_0,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({1, 0}));

  // C2R plan
  KokkosFFT::Plan plan_c2r_axes_0_1(execution_space(), x_c_axis_1, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2r_axes_1_0(execution_space(), x_c_axis_0, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({1, 0}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axes_0_1(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_1_0(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({1, 0}));

  // Check if errors are correctly raised against wrong direction
  // Input Real, Output Complex -> must be forward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_0_1(
            execution_space(), x, x_c_axis_1, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({0, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_1_0(
            execution_space(), x, x_c_axis_0, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({1, 0}));
      },
      std::runtime_error);

  // Input Complex, Output Real -> must be backward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_0_1(
            execution_space(), x_c_axis_1, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({0, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_1_0(
            execution_space(), x_c_axis_0, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({1, 0}));
      },
      std::runtime_error);
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
  KokkosFFT::Plan plan_r2c_axes_0_1(execution_space(), x, x_c_axis_1,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_r2c_axes_0_2(execution_space(), x, x_c_axis_2,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Plan plan_r2c_axes_1_0(execution_space(), x, x_c_axis_0,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Plan plan_r2c_axes_1_2(execution_space(), x, x_c_axis_2,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Plan plan_r2c_axes_2_0(execution_space(), x, x_c_axis_0,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Plan plan_r2c_axes_2_1(execution_space(), x, x_c_axis_1,
                                    KokkosFFT::Direction::forward,
                                    /*axes=*/axes_type<2>({2, 1}));

  // C2R plan
  KokkosFFT::Plan plan_c2r_axes_0_1(execution_space(), x_c_axis_1, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2r_axes_0_2(execution_space(), x_c_axis_2, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Plan plan_c2r_axes_1_0(execution_space(), x_c_axis_0, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Plan plan_c2r_axes_1_2(execution_space(), x_c_axis_2, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Plan plan_c2r_axes_2_0(execution_space(), x_c_axis_0, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Plan plan_c2r_axes_2_1(execution_space(), x_c_axis_1, x,
                                    KokkosFFT::Direction::backward,
                                    /*axes=*/axes_type<2>({2, 1}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axes_0_1(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({0, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_0_2(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({0, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_1_0(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({1, 0}));
  KokkosFFT::Plan plan_c2c_f_axes_1_2(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({1, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_2_0(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({2, 0}));
  KokkosFFT::Plan plan_c2c_f_axes_2_1(execution_space(), x_cin, x_cout,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<2>({2, 1}));

  // Check if errors are correctly raised against wrong direction
  // Input Real, Output Complex -> must be forward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_0_1(
            execution_space(), x, x_c_axis_1, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({0, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_0_2(
            execution_space(), x, x_c_axis_2, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({0, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_1_0(
            execution_space(), x, x_c_axis_0, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({1, 0}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_1_2(
            execution_space(), x, x_c_axis_2, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({1, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_2_0(
            execution_space(), x, x_c_axis_0, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({2, 0}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_2_1(
            execution_space(), x, x_c_axis_1, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<2>({2, 1}));
      },
      std::runtime_error);

  // Input Complex, Output Real -> must be backward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_0_1(
            execution_space(), x_c_axis_1, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({0, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_0_2(
            execution_space(), x_c_axis_2, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({0, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_1_0(
            execution_space(), x_c_axis_0, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({1, 0}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_1_2(
            execution_space(), x_c_axis_2, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({1, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_2_0(
            execution_space(), x_c_axis_0, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({2, 0}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_2_1(
            execution_space(), x_c_axis_1, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<2>({2, 1}));
      },
      std::runtime_error);
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
  KokkosFFT::Plan plan_r2c_axes_0_1_2(execution_space(), x, x_c_axis_2,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Plan plan_r2c_axes_0_2_1(execution_space(), x, x_c_axis_1,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Plan plan_r2c_axes_1_0_2(execution_space(), x, x_c_axis_2,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Plan plan_r2c_axes_1_2_0(execution_space(), x, x_c_axis_0,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Plan plan_r2c_axes_2_0_1(execution_space(), x, x_c_axis_1,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Plan plan_r2c_axes_2_1_0(execution_space(), x, x_c_axis_0,
                                      KokkosFFT::Direction::forward,
                                      /*axes=*/axes_type<3>({2, 1, 0}));

  // C2R plan
  KokkosFFT::Plan plan_c2r_axes_0_1_2(execution_space(), x_c_axis_2, x,
                                      KokkosFFT::Direction::backward,
                                      /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Plan plan_c2r_axes_0_2_1(execution_space(), x_c_axis_1, x,
                                      KokkosFFT::Direction::backward,
                                      /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Plan plan_c2r_axes_1_0_2(execution_space(), x_c_axis_2, x,
                                      KokkosFFT::Direction::backward,
                                      /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Plan plan_c2r_axes_1_2_0(execution_space(), x_c_axis_0, x,
                                      KokkosFFT::Direction::backward,
                                      /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Plan plan_c2r_axes_2_0_1(execution_space(), x_c_axis_1, x,
                                      KokkosFFT::Direction::backward,
                                      /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Plan plan_c2r_axes_2_1_0(execution_space(), x_c_axis_0, x,
                                      KokkosFFT::Direction::backward,
                                      /*axes=*/axes_type<3>({2, 1, 0}));

  // C2C plan
  KokkosFFT::Plan plan_c2c_f_axes_0_1_2(execution_space(), x_cin, x_cout,
                                        KokkosFFT::Direction::forward,
                                        /*axes=*/axes_type<3>({0, 1, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_0_2_1(execution_space(), x_cin, x_cout,
                                        KokkosFFT::Direction::forward,
                                        /*axes=*/axes_type<3>({0, 2, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_1_0_2(execution_space(), x_cin, x_cout,
                                        KokkosFFT::Direction::forward,
                                        /*axes=*/axes_type<3>({1, 0, 2}));
  KokkosFFT::Plan plan_c2c_f_axes_1_2_0(execution_space(), x_cin, x_cout,
                                        KokkosFFT::Direction::forward,
                                        /*axes=*/axes_type<3>({1, 2, 0}));
  KokkosFFT::Plan plan_c2c_f_axes_2_0_1(execution_space(), x_cin, x_cout,
                                        KokkosFFT::Direction::forward,
                                        /*axes=*/axes_type<3>({2, 0, 1}));
  KokkosFFT::Plan plan_c2c_f_axes_2_1_0(execution_space(), x_cin, x_cout,
                                        KokkosFFT::Direction::forward,
                                        /*axes=*/axes_type<3>({2, 1, 0}));

  // Check if errors are correctly raised against wrong direction
  // Input Real, Output Complex -> must be forward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_0_1_2(
            execution_space(), x, x_c_axis_2, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<3>({0, 1, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_0_2_1(
            execution_space(), x, x_c_axis_1, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<3>({0, 2, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_1_0_2(
            execution_space(), x, x_c_axis_2, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<3>({1, 0, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_1_2_0(
            execution_space(), x, x_c_axis_0, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<3>({1, 2, 0}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_2_0_1(
            execution_space(), x, x_c_axis_1, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<3>({2, 0, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_r2c_axes_2_1_0(
            execution_space(), x, x_c_axis_0, KokkosFFT::Direction::backward,
            /*axes=*/axes_type<3>({2, 1, 0}));
      },
      std::runtime_error);

  // Input Complex, Output Real -> must be backward plan
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_0_1_2(
            execution_space(), x_c_axis_2, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<3>({0, 1, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_0_2_1(
            execution_space(), x_c_axis_1, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<3>({0, 2, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_1_0_2(
            execution_space(), x_c_axis_2, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<3>({1, 0, 2}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_1_2_0(
            execution_space(), x_c_axis_0, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<3>({1, 2, 0}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_2_0_1(
            execution_space(), x_c_axis_1, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<3>({2, 0, 1}));
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        KokkosFFT::Plan wrong_plan_c2r_axes_2_1_0(
            execution_space(), x_c_axis_0, x, KokkosFFT::Direction::forward,
            /*axes=*/axes_type<3>({2, 1, 0}));
      },
      std::runtime_error);
}

}  // namespace

TYPED_TEST_SUITE(ExecutionSpaceType, execution_spaces);
TYPED_TEST_SUITE(Plans1D, test_types);
TYPED_TEST_SUITE(Plans2D, test_types);
TYPED_TEST_SUITE(Plans3D, test_types);

// Tests for plan constructiblility
TYPED_TEST(ExecutionSpaceType, is_allowed_space) {
  using execution_space_type = typename TestFixture::execution_space_type;
  test_allowed_exec_space<execution_space_type>();
}

// Tests for 1D FFT plan on 1D View
TYPED_TEST(ExecutionSpaceType, is_constrcutrible) {
  using execution_space_type = typename TestFixture::execution_space_type;
  test_plan_constructible<execution_space_type>();
}

// Tests for 1D FFT plan on 1D View
TYPED_TEST(Plans1D, 1DFFT_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_1dfft_1dview<float_type, layout_type>();
}

// Tests for 1D batched FFT plan on 2D View
TYPED_TEST(Plans1D, 1DFFT_batched_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_1dfft_2dview<float_type, layout_type>();
}

// Tests for 1D batched FFT plan on 3D View
TYPED_TEST(Plans1D, 1DFFT_batched_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_1dfft_3dview<float_type, layout_type>();
}

// Tests for 2D FFT plan on 2D View
TYPED_TEST(Plans2D, 2DFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_2dfft_2dview<float_type, layout_type>();
}

// Tests for 2D batched FFT plan on 3D View
TYPED_TEST(Plans2D, 2DFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_2dfft_3dview<float_type, layout_type>();
}

// Tests for 3D FFT plan on 3D View
TYPED_TEST(Plans3D, 3DFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan_3dfft_3dview<float_type, layout_type>();
}
