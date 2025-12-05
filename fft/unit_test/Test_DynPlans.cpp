// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_Plans.hpp"
#include "KokkosFFT_DynPlans.hpp"
#include "KokkosFFT_Transform.hpp"
#include "Test_Utils.hpp"

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
struct CompileTestDynPlan : public ::testing::Test {
  using execution_space_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestGetWorkSpaceSize : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct TestDynPlans1D : public ::testing::Test {
  using float_type   = typename T::first_type;
  using complex_type = Kokkos::complex<float_type>;
  using layout_type  = typename T::second_type;
};

template <typename T>
struct TestDynPlans2D : public ::testing::Test {
  using float_type   = typename T::first_type;
  using complex_type = Kokkos::complex<float_type>;
  using layout_type  = typename T::second_type;
};

template <typename T>
struct TestDynPlans3D : public ::testing::Test {
  using float_type   = typename T::first_type;
  using complex_type = Kokkos::complex<float_type>;
  using layout_type  = typename T::second_type;
};

template <typename LayoutType, std::size_t DIM, std::size_t FFT_DIM>
auto get_in_out_extents(const bool is_R2C) {
  std::array<std::size_t, DIM> in_extents{};
  for (std::size_t i = 0; i < in_extents.size(); i++) {
    in_extents.at(i) = i % 2 == 0 ? 6 : 5;
  }

  int inner_most_axis =
      std::is_same_v<LayoutType, typename Kokkos::LayoutLeft> ? 0 : (DIM - 1);

  std::array<std::size_t, DIM> out_extents = in_extents;
  out_extents.at(inner_most_axis) = KokkosFFT::Impl::extent_after_transform(
      in_extents.at(inner_most_axis), is_R2C);

  if constexpr (FFT_DIM == 1) {
    return std::make_tuple(in_extents, out_extents, inner_most_axis);
  } else {
    KokkosFFT::axis_type<FFT_DIM> axes;
    if constexpr (std::is_same_v<LayoutType, typename Kokkos::LayoutLeft>) {
      axes = KokkosFFT::Impl::index_sequence<int, FFT_DIM, 0>();
      std::reverse(axes.begin(), axes.end());
    } else {
      axes = KokkosFFT::Impl::index_sequence<int, FFT_DIM,
                                             -static_cast<int>(FFT_DIM)>();
    }
    return std::make_tuple(in_extents, out_extents, axes);
  }
}

// Tests for execution space
template <typename ExecutionSpace, typename InValueType, typename OutValueType>
void test_dynplan_constructible() {
  using InView1DType  = Kokkos::View<InValueType*, ExecutionSpace>;
  using OutView1DType = Kokkos::View<OutValueType*, ExecutionSpace>;
  using DynPlanType =
      KokkosFFT::DynPlan<ExecutionSpace, InView1DType, OutView1DType>;

#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
  // A plan can be constructible from Kokkos::DefaultExecutionSpace,
  // Kokkos::DefaultHostExecutionSpace or Kokkos::Serial (if enabled)
  static_assert(
      std::is_constructible_v<DynPlanType, const ExecutionSpace&, InView1DType&,
                              OutView1DType&, KokkosFFT::Direction, int>);
#else
  // Only device backend library is available
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    static_assert(std::is_constructible_v<DynPlanType, const ExecutionSpace&,
                                          InView1DType&, OutView1DType&,
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
    static_assert(std::is_constructible_v<DynPlanType, const ExecutionSpace&,
                                          InView1DType&, OutView1DType&,
                                          KokkosFFT::Direction, int>);
  }
#else
  // A plan can be constructible from HostSpace
  static_assert(
      std::is_constructible_v<DynPlanType, const ExecutionSpace&, InView1DType&,
                              OutView1DType&, KokkosFFT::Direction, int>);
#endif
#endif
}

template <typename T, typename LayoutType, std::size_t DIM, std::size_t FFT_DIM>
void test_get_required_workspace_size() {
  using complex_type   = Kokkos::complex<T>;
  using view_data_type = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using complex_view_data_type =
      KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using ViewType = Kokkos::View<view_data_type, LayoutType, execution_space>;
  using ComplexViewType =
      Kokkos::View<complex_view_data_type, LayoutType, execution_space>;
  [[maybe_unused]] auto [in_extents, out_extents, axes] =
      get_in_out_extents<LayoutType, DIM, FFT_DIM>(true);

  // Make a reference with a basic-API
  auto in_layout  = KokkosFFT::Impl::create_layout<LayoutType>(in_extents);
  auto out_layout = KokkosFFT::Impl::create_layout<LayoutType>(out_extents);
  ViewType u("u", in_layout), u_inv("u_inv", in_layout),
      u_ref("u_ref", in_layout);
  ComplexViewType u_hat("u_hat", out_layout),
      u_hat_ref("u_hat_ref", out_layout);

  // Calculate with DynPlan
  execution_space exec;
  KokkosFFT::DynPlan plan_r2c(exec, u, u_hat, KokkosFFT::Direction::forward,
                              FFT_DIM);
  KokkosFFT::DynPlan plan_c2r(exec, u_hat, u_inv,
                              KokkosFFT::Direction::backward, FFT_DIM);
  KokkosFFT::DynPlan plan_c2c(exec, u_hat, u_hat, KokkosFFT::Direction::forward,
                              FFT_DIM);

  // Using aligned data
  auto workspace_size_r2c =
      KokkosFFT::compute_required_workspace_size<complex_type>(plan_r2c);
  auto workspace_size_r2c_c2r =
      KokkosFFT::compute_required_workspace_size<complex_type>(plan_r2c,
                                                               plan_c2r);
  auto workspace_size =
      KokkosFFT::compute_required_workspace_size<complex_type>(
          plan_r2c, plan_c2r, plan_c2c);

  // Compute reference
  auto ws_r2c = plan_r2c.workspace_size(sizeof(complex_type));
  auto ws_c2r = plan_c2r.workspace_size(sizeof(complex_type));
  auto ws_c2c = plan_c2c.workspace_size(sizeof(complex_type));

  auto ref_workspace_size_r2c     = ws_r2c;
  auto ref_workspace_size_r2c_c2r = std::max(ws_r2c, ws_c2r);
  auto ref_workspace_size         = std::max({ws_r2c, ws_c2r, ws_c2c});

  EXPECT_EQ(workspace_size_r2c, ref_workspace_size_r2c);
  EXPECT_EQ(workspace_size_r2c_c2r, ref_workspace_size_r2c_c2r);
  EXPECT_EQ(workspace_size, ref_workspace_size);
}

template <typename T, typename LayoutType, std::size_t DIM, std::size_t FFT_DIM>
void test_dynplan() {
  using float_type     = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type   = Kokkos::complex<float_type>;
  using view_data_type = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using complex_view_data_type =
      KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using ViewType = Kokkos::View<view_data_type, LayoutType, execution_space>;
  using ComplexViewType =
      Kokkos::View<complex_view_data_type, LayoutType, execution_space>;
  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;
  auto [in_extents, out_extents, axes] =
      get_in_out_extents<LayoutType, DIM, FFT_DIM>(is_R2C);

  // Make a reference with a basic-API
  auto in_layout  = KokkosFFT::Impl::create_layout<LayoutType>(in_extents);
  auto out_layout = KokkosFFT::Impl::create_layout<LayoutType>(out_extents);
  ViewType u("u", in_layout), u_inv("u_inv", in_layout),
      u_ref("u_ref", in_layout);
  ComplexViewType u_hat("u_hat", out_layout),
      u_hat_ref("u_hat_ref", out_layout);

  // Initialization
  execution_space exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(exec, u, random_pool, 1.0);
  exec.fence();
  Kokkos::deep_copy(u_ref, u);

  KokkosFFT::Plan plan(exec, u, u_hat_ref, KokkosFFT::Direction::forward, axes);
  KokkosFFT::execute(plan, u, u_hat_ref, KokkosFFT::Normalization::backward);

  // Since HIP FFT destructs the input data, we need to recover the input data
  Kokkos::deep_copy(u, u_ref);

  // Calculate with DynPlan
  KokkosFFT::DynPlan plan_f(exec, u, u_hat, KokkosFFT::Direction::forward,
                            FFT_DIM);
  KokkosFFT::DynPlan plan_b(exec, u_hat, u_inv, KokkosFFT::Direction::backward,
                            FFT_DIM);

  // Using aligned data
  auto workspace_size =
      KokkosFFT::compute_required_workspace_size<complex_type>(plan_f, plan_b);

  // Allocate a 1D data buffer and set work areas
  using BufferType = Kokkos::View<complex_type*, execution_space>;
  BufferType buffer("buffer", workspace_size);
  plan_f.set_work_area(buffer);
  plan_b.set_work_area(buffer);

  auto atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  KokkosFFT::execute(plan_f, u, u_hat);
  exec.fence();
  EXPECT_TRUE(allclose(exec, u_hat, u_hat_ref, 1.e-5, atol));

#if !defined(KOKKOS_ENABLE_SYCL)
  // FIXME_SYCL: if we touch buffer, it gives the wrong result
  Kokkos::fill_random(exec, buffer, random_pool, 1.0);
#endif

  KokkosFFT::execute(plan_b, u_hat, u_inv);
  exec.fence();
  EXPECT_TRUE(allclose(exec, u_inv, u_ref, 1.e-5, atol));

  // Check if errors are correctly raised against too small buffer size
  if (workspace_size > 0) {
    BufferType small_buffer("small_buffer", workspace_size - 1);
    EXPECT_THROW({ plan_f.set_work_area(small_buffer); }, std::runtime_error);

    EXPECT_THROW({ plan_b.set_work_area(small_buffer); }, std::runtime_error);
  }
}

}  // namespace

TYPED_TEST_SUITE(CompileTestDynPlan, execution_spaces);
TYPED_TEST_SUITE(TestGetWorkSpaceSize, test_types);
TYPED_TEST_SUITE(TestDynPlans1D, test_types);
TYPED_TEST_SUITE(TestDynPlans2D, test_types);
TYPED_TEST_SUITE(TestDynPlans3D, test_types);

// Tests for plan constructiblility
TYPED_TEST(CompileTestDynPlan, is_constrcutrible) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using real_type            = double;
  using complex_type         = Kokkos::complex<real_type>;
  test_dynplan_constructible<execution_space_type, real_type, complex_type>();
  test_dynplan_constructible<execution_space_type, complex_type, real_type>();
  test_dynplan_constructible<execution_space_type, complex_type,
                             complex_type>();
}

// Tests for get_required_workspace_size
TYPED_TEST(TestGetWorkSpaceSize, FFT1D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_required_workspace_size<float_type, layout_type, 1, 1>();
  test_get_required_workspace_size<float_type, layout_type, 2, 1>();
  test_get_required_workspace_size<float_type, layout_type, 3, 1>();
  test_get_required_workspace_size<float_type, layout_type, 4, 1>();
  test_get_required_workspace_size<float_type, layout_type, 5, 1>();
  test_get_required_workspace_size<float_type, layout_type, 6, 1>();
  test_get_required_workspace_size<float_type, layout_type, 7, 1>();
  test_get_required_workspace_size<float_type, layout_type, 8, 1>();
}

TYPED_TEST(TestGetWorkSpaceSize, FFT2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_required_workspace_size<float_type, layout_type, 2, 2>();
  test_get_required_workspace_size<float_type, layout_type, 3, 2>();
  test_get_required_workspace_size<float_type, layout_type, 4, 2>();
  test_get_required_workspace_size<float_type, layout_type, 5, 2>();
  test_get_required_workspace_size<float_type, layout_type, 6, 2>();
  test_get_required_workspace_size<float_type, layout_type, 7, 2>();
  test_get_required_workspace_size<float_type, layout_type, 8, 2>();
}

TYPED_TEST(TestGetWorkSpaceSize, FFT3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_required_workspace_size<float_type, layout_type, 3, 3>();
  test_get_required_workspace_size<float_type, layout_type, 4, 3>();
  test_get_required_workspace_size<float_type, layout_type, 5, 3>();
  test_get_required_workspace_size<float_type, layout_type, 6, 3>();
  test_get_required_workspace_size<float_type, layout_type, 7, 3>();
  test_get_required_workspace_size<float_type, layout_type, 8, 3>();
}

// Tests for 1D FFTs
TYPED_TEST(TestDynPlans1D, View1D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 1, 1>();
}

TYPED_TEST(TestDynPlans1D, View1D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 1, 1>();
}

TYPED_TEST(TestDynPlans1D, View2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 2, 1>();
}

TYPED_TEST(TestDynPlans1D, View2D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 2, 1>();
}

TYPED_TEST(TestDynPlans1D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 3, 1>();
}

TYPED_TEST(TestDynPlans1D, View3D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 3, 1>();
}

TYPED_TEST(TestDynPlans1D, View4D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 4, 1>();
}

TYPED_TEST(TestDynPlans1D, View4D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 4, 1>();
}

TYPED_TEST(TestDynPlans1D, View5D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 5, 1>();
}

TYPED_TEST(TestDynPlans1D, View5D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 5, 1>();
}

TYPED_TEST(TestDynPlans1D, View6D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 6, 1>();
}

TYPED_TEST(TestDynPlans1D, View6D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 6, 1>();
}

TYPED_TEST(TestDynPlans1D, View7D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 7, 1>();
}

TYPED_TEST(TestDynPlans1D, View7D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 7, 1>();
}

TYPED_TEST(TestDynPlans1D, View8D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 8, 1>();
}

TYPED_TEST(TestDynPlans1D, View8D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 8, 1>();
}

// Tests for 2D FFTs
TYPED_TEST(TestDynPlans2D, View2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 2, 2>();
}

TYPED_TEST(TestDynPlans2D, View2D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 2, 2>();
}

TYPED_TEST(TestDynPlans2D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 3, 2>();
}

TYPED_TEST(TestDynPlans2D, View3D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 3, 2>();
}

TYPED_TEST(TestDynPlans2D, View4D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 4, 2>();
}

TYPED_TEST(TestDynPlans2D, View4D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 4, 2>();
}

TYPED_TEST(TestDynPlans2D, View5D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 5, 2>();
}

TYPED_TEST(TestDynPlans2D, View5D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 5, 2>();
}

TYPED_TEST(TestDynPlans2D, View6D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 6, 2>();
}

TYPED_TEST(TestDynPlans2D, View6D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 6, 2>();
}

TYPED_TEST(TestDynPlans2D, View7D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 7, 2>();
}

TYPED_TEST(TestDynPlans2D, View7D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 7, 2>();
}

TYPED_TEST(TestDynPlans2D, View8D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 8, 2>();
}

TYPED_TEST(TestDynPlans2D, View8D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 8, 2>();
}

// Tests for 3D FFTs
TYPED_TEST(TestDynPlans3D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 3, 3>();
}

TYPED_TEST(TestDynPlans3D, View3D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 3, 3>();
}

TYPED_TEST(TestDynPlans3D, View4D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 4, 3>();
}

TYPED_TEST(TestDynPlans3D, View4D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 4, 3>();
}

TYPED_TEST(TestDynPlans3D, View5D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 5, 3>();
}

TYPED_TEST(TestDynPlans3D, View5D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 5, 3>();
}

TYPED_TEST(TestDynPlans3D, View6D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 6, 3>();
}

TYPED_TEST(TestDynPlans3D, View6D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 6, 3>();
}

TYPED_TEST(TestDynPlans3D, View7D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 7, 3>();
}

TYPED_TEST(TestDynPlans3D, View7D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 7, 3>();
}

TYPED_TEST(TestDynPlans3D, View8D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 8, 3>();
}

TYPED_TEST(TestDynPlans3D, View8D_C2C) {
  using float_type  = typename TestFixture::complex_type;
  using layout_type = typename TestFixture::layout_type;

  test_dynplan<float_type, layout_type, 8, 3>();
}
