#include "KokkosFFT_Batched.hpp"
#include "Test_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <concepts>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using KokkosFFT::Batched::AxisTag;
using KokkosFFT::Batched::Plan;
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

const std::vector<std::size_t> test_lengths = {
    1, 2, 3, 4, 5, 7, 8, 12, 16, 30, 97, 128, 210, 1009, 4096};

template <typename T>
struct TestPlan : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

TYPED_TEST_SUITE(TestPlan, test_types);

void test_factorize(std::size_t n) {
  const auto radices = KokkosFFT::Batched::Impl::factorize(n);
  const std::size_t product =
      std::accumulate(radices.begin(), radices.end(), std::size_t(1),
                      [](std::size_t a, std::size_t b) { return a * b; });
  EXPECT_EQ(product, n) << "n = " << n;
  for (auto r : radices) {
    EXPECT_GT(r, 1u) << "n = " << n;
  }
  // An even stage count is only kept if no radix 4 can be split
  if (radices.size() % 2 == 0) {
    EXPECT_EQ(std::count(radices.begin(), radices.end(), std::size_t(4)), 0)
        << "n = " << n;
  }
}

/// \brief Sum of the generic-radix primes, i.e. the size of the roots table
std::size_t generic_roots_size(const std::vector<std::size_t> &radices) {
  std::size_t size = 0;
  for (auto r : radices) {
    if (!KokkosFFT::Batched::Impl::is_specialized_radix(r)) size += r;
  }
  return size;
}

template <typename T, typename LayoutType, int Axis>
void test_plan_c2c_1d(std::size_t n) {
  using View2DType =
      Kokkos::View<Kokkos::complex<T> **, LayoutType, execution_space>;
  constexpr std::size_t nbatch = 3;
  View2DType x("x", Axis == 0 ? n : nbatch, Axis == 0 ? nbatch : n);
  View2DType x_hat("x_hat", x.extent(0), x.extent(1));

  execution_space exec;
  Plan plan(exec, x, x_hat, AxisTag<Axis>{});  // CTAD
  static_assert(std::same_as<decltype(plan), Plan<execution_space, View2DType,
                                                  View2DType, AxisTag<Axis>>>);

  const auto expected = KokkosFFT::Batched::Impl::factorize(n);
  EXPECT_EQ(plan.length(0), n);
  EXPECT_EQ(plan.in_extent(0), n);
  EXPECT_EQ(plan.out_extent(0), n);
  EXPECT_EQ(plan.fft_size(), n);
  EXPECT_EQ(plan.get_scratch_size(0), 0u);
  ASSERT_EQ(plan.nstages(), static_cast<int>(expected.size()));

  // Stage twiddles: n - 1 entries; roots: the generic-radix primes
  EXPECT_EQ(plan.twiddles().extent(0), n - 1);
  EXPECT_EQ(plan.roots().extent(0), generic_roots_size(expected));

  auto h_radices =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, plan.radices());
  for (std::size_t s = 0; s < expected.size(); ++s) {
    EXPECT_EQ(h_radices(s), expected[s]);
  }

  // First twiddle of every stage is w^0 = 1 (p = 0, j = 1)
  auto h_twiddles =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, plan.twiddles());
  if (n > 1) {
    EXPECT_EQ(h_twiddles(0), Kokkos::complex<T>(1));
  }
}

template <typename T, typename LayoutType>
void test_plan_errors() {
  using View2DType =
      Kokkos::View<Kokkos::complex<T> **, LayoutType, execution_space>;
  execution_space exec;
  View2DType x("x", 8, 3);

  // FFT extents differ
  View2DType y0("y0", 9, 3);
  EXPECT_THROW(Plan(exec, x, y0, AxisTag<0>{}), std::runtime_error);

  // Batch extents differ
  View2DType y1("y1", 8, 4);
  EXPECT_THROW(Plan(exec, x, y1, AxisTag<0>{}), std::runtime_error);

  // Zero length
  View2DType z("z", 0, 3);
  EXPECT_THROW(Plan(exec, z, z, AxisTag<0>{}), std::runtime_error);

  using RealView2DType = Kokkos::View<T **, LayoutType, execution_space>;
  RealView2DType r8("r8", 8, 3), r7("r7", 7, 3);

  // R2C/C2R: the complex extent must be n/2+1
  View2DType c4("c4", 4, 3), c5("c5", 5, 3);
  EXPECT_THROW(Plan(exec, r8, c4, AxisTag<0>{}), std::runtime_error);
  EXPECT_THROW(Plan(exec, c4, r8, AxisTag<0>{}), std::runtime_error);
  EXPECT_NO_THROW(Plan(exec, r8, c5, AxisTag<0>{}));
  EXPECT_NO_THROW(Plan(exec, c5, r8, AxisTag<0>{}));

  // Odd lengths: the complex extent is (n-1)/2+1 = n/2+1 as well
  View2DType c3("c3", 3, 3);
  EXPECT_THROW(Plan(exec, r7, c5, AxisTag<0>{}), std::runtime_error);
  EXPECT_NO_THROW(Plan(exec, r7, c4, AxisTag<0>{}));
  EXPECT_NO_THROW(Plan(exec, c4, r7, AxisTag<0>{}));
  EXPECT_THROW(Plan(exec, c3, r7, AxisTag<0>{}), std::runtime_error);
}

/// \brief R2C and C2R plans of length n: extents and table sizes.
/// Even n runs a complex FFT of length n/2 with pre/post-processing; odd n
/// runs real half-complex stages of length n with one W_n^t table.
template <typename T, typename LayoutType, int Axis>
void test_plan_real_1d(std::size_t n) {
  using RealView2DType = Kokkos::View<T **, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T> **, LayoutType, execution_space>;
  constexpr std::size_t nbatch = 3;
  const bool is_odd            = n % 2 == 1;
  const std::size_t h          = n / 2;
  const std::size_t n_fft      = is_odd ? n : h;
  RealView2DType x("x", Axis == 0 ? n : nbatch, Axis == 0 ? nbatch : n);
  ComplexView2DType x_hat("x_hat", Axis == 0 ? h + 1 : nbatch,
                          Axis == 0 ? nbatch : h + 1);

  execution_space exec;
  Plan r2c(exec, x, x_hat, AxisTag<Axis>{});
  Plan c2r(exec, x_hat, x, AxisTag<Axis>{});
  static_assert(decltype(r2c)::kind == KokkosFFT::Batched::TransformKind::R2C);
  static_assert(decltype(c2r)::kind == KokkosFFT::Batched::TransformKind::C2R);

  EXPECT_EQ(r2c.length(0), n);
  EXPECT_EQ(r2c.fft_size(), n);
  EXPECT_EQ(r2c.n_fft(), n_fft);
  EXPECT_EQ(r2c.in_extent(0), n);
  EXPECT_EQ(r2c.out_extent(0), h + 1);
  EXPECT_EQ(c2r.length(0), n);
  EXPECT_EQ(c2r.n_fft(), n_fft);
  EXPECT_EQ(c2r.in_extent(0), h + 1);
  EXPECT_EQ(c2r.out_extent(0), n);

  const auto expected = KokkosFFT::Batched::Impl::factorize(n_fft);
  EXPECT_EQ(r2c.nstages(), static_cast<int>(expected.size()));
  if (is_odd) {
    // Only W_n^t, t = 0..n-1; no complex stage tables
    EXPECT_EQ(r2c.twiddles().extent(0), 0u);
    EXPECT_EQ(r2c.roots().extent(0), 0u);
    EXPECT_EQ(r2c.real_twiddles().extent(0), n);
    EXPECT_EQ(c2r.real_twiddles().extent(0), n);
  } else {
    // Stage twiddles of the half-length FFT, and W^k for k = 0..h/2
    EXPECT_EQ(r2c.twiddles().extent(0), h - 1);
    EXPECT_EQ(r2c.roots().extent(0), generic_roots_size(expected));
    EXPECT_EQ(r2c.real_twiddles().extent(0), h / 2 + 1);
    EXPECT_EQ(c2r.real_twiddles().extent(0), h / 2 + 1);
  }
}

/// \brief 3-D plans over axes (2, 0, 1) of a 4-D view batched along dim 3:
/// lengths, extents, children, and the real-axis requirement of N-D R2C/C2R
template <typename T, typename LayoutType>
void test_plan_nd() {
  using Axes = AxisTag<2, 0, 1>;
  using ComplexView4D =
      Kokkos::View<Kokkos::complex<T> ****, LayoutType, execution_space>;
  using RealView4D = Kokkos::View<T ****, LayoutType, execution_space>;
  execution_space exec;

  // FFT lengths (in the order of Axes): axis 2 -> 8, axis 0 -> 7, axis 1 -> 6
  ComplexView4D x("x", 7, 6, 8, 3), x_hat("x_hat", 7, 6, 8, 3);
  Plan c2c(exec, x, x_hat, Axes{});
  EXPECT_EQ(c2c.length(0), 8u);
  EXPECT_EQ(c2c.length(1), 7u);
  EXPECT_EQ(c2c.length(2), 6u);
  EXPECT_EQ(c2c.fft_size(), 8u * 7u * 6u);
  EXPECT_EQ(c2c.heads_plan().length(0), 8u);
  EXPECT_EQ(c2c.heads_plan().length(1), 7u);
  EXPECT_EQ(c2c.last_plan().length(0), 6u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(c2c.in_extent(i), c2c.length(i));
    EXPECT_EQ(c2c.out_extent(i), c2c.length(i));
  }

  // R2C/C2R: the last axis (axis 1, length 6) is halved to 4
  RealView4D r("r", 7, 6, 8, 3);
  ComplexView4D r_hat("r_hat", 7, 4, 8, 3);
  Plan r2c(exec, r, r_hat, Axes{});
  Plan c2r(exec, r_hat, r, Axes{});
  EXPECT_EQ(r2c.length(2), 6u);
  EXPECT_EQ(r2c.in_extent(2), 6u);
  EXPECT_EQ(r2c.out_extent(2), 4u);
  EXPECT_EQ(r2c.out_extent(0), 8u);  // heads are C2C
  EXPECT_EQ(c2r.in_extent(2), 4u);
  EXPECT_EQ(c2r.out_extent(2), 6u);
  EXPECT_EQ(r2c.fft_size(), 8u * 7u * 6u);
  EXPECT_EQ(r2c.last_plan().n_fft(), 3u);  // even: half-length complex FFT

  // A length-1 heads axis is fine
  RealView4D r1("r1", 1, 6, 8, 3);
  ComplexView4D r1_hat("r1_hat", 1, 4, 8, 3);
  EXPECT_NO_THROW(Plan(exec, r1, r1_hat, Axes{}));

  // N-D R2C/C2R needs a real axis of length >= 2 (scratch capacity, §2.6)
  RealView4D rl("rl", 7, 1, 8, 3);
  ComplexView4D rl_hat("rl_hat", 7, 1, 8, 3);
  EXPECT_THROW(Plan(exec, rl, rl_hat, Axes{}), std::runtime_error);
  EXPECT_THROW(Plan(exec, rl_hat, rl, Axes{}), std::runtime_error);
  // ... while C2C accepts length 1 on any axis
  ComplexView4D c1("c1", 7, 1, 8, 3), c1_hat("c1_hat", 7, 1, 8, 3);
  EXPECT_NO_THROW(Plan(exec, c1, c1_hat, Axes{}));

  // Batch extents must agree
  ComplexView4D y("y", 7, 6, 8, 4);
  EXPECT_THROW(Plan(exec, x, y, Axes{}), std::runtime_error);
}

template <typename T, typename LayoutType>
void test_plan_allocations() {
  using View2DType =
      Kokkos::View<Kokkos::complex<T> **, LayoutType, execution_space>;
  View2DType x("x", 210, 3), x_hat("x_hat", 210, 3);
  execution_space exec;

  TestUtils::AllocationRecorder recorder;
  Plan plan(exec, x, x_hat, AxisTag<0>{});
  ASSERT_FALSE(recorder.recorded().empty());
  for (const auto &label : recorder.recorded()) {
    EXPECT_EQ(label.rfind("KokkosFFT::Batched::Plan::", 0), 0u)
        << "unexpected allocation: " << label;
  }
}
}  // namespace

TEST(TestFactorize, Radices) {
  for (std::size_t n = 1; n <= 4096; ++n) {
    test_factorize(n);
  }
  using KokkosFFT::Batched::Impl::factorize;
  EXPECT_TRUE(factorize(1).empty());
  EXPECT_EQ(factorize(8), (std::vector<std::size_t>{2, 2, 2}));
  EXPECT_EQ(factorize(30), (std::vector<std::size_t>{2, 3, 5}));
  EXPECT_EQ(factorize(1009), (std::vector<std::size_t>{1009}));
  EXPECT_EQ(factorize(4096).size() % 2, 1u);
}

TYPED_TEST(TestPlan, C2C1D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  for (auto n : test_lengths) {
    test_plan_c2c_1d<float_type, layout_type, 0>(n);
    test_plan_c2c_1d<float_type, layout_type, 1>(n);
  }
}

TYPED_TEST(TestPlan, Real1D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  for (std::size_t n :
       {1, 2, 3, 4, 6, 7, 8, 9, 16, 30, 105, 128, 1009, 2018, 4096}) {
    test_plan_real_1d<float_type, layout_type, 0>(n);
    test_plan_real_1d<float_type, layout_type, 1>(n);
  }
}

TYPED_TEST(TestPlan, Errors) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_plan_errors<float_type, layout_type>();
}

TYPED_TEST(TestPlan, ND) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_plan_nd<float_type, layout_type>();
}

TYPED_TEST(TestPlan, AllocationLabels) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_plan_allocations<float_type, layout_type>();
}
