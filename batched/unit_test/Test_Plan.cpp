#include "KokkosFFT_Batched.hpp"
#include "Test_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
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

void test_factorize(std::size_t n, bool odd_stages) {
  const auto radices = KokkosFFT::Batched::Impl::factorize(n, odd_stages);
  const std::size_t product =
      std::accumulate(radices.begin(), radices.end(), std::size_t(1),
                      [](std::size_t a, std::size_t b) { return a * b; });
  EXPECT_EQ(product, n) << "n = " << n;
  for (auto r : radices) {
    EXPECT_GT(r, 1u) << "n = " << n;
  }
  // The power of two 2^k of n takes the fewest stages, ceil(k / 3): radix 8,
  // plus 4 x 4, 4 or 2 for the rest; radix 2 only when k = 1
  std::size_t k = 0;
  for (std::size_t m = n; m % 2 == 0; m /= 2) ++k;
  const auto pow2_stages = static_cast<std::size_t>(
      std::count_if(radices.begin(), radices.end(),
                    [](std::size_t r) { return r == 2 || r == 4 || r == 8; }));
  const std::size_t min_stages = (k + 2) / 3;
  const std::size_t n_others   = radices.size() - pow2_stages;
  // G2e: one more stage only if none of them needs radix 2
  if (!odd_stages || (min_stages + n_others) % 2 == 1 ||
      2 * (min_stages + 1) > k) {
    EXPECT_EQ(pow2_stages, min_stages) << "n = " << n;
    if (k != 1) {
      EXPECT_EQ(std::count(radices.begin(), radices.end(), std::size_t(2)), 0)
          << "n = " << n;
    }
  } else {
    // G2c: one more stage makes the count odd, balanced (radices differ by
    // at most a factor 2), and no radix 2 (G2e)
    EXPECT_EQ(pow2_stages, min_stages + 1) << "n = " << n;
    EXPECT_EQ(radices.size() % 2, 1u) << "n = " << n;
    EXPECT_LE(radices[pow2_stages - 1], 2 * radices[0]) << "n = " << n;
    EXPECT_GE(radices[0], 4u) << "n = " << n;
  }
  // Powers of two first, in ascending order
  EXPECT_TRUE(std::is_sorted(radices.begin(), radices.begin() + pow2_stages))
      << "n = " << n;
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

  // A 1-D C2C plan prefers an odd stage count (G2c)
  const auto expected = KokkosFFT::Batched::Impl::factorize(n, true);
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

  // R2C keeps the fewest stages; C2R prefers an odd count (G2c)
  const auto expected = KokkosFFT::Batched::Impl::factorize(n_fft);
  EXPECT_EQ(r2c.nstages(), static_cast<int>(expected.size()));
  EXPECT_EQ(c2r.nstages(),
            static_cast<int>(
                KokkosFFT::Batched::Impl::factorize(n_fft, true).size()));
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

/// \brief Radices of a leaf plan, on the host
template <typename LeafPlanType>
std::vector<std::size_t> leaf_radices(const LeafPlanType &leaf) {
  auto h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, leaf.radices());
  return std::vector<std::size_t>(h.data(), h.data() + h.extent(0));
}

/// \brief Radix 8 in N-D plans (G2d): the C2C leaves of a team plan with 3-D
/// slices factorise without it, as their merged kernels do not have it. Serial
/// plans, 2-D team plans and real 3-D team plans keep it.
template <typename T, typename LayoutType>
void test_plan_radix8() {
  using KokkosFFT::Batched::Impl::factorize;
  using ComplexView4D =
      Kokkos::View<Kokkos::complex<T> ****, LayoutType, execution_space>;
  using ComplexView3D =
      Kokkos::View<Kokkos::complex<T> ***, LayoutType, execution_space>;
  using RealView4D = Kokkos::View<T ****, LayoutType, execution_space>;
  const auto no8   = [](std::size_t n) { return factorize(n, false, false); };
  const auto with8 = [](std::size_t n) { return factorize(n); };
  EXPECT_EQ(no8(8), (std::vector<std::size_t>{2, 4}));
  EXPECT_EQ(no8(32), (std::vector<std::size_t>{2, 4, 4}));
  EXPECT_EQ(no8(64), (std::vector<std::size_t>{4, 4, 4}));

  // Lengths (in the order of the axes 2, 0, 1): 8, 32, 64
  ComplexView4D x("x", 32, 64, 8, 2), x_hat("x_hat", 32, 64, 8, 2);
  using Axes3 = AxisTag<2, 0, 1>;
  Plan team(TestUtils::plan_policy<true>(), x, x_hat, Axes3{});
  static_assert(!decltype(team)::root_radix8);
  EXPECT_EQ(leaf_radices(team.heads_plan().heads_plan()), no8(8));
  EXPECT_EQ(leaf_radices(team.heads_plan().last_plan()), no8(32));
  EXPECT_EQ(leaf_radices(team.last_plan()), no8(64));

  Plan serial(execution_space(), x, x_hat, Axes3{});
  static_assert(decltype(serial)::root_radix8);
  EXPECT_EQ(leaf_radices(serial.heads_plan().heads_plan()), with8(8));
  EXPECT_EQ(leaf_radices(serial.last_plan()), with8(64));

  // R2C 3-D (team): the heads run another kernel, which has radix 8
  RealView4D r("r", 32, 64, 8, 2);
  ComplexView4D r_hat("r_hat", 32, 33, 8, 2);
  Plan r2c(TestUtils::plan_policy<true>(), r, r_hat, Axes3{});
  static_assert(decltype(r2c)::root_radix8);
  EXPECT_EQ(leaf_radices(r2c.heads_plan().heads_plan()), with8(8));

  // 2-D slices (team): radix 8
  ComplexView3D y("y", 8, 64, 2), y_hat("y_hat", 8, 64, 2);
  Plan team2d(TestUtils::plan_policy<true>(), y, y_hat, AxisTag<0, 1>{});
  static_assert(decltype(team2d)::root_radix8);
  EXPECT_EQ(leaf_radices(team2d.heads_plan()), with8(8));
  EXPECT_EQ(leaf_radices(team2d.last_plan()), with8(64));
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
    test_factorize(n, false);
    test_factorize(n, true);
    // Without radix 8 (G2d): ceil(k / 2) stages of radix 4 or 2, radix 2 at
    // most once, and the same factors
    const auto no8 = KokkosFFT::Batched::Impl::factorize(n, false, false);
    const auto ref = KokkosFFT::Batched::Impl::factorize(n);
    std::size_t k  = 0;
    for (std::size_t m = n; m % 2 == 0; m /= 2) ++k;
    EXPECT_EQ(std::count(no8.begin(), no8.end(), std::size_t(8)), 0);
    EXPECT_LE(std::count(no8.begin(), no8.end(), std::size_t(2)), 1);
    EXPECT_EQ(std::count_if(no8.begin(), no8.end(),
                            [](std::size_t r) { return r == 2 || r == 4; }),
              static_cast<std::ptrdiff_t>((k + 1) / 2))
        << "n = " << n;
    EXPECT_TRUE(std::equal(no8.end() - (no8.size() - (k + 1) / 2), no8.end(),
                           ref.end() - (no8.size() - (k + 1) / 2)))
        << "n = " << n;
  }
  using KokkosFFT::Batched::Impl::factorize;
  EXPECT_TRUE(factorize(1).empty());
  EXPECT_EQ(factorize(8), (std::vector<std::size_t>{8}));
  EXPECT_EQ(factorize(16), (std::vector<std::size_t>{4, 4}));
  EXPECT_EQ(factorize(32), (std::vector<std::size_t>{4, 8}));
  EXPECT_EQ(factorize(128), (std::vector<std::size_t>{4, 4, 8}));
  EXPECT_EQ(factorize(30), (std::vector<std::size_t>{2, 3, 5}));
  EXPECT_EQ(factorize(1009), (std::vector<std::size_t>{1009}));
  EXPECT_EQ(factorize(4096), (std::vector<std::size_t>{8, 8, 8, 8}));
  // Odd stage count (G2c)
  EXPECT_EQ(factorize(8, true), (std::vector<std::size_t>{8}));
  // G2e: one more stage would need radix 2, the count stays even
  EXPECT_EQ(factorize(16, true), (std::vector<std::size_t>{4, 4}));
  EXPECT_EQ(factorize(32, true), (std::vector<std::size_t>{4, 8}));
  EXPECT_EQ(factorize(12, true), (std::vector<std::size_t>{4, 3}));
  EXPECT_EQ(factorize(64, true), (std::vector<std::size_t>{4, 4, 4}));
  EXPECT_EQ(factorize(128, true), (std::vector<std::size_t>{4, 4, 8}));
  EXPECT_EQ(factorize(1024, true), (std::vector<std::size_t>{4, 4, 4, 4, 4}));
  EXPECT_EQ(factorize(4096, true), (std::vector<std::size_t>{4, 4, 4, 8, 8}));
  EXPECT_EQ(factorize(48, true), (std::vector<std::size_t>{4, 4, 3}));
  EXPECT_EQ(factorize(30, true), (std::vector<std::size_t>{2, 3, 5}));
  // k = 1 or 0: no room for one more stage, the count stays even
  EXPECT_EQ(factorize(210, true), (std::vector<std::size_t>{2, 3, 5, 7}));
  EXPECT_EQ(factorize(15, true), (std::vector<std::size_t>{3, 5}));
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

TYPED_TEST(TestPlan, Radix8ND) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_plan_radix8<float_type, layout_type>();
}

TYPED_TEST(TestPlan, AllocationLabels) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_plan_allocations<float_type, layout_type>();
}
