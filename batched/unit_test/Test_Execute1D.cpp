#include "KokkosFFT_Batched.hpp"
#include "Test_Utils.hpp"
#include <KokkosFFT.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {
using KokkosFFT::Direction;
using KokkosFFT::Normalization;
using KokkosFFT::Batched::AxisTag;
using KokkosFFT::Batched::Plan;
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

const std::vector<std::size_t> test_lengths = {
    1, 2, 3, 4, 5, 7, 8, 12, 16, 30, 97, 128, 210, 1009, 4096};
const std::vector<Normalization> test_norms = {
    Normalization::forward, Normalization::backward, Normalization::ortho,
    Normalization::none};

template <typename T>
struct TestExecute1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

TYPED_TEST_SUITE(TestExecute1D, test_types);

/// \brief Run `execute` on every batch slice of the 2D views, from a kernel.
/// The plan is captured by value. Serial plans: one batch per iteration of a
/// RangePolicy; team plans: one batch per team of a TeamPolicy
/// (TestUtils::team_config()).
template <int Axis, typename PlanType, typename ViewType>
void batched_execute(const PlanType &plan, const ViewType &in,
                     const ViewType &out, Direction dir, Normalization norm) {
  const std::size_t nbatch = in.extent(1 - Axis);
  if constexpr (PlanType::is_team) {
    const auto f = KOKKOS_LAMBDA(const TestUtils::member_type &member) {
      const std::size_t ib = member.league_rank();
      auto sub_in          = TestUtils::batch_slice<Axis>(in, ib);
      auto sub_out         = TestUtils::batch_slice<Axis>(out, ib);
      KokkosFFT::Batched::execute(member, plan, sub_in, sub_out, dir, norm);
    };
    Kokkos::parallel_for("test_batched_execute_team",
                         TestUtils::make_team_policy(nbatch, f), f);
  } else {
    Kokkos::parallel_for(
        "test_batched_execute",
        Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
            0, nbatch),
        KOKKOS_LAMBDA(const std::size_t ib) {
          auto sub_in  = TestUtils::batch_slice<Axis>(in, ib);
          auto sub_out = TestUtils::batch_slice<Axis>(out, ib);
          KokkosFFT::Batched::execute(plan, sub_in, sub_out, dir, norm);
        });
  }
  Kokkos::fence();
}

/// \brief C2C 1D along `Axis` of a (n, nbatch) or (nbatch, n) view, compared
/// with KokkosFFT. The same plan runs forward then backward.
template <typename T, typename LayoutType, int Axis, bool Team>
void test_c2c_1d(std::size_t n, Normalization norm) {
  using View2DType =
      Kokkos::View<Kokkos::complex<T> **, LayoutType, execution_space>;
  constexpr std::size_t nbatch = 5;
  const std::size_t n0         = Axis == 0 ? n : nbatch;
  const std::size_t n1         = Axis == 0 ? nbatch : n;
  View2DType x0("x0", n0, n1), x("x", n0, n1), x_hat("x_hat", n0, n1),
      x_back("x_back", n0, n1), ref_hat("ref_hat", n0, n1),
      ref_back("ref_back", n0, n1);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x0, random_pool, Kokkos::complex<T>(1, 1));
  Kokkos::deep_copy(exec, x, x0);

  Plan plan(TestUtils::plan_policy<Team>(), x, x_hat, AxisTag<Axis>{});

  // Forward: x is overwritten, the result is in x_hat
  batched_execute<Axis>(plan, x, x_hat, Direction::forward, norm);
  KokkosFFT::fft(exec, x0, ref_hat, norm, Axis);
  exec.fence();
  const double tol = TestUtils::fft_tolerance<T>(n);
  EXPECT_LE(TestUtils::relative_l2_error(x_hat, ref_hat), tol)
      << "forward, n = " << n << ", norm = " << static_cast<int>(norm);

  // Backward with the same plan: x_hat is overwritten
  KokkosFFT::ifft(exec, ref_hat, ref_back, norm, Axis);
  exec.fence();
  batched_execute<Axis>(plan, x_hat, x_back, Direction::backward, norm);
  EXPECT_LE(TestUtils::relative_l2_error(x_back, ref_back), tol)
      << "backward, n = " << n << ", norm = " << static_cast<int>(norm);

  // Round trip (none leaves a factor n)
  if (norm != Normalization::none) {
    EXPECT_LE(TestUtils::relative_l2_error(x_back, x0), tol)
        << "round trip, n = " << n << ", norm = " << static_cast<int>(norm);
  }
}

template <typename T, typename LayoutType, bool Team = false>
void test_c2c_1d_all() {
  for (auto n : test_lengths) {
    for (auto norm : test_norms) {
      test_c2c_1d<T, LayoutType, 0, Team>(n, norm);
      test_c2c_1d<T, LayoutType, 1, Team>(n, norm);
    }
  }
}

// Lengths for R2C/C2R.
// Even n: the half lengths h = n/2 cover both stage-count parities (e.g.
// h = 6 = 2 x 3 and h = 15 = 3 x 5 are even), so both branches of the passes
// (result in `in` or in `out`) are exercised.
// Odd n: n = 1 (no stage), primes (one direct stage), and both stage-count
// parities of n (e.g. 9 = 3 x 3 and 1155 = 3 x 5 x 7 x 11 are even, which
// takes the copy-back paths).
const std::vector<std::size_t> real_test_lengths = {
    2, 4, 6, 8, 10, 12, 16, 30, 32, 42, 60, 128, 210, 1024, 2018, 4096,
    1, 3, 5, 7, 9,  15, 21, 25, 27, 45, 97, 105, 243, 1009, 1155, 3027};

/// \brief Run the direction-less `execute` (R2C/C2R) on every batch slice
template <int Axis, typename PlanType, typename InViewType,
          typename OutViewType>
void batched_execute_real(const PlanType &plan, const InViewType &in,
                          const OutViewType &out, Normalization norm) {
  const std::size_t nbatch = in.extent(1 - Axis);
  if constexpr (PlanType::is_team) {
    const auto f = KOKKOS_LAMBDA(const TestUtils::member_type &member) {
      const std::size_t ib = member.league_rank();
      auto sub_in          = TestUtils::batch_slice<Axis>(in, ib);
      auto sub_out         = TestUtils::batch_slice<Axis>(out, ib);
      KokkosFFT::Batched::execute(member, plan, sub_in, sub_out, norm);
    };
    Kokkos::parallel_for("test_batched_execute_real_team",
                         TestUtils::make_team_policy(nbatch, f), f);
  } else {
    Kokkos::parallel_for(
        "test_batched_execute_real",
        Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
            0, nbatch),
        KOKKOS_LAMBDA(const std::size_t ib) {
          auto sub_in  = TestUtils::batch_slice<Axis>(in, ib);
          auto sub_out = TestUtils::batch_slice<Axis>(out, ib);
          KokkosFFT::Batched::execute(plan, sub_in, sub_out, norm);
        });
  }
  Kokkos::fence();
}

/// \brief Zero the imaginary parts of the bins a C2R of length n ignores:
/// X[0], and the Nyquist bin X[n/2] if n is even. For odd n the last bin
/// X[(n-1)/2] is a regular bin and keeps its imaginary part.
template <int Axis, typename ViewType>
void zero_dc_nyquist_imag(const ViewType &x_hat, std::size_t n) {
  using value_type = typename ViewType::non_const_value_type;
  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  const std::size_t nbatch      = x_hat.extent(1 - Axis);
  std::vector<std::size_t> bins = {0};
  if (n % 2 == 0) bins.push_back(n / 2);
  for (std::size_t ib = 0; ib < nbatch; ++ib) {
    for (std::size_t k : bins) {
      auto &v = Axis == 0 ? h_x(k, ib) : h_x(ib, k);
      v       = value_type(v.real(), 0);
    }
  }
  Kokkos::deep_copy(x_hat, h_x);
}

/// \brief R2C then C2R 1D along `Axis`, compared with KokkosFFT rfft/irfft
template <typename T, typename LayoutType, int Axis, bool Team>
void test_r2c_c2r_1d(std::size_t n, Normalization norm) {
  using RealView2DType = Kokkos::View<T **, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T> **, LayoutType, execution_space>;
  constexpr std::size_t nbatch = 5;
  const std::size_t h          = n / 2;
  const std::size_t r0 = Axis == 0 ? n : nbatch, r1 = Axis == 0 ? nbatch : n;
  const std::size_t c0 = Axis == 0 ? h + 1 : nbatch,
                    c1 = Axis == 0 ? nbatch : h + 1;
  RealView2DType x0("x0", r0, r1), x("x", r0, r1), x_back("x_back", r0, r1),
      ref_back("ref_back", r0, r1);
  ComplexView2DType x_hat("x_hat", c0, c1), ref_hat("ref_hat", c0, c1),
      ref_in("ref_in", c0, c1);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x0, random_pool, T(1));
  Kokkos::deep_copy(exec, x, x0);

  Plan r2c(TestUtils::plan_policy<Team>(), x, x_hat, AxisTag<Axis>{});
  Plan c2r(TestUtils::plan_policy<Team>(), x_hat, x_back, AxisTag<Axis>{});
  const double tol = TestUtils::fft_tolerance<T>(n);

  // R2C: x is overwritten, the result is in x_hat
  batched_execute_real<Axis>(r2c, x, x_hat, norm);
  KokkosFFT::rfft(exec, x0, ref_hat, norm, Axis);
  exec.fence();
  EXPECT_LE(TestUtils::relative_l2_error(x_hat, ref_hat), tol)
      << "R2C, n = " << n << ", norm = " << static_cast<int>(norm);

  // C2R of the R2C result: x_hat is overwritten
  Kokkos::deep_copy(exec, ref_in, ref_hat);
  KokkosFFT::irfft(exec, ref_in, ref_back, norm, Axis);
  exec.fence();
  batched_execute_real<Axis>(c2r, x_hat, x_back, norm);
  EXPECT_LE(TestUtils::relative_l2_error(x_back, ref_back), tol)
      << "C2R, n = " << n << ", norm = " << static_cast<int>(norm);
  if (norm != Normalization::none) {
    EXPECT_LE(TestUtils::relative_l2_error(x_back, x0), tol)
        << "round trip, n = " << n << ", norm = " << static_cast<int>(norm);
  }

  // C2R of an arbitrary spectrum: Im X[0] (and Im X[n/2] for even n) are
  // ignored, as in numpy, so the reference sees them zeroed
  Kokkos::fill_random(exec, x_hat, random_pool, Kokkos::complex<T>(1, 1));
  Kokkos::deep_copy(exec, ref_in, x_hat);
  exec.fence();
  zero_dc_nyquist_imag<Axis>(ref_in, n);
  KokkosFFT::irfft(exec, ref_in, ref_back, norm, Axis);
  exec.fence();
  batched_execute_real<Axis>(c2r, x_hat, x_back, norm);
  EXPECT_LE(TestUtils::relative_l2_error(x_back, ref_back), tol)
      << "C2R (arbitrary spectrum), n = " << n
      << ", norm = " << static_cast<int>(norm);
}

template <typename T, typename LayoutType, bool Team = false>
void test_r2c_c2r_1d_all() {
  for (auto n : real_test_lengths) {
    for (auto norm : test_norms) {
      test_r2c_c2r_1d<T, LayoutType, 0, Team>(n, norm);
      test_r2c_c2r_1d<T, LayoutType, 1, Team>(n, norm);
    }
  }
}

/// \brief R5: execute never allocates
template <typename T, typename LayoutType, bool Team = false>
void test_execute_no_allocation() {
  using View2DType =
      Kokkos::View<Kokkos::complex<T> **, LayoutType, execution_space>;
  const std::size_t n = 210, nbatch = 4;
  View2DType x("x", n, nbatch), x_hat("x_hat", n, nbatch);
  Plan plan(TestUtils::plan_policy<Team>(), x, x_hat, AxisTag<0>{});

  // Warm-up: the first TeamPolicy launch makes the Kokkos runtime allocate
  // its team scratch buffer (e.g. "Kokkos::Serial::scratch_mem"). That is
  // the backend's launch machinery, not execute, so it happens before
  // recording; later launches reuse the buffer.
  batched_execute<0>(plan, x, x_hat, Direction::forward,
                     Normalization::backward);

  TestUtils::AllocationRecorder recorder;
  // Sanity check: the recorder sees allocations
  { View2DType probe("probe", 1, 1); }
  ASSERT_EQ(recorder.recorded().size(), 1u);
  recorder.clear();

  batched_execute<0>(plan, x, x_hat, Direction::forward,
                     Normalization::backward);
  batched_execute<0>(plan, x_hat, x, Direction::backward,
                     Normalization::backward);
  EXPECT_TRUE(recorder.recorded().empty())
      << "C2C execute allocated: " << recorder.recorded().front();

  // R2C / C2R
  using RealView2DType = Kokkos::View<T **, LayoutType, execution_space>;
  RealView2DType r("r", n, nbatch);
  View2DType r_hat("r_hat", n / 2 + 1, nbatch);
  recorder.clear();
  Plan r2c(TestUtils::plan_policy<Team>(), r, r_hat, AxisTag<0>{});
  Plan c2r(TestUtils::plan_policy<Team>(), r_hat, r, AxisTag<0>{});
  batched_execute_real<0>(r2c, r, r_hat, Normalization::backward);  // warm-up
  recorder.clear();
  batched_execute_real<0>(r2c, r, r_hat, Normalization::backward);
  batched_execute_real<0>(c2r, r_hat, r, Normalization::backward);
  EXPECT_TRUE(recorder.recorded().empty())
      << "R2C/C2R execute allocated: " << recorder.recorded().front();
}
}  // namespace

TEST(TestExecute1DLengths, RealParityCoverage) {
  // Both stage-count parities of the even-n and odd-n paths of pass_r2c and
  // pass_c2r must be covered by real_test_lengths. C2R prefers an odd count
  // (G2c), so its parities are checked with its own factorisation.
  using KokkosFFT::Batched::Impl::factorize;
  for (bool c2r : {false, true}) {
    bool even_n_odd_stages = false, even_n_even_stages = false;
    bool odd_n_odd_stages = false, odd_n_even_stages = false;
    for (auto n : real_test_lengths) {
      if (n % 2 == 0) {
        const auto nstages = factorize(n / 2, c2r).size();
        even_n_odd_stages |= nstages % 2 == 1;
        even_n_even_stages |= nstages % 2 == 0;
      } else if (n > 1) {
        const auto nstages = factorize(n, c2r).size();
        odd_n_odd_stages |= nstages % 2 == 1;
        odd_n_even_stages |= nstages % 2 == 0;
      }
    }
    EXPECT_TRUE(even_n_odd_stages) << "c2r = " << c2r;
    EXPECT_TRUE(even_n_even_stages) << "c2r = " << c2r;
    EXPECT_TRUE(odd_n_odd_stages) << "c2r = " << c2r;
    EXPECT_TRUE(odd_n_even_stages) << "c2r = " << c2r;
  }
}

TEST(TestExecute1DLengths, C2CParityCoverage) {
  // A 1-D C2C plan prefers an odd stage count (G2c); test_lengths must still
  // cover an even count (the copy in finalize) and an odd one
  bool odd_stages = false, even_stages = false;
  for (auto n : test_lengths) {
    const auto nstages = KokkosFFT::Batched::Impl::factorize(n, true).size();
    odd_stages |= nstages % 2 == 1;
    even_stages |= n > 1 && nstages % 2 == 0;
  }
  EXPECT_TRUE(odd_stages);
  EXPECT_TRUE(even_stages);
}

TYPED_TEST(TestExecute1D, C2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_c2c_1d_all<float_type, layout_type>();
}

TYPED_TEST(TestExecute1D, R2CC2R) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_r2c_c2r_1d_all<float_type, layout_type>();
}

TYPED_TEST(TestExecute1D, NoAllocationInExecute) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_execute_no_allocation<float_type, layout_type>();
}

// ---- Team plans: the same checks with one team per batch, for each team
// configuration (team size AUTO / 1 / 4, and 4 with the maximum vector
// length) ----

TYPED_TEST(TestExecute1D, C2CTeam) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  for (auto config : TestUtils::team_configs()) {
    TestUtils::team_config() = config;
    test_c2c_1d_all<float_type, layout_type, true>();
  }
}

TYPED_TEST(TestExecute1D, R2CC2RTeam) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  for (auto config : TestUtils::team_configs()) {
    TestUtils::team_config() = config;
    test_r2c_c2r_1d_all<float_type, layout_type, true>();
  }
}

TYPED_TEST(TestExecute1D, NoAllocationInExecuteTeam) {
  using float_type         = typename TestFixture::float_type;
  using layout_type        = typename TestFixture::layout_type;
  TestUtils::team_config() = {4, false};
  test_execute_no_allocation<float_type, layout_type, true>();
}
