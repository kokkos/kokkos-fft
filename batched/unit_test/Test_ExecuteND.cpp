#include "KokkosFFT_Batched.hpp"
#include "Test_Utils.hpp"
#include <KokkosFFT.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <array>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

// N-D batched transforms: 2-D slices of 3-D views and 3-D slices of 4-D
// views, with the batch dimension at various positions and the FFT axes in
// various orders, compared with KokkosFFT fftn/ifftn/rfftn/irfftn.
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

const std::vector<Normalization> test_norms = {
    Normalization::forward, Normalization::backward, Normalization::ortho,
    Normalization::none};

template <typename T>
struct TestExecuteND : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

TYPED_TEST_SUITE(TestExecuteND, test_types);

constexpr std::size_t nbatch = 3;

/// \brief Slice of a rank-3/4 batched view with dimension B fixed to ib
template <int B, typename ViewType>
KOKKOS_INLINE_FUNCTION auto batch_slice(const ViewType &v, std::size_t ib) {
  constexpr std::size_t rank = ViewType::rank();
  static_assert(rank == 3 || rank == 4);
  if constexpr (rank == 3) {
    if constexpr (B == 0) {
      return Kokkos::subview(v, ib, Kokkos::ALL, Kokkos::ALL);
    } else if constexpr (B == 1) {
      return Kokkos::subview(v, Kokkos::ALL, ib, Kokkos::ALL);
    } else {
      return Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, ib);
    }
  } else {
    if constexpr (B == 0) {
      return Kokkos::subview(v, ib, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    } else if constexpr (B == 1) {
      return Kokkos::subview(v, Kokkos::ALL, ib, Kokkos::ALL, Kokkos::ALL);
    } else if constexpr (B == 2) {
      return Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, ib, Kokkos::ALL);
    } else {
      return Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ib);
    }
  }
}

template <typename ViewType, std::size_t Rank>
ViewType make_view(const std::string &label,
                   const std::array<std::size_t, Rank> &e) {
  if constexpr (Rank == 3) {
    return ViewType(label, e[0], e[1], e[2]);
  } else {
    return ViewType(label, e[0], e[1], e[2], e[3]);
  }
}

/// \brief Extents of the batched view: FFT axis Axes[i] gets lengths[i],
/// dimension B gets nbatch
template <typename Axes, int B>
std::array<std::size_t, Axes::rank + 1> full_extents(
    const std::array<std::size_t, Axes::rank> &lengths) {
  constexpr auto axes = KokkosFFT::Batched::Impl::axis_values<Axes>::value;
  std::array<std::size_t, Axes::rank + 1> extents{};
  for (std::size_t i = 0; i < Axes::rank; ++i) extents[axes[i]] = lengths[i];
  extents[B] = nbatch;
  return extents;
}

template <typename Axes>
KokkosFFT::axis_type<Axes::rank> kokkosfft_axes() {
  KokkosFFT::axis_type<Axes::rank> axes{};
  constexpr auto values = KokkosFFT::Batched::Impl::axis_values<Axes>::value;
  for (std::size_t i = 0; i < Axes::rank; ++i) axes[i] = values[i];
  return axes;
}

/// \brief Run `execute` on every batch slice. Serial plans: one batch per
/// iteration of a RangePolicy; team plans: one batch per team of a
/// TeamPolicy (TestUtils::team_config()).
template <int B, typename PlanType, typename InViewType, typename OutViewType>
void batched_execute(const PlanType &plan, const InViewType &in,
                     const OutViewType &out, Direction dir,
                     Normalization norm) {
  if constexpr (PlanType::is_team) {
    const auto f = KOKKOS_LAMBDA(const TestUtils::member_type &member) {
      const std::size_t ib = member.league_rank();
      KokkosFFT::Batched::execute(member, plan, batch_slice<B>(in, ib),
                                  batch_slice<B>(out, ib), dir, norm);
    };
    Kokkos::parallel_for("test_batched_execute_nd_team",
                         TestUtils::make_team_policy(nbatch, f), f);
  } else {
    Kokkos::parallel_for(
        "test_batched_execute_nd",
        Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
            0, nbatch),
        KOKKOS_LAMBDA(const std::size_t ib) {
          KokkosFFT::Batched::execute(plan, batch_slice<B>(in, ib),
                                      batch_slice<B>(out, ib), dir, norm);
        });
  }
  Kokkos::fence();
}

template <int B, typename PlanType, typename InViewType, typename OutViewType>
void batched_execute_real(const PlanType &plan, const InViewType &in,
                          const OutViewType &out, Normalization norm) {
  if constexpr (PlanType::is_team) {
    const auto f = KOKKOS_LAMBDA(const TestUtils::member_type &member) {
      const std::size_t ib = member.league_rank();
      KokkosFFT::Batched::execute(member, plan, batch_slice<B>(in, ib),
                                  batch_slice<B>(out, ib), norm);
    };
    Kokkos::parallel_for("test_batched_execute_nd_real_team",
                         TestUtils::make_team_policy(nbatch, f), f);
  } else {
    Kokkos::parallel_for(
        "test_batched_execute_nd_real",
        Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
            0, nbatch),
        KOKKOS_LAMBDA(const std::size_t ib) {
          KokkosFFT::Batched::execute(plan, batch_slice<B>(in, ib),
                                      batch_slice<B>(out, ib), norm);
        });
  }
  Kokkos::fence();
}

template <std::size_t N>
std::string describe(const std::array<std::size_t, N> &lengths,
                     Normalization norm) {
  std::string s = "lengths = (";
  for (std::size_t i = 0; i < N; ++i) {
    s += std::to_string(lengths[i]) + (i + 1 < N ? ", " : ")");
  }
  return s + ", norm = " + std::to_string(static_cast<int>(norm));
}

template <std::size_t N>
std::size_t total_length(const std::array<std::size_t, N> &lengths) {
  std::size_t total = 1;
  for (auto n : lengths) total *= n;
  return total;
}

/// \brief C2C over Axes, batched along B; one plan forward then backward
template <typename T, typename LayoutType, typename Axes, int B, bool Team>
void test_c2c_nd(const std::array<std::size_t, Axes::rank> &lengths,
                 Normalization norm) {
  constexpr std::size_t rank = Axes::rank + 1;
  using ViewType =
      Kokkos::View<KokkosFFT::Impl::add_pointer_n_t<Kokkos::complex<T>, rank>,
                   LayoutType, execution_space>;
  const auto e = full_extents<Axes, B>(lengths);
  auto x0 = make_view<ViewType>("x0", e), x = make_view<ViewType>("x", e),
       x_hat    = make_view<ViewType>("x_hat", e),
       x_back   = make_view<ViewType>("x_back", e),
       ref_hat  = make_view<ViewType>("ref_hat", e),
       ref_in   = make_view<ViewType>("ref_in", e),
       ref_back = make_view<ViewType>("ref_back", e);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x0, random_pool, Kokkos::complex<T>(1, 1));
  Kokkos::deep_copy(exec, x, x0);

  Plan plan(TestUtils::plan_policy<Team>(), x, x_hat, Axes{});
  const auto axes  = kokkosfft_axes<Axes>();
  const double tol = TestUtils::fft_tolerance<T>(total_length(lengths));

  batched_execute<B>(plan, x, x_hat, Direction::forward, norm);
  KokkosFFT::fftn(exec, x0, ref_hat, axes, norm);
  exec.fence();
  EXPECT_LE(TestUtils::relative_l2_error(x_hat, ref_hat), tol)
      << "C2C forward, " << describe(lengths, norm);

  Kokkos::deep_copy(exec, ref_in, ref_hat);
  KokkosFFT::ifftn(exec, ref_in, ref_back, axes, norm);
  exec.fence();
  batched_execute<B>(plan, x_hat, x_back, Direction::backward, norm);
  EXPECT_LE(TestUtils::relative_l2_error(x_back, ref_back), tol)
      << "C2C backward, " << describe(lengths, norm);
  if (norm != Normalization::none) {
    EXPECT_LE(TestUtils::relative_l2_error(x_back, x0), tol)
        << "C2C round trip, " << describe(lengths, norm);
  }
}

/// \brief R2C then C2R over Axes (the last one is the real axis), batched
/// along B
template <typename T, typename LayoutType, typename Axes, int B, bool Team>
void test_r2c_c2r_nd(const std::array<std::size_t, Axes::rank> &lengths,
                     Normalization norm) {
  constexpr std::size_t rank = Axes::rank + 1;
  using RealViewType = Kokkos::View<KokkosFFT::Impl::add_pointer_n_t<T, rank>,
                                    LayoutType, execution_space>;
  using ComplexViewType =
      Kokkos::View<KokkosFFT::Impl::add_pointer_n_t<Kokkos::complex<T>, rank>,
                   LayoutType, execution_space>;
  const auto e     = full_extents<Axes, B>(lengths);
  auto ec          = e;
  ec[Axes::last_v] = lengths[Axes::rank - 1] / 2 + 1;

  auto x0       = make_view<RealViewType>("x0", e),
       x        = make_view<RealViewType>("x", e),
       x_back   = make_view<RealViewType>("x_back", e),
       ref_back = make_view<RealViewType>("ref_back", e);
  auto x_hat    = make_view<ComplexViewType>("x_hat", ec),
       ref_hat  = make_view<ComplexViewType>("ref_hat", ec),
       ref_in   = make_view<ComplexViewType>("ref_in", ec);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x0, random_pool, T(1));
  Kokkos::deep_copy(exec, x, x0);

  Plan r2c(TestUtils::plan_policy<Team>(), x, x_hat, Axes{});
  Plan c2r(TestUtils::plan_policy<Team>(), x_hat, x_back, Axes{});
  const auto axes  = kokkosfft_axes<Axes>();
  const double tol = TestUtils::fft_tolerance<T>(total_length(lengths));

  batched_execute_real<B>(r2c, x, x_hat, norm);
  KokkosFFT::rfftn(exec, x0, ref_hat, axes, norm);
  exec.fence();
  EXPECT_LE(TestUtils::relative_l2_error(x_hat, ref_hat), tol)
      << "R2C, " << describe(lengths, norm);

  Kokkos::deep_copy(exec, ref_in, ref_hat);
  KokkosFFT::irfftn(exec, ref_in, ref_back, axes, norm);
  exec.fence();
  batched_execute_real<B>(c2r, x_hat, x_back, norm);
  EXPECT_LE(TestUtils::relative_l2_error(x_back, ref_back), tol)
      << "C2R, " << describe(lengths, norm);
  if (norm != Normalization::none) {
    EXPECT_LE(TestUtils::relative_l2_error(x_back, x0), tol)
        << "R2C/C2R round trip, " << describe(lengths, norm);
  }
}

template <typename T, typename LayoutType, typename Axes, int B,
          bool Team = false>
void test_nd(const std::vector<std::array<std::size_t, Axes::rank>> &sets) {
  for (const auto &lengths : sets) {
    for (auto norm : test_norms) {
      test_c2c_nd<T, LayoutType, Axes, B, Team>(lengths, norm);
      test_r2c_c2r_nd<T, LayoutType, Axes, B, Team>(lengths, norm);
    }
  }
}

// Lengths per FFT axis (in the order of Axes, the last one is the real axis
// of R2C/C2R): even/odd/prime mixes, a minimal real axis (2), and length 1 on
// a heads axis.
const std::vector<std::array<std::size_t, 2>> lengths_2d = {
    {7, 6}, {12, 9}, {5, 2}, {11, 4}, {1, 8}};
const std::vector<std::array<std::size_t, 3>> lengths_3d = {
    {8, 7, 6}, {5, 12, 9}, {3, 1, 4}};

/// \brief R5 for N-D: execute never allocates
template <typename T, typename LayoutType, bool Team = false>
void test_nd_no_allocation() {
  using Axes      = AxisTag<2, 0, 1>;
  constexpr int B = 3;
  using ComplexViewType =
      Kokkos::View<Kokkos::complex<T> ****, LayoutType, execution_space>;
  using RealViewType = Kokkos::View<T ****, LayoutType, execution_space>;
  const std::array<std::size_t, 3> lengths = {6, 5, 8};
  const auto e                             = full_extents<Axes, B>(lengths);
  auto ec                                  = e;
  ec[Axes::last_v]                         = lengths[2] / 2 + 1;
  auto x                                   = make_view<ComplexViewType>("x", e),
       x_hat = make_view<ComplexViewType>("x_hat", e);
  auto r     = make_view<RealViewType>("r", e);
  auto r_hat = make_view<ComplexViewType>("r_hat", ec);

  Plan c2c(TestUtils::plan_policy<Team>(), x, x_hat, Axes{});
  Plan r2c(TestUtils::plan_policy<Team>(), r, r_hat, Axes{});
  Plan c2r(TestUtils::plan_policy<Team>(), r_hat, r, Axes{});

  // Warm-up: the first TeamPolicy launch makes the Kokkos runtime allocate
  // its team scratch buffer; that is not execute, so it is not recorded
  batched_execute<B>(c2c, x, x_hat, Direction::forward, Normalization::ortho);

  TestUtils::AllocationRecorder recorder;
  batched_execute<B>(c2c, x, x_hat, Direction::forward, Normalization::ortho);
  batched_execute<B>(c2c, x_hat, x, Direction::backward, Normalization::ortho);
  batched_execute_real<B>(r2c, r, r_hat, Normalization::backward);
  batched_execute_real<B>(c2r, r_hat, r, Normalization::backward);
  EXPECT_TRUE(recorder.recorded().empty())
      << "N-D execute allocated: " << recorder.recorded().front();
}
}  // namespace

TYPED_TEST(TestExecuteND, TwoD) {
  using T = typename TestFixture::float_type;
  using L = typename TestFixture::layout_type;
  test_nd<T, L, AxisTag<0, 1>, 2>(lengths_2d);
  test_nd<T, L, AxisTag<1, 0>, 2>(lengths_2d);
  test_nd<T, L, AxisTag<1, 2>, 0>(lengths_2d);
  test_nd<T, L, AxisTag<2, 1>, 0>(lengths_2d);
  test_nd<T, L, AxisTag<0, 2>, 1>(lengths_2d);
  test_nd<T, L, AxisTag<2, 0>, 1>(lengths_2d);
}

TYPED_TEST(TestExecuteND, ThreeD) {
  using T = typename TestFixture::float_type;
  using L = typename TestFixture::layout_type;
  test_nd<T, L, AxisTag<0, 1, 2>, 3>(lengths_3d);
  test_nd<T, L, AxisTag<2, 0, 1>, 3>(lengths_3d);
  test_nd<T, L, AxisTag<1, 2, 0>, 3>(lengths_3d);
  test_nd<T, L, AxisTag<1, 2, 3>, 0>(lengths_3d);
  test_nd<T, L, AxisTag<3, 1, 2>, 0>(lengths_3d);
}

TYPED_TEST(TestExecuteND, NoAllocationInExecute) {
  using T = typename TestFixture::float_type;
  using L = typename TestFixture::layout_type;
  test_nd_no_allocation<T, L>();
}

// ---- Team plans: the same checks with one team per batch, for each team
// configuration. With 4 threads, passes with at least 4 lines distribute
// their lines over the team, in rounds when the real scratch of an R2C/C2R
// plan cannot hold all lines at once (e.g. lengths (8, 7, 6): 28 heads lines
// of length 8, 21 scratch regions) ----

TYPED_TEST(TestExecuteND, TwoDTeam) {
  using T = typename TestFixture::float_type;
  using L = typename TestFixture::layout_type;
  for (auto config : TestUtils::team_configs()) {
    TestUtils::team_config() = config;
    test_nd<T, L, AxisTag<0, 1>, 2, true>(lengths_2d);
    test_nd<T, L, AxisTag<1, 0>, 2, true>(lengths_2d);
    test_nd<T, L, AxisTag<2, 0>, 1, true>(lengths_2d);
  }
}

TYPED_TEST(TestExecuteND, ThreeDTeam) {
  using T = typename TestFixture::float_type;
  using L = typename TestFixture::layout_type;
  for (auto config : TestUtils::team_configs()) {
    TestUtils::team_config() = config;
    test_nd<T, L, AxisTag<0, 1, 2>, 3, true>(lengths_3d);
    test_nd<T, L, AxisTag<2, 0, 1>, 3, true>(lengths_3d);
    test_nd<T, L, AxisTag<3, 1, 2>, 0, true>(lengths_3d);
  }
}

TYPED_TEST(TestExecuteND, NoAllocationInExecuteTeam) {
  using T                  = typename TestFixture::float_type;
  using L                  = typename TestFixture::layout_type;
  TestUtils::team_config() = {4, false};
  test_nd_no_allocation<T, L, true>();
}
