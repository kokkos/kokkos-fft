#include "KokkosFFT_Batched_Plan.hpp"
#include "KokkosFFT_Batched_Traits.hpp"
#include <Kokkos_Core.hpp>
#include <concepts>
#include <gtest/gtest.h>

namespace {
using KokkosFFT::Batched::AxisTag;
using KokkosFFT::Batched::Plan;
using KokkosFFT::Batched::TransformKind;
using execution_space  = Kokkos::DefaultExecutionSpace;
using team_policy_type = Kokkos::TeamPolicy<execution_space>;

using float_types = ::testing::Types<float, double>;

template <typename T>
using complex_t = Kokkos::complex<T>;

template <typename T>
struct CompileTestTransformKind : public ::testing::Test {
  using float_type = T;
};

template <typename T>
struct CompileTestPlan : public ::testing::Test {
  using float_type = T;
};

TYPED_TEST_SUITE(CompileTestTransformKind, float_types);
TYPED_TEST_SUITE(CompileTestPlan, float_types);

void test_policy_traits() {
  using KokkosFFT::Batched::Impl::is_team_policy_v;
  using KokkosFFT::Batched::Impl::policy_execution_space_t;
  static_assert(!is_team_policy_v<execution_space>);
  static_assert(is_team_policy_v<team_policy_type>);
  static_assert(
      is_team_policy_v<Kokkos::TeamPolicy<execution_space,
                                          Kokkos::Schedule<Kokkos::Dynamic>>>);
  static_assert(!is_team_policy_v<Kokkos::RangePolicy<execution_space>>);
  static_assert(
      std::same_as<policy_execution_space_t<execution_space>, execution_space>);
  static_assert(std::same_as<policy_execution_space_t<team_policy_type>,
                             execution_space>);

  // get_space returns an execution space instance for both
  auto space0 = KokkosFFT::Batched::Impl::get_space(execution_space{});
  auto space1 = KokkosFFT::Batched::Impl::get_space(team_policy_type(1, 1));
  static_assert(std::same_as<decltype(space0), execution_space>);
  static_assert(std::same_as<decltype(space1), execution_space>);
}

template <typename T>
void test_transform_kind() {
  using KokkosFFT::Batched::Impl::transform_kind_v;
  static_assert(transform_kind_v<complex_t<T>, complex_t<T>> ==
                TransformKind::C2C);
  static_assert(transform_kind_v<T, complex_t<T>> == TransformKind::R2C);
  static_assert(transform_kind_v<complex_t<T>, T> == TransformKind::C2R);
}

void test_axes() {
  using KokkosFFT::Batched::Impl::are_valid_axes;
  static_assert(are_valid_axes<AxisTag<0>, 1>());
  static_assert(are_valid_axes<AxisTag<2, 0>, 3>());
  static_assert(!are_valid_axes<AxisTag<1>, 1>());     // out of range
  static_assert(!are_valid_axes<AxisTag<0, 0>, 2>());  // duplicated
  static_assert(!are_valid_axes<AxisTag<-1>, 2>());    // negative

  // Axes {2, 0} of a rank-3 view: the slice keeps dims 0 and 2 in order
  using KokkosFFT::Batched::Impl::slice_dim_v;
  static_assert(slice_dim_v<AxisTag<2, 0>, 0> == 0);
  static_assert(slice_dim_v<AxisTag<2, 0>, 2> == 1);
  static_assert(slice_dim_v<AxisTag<1, 3, 2>, 1> == 0);
  static_assert(slice_dim_v<AxisTag<1, 3, 2>, 2> == 1);
  static_assert(slice_dim_v<AxisTag<1, 3, 2>, 3> == 2);
  static_assert(slice_dim_v<AxisTag<1, 3>, 2> == -1);  // not an FFT axis
}

template <typename ExecPolicy, typename T>
void test_plan_tree() {
  using RealView3D    = Kokkos::View<T ***, execution_space>;
  using ComplexView3D = Kokkos::View<complex_t<T> ***, execution_space>;
  using ComplexStride3D =
      Kokkos::View<complex_t<T> ***, Kokkos::LayoutStride, execution_space>;
  using RealStride3D =
      Kokkos::View<T ***, Kokkos::LayoutStride, execution_space>;
  using Axes = AxisTag<2, 0, 1>;

  constexpr bool is_team =
      KokkosFFT::Batched::Impl::is_team_policy_v<ExecPolicy>;

  // C2C: every node is C2C
  using C2CPlan = Plan<ExecPolicy, ComplexView3D, ComplexView3D, Axes>;
  static_assert(KokkosFFT::Batched::Planable<C2CPlan>);
  static_assert(C2CPlan::kind == TransformKind::C2C);
  static_assert(C2CPlan::rank == 3 && C2CPlan::view_rank == 3);
  static_assert(C2CPlan::is_team == is_team);
  static_assert(
      std::same_as<typename C2CPlan::execution_space, execution_space>);
  static_assert(C2CPlan::heads_plan_type::kind == TransformKind::C2C);
  static_assert(C2CPlan::last_plan_type::kind == TransformKind::C2C);
  static_assert(C2CPlan::last_plan_type::axis == 1);
  static_assert(std::same_as<typename C2CPlan::heads_plan_type::axes_type,
                             AxisTag<2, 0>>);

  // R2C: the last axis is R2C, the heads are C2C
  using R2CPlan = Plan<ExecPolicy, RealView3D, ComplexView3D, Axes>;
  static_assert(R2CPlan::kind == TransformKind::R2C);
  static_assert(R2CPlan::last_plan_type::kind == TransformKind::R2C);
  static_assert(R2CPlan::heads_plan_type::kind == TransformKind::C2C);
  static_assert(R2CPlan::heads_plan_type::last_plan_type::kind ==
                TransformKind::C2C);
  static_assert(std::same_as<typename R2CPlan::last_plan_type::in_view_type,
                             RealStride3D>);
  static_assert(std::same_as<typename R2CPlan::last_plan_type::out_view_type,
                             ComplexStride3D>);

  // C2R: the heads are C2C, the last axis is C2R
  using C2RPlan = Plan<ExecPolicy, ComplexView3D, RealView3D, Axes>;
  static_assert(C2RPlan::kind == TransformKind::C2R);
  static_assert(C2RPlan::last_plan_type::kind == TransformKind::C2R);
  static_assert(C2RPlan::heads_plan_type::kind == TransformKind::C2C);
  static_assert(std::same_as<typename C2RPlan::heads_plan_type::out_view_type,
                             ComplexStride3D>);
  static_assert(std::same_as<typename C2RPlan::last_plan_type::out_view_type,
                             RealStride3D>);

  // Children keep the full view rank and the level of the root
  static_assert(C2RPlan::heads_plan_type::view_rank == 3);
  static_assert(C2RPlan::last_plan_type::view_rank == 3);
  static_assert(C2RPlan::heads_plan_type::is_team == is_team);
  static_assert(C2RPlan::last_plan_type::is_team == is_team);
}

}  // namespace

TEST(CompileTestTraits, PolicyTraits) { test_policy_traits(); }

TEST(CompileTestTraits, Axes) { test_axes(); }

TYPED_TEST(CompileTestTransformKind, Deduction) {
  using float_type = typename TestFixture::float_type;
  test_transform_kind<float_type>();
}

TYPED_TEST(CompileTestPlan, TreeSerial) {
  using float_type = typename TestFixture::float_type;
  test_plan_tree<execution_space, float_type>();
}

TYPED_TEST(CompileTestPlan, TreeTeam) {
  using float_type = typename TestFixture::float_type;
  test_plan_tree<team_policy_type, float_type>();
}
