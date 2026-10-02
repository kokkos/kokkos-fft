#include "KokkosFFT_Batched.hpp"
#include <Kokkos_Core.hpp>
#include <concepts>
#include <gtest/gtest.h>

// These tests check the overload set of KokkosFFT::Batched::execute with
// requires expressions only, which do not instantiate the function bodies.
namespace {
using KokkosFFT::Direction;
using KokkosFFT::Normalization;
using KokkosFFT::Batched::AxisTag;
using KokkosFFT::Batched::Plan;
using execution_space = Kokkos::DefaultExecutionSpace;
using team_policy_type = Kokkos::TeamPolicy<execution_space>;
using member_type = typename team_policy_type::member_type;
using float_types = ::testing::Types<float, double>;

template <typename T> struct CompileTestConcepts : public ::testing::Test {
  using float_type = T;
};

TYPED_TEST_SUITE(CompileTestConcepts, float_types);

template <typename PlanType, typename InViewType, typename OutViewType>
concept SerialWithDirection = requires(
    const PlanType &plan, const InViewType &in, const OutViewType &out) {
  KokkosFFT::Batched::execute(plan, in, out, Direction::forward);
  KokkosFFT::Batched::execute(plan, in, out, Direction::backward,
                              Normalization::ortho);
};

template <typename PlanType, typename InViewType, typename OutViewType>
concept SerialWithoutDirection = requires(
    const PlanType &plan, const InViewType &in, const OutViewType &out) {
  KokkosFFT::Batched::execute(plan, in, out);
  KokkosFFT::Batched::execute(plan, in, out, Normalization::forward);
};

template <typename PlanType, typename InViewType, typename OutViewType>
concept TeamWithDirection =
    requires(const member_type &member, const PlanType &plan,
             const InViewType &in, const OutViewType &out) {
      KokkosFFT::Batched::execute(member, plan, in, out, Direction::forward);
    };

template <typename PlanType, typename InViewType, typename OutViewType>
concept TeamWithoutDirection =
    requires(const member_type &member, const PlanType &plan,
             const InViewType &in, const OutViewType &out) {
      KokkosFFT::Batched::execute(member, plan, in, out);
    };

template <typename T> void test_planable() {
  using View2D = Kokkos::View<Kokkos::complex<T> **, execution_space>;
  static_assert(KokkosFFT::Batched::Planable<
                Plan<execution_space, View2D, View2D, AxisTag<0>>>);
  static_assert(KokkosFFT::Batched::Planable<
                Plan<team_policy_type, View2D, View2D, AxisTag<1, 0>>>);
  static_assert(!KokkosFFT::Batched::Planable<View2D>);

  static_assert(
      KokkosFFT::Batched::ExecutionSpaceOrTeamPolicy<execution_space>);
  static_assert(
      KokkosFFT::Batched::ExecutionSpaceOrTeamPolicy<team_policy_type>);
  static_assert(!KokkosFFT::Batched::ExecutionSpaceOrTeamPolicy<
                Kokkos::RangePolicy<execution_space>>);

  static_assert(KokkosFFT::Batched::AxesSelectable<AxisTag<0>>);
  static_assert(KokkosFFT::Batched::AxesSelectable<AxisTag<0, 1>>);
  static_assert(!KokkosFFT::Batched::AxesSelectable<AxisTag<>>);
}

template <typename T> void test_slice_views() {
  using ComplexView2D = Kokkos::View<Kokkos::complex<T> **, execution_space>;
  using RealView2D = Kokkos::View<T **, execution_space>;
  using R2CPlan = Plan<execution_space, RealView2D, ComplexView2D, AxisTag<0>>;

  using RealSlice = Kokkos::View<T *, Kokkos::LayoutStride, execution_space>;
  using ComplexSlice =
      Kokkos::View<Kokkos::complex<T> *, Kokkos::LayoutStride, execution_space>;
  using ConstRealSlice =
      Kokkos::View<const T *, Kokkos::LayoutStride, execution_space>;

  static_assert(KokkosFFT::Batched::InSliceView<RealSlice, R2CPlan>);
  static_assert(KokkosFFT::Batched::OutSliceView<ComplexSlice, R2CPlan>);
  // Any layout is accepted
  static_assert(
      KokkosFFT::Batched::InSliceView<Kokkos::View<T *, execution_space>,
                                      R2CPlan>);
  // Input is a work buffer: it must be non-const
  static_assert(!KokkosFFT::Batched::InSliceView<ConstRealSlice, R2CPlan>);
  // Wrong value type
  static_assert(!KokkosFFT::Batched::InSliceView<ComplexSlice, R2CPlan>);
  static_assert(!KokkosFFT::Batched::OutSliceView<RealSlice, R2CPlan>);
  // Wrong rank: the slice rank is the FFT rank, not the view rank
  static_assert(!KokkosFFT::Batched::InSliceView<RealView2D, R2CPlan>);
}

template <typename T> void test_execute_overloads() {
  using ComplexView2D = Kokkos::View<Kokkos::complex<T> **, execution_space>;
  using RealView2D = Kokkos::View<T **, execution_space>;
  using ComplexSlice =
      Kokkos::View<Kokkos::complex<T> *, Kokkos::LayoutStride, execution_space>;
  using RealSlice = Kokkos::View<T *, Kokkos::LayoutStride, execution_space>;
  using ConstComplexSlice = Kokkos::View<const Kokkos::complex<T> *,
                                         Kokkos::LayoutStride, execution_space>;

  using SerialC2C =
      Plan<execution_space, ComplexView2D, ComplexView2D, AxisTag<0>>;
  using SerialR2C =
      Plan<execution_space, RealView2D, ComplexView2D, AxisTag<0>>;
  using SerialC2R =
      Plan<execution_space, ComplexView2D, RealView2D, AxisTag<0>>;
  using TeamC2C =
      Plan<team_policy_type, ComplexView2D, ComplexView2D, AxisTag<0>>;
  using TeamR2C = Plan<team_policy_type, RealView2D, ComplexView2D, AxisTag<0>>;

  // C2C needs a direction, and one plan serves both directions
  static_assert(SerialWithDirection<SerialC2C, ComplexSlice, ComplexSlice>);
  static_assert(!SerialWithoutDirection<SerialC2C, ComplexSlice, ComplexSlice>);

  // R2C/C2R take no direction
  static_assert(SerialWithoutDirection<SerialR2C, RealSlice, ComplexSlice>);
  static_assert(!SerialWithDirection<SerialR2C, RealSlice, ComplexSlice>);
  static_assert(SerialWithoutDirection<SerialC2R, ComplexSlice, RealSlice>);
  static_assert(!SerialWithDirection<SerialC2R, ComplexSlice, RealSlice>);

  // Swapped in/out value types
  static_assert(!SerialWithoutDirection<SerialR2C, ComplexSlice, RealSlice>);

  // Const input is rejected (input is used as a work buffer)
  static_assert(
      !SerialWithDirection<SerialC2C, ConstComplexSlice, ComplexSlice>);

  // Wrong slice rank
  static_assert(!SerialWithDirection<SerialC2C, ComplexView2D, ComplexView2D>);

  // Team plans are executed with the team member
  static_assert(TeamWithDirection<TeamC2C, ComplexSlice, ComplexSlice>);
  static_assert(!TeamWithoutDirection<TeamC2C, ComplexSlice, ComplexSlice>);
  static_assert(TeamWithoutDirection<TeamR2C, RealSlice, ComplexSlice>);

  // Level mismatches are rejected
  static_assert(!SerialWithDirection<TeamC2C, ComplexSlice, ComplexSlice>);
  static_assert(!SerialWithoutDirection<TeamR2C, RealSlice, ComplexSlice>);
  static_assert(!TeamWithDirection<SerialC2C, ComplexSlice, ComplexSlice>);
  static_assert(!TeamWithoutDirection<SerialR2C, RealSlice, ComplexSlice>);
}
} // namespace

TYPED_TEST(CompileTestConcepts, Planable) {
  using float_type = typename TestFixture::float_type;
  test_planable<float_type>();
}

TYPED_TEST(CompileTestConcepts, SliceViews) {
  using float_type = typename TestFixture::float_type;
  test_slice_views<float_type>();
}

TYPED_TEST(CompileTestConcepts, ExecuteOverloads) {
  using float_type = typename TestFixture::float_type;
  test_execute_overloads<float_type>();
}
