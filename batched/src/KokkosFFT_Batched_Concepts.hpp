#ifndef KOKKOSFFT_BATCHED_CONCEPTS_HPP
#define KOKKOSFFT_BATCHED_CONCEPTS_HPP

#include "KokkosFFT_Batched_Base_Types.hpp"
#include "KokkosFFT_Batched_Traits.hpp"
#include <Kokkos_Core.hpp>
#include <concepts>
#include <cstddef>

namespace KokkosFFT {
namespace Batched {

/// \brief Concept to check if a type is an AxisTag.
/// This concept is satisfied if the type is an instantiation of the AxisTag
/// template.
template <typename T>
concept AxesSelectable = requires {
  typename T::head;  // Must have a nested type 'head'
  typename T::tail;  // Must have a nested type 'tail'
  {
    T::rank
  } -> std::convertible_to<std::size_t>;  // Must have a static member 'rank'
                                          // convertible to size_t
};

/// \brief The first argument of a Plan: an execution space (serial plan) or a
/// Kokkos::TeamPolicy (team plan)
template <typename T>
concept ExecutionSpaceOrTeamPolicy =
    Kokkos::is_execution_space_v<T> || Impl::is_team_policy_v<T>;

template <typename T>
concept Planable = requires {
  typename T::policy_type;
  typename T::execution_space;
  typename T::in_view_type;
  typename T::out_view_type;
  typename T::in_value_type;
  typename T::out_value_type;
  typename T::axes_type;
  { T::rank } -> std::convertible_to<std::size_t>;
  { T::is_team } -> std::convertible_to<bool>;
  { T::kind } -> std::convertible_to<TransformKind>;
};

namespace Impl {
template <typename ViewType, typename PlanType, typename ValueType>
concept SliceViewOf =
    Planable<PlanType> && Kokkos::is_view_v<ViewType> &&
    (ViewType::rank() == PlanType::rank) &&
    std::same_as<typename ViewType::value_type, ValueType> &&
    // accessible is an enum: an atomic constraint must be exactly bool
    static_cast<bool>(Kokkos::SpaceAccessibility<
                      typename PlanType::execution_space,
                      typename ViewType::memory_space>::accessible);
}  // namespace Impl

/// \brief A per-batch input slice for `PlanType`: a View of rank
/// PlanType::rank with the plan's (non-const) input value type, any layout.
/// It is non-const because the input is used as a work buffer.
template <typename ViewType, typename PlanType>
concept InSliceView =
    Impl::SliceViewOf<ViewType, PlanType, typename PlanType::in_value_type>;

/// \brief A per-batch output slice for `PlanType`: a View of rank
/// PlanType::rank with the plan's (non-const) output value type, any layout.
template <typename ViewType, typename PlanType>
concept OutSliceView =
    Impl::SliceViewOf<ViewType, PlanType, typename PlanType::out_value_type>;

}  // namespace Batched
}  // namespace KokkosFFT

#endif
