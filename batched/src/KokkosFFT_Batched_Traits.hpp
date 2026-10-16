#ifndef KOKKOSFFT_BATCHED_TRAITS_HPP
#define KOKKOSFFT_BATCHED_TRAITS_HPP

#include "KokkosFFT_Batched_Base_Types.hpp"
#include <KokkosFFT_Traits.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <type_traits>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

template <typename... Ts>
inline constexpr bool always_false_v = false;

/// \brief Checks whether T is a Kokkos::TeamPolicy
template <typename T>
struct is_team_policy : std::false_type {};

template <typename... Properties>
struct is_team_policy<Kokkos::TeamPolicy<Properties...>> : std::true_type {};

template <typename T>
inline constexpr bool is_team_policy_v = is_team_policy<T>::value;

/// \brief Execution space of an execution space (itself) or of a TeamPolicy
template <typename T>
struct policy_execution_space {
  using type = T;
};

template <typename... Properties>
struct policy_execution_space<Kokkos::TeamPolicy<Properties...>> {
  using type = typename Kokkos::TeamPolicy<Properties...>::execution_space;
};

template <typename T>
using policy_execution_space_t = typename policy_execution_space<T>::type;

/// \brief Execution space instance from an execution space or a TeamPolicy
template <typename ExecPolicy>
auto get_space(const ExecPolicy &exec_policy) {
  if constexpr (is_team_policy_v<ExecPolicy>) {
    return exec_policy.space();
  } else {
    return exec_policy;
  }
}

/// \brief Value types accepted for FFTs: float, double and their complex
template <typename T>
inline constexpr bool is_fft_value_v =
    KokkosFFT::Impl::is_real_v<T> || KokkosFFT::Impl::is_complex_v<T>;

/// \brief Deduce the transform kind from the in/out value types.
/// real -> real is not a valid FFT and is rejected at compile time.
template <typename InValueType, typename OutValueType>
consteval TransformKind deduce_transform_kind() {
  using KokkosFFT::Impl::is_complex_v;
  using KokkosFFT::Impl::is_real_v;
  if constexpr (is_complex_v<InValueType> && is_complex_v<OutValueType>) {
    return TransformKind::C2C;
  } else if constexpr (is_real_v<InValueType> && is_complex_v<OutValueType>) {
    return TransformKind::R2C;
  } else if constexpr (is_complex_v<InValueType> && is_real_v<OutValueType>) {
    return TransformKind::C2R;
  } else {
    static_assert(
        always_false_v<InValueType, OutValueType>,
        "KokkosFFT::Batched::Plan: in/out value types must be complex->complex "
        "(C2C), real->complex (R2C) or complex->real (C2R) of "
        "float or double");
    return TransformKind::C2C;
  }
}

template <typename InValueType, typename OutValueType>
inline constexpr TransformKind transform_kind_v =
    deduce_transform_kind<InValueType, OutValueType>();

/// \brief Axis values of an AxisTag as a std::array
template <typename Axes>
struct axis_values;

template <int... Tags>
struct axis_values<AxisTag<Tags...>> {
  static constexpr std::array<int, sizeof...(Tags)> value{Tags...};
};

/// \brief Axes are non-negative, smaller than the view rank and distinct.
/// Negative axes are not supported for the moment.
template <typename Axes, std::size_t ViewRank>
consteval bool are_valid_axes() {
  constexpr auto axes = axis_values<Axes>::value;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0 || static_cast<std::size_t>(axes[i]) >= ViewRank)
      return false;
    for (std::size_t j = i + 1; j < axes.size(); ++j) {
      if (axes[i] == axes[j]) return false;
    }
  }
  return true;
}

/// \brief Dimension of axis `Axis` inside a slice that keeps only the FFT
/// axes `RootAxes` of the full batched view (in their original order).
/// e.g. RootAxes = AxisTag<2, 0>: axis 0 -> slice dim 0, axis 2 -> slice dim 1
template <typename RootAxes, int Axis>
consteval int slice_dim() {
  constexpr auto axes = axis_values<RootAxes>::value;
  int dim             = 0;
  bool found          = false;
  for (auto a : axes) {
    if (a < Axis) ++dim;
    if (a == Axis) found = true;
  }
  return found ? dim : -1;
}

template <typename RootAxes, int Axis>
inline constexpr int slice_dim_v = slice_dim<RootAxes, Axis>();

/// \brief LayoutStride view of rank `Rank` used for the synthesized child
/// view types of a recursive plan. These view types are never materialized.
template <typename ExecutionSpace, typename ValueType, std::size_t Rank>
using stride_view_t =
    Kokkos::View<KokkosFFT::Impl::add_pointer_n_t<ValueType, Rank>,
                 Kokkos::LayoutStride, ExecutionSpace>;

/// \brief View types of the heads/last children of a recursive plan.
/// The last child keeps the parent's value types (so it is the real
/// transform of an R2C/C2R plan) and the heads child is always C2C.
/// Both keep the full view rank so that axis indices stay valid.
///
/// | Parent | last child | heads child |
/// | C2C    | C2C        | C2C         |
/// | R2C    | R2C        | C2C         |
/// | C2R    | C2R        | C2C         |
template <typename ExecutionSpace, typename InValueType, typename OutValueType,
          std::size_t ViewRank>
struct split_view_types {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<InValueType>;
  using complex_type = Kokkos::complex<float_type>;

  using heads_in_view_type =
      stride_view_t<ExecutionSpace, complex_type, ViewRank>;
  using heads_out_view_type =
      stride_view_t<ExecutionSpace, complex_type, ViewRank>;
  using last_in_view_type =
      stride_view_t<ExecutionSpace, InValueType, ViewRank>;
  using last_out_view_type =
      stride_view_t<ExecutionSpace, OutValueType, ViewRank>;
};

}  // namespace Impl
}  // namespace Batched
}  // namespace KokkosFFT

#endif
