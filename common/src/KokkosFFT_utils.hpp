// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_UTILS_HPP
#define KOKKOSFFT_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_common_types.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename ScalarType1, typename ScalarType2>
bool are_aliasing(const ScalarType1* ptr1, const ScalarType2* ptr2) {
  return (static_cast<const void*>(ptr1) == static_cast<const void*>(ptr2));
}

template <typename IntType, std::size_t Rank>
IntType convert_negative_axis(IntType axis) {
  static_assert(
      std::is_integral_v<IntType> && std::is_signed_v<IntType>,
      "convert_negative_axis: IntType must be a signed integer type.");
  const IntType rank = static_cast<IntType>(Rank);
  KOKKOSFFT_THROW_IF(axis < -rank || axis >= rank,
                     "Axis must be in [-rank, rank-1]");

  return axis < 0 ? rank + axis : axis;
}

template <typename ViewType>
auto convert_negative_shift(const ViewType& view, int shift, int axis) {
  static_assert(Kokkos::is_view_v<ViewType>,
                "convert_negative_shift: ViewType must be a Kokkos::View.");
  int non_negative_axis = convert_negative_axis<int, ViewType::rank()>(axis);
  int extent            = view.extent(non_negative_axis);
  int shift0 = 0, shift1 = 0, shift2 = extent / 2;

  if (shift < 0) {
    shift0 = -shift + extent % 2;  // add 1 for odd case
    shift1 = -shift;
    shift2 = 0;
  } else if (shift > 0) {
    shift0 = shift;
    shift1 = shift + extent % 2;  // add 1 for odd case
    shift2 = 0;
  }

  return std::tuple<int, int, int>(shift0, shift1, shift2);
}

template <typename ContainerType, typename ValueType>
bool is_found(const ContainerType& values, const ValueType value) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  static_assert(std::is_same_v<value_type, ValueType>,
                "is_found: Container value type must match ValueType");
  return std::find(values.begin(), values.end(), value) != values.end();
}

template <typename ContainerType>
bool has_duplicate_values(const ContainerType& values) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  std::set<value_type> set_values(values.begin(), values.end());
  return set_values.size() < values.size();
}

template <typename ContainerType, typename IntType>
bool is_out_of_range_value_included(const ContainerType& values, IntType max) {
  static_assert(
      std::is_integral_v<IntType>,
      "is_out_of_range_value_included: IntType must be an integral type");
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  static_assert(std::is_same_v<value_type, IntType>,
                "is_out_of_range_value_included: Container value type must "
                "match IntType");
  if constexpr (std::is_signed_v<value_type>) {
    KOKKOSFFT_THROW_IF(
        std::any_of(values.begin(), values.end(),
                    [](value_type value) { return value < 0; }),
        "is_out_of_range_value_included: values must be non-negative");
  }
  return std::any_of(values.begin(), values.end(),
                     [max](value_type value) { return value >= max; });
}

template <
    typename ViewType, template <typename, std::size_t> class ArrayType,
    typename IntType, std::size_t DIM = 1,
    std::enable_if_t<Kokkos::is_view_v<ViewType> && std::is_integral_v<IntType>,
                     std::nullptr_t> = nullptr>
bool are_valid_axes(const ViewType& view, const ArrayType<IntType, DIM>& axes) {
  static_assert(Kokkos::is_view_v<ViewType>,
                "are_valid_axes: ViewType must be a Kokkos::View");
  static_assert(std::is_integral_v<IntType>,
                "are_valid_axes: IntType must be an integral type");
  static_assert(
      DIM >= 1 && DIM <= ViewType::rank(),
      "are_valid_axes: the Rank of FFT axes must be between 1 and View rank");

  // Convert the input axes to be in the range of [0, rank-1]
  // int type is chosen for consistency with the rest of the code
  // the axes are defined with int type
  std::array<int, DIM> non_negative_axes;

  // In case axis is out of range, 'convert_negative_axis' will throw an
  // runtime_error and we will return false. Without runtime_error, it is
  // ensured that the 'non_negative_axes' are in the range of [0, rank-1]
  try {
    for (std::size_t i = 0; i < DIM; i++) {
      int axis = axes[i];
      non_negative_axes[i] =
          KokkosFFT::Impl::convert_negative_axis<int, ViewType::rank()>(axis);
    }
  } catch (std::runtime_error& e) {
    return false;
  }

  bool is_valid = !KokkosFFT::Impl::has_duplicate_values(non_negative_axes);
  return is_valid;
}

template <std::size_t DIM = 1>
bool is_transpose_needed(const std::array<int, DIM>& map) {
  std::array<int, DIM> contiguous_map;
  std::iota(contiguous_map.begin(), contiguous_map.end(), 0);
  return map != contiguous_map;
}

template <typename ContainerType, typename ValueType>
std::size_t get_index(const ContainerType& values, const ValueType value) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  static_assert(std::is_same_v<value_type, ValueType>,
                "get_index: Container value type must match ValueType");
  auto it = std::find(values.begin(), values.end(), value);
  KOKKOSFFT_THROW_IF(it == values.end(), "value is not included in values");
  return it - values.begin();
}

template <typename IntType, std::size_t N, IntType start>
constexpr std::array<IntType, N> index_sequence() {
  static_assert(std::is_integral_v<IntType>,
                "index_sequence: IntType must be an integral type.");
  std::array<IntType, N> sequence{};
  for (std::size_t i = 0; i < N; ++i) {
    sequence[i] = start + static_cast<IntType>(i);
  }
  return sequence;
}

template <typename ElementType>
inline std::vector<ElementType> arange(const ElementType start,
                                       const ElementType stop,
                                       const ElementType step = 1) {
  const size_t length = ceil((stop - start) / step);
  std::vector<ElementType> result;
  ElementType delta = (stop - start) / length;

  // thrust::sequence
  for (std::size_t i = 0; i < length; i++) {
    ElementType value = start + delta * i;
    result.push_back(value);
  }
  return result;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void conjugate(const ExecutionSpace& exec_space, const InViewType& in,
               OutViewType& out) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "conjugate: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "conjugate: OutViewType must be complex");
  std::size_t size = in.size();
  out              = OutViewType("out", in.layout());

  auto* in_data  = in.data();
  auto* out_data = out.data();

  Kokkos::parallel_for(
      "KokkosFFT::conjugate",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, size),
      KOKKOS_LAMBDA(std::size_t i) { out_data[i] = Kokkos::conj(in_data[i]); });
}

template <typename ViewType>
auto extract_extents(const ViewType& view) {
  static_assert(Kokkos::is_view_v<ViewType>,
                "extract_extents: ViewType is not a Kokkos::View.");
  constexpr std::size_t rank = ViewType::rank();
  std::array<std::size_t, rank> extents;
  for (std::size_t i = 0; i < rank; i++) {
    extents.at(i) = view.extent(i);
  }
  return extents;
}

template <typename Layout, typename IntType, std::size_t N>
Layout create_layout(const std::array<IntType, N>& extents) {
  static_assert(std::is_integral_v<IntType>,
                "create_layout: IntType must be an integral type");
  static_assert(std::is_same_v<Layout, Kokkos::LayoutLeft> ||
                    std::is_same_v<Layout, Kokkos::LayoutRight>,
                "create_layout: Layout must be either Kokkos::LayoutLeft or "
                "Kokkos::LayoutRight.");
  Layout layout;
  std::copy_n(extents.begin(), N, layout.dimension);
  return layout;
}

template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_lvalue(
    std::array<T, N>& a, std::index_sequence<Is...>) {
  return {{a[Is]...}};
}
template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_rvalue(
    std::array<T, N>&& a, std::index_sequence<Is...>) {
  return {{std::move(a[Is])...}};
}

template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>& a) {
  return to_array_lvalue(a, std::make_index_sequence<N>());
}
template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>&& a) {
  return to_array_rvalue(std::move(a), std::make_index_sequence<N>());
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
