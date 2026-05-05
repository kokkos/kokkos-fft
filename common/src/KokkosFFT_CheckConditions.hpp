// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CHECKCONDITIONS_HPP
#define KOKKOSFFT_CHECKCONDITIONS_HPP

#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <type_traits>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Mapping.hpp"

namespace KokkosFFT {
namespace Impl {

/// \brief Check if two pointers are aliasing (pointing to the same memory
/// location) Pointers can be of different types, but they are considered
/// aliasing if they point to the same memory location. Examples:
/// - are_aliasing(ptr1, ptr1) returns true (same pointer)
/// - are_aliasing(ptr1, ptr2) returns false (different pointers)
///
/// \tparam ScalarType1 The type of the first pointer
/// \tparam ScalarType2 The type of the second pointer
/// \param[in] ptr1 The first pointer
/// \param[in] ptr2 The second pointer
/// \return True if the two pointers are aliasing, false otherwise
template <typename ScalarType1, typename ScalarType2>
bool are_aliasing(const ScalarType1* ptr1, const ScalarType2* ptr2) {
  return (static_cast<const void*>(ptr1) == static_cast<const void*>(ptr2));
}

/// \brief Check if a container has duplicate values
/// Examples:
/// - has_duplicate_values({1, 2, 3}) returns false
/// - has_duplicate_values({1, 2, 2}) returns true
///
/// \tparam ContainerType The type of the container
/// \param[in] values The container to check for duplicate values
/// \return True if the container has duplicate values, false otherwise
template <typename ContainerType>
bool has_duplicate_values(const ContainerType& values) {
  using value_type = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
  std::set<value_type> set_values(values.begin(), values.end());
  return set_values.size() < values.size();
}

/// \brief Check if a container includes any out-of-range values (negative or
/// greater than or equal to max) Examples:
/// - is_out_of_range_value_included({0, 1, 2}, 3) returns false
/// - is_out_of_range_value_included({-1, 0, 1}, 3) throws a runtime_error
/// (negative value)
/// - is_out_of_range_value_included({0, 1, 3}, 3) returns true (value equal to
/// max)
/// - is_out_of_range_value_included({0, 1, 4}, 3) returns true (value greater
/// than max)
///
/// \tparam ContainerType The type of the container
/// \tparam IntType The type of the integer values in the container
/// \param[in] values The container to check for out-of-range values
/// \param[in] max The maximum allowed value (exclusive)
/// \return True if the container includes any out-of-range values
/// \throws a runtime_error if the container includes any negative values (only
/// for signed integer types)
template <typename ContainerType, typename IntType>
bool is_out_of_range_value_included(const ContainerType& values, IntType max) {
  static_assert(
      std::is_integral_v<IntType>,
      "is_out_of_range_value_included: IntType must be an integral type");
  using value_type = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
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

/// \brief Check if the given axes are valid for the given view
/// Valid axes must satisfy the following conditions:
/// 1. The axes must be in the range of [0, rank-1] after converting negative
/// axes to non-negative axes, where rank is the rank of the view
/// 2. The axes must not contain duplicate values
/// Examples:
/// - are_valid_axes(view2d, {0, 1}) returns true (valid axes)
/// - are_valid_axes(view2d, {0, 0}) returns false (duplicate axes)
/// - are_valid_axes(view, {-1}) returns true (negative axis is converted to
/// non-negative axis)
/// - are_valid_axes(view, {rank}) returns false (axis equal to rank is out of
/// range)
/// - are_valid_axes(view, {rank + 1}) returns false (axis greater than rank is
/// out of range)
///
/// \tparam ViewType The type of the view
/// \tparam ArrayType The type of the array containing the axes
/// \tparam IntType The type of the integer values in the array
/// \tparam DIM The dimension of the array containing the axes
///
/// \param[in] view The view to check for valid axes
/// \param[in] axes The array containing the axes to check
/// \return True if the axes are valid for the view, false otherwise
template <
    typename ViewType, template <typename, std::size_t> class ArrayType,
    typename IntType, std::size_t DIM = 1,
    std::enable_if_t<Kokkos::is_view_v<ViewType> && std::is_integral_v<IntType>,
                     std::nullptr_t> = nullptr>
bool are_valid_axes(const ViewType& /*view*/,
                    const ArrayType<IntType, DIM>& axes) {
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
  std::array<IntType, DIM> non_negative_axes{};

  // In case axis is out of range, 'convert_negative_axes' will throw an
  // runtime_error and we will return false. Without runtime_error, it is
  // ensured that the 'non_negative_axes' are in the range of [0, rank-1]
  try {
    non_negative_axes = convert_negative_axes(axes, ViewType::rank());
  } catch (const std::runtime_error&) {
    return false;
  }

  bool is_valid = !has_duplicate_values(non_negative_axes);
  return is_valid;
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
