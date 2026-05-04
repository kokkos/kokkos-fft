// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_MAPPING_HPP
#define KOKKOSFFT_MAPPING_HPP

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"

namespace KokkosFFT {
namespace Impl {

/// \brief Check if a value is found in a container
/// Examples:
/// - is_found({1, 2, 3}, 2) returns true
/// - is_found({1, 2, 3}, 4) returns false
///
/// \tparam ContainerType The type of the container
/// \tparam ValueType The type of the value to search for
/// \param[in] values The container to search in
/// \param[in] value The value to search for
/// \return True if the value is found in the container, false otherwise
template <typename ContainerType, typename ValueType>
bool is_found(const ContainerType& values, const ValueType value) {
  using value_type = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
  static_assert(std::is_same_v<value_type, ValueType>,
                "is_found: Container value type must match ValueType");
  return std::find(values.begin(), values.end(), value) != values.end();
}

/// \brief Get the index of a value in a container
/// Examples:
/// - get_index({1, 2, 3}, 2) returns 1
/// - get_index({1, 2, 3}, 4) throws an exception
///
/// \tparam ContainerType The type of the container
/// \tparam ValueType The type of the value to search for
/// \param[in] values The container to search in
/// \param[in] value The value to search for
/// \return The index of the value in the container
/// \throws a runtime_error if the value is not found in the container
template <typename ContainerType, typename ValueType>
std::size_t get_index(const ContainerType& values, const ValueType value) {
  using value_type = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
  static_assert(std::is_same_v<value_type, ValueType>,
                "get_index: Container value type must match ValueType");
  auto it = std::find(values.begin(), values.end(), value);
  KOKKOSFFT_THROW_IF(it == values.end(), "value is not included in values");
  return it - values.begin();
}

/// \brief Converts axes in [-rank, rank-1] to [0, rank-1]
/// \tparam IntType The integer type used for axis
/// \tparam DIM The dimensionality of the axes
///
/// \param[in] axes The axes to be converted
/// \param[in] rank The rank of the view
/// \return The converted axes
/// \throws a runtime_error if any axis is out of range
template <typename IntType, std::size_t DIM>
auto convert_negative_axes(const std::array<IntType, DIM>& axes,
                           std::size_t rank) {
  static_assert(std::is_integral_v<IntType>,
                "convert_negative_axes: IntType must be an integral type.");
  std::array<IntType, DIM> non_negative_axes{};

  const IntType irank        = static_cast<IntType>(rank);
  auto convert_negative_axis = [irank](IntType axis) -> IntType {
    if constexpr (std::is_signed_v<IntType>) {
      KOKKOSFFT_THROW_IF(axis < -irank || axis >= irank,
                         "All axes must be in [-rank, rank-1]");
      return axis < 0 ? irank + axis : axis;
    } else {
      KOKKOSFFT_THROW_IF(axis >= irank, "All axes must be in [0, rank-1]");
      return axis;
    }
  };

  std::transform(axes.begin(), axes.end(), non_negative_axes.begin(),
                 convert_negative_axis);

  return non_negative_axes;
}

/// \brief Mapping axes for transpose. With this mapping,
/// the input view is transposed into the contiguous order which is expected by
/// the FFT plan.
///
/// \tparam Layout The layout of the input view
/// \tparam DIM The dimensionality of the input view
/// \tparam IntType The type of axes
/// \tparam FFT_DIM The dimensionality of the FFT axes
///
/// \param[in] axes Axes over which FFT is performed
/// \return The mapping axes and inverse mapping axes as a tuple
/// \throws std::runtime_error if any axis is out of range
/// \pre axes must not contain duplicates after negative-axis conversion
template <typename Layout, std::size_t DIM, typename IntType,
          std::size_t FFT_DIM>
auto get_map_axes(const std::array<IntType, FFT_DIM>& axes) {
  static_assert(std::is_integral_v<IntType>,
                "get_map_axes: IntType must be an integral type.");
  static_assert(
      FFT_DIM >= 1 && FFT_DIM <= DIM,
      "get_map_axes: the Rank of FFT axes must be between 1 and View rank");

  // Convert the input axes to be in the range [0, rank-1]
  auto non_negative_axes = convert_negative_axes(axes, DIM);

  // How indices are mapped
  // For 5D View and axes (2,3), the map would be (0, 1, 4, 2, 3)
  constexpr IntType rank = static_cast<IntType>(DIM);
  std::vector<IntType> map;
  map.reserve(rank);

  if (std::is_same_v<Layout, Kokkos::LayoutRight>) {
    // Stack axes not specified by axes (0, 1, 4)
    for (IntType i = 0; i < rank; i++) {
      if (!is_found(non_negative_axes, i)) {
        map.push_back(i);
      }
    }

    // Stack axes on the map (For layout Right)
    // Then stack (2, 3) to have (0, 1, 4, 2, 3)
    for (auto axis : non_negative_axes) {
      map.push_back(axis);
    }
  } else {
    // For layout Left, stack innermost axes first
    std::reverse(non_negative_axes.begin(), non_negative_axes.end());
    for (auto axis : non_negative_axes) {
      map.push_back(axis);
    }

    // Then stack remaining axes
    for (IntType i = 0; i < rank; i++) {
      if (!is_found(non_negative_axes, i)) {
        map.push_back(i);
      }
    }
  }

  using full_axis_type     = std::array<IntType, rank>;
  full_axis_type array_map = {}, array_map_inv = {};
  std::copy_n(map.begin(), rank, array_map.begin());

  // Construct inverse map
  for (IntType i = 0; i < rank; i++) {
    array_map_inv.at(i) = get_index(array_map, i);
  }

  return std::make_tuple(array_map, array_map_inv);
}

/// \brief Mapping axes for transpose. With this mapping,
/// the input view is transposed into the contiguous order which is expected by
/// the FFT plan.
///
/// \tparam ViewType The type of the input view
/// \tparam FFT_DIM The dimensionality of the FFT axes
///
/// \param[in] view The input view (used for type deduction)
/// \param[in] axes Axes over which FFT is performed
/// \return The mapping axes and inverse mapping axes as a tuple
template <typename ViewType, std::size_t FFT_DIM>
auto get_map_axes(const ViewType& /*view*/, const axis_type<FFT_DIM>& axes) {
  using LayoutType = typename ViewType::array_layout;
  return get_map_axes<LayoutType, ViewType::rank()>(axes);
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
