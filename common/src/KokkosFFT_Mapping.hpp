// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_MAPPING_HPP
#define KOKKOSFFT_MAPPING_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include <numeric>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {

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
/// \throws if axes are not valid for the view
template <typename Layout, std::size_t DIM, typename IntType,
          std::size_t FFT_DIM>
auto get_map_axes(const std::array<IntType, FFT_DIM>& axes) {
  static_assert(std::is_integral_v<IntType>,
                "get_map_axes: IntType must be an integral type.");
  static_assert(
      FFT_DIM >= 1 && FFT_DIM <= DIM,
      "get_map_axes: the Rank of FFT axes must be between 1 and View rank");

  // Convert the input axes to be in the range of [0, rank-1]
  auto non_negative_axes = convert_negative_axes(axes, DIM);

  // how indices are map
  // For 5D View and axes are (2,3), map would be (0, 1, 4, 2, 3)
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
/// \param[in] axes Axes over which FFT is performed
/// \return The mapping axes and inverse mapping axes as a tuple
/// \throws if axes are not valid for the view
template <typename ViewType, std::size_t FFT_DIM>
auto get_map_axes(const ViewType& view, const axis_type<FFT_DIM>& axes) {
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(view, axes),
                     "get_map_axes: input axes are not valid for the view");
  using LayoutType = typename ViewType::array_layout;
  return get_map_axes<LayoutType, ViewType::rank()>(axes);
}

/// \brief Mapping axes for transpose. With this mapping,
/// the input view is transposed into the contiguous order which is expected by
/// the FFT plan.
///
/// \tparam ViewType The type of the input view
///
/// \param[in] axis Axis over which FFT is performed
/// \return The mapping axes and inverse mapping axes as a tuple
/// \throws if axes are not valid for the view
template <typename ViewType>
auto get_map_axes(const ViewType& view, int axis) {
  return get_map_axes(view, axis_type<1>({axis}));
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
