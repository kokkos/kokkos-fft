// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_MAPPING_HPP
#define KOKKOSFFT_DISTRIBUTED_MAPPING_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Get the mapping of the destination view from
/// src mapping. In the middle of the parallel FFTs,
/// the axis of the view can be changed which is stored in
/// the src_map. The dst_map is the mapping that is ready
/// for FFTs along the innermost direction.
///
/// E.g. Src Mapping (0, 1, 2) -> (0, 2, 1)
///      This corresponds to the mapping of
///      x -> x, y -> z, z -> y
///
///      Layout Left
///      axis == 0 -> (0, 2, 1)
///      axis == 1 -> (1, 0, 2)
///      axis == 2 -> (2, 0, 1)
///
///      Layout Right
///      axis == 0 -> (2, 1, 0)
///      axis == 1 -> (0, 2, 1)
///      axis == 2 -> (0, 1, 2)
/// [TO DO] Add a test case with src_map is not
/// in ascending order
/// \tparam LayoutType The layout type of the view
/// \tparam DIM        The dimensionality of the map
///
/// \param[in] src_map The axis map of the input view
/// \param[in] axis    The axis to be merged/split
template <typename LayoutType, typename ContainerType, typename iType,
          std::size_t DIM>
std::array<iType, DIM> permute_map_by_axes(
    const std::array<iType, DIM>& src_map, const ContainerType& axes) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*axes.begin())>>;
  static_assert(std::is_same_v<value_type, iType>,
                "permute_map_by_axes: Container value type must match iType");

  std::vector<iType> map;
  map.reserve(DIM);
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    for (auto src_idx : src_map) {
      if (!KokkosFFT::Impl::is_found(axes, src_idx)) {
        map.push_back(src_idx);
      }
    }
    for (auto axis : axes) {
      map.push_back(axis);
    }
  } else {
    // For layout Left, stack innermost axes first
    auto axes_reversed = KokkosFFT::Impl::reversed(axes);
    for (auto axis : axes_reversed) {
      map.push_back(axis);
    }

    // Then stack remaining axes
    for (auto src_idx : src_map) {
      if (!KokkosFFT::Impl::is_found(axes_reversed, src_idx)) {
        map.push_back(src_idx);
      }
    }
  }

  using full_axis_type = std::array<iType, DIM>;
  full_axis_type dst_map{};
  std::copy_n(map.begin(), DIM, dst_map.begin());

  return dst_map;
}

/// \brief Get the mapping of the destination view from
/// src mapping. In the middle of the parallel FFTs,
/// the axis of the view can be changed which is stored in
/// the src_map. The dst_map is the mapping that is ready
/// for FFTs along the innermost direction.
///
/// E.g. Src Mapping (0, 1, 2) -> (0, 2, 1)
///      This corresponds to the mapping of
///      x -> x, y -> z, z -> y
///
///      Layout Left
///      axis == 0 -> (0, 2, 1)
///      axis == 1 -> (1, 0, 2)
///      axis == 2 -> (2, 0, 1)
///
///      Layout Right
///      axis == 0 -> (2, 1, 0)
///      axis == 1 -> (0, 2, 1)
///      axis == 2 -> (0, 1, 2)
///
/// \tparam LayoutType The layout type of the view
/// \tparam DIM        The dimensionality of the map
///
/// \param[in] src_map The axis map of the input view
/// \param[in] axis    The axis to be merged/split
template <typename LayoutType, typename iType, std::size_t DIM>
auto permute_map_by_axes(const std::array<iType, DIM>& src_map, iType axis) {
  return permute_map_by_axes<LayoutType>(src_map, std::vector<iType>{axis});
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
