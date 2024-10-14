// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_LAYOUTS_HPP
#define KOKKOSFFT_LAYOUTS_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include <numeric>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_padding.hpp"

namespace KokkosFFT {
namespace Impl {
/* Input and output extents exposed to the fft library
   i.e extents are converted into Layout Right
*/
template <typename InViewType, typename OutViewType, std::size_t DIM = 1>
auto get_extents(const InViewType& in, const OutViewType& out,
                 axis_type<DIM> axes, shape_type<DIM> shape = {0}) {
  using in_value_type     = typename InViewType::non_const_value_type;
  using out_value_type    = typename OutViewType::non_const_value_type;
  using array_layout_type = typename InViewType::array_layout;

  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "input axes are not valid for the view");

  constexpr std::size_t rank = InViewType::rank;
  [[maybe_unused]] int inner_most_axis =
      std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft>
          ? 0
          : (rank - 1);

  // index map after transpose over axis
  auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(in, axes);

  // Get new shape based on shape parameter
  auto modified_in_shape =
      KokkosFFT::Impl::get_modified_shape(in, out, shape, axes);

  // Get extents for the inner most axes in LayoutRight
  // If we allow the FFT on the layoutLeft, this part should be modified
  std::vector<int> _in_extents, _out_extents, _fft_extents;
  for (std::size_t i = 0; i < rank; i++) {
    auto _idx       = map.at(i);
    auto in_extent  = modified_in_shape.at(_idx);
    auto out_extent = out.extent(_idx);
    _in_extents.push_back(in_extent);
    _out_extents.push_back(out_extent);

    // The extent for transform is always equal to the extent
    // of the extent of real type (R2C or C2R)
    // For C2C, the in and out extents are the same.
    // In the end, we can just use the largest extent among in and out extents.
    auto fft_extent = std::max(in_extent, out_extent);
    _fft_extents.push_back(fft_extent);
  }

  static_assert(!(is_real_v<in_value_type> && is_real_v<out_value_type>),
                "get_extents: real to real transform is not supported");

  if constexpr (is_real_v<in_value_type>) {
    // Then R2C
    KOKKOSFFT_THROW_IF(
        _out_extents.at(inner_most_axis) !=
            _in_extents.at(inner_most_axis) / 2 + 1,
        "For R2C, the 'output extent' of transform must be equal to "
        "'input extent'/2 + 1");
  }

  if constexpr (is_real_v<out_value_type>) {
    // Then C2R
    KOKKOSFFT_THROW_IF(
        _in_extents.at(inner_most_axis) !=
            _out_extents.at(inner_most_axis) / 2 + 1,
        "For C2R, the 'input extent' of transform must be equal to "
        "'output extent' / 2 + 1");
  }

  if constexpr (std::is_same_v<array_layout_type, Kokkos::LayoutLeft>) {
    std::reverse(_in_extents.begin(), _in_extents.end());
    std::reverse(_out_extents.begin(), _out_extents.end());
    std::reverse(_fft_extents.begin(), _fft_extents.end());
  }

  // Define subvectors starting from last - DIM
  // Dimensions relevant to FFTs
  std::vector<int> in_extents(_in_extents.end() - DIM, _in_extents.end());
  std::vector<int> out_extents(_out_extents.end() - DIM, _out_extents.end());
  std::vector<int> fft_extents(_fft_extents.end() - DIM, _fft_extents.end());

  int total_fft_size = std::accumulate(_fft_extents.begin(), _fft_extents.end(),
                                       1, std::multiplies<>());
  int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<>());
  int howmany  = total_fft_size / fft_size;

  return std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, int>(
      in_extents, out_extents, fft_extents, howmany);
}
}  // namespace Impl
};  // namespace KokkosFFT

#endif
