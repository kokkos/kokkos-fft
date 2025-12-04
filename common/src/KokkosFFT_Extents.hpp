// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_EXTENTS_HPP
#define KOKKOSFFT_EXTENTS_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include <numeric>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_transpose.hpp"

namespace KokkosFFT {
namespace Impl {
/// \brief Compute the extent after FFT
/// For real to complex case, the output extent is
/// out_extent = in_extent / 2 + 1
/// For complex to complex case, the output extent is
/// out_extent = in_extent
///
/// \param[in] extent The input extent
/// \param[in] is_R2C Whether it is real to complex or not
/// \return The output extent
inline auto extent_after_transform(std::size_t extent, bool is_R2C) {
  return is_R2C ? (extent / 2 + 1) : extent;
}

/// \brief Return a new shape of the input view based on the
/// specified input shape and axes.
///
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam DIM         The dimensionality of the shape and axes
///
/// \param in [in] Input view from which to derive the new shape
/// \param out [in] Output view (unused but necessary for type deduction)
/// \param shape [in] The new shape of the input view. If the shape is zero,
/// no modifications are made.
/// \param axes [in] Axes over which the shape modification is applied.
template <typename InViewType, typename OutViewType, std::size_t DIM>
auto get_modified_shape(const InViewType in, const OutViewType /* out */,
                        shape_type<DIM> shape, axis_type<DIM> axes) {
  static_assert(
      KokkosFFT::Impl::have_same_rank_v<InViewType, OutViewType>,
      "get_modified_shape: Input View and Output View must have the same rank");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "input axes are not valid for the view");

  shape_type<DIM> zeros{};  // default shape means no crop or pad
  if (shape == zeros) {
    return KokkosFFT::Impl::extract_extents(in);
  }

  // Convert the input axes to be in the range of [0, rank-1]
  constexpr std::size_t rank = InViewType::rank();
  auto non_negative_axes     = convert_negative_axes(axes, rank);

  using full_shape_type = shape_type<rank>;
  full_shape_type modified_shape;
  for (std::size_t i = 0; i < rank; i++) {
    modified_shape.at(i) = in.extent(i);
  }

  // Update shapes based on newly given shape
  for (std::size_t i = 0; i < DIM; i++) {
    auto non_negative_axis = non_negative_axes.at(i);
    KOKKOSFFT_THROW_IF(shape.at(i) <= 0,
                       "get_modified_shape: shape must be greater than 0");
    modified_shape.at(non_negative_axis) = shape.at(i);
  }

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  bool is_C2R = is_complex_v<in_value_type> && is_real_v<out_value_type>;

  if (is_C2R) {
    auto reshaped_axis               = non_negative_axes.back();
    modified_shape.at(reshaped_axis) = modified_shape.at(reshaped_axis) / 2 + 1;
  }

  return modified_shape;
}

/// \brief Compute input, output and fft extents required for FFT
/// libraries based on the input view, output view, axes and shape.
/// Extents are converted into Layout Right
///
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
///
/// \param[in] in Input view
/// \param[in] out Output view
/// \param[in] map mapping for permutation
/// \param[in] axes Axes over which the FFT operations are performed
/// \param[in] modified_in_shape The new shape of the input view
/// \param[in] is_inplace Whether the FFT is inplace or not
template <typename InViewType, typename OutViewType>
auto get_extents(const InViewType& in, const OutViewType& out,
                 const std::vector<int>& map,
                 [[maybe_unused]] const std::vector<int>& axes,
                 const std::vector<std::size_t>& modified_in_shape,
                 [[maybe_unused]] bool is_inplace) {
  using in_value_type     = typename InViewType::non_const_value_type;
  using out_value_type    = typename OutViewType::non_const_value_type;
  using array_layout_type = typename InViewType::array_layout;

  static_assert(!(is_real_v<in_value_type> && is_real_v<out_value_type>),
                "get_extents: real to real transform is not supported");

  constexpr std::size_t rank = InViewType::rank;
  [[maybe_unused]] int inner_most_axis =
      std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft>
          ? 0
          : (rank - 1);

  // Get extents for the inner most axes in LayoutRight
  // If we allow the FFT on the layoutLeft, this part should be modified
  std::vector<int> in_extents_full, out_extents_full, fft_extents_full;
  for (std::size_t i = 0; i < rank; i++) {
    auto idx        = map.at(i);
    auto in_extent  = modified_in_shape.at(idx);
    auto out_extent = out.extent(idx);
    in_extents_full.push_back(in_extent);
    out_extents_full.push_back(out_extent);

    // The extent for transform is always equal to the extent
    // of the extent of real type (R2C or C2R)
    // For C2C, the in and out extents are the same.
    // In the end, we can just use the largest extent among in and out extents.
    auto fft_extent = std::max(in_extent, out_extent);
    fft_extents_full.push_back(fft_extent);
  }

  auto mismatched_extents = [&in, &out, &axes]() -> std::string {
    std::string message;
    message += in.label();
    message += "(";
    message += std::to_string(in.extent(0));
    for (std::size_t r = 1; r < rank; r++) {
      message += ",";
      message += std::to_string(in.extent(r));
    }
    message += "), ";
    message += out.label();
    message += "(";
    message += std::to_string(out.extent(0));
    for (std::size_t r = 1; r < rank; r++) {
      message += ",";
      message += std::to_string(out.extent(r));
    }
    message += "), with axes (";
    message += std::to_string(axes.at(0));
    for (std::size_t i = 1; i < axes.size(); i++) {
      message += ",";
      message += std::to_string(axes.at(i));
    }
    message += ")";
    return message;
  };

  for (std::size_t i = 0; i < rank; i++) {
    // The requirement for inner_most_axis is different for transform type
    if (static_cast<int>(i) == inner_most_axis) continue;
    KOKKOSFFT_THROW_IF(in_extents_full.at(i) != out_extents_full.at(i),
                       "input and output extents must be the same except for "
                       "the transform axis: " +
                           mismatched_extents());
  }

  if constexpr (is_complex_v<in_value_type> && is_complex_v<out_value_type>) {
    // Then C2C
    KOKKOSFFT_THROW_IF(
        in_extents_full.at(inner_most_axis) !=
            out_extents_full.at(inner_most_axis),
        "input and output extents must be the same for C2C transform: " +
            mismatched_extents());
  }

  if constexpr (is_real_v<in_value_type>) {
    // Then R2C
    if (is_inplace) {
      in_extents_full.at(inner_most_axis) =
          out_extents_full.at(inner_most_axis) * 2;
    } else {
      KOKKOSFFT_THROW_IF(
          out_extents_full.at(inner_most_axis) !=
              in_extents_full.at(inner_most_axis) / 2 + 1,
          "For R2C, the 'output extent' of transform must be equal to "
          "'input extent'/2 + 1: " +
              mismatched_extents());
    }
  }

  if constexpr (is_real_v<out_value_type>) {
    // Then C2R
    if (is_inplace) {
      out_extents_full.at(inner_most_axis) =
          in_extents_full.at(inner_most_axis) * 2;
    } else {
      KOKKOSFFT_THROW_IF(
          in_extents_full.at(inner_most_axis) !=
              out_extents_full.at(inner_most_axis) / 2 + 1,
          "For C2R, the 'input extent' of transform must be equal to "
          "'output extent' / 2 + 1: " +
              mismatched_extents());
    }
  }

  if constexpr (std::is_same_v<array_layout_type, Kokkos::LayoutLeft>) {
    std::reverse(in_extents_full.begin(), in_extents_full.end());
    std::reverse(out_extents_full.begin(), out_extents_full.end());
    std::reverse(fft_extents_full.begin(), fft_extents_full.end());
  }

  // Define subvectors starting from last - DIM
  // Dimensions relevant to FFTs
  const std::size_t DIM = axes.size();
  std::vector<int> in_extents(in_extents_full.end() - DIM,
                              in_extents_full.end());
  std::vector<int> out_extents(out_extents_full.end() - DIM,
                               out_extents_full.end());
  std::vector<int> fft_extents(fft_extents_full.end() - DIM,
                               fft_extents_full.end());

  auto total_fft_size = total_size(fft_extents_full);
  auto fft_size       = total_size(fft_extents);
  auto howmany        = total_fft_size / fft_size;

  return std::make_tuple(in_extents, out_extents, fft_extents, howmany);
}

/// \brief Compute input, output and fft extents required for FFT
/// libraries based on the input view, output view, axes and shape.
/// Extents are converted into Layout Right
///
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam DIM         The dimensionality of the axes
///
/// \param[in] in Input view
/// \param[in] out Output view
/// \param[in] axes Axes over which the FFT operations are performed.
/// \param[in] shape The new shape of the input view. If the shape is zero,
/// no modifications are made.
/// \param[in] is_inplace Whether the FFT is inplace or not
template <typename InViewType, typename OutViewType, std::size_t DIM = 1>
auto get_extents(const InViewType& in, const OutViewType& out,
                 axis_type<DIM> axes, shape_type<DIM> shape = {},
                 bool is_inplace = false) {
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "input axes are not valid for the view");

  // index map after transpose over axis
  [[maybe_unused]] auto [map, map_inv] =
      KokkosFFT::Impl::get_map_axes(in, axes);

  // Get new shape based on shape parameter
  auto modified_in_shape =
      KokkosFFT::Impl::get_modified_shape(in, out, shape, axes);

  auto map_vec               = to_vector(map);
  auto axes_vec              = to_vector(axes);
  auto modified_in_shape_vec = to_vector(modified_in_shape);

  return get_extents(in, out, map_vec, axes_vec, modified_in_shape_vec,
                     is_inplace);
}

/// \brief Compute input, output and fft extents required for FFT
/// libraries based on the input view, output view, axes and shape.
/// Extents are converted into Layout Right
///
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
///
/// \param[in] in Input view
/// \param[in] out Output view
/// \param[in] dim The dimensionality of FFT
/// \param[in] is_inplace  Whether the FFT is inplace or not
template <typename InViewType, typename OutViewType>
auto get_extents(const InViewType& in, const OutViewType& out, std::size_t dim,
                 bool is_inplace = false) {
  using LayoutType = typename InViewType::array_layout;

  // Contiguous map (no permutation)
  std::vector<int> map(InViewType::rank());
  std::iota(map.begin(), map.end(), 0);

  std::vector<int> axes(dim);
  if constexpr (std::is_same_v<LayoutType, typename Kokkos::LayoutLeft>) {
    std::iota(axes.begin(), axes.end(), 0);
    std::reverse(axes.begin(), axes.end());
  } else {
    std::iota(axes.begin(), axes.end(), -dim);
  }

  auto in_shape     = extract_extents(in);
  auto in_shape_vec = to_vector(in_shape);

  return get_extents(in, out, map, axes, in_shape_vec, is_inplace);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
