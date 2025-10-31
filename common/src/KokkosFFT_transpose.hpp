// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TRANSPOSE_HPP
#define KOKKOSFFT_TRANSPOSE_HPP

#include <numeric>
#include <tuple>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_padding.hpp"

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

template <class ViewType>
axis_type<ViewType::rank()> compute_transpose_extents(
    ViewType const& view, axis_type<ViewType::rank()> const& map) {
  static_assert(Kokkos::is_view_v<ViewType>,
                "compute_transpose_extents: ViewType must be a Kokkos::View.");
  constexpr std::size_t rank = ViewType::rank();

  axis_type<rank> out_extents;
  for (std::size_t i = 0; i < rank; ++i) {
    out_extents.at(i) = view.extent(map.at(i));
  }

  return out_extents;
}

struct BoundsCheck {
  struct On {};
  struct Off {};
};

/// \brief Transpose functor for out-of-place transpose operations.
/// This struct implements a functor that applies a transpose on a Kokkos view.
/// Before FFT, the input view is transposed into the order which is expected by
/// the FFT plan. After FFT, the input view is transposed back into the original
/// order.
///
/// \tparam ExecutionSpace The type of Kokkos execution space.
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam IndexType The index type used for the map
/// \tparam ArgBoundsCheck The bounds check type (default is Off)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename IndexType, typename ArgBoundsCheck = BoundsCheck::Off>
struct Transpose {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(InViewType::rank(), std::size_t(6));

  using ArrayType = Kokkos::Array<int, InViewType::rank()>;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] x The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const InViewType& x) const {
    if constexpr (InViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<IndexType>>;
      return range_policy_type(space, 0, x.extent(0));
    } else {
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, Kokkos::Iterate::Default,
                       Kokkos::Iterate::Default>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<IndexType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins = {};
      Kokkos::Array<std::size_t, m_rank_truncated> ends   = {};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = x.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructor for the Transpose functor.
  ///
  /// \param[in] in The input Kokkos view to be transposed.
  /// \param[out] out The output Kokkos view after transpose.
  /// \param[in] map The indices mapping of transpose
  /// \param[in] exec_space The Kokkos execution space to be used (defaults
  /// to ExecutionSpace()).
  Transpose(const InViewType& in, const OutViewType& out, const ArrayType& map,
            const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::transpose", get_policy(exec_space, in),
                         TransposeInternal(in, out, map));
  }

  /// \brief Helper functor to perform the transpose operation
  struct TransposeInternal {
    InViewType m_in;
    OutViewType m_out;
    ArrayType m_map;

    TransposeInternal(const InViewType& in, const OutViewType& out,
                      const ArrayType& map)
        : m_in(in), m_out(out), m_map(map) {}

    template <typename... IndicesType>
    KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
      if constexpr (InViewType::rank() <= 6) {
        IndexType src_indices[InViewType::rank()] = {
            static_cast<IndexType>(indices)...};
        transpose_internal(src_indices,
                           std::make_index_sequence<InViewType::rank()>{});
      } else if constexpr (InViewType::rank() == 7) {
        for (IndexType i6 = 0; i6 < IndexType(m_in.extent(6)); i6++) {
          IndexType src_indices[InViewType::rank()] = {
              static_cast<IndexType>(indices)..., i6};
          transpose_internal(src_indices,
                             std::make_index_sequence<InViewType::rank()>{});
        }
      } else if constexpr (InViewType::rank() == 8) {
        for (IndexType i6 = 0; i6 < IndexType(m_in.extent(6)); i6++) {
          for (IndexType i7 = 0; i7 < IndexType(m_in.extent(7)); i7++) {
            IndexType src_indices[InViewType::rank()] = {
                static_cast<IndexType>(indices)..., i6, i7};
            transpose_internal(src_indices,
                               std::make_index_sequence<InViewType::rank()>{});
          }
        }
      }
    }

    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION void transpose_internal(
        IndexType src_idx[], std::index_sequence<Is...>) const {
      if constexpr (std::is_same_v<ArgBoundsCheck, BoundsCheck::On>) {
        // Bounds check
        bool in_bounds = true;
        for (std::size_t i = 0; i < InViewType::rank(); ++i) {
          if (src_idx[m_map[i]] >= IndexType(m_out.extent(i)))
            in_bounds = false;
        }

        if (in_bounds) {
          m_out(src_idx[m_map[Is]]...) = m_in(src_idx[Is]...);
        }
      } else {
        m_out(src_idx[m_map[Is]]...) = m_in(src_idx[Is]...);
      }
    }
  };
};

/// \brief Make the axis direction to the inner most direction
/// axis should be the range in [-(rank-1), rank-1], where
/// negative number is interpreted as rank + axis.
/// E.g. axis = -1 for rank 3 matrix is interpreted as axis = 2
///
/// E.g.
///      Layout Left
///      A (3, 4, 2) and axis = 1 -> A' (4, 3, 2)
///      B (2, 4, 3, 5) and axis = 2 -> B' (3, 2, 4, 5)
///      C (8, 6, 3) and axis = 0 -> C' (8, 6, 3)
///      D (7, 5) and axis = -1 -> D' (5, 7)
///
///      Layout Right
///      A (3, 4, 2) and axis = 1 -> A' (3, 2, 4)
///      B (2, 4, 3, 5) and axis = 2 -> B' (2, 4, 5, 3)
///      C (8, 6, 3) and axis = 0 -> C' (6, 3, 8)
///      D (5, 7) and axis = -1 -> D' (5, 7)
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam IndexType The index type used for the view
///
/// \param[in] exec_space execution space instance
/// \param[in] in The input view
/// \param[out] out The output view
/// \param[in] map The axis map for transpose
/// \param[in] bounds_check Perform bounds checking on the output view (default:
/// false)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename IndexType>
void transpose(const ExecutionSpace& exec_space, const InViewType& in,
               const OutViewType& out,
               std::array<IndexType, InViewType::rank()> map,
               bool bounds_check = false) {
  static_assert(is_operatable_view_v<ExecutionSpace, InViewType>,
                "transpose: In View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "ExecutionSpace must be able to access data in ViewType");

  static_assert(is_operatable_view_v<ExecutionSpace, OutViewType>,
                "transpose: Out View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "ExecutionSpace must be able to access data in ViewType");

  static_assert(have_same_rank_v<InViewType, OutViewType>,
                "transpose: In and Out View must have the same rank.");

  static_assert(
      have_same_base_floating_point_type_v<InViewType, OutViewType>,
      "transpose: In and Out View must have the same base floating point "
      "type.");

  if (!is_transpose_needed(map)) {
    // Just perform deep_copy (Layout may change)
    KokkosFFT::Impl::crop_or_pad_impl(
        exec_space, in, out, std::make_index_sequence<InViewType::rank()>{});
    return;
  }

  Kokkos::Array<IndexType, InViewType::rank()> map_array = to_array(map);
  if ((in.span() >= std::size_t(std::numeric_limits<int>::max())) ||
      (out.span() >= std::size_t(std::numeric_limits<int>::max()))) {
    if (bounds_check) {
      Transpose<ExecutionSpace, InViewType, OutViewType, int64_t,
                BoundsCheck::On>(in, out, map_array, exec_space);
    } else {
      Transpose<ExecutionSpace, InViewType, OutViewType, int64_t,
                BoundsCheck::Off>(in, out, map_array, exec_space);
    }
  } else {
    if (bounds_check) {
      Transpose<ExecutionSpace, InViewType, OutViewType, int, BoundsCheck::On>(
          in, out, map_array, exec_space);
    } else {
      Transpose<ExecutionSpace, InViewType, OutViewType, int, BoundsCheck::Off>(
          in, out, map_array, exec_space);
    }
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
