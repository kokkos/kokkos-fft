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
#include "KokkosFFT_Layout.hpp"

namespace KokkosFFT {
namespace Impl {
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
      using LayoutType = typename InViewType::array_layout;
      static const Kokkos::Iterate outer_iteration_pattern =
          layout_iterate_type_selector<LayoutType>::outer_iteration_pattern;
      static const Kokkos::Iterate inner_iteration_pattern =
          layout_iterate_type_selector<LayoutType>::inner_iteration_pattern;
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, outer_iteration_pattern,
                       inner_iteration_pattern>;
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
  static_assert(
      is_operatable_view_v<ExecutionSpace, InViewType>,
      "transpose: In View value type must be float, double, "
      "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
      "Layout must be either LayoutLeft or LayoutRight. "
      "The data in InViewType must be accessible from ExecutionSpace.");

  static_assert(
      is_operatable_view_v<ExecutionSpace, OutViewType>,
      "transpose: Out View value type must be float, double, "
      "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
      "Layout must be either LayoutLeft or LayoutRight. "
      "The data in OutViewType must be accessible from ExecutionSpace.");

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
