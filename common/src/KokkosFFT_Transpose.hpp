// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TRANSPOSE_HPP
#define KOKKOSFFT_TRANSPOSE_HPP

#include <array>
#include <limits>
#include <numeric>
#include <utility>
#include <type_traits>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_Layout.hpp"
#include "KokkosFFT_MDOperations.hpp"
#include "KokkosFFT_Padding.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {
struct BoundsCheck {
  struct On {};
  struct Off {};
};

/// \brief Check if transpose is needed or not
/// If a map is contiguous and in ascending order (e.g. {0, 1, 2}),
/// we do not need transpose
/// \tparam IndexType The integer type used for map
/// \tparam DIM The dimensionality of the axes
///
/// \param[in] map The map used for permutation
/// \return true if transpose is needed, false otherwise
template <typename IndexType, std::size_t DIM>
bool is_transpose_needed(const std::array<IndexType, DIM>& map) {
  static_assert(std::is_integral_v<IndexType>,
                "is_transpose_needed: IndexType must be an integral type.");
  std::array<IndexType, DIM> contiguous_map;
  std::iota(contiguous_map.begin(), contiguous_map.end(), 0);
  return map != contiguous_map;
}

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
  using ArrayType = Kokkos::Array<int, InViewType::rank()>;

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
    Kokkos::parallel_for(
        "KokkosFFT::transpose",
        KokkosFFT::Impl::get_mdpolicy<IndexType>(exec_space, in),
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
        for (std::size_t i = 0; i < InViewType::rank(); ++i) {
          if (src_idx[m_map[i]] >= IndexType(m_out.extent(i))) {
            // Quick return for out-of-bounds
            return;
          }
        }
      }
      m_out(src_idx[m_map[Is]]...) = m_in(src_idx[Is]...);
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
