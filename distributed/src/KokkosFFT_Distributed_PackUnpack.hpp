// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_PACKUNPACK_HPP
#define KOKKOSFFT_DISTRIBUTED_PACKUNPACK_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Mapping.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Helper function to compute the start index and length of a bin
/// when dividing a range [0, N) into `nbins` bins. The `ibin`-th bin
/// will contain the indices from `start` to `start + length - 1`.
/// The bins are distributed as evenly as possible, with the first `remainder`
/// bins containing one extra element if `N` is not perfectly divisible by
/// `nbins`.
/// For example, if N=10 and nbins=3, the bins will be:
/// [0, 4), [4, 7), [7, 10)
/// which corresponds to (start=0, length=4), (start=4, length=3),
/// (start=7, length=3).
/// Note, `nbins` corresponds to the number of processes in the communicator and
/// `ibin` corresponds to the rank of the process.
///
/// \tparam iType Integer type for indices
/// \param[in] N Total number of elements
/// \param[in] nbins Number of bins to divide into
/// \param[in] ibin Index of the bin to compute (0-based)
/// \return A pair containing the start index and length of the `ibin`-th bin
template <typename iType>
KOKKOS_INLINE_FUNCTION auto bin_mapping(iType N, iType nbins, iType ibin) {
  iType base_size = N / nbins, remainder = N % nbins;
  iType length = base_size + (ibin < remainder ? 1 : 0);
  iType start  = ibin * base_size + Kokkos::min(ibin, remainder);
  return Kokkos::pair<iType, iType>{start, length};
}

/// \brief If `axis` is not equal to `merged_axis`, it returns `idx` unchanged.
/// If `axis` is equal to `merged_axis`, it merges the index `idx` with the
/// `start` and `extent` of a bin along the specified `axis`. If `idx` is
/// outside the range of the bin, it returns -1. Otherwise, it returns the
/// global index by adding `start` to `idx`.
/// `start` and `extent` are obtained from `bin_mapping` function defined above
///
/// \tparam iType Integer type for indices
/// \param[in] idx Local index to be merged
/// \param[in] start Start index of the bin
/// \param[in] extent Length of the bin
/// \param[in] axis Axis along which the bin is defined
/// \param[in] merged_axis Axis that is being merged
/// \return The global index if `idx` is within the bin, otherwise -1
template <typename iType>
KOKKOS_INLINE_FUNCTION iType merge_indices(iType idx, iType start, iType extent,
                                           std::size_t axis,
                                           std::size_t merged_axis) {
  if (axis == merged_axis) {
    return idx >= extent ? -1 : idx + start;
  } else {
    return idx;
  }
}

/// \brief Pack data from ND source view to N+1D destination view by splitting
/// along a specified axis. The destination view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam SrcViewType Kokkos View type of the source data
/// \tparam DstViewType Kokkos View type of the destination data
/// \tparam iType Integer type for indices
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          typename iType>
struct Pack {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(DstViewType::rank(), std::size_t(6));

  using ShapeType = Kokkos::Array<std::size_t, SrcViewType::rank()>;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] x The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const DstViewType& x) const {
    if constexpr (DstViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, x.extent(0));
    } else {
      using LayoutType = typename DstViewType::array_layout;
      static const Kokkos::Iterate outer_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::outer_iteration_pattern;
      static const Kokkos::Iterate inner_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::inner_iteration_pattern;
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, outer_iteration_pattern,
                       inner_iteration_pattern>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins = {};
      Kokkos::Array<std::size_t, m_rank_truncated> ends   = {};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = x.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructor for the Pack functor.
  ///
  /// \param[in] src The input Kokkos view to be packed
  /// \param[out] dst The output Kokkos view to be packed
  /// \param[in] axis The axis to be split
  /// \param[in] merged_size The extent of the dimension to be split
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Pack(const SrcViewType& src, const DstViewType& dst, const ShapeType& map,
       const std::size_t axis, const std::size_t merged_size,
       const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::Distributed::Pack",
                         get_policy(exec_space, dst),
                         PackInternal(src, dst, map, axis, merged_size));
  }

  struct PackInternal {
    using LayoutType = typename DstViewType::array_layout;
    using ValueType  = typename SrcViewType::non_const_value_type;
    SrcViewType m_src;
    DstViewType m_dst;
    std::size_t m_axis, m_merged_size, m_nprocs = 1;
    ShapeType m_map;
    ShapeType m_dst_extents;
    ShapeType m_src_extents;

    PackInternal(const SrcViewType& src, const DstViewType& dst,
                 const ShapeType& map, const std::size_t axis,
                 const std::size_t merged_size)
        : m_src(src),
          m_dst(dst),
          m_axis(axis),
          m_merged_size(merged_size),
          m_map(map) {
      for (std::size_t i = 0; i < SrcViewType::rank(); ++i) {
        m_dst_extents[i] = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                               ? dst.extent(i + 1)
                               : dst.extent(i);
      }
      m_nprocs = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                     ? dst.extent(0)
                     : dst.extent(DstViewType::rank() - 1);
      for (std::size_t i = 0; i < SrcViewType::rank(); ++i) {
        m_src_extents[i] = src.extent(i);
      }
    }

    template <typename... IndicesType>
    KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
      if constexpr (DstViewType::rank() <= 6) {
        iType dst_indices[DstViewType::rank()] = {
            static_cast<iType>(indices)...};
        m_dst(indices...) = get_src_value(
            dst_indices, std::make_index_sequence<DstViewType::rank() - 1>{});
      } else if constexpr (DstViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          iType dst_indices[DstViewType::rank()] = {
              static_cast<iType>(indices)..., i6};
          m_dst(indices..., i6) = get_src_value(
              dst_indices, std::make_index_sequence<DstViewType::rank() - 1>{});
        }
      } else if constexpr (DstViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_dst.extent(7)); i7++) {
            iType dst_indices[DstViewType::rank()] = {
                static_cast<iType>(indices)..., i6, i7};
            m_dst(indices..., i6, i7) = get_src_value(
                dst_indices,
                std::make_index_sequence<DstViewType::rank() - 1>{});
          }
        }
      }
    }

    /// \brief Get the value from the source view corresponding to the given
    /// destination indices.
    /// This function computes the source indices by merging the destination
    /// indices with the start and extent of the bin along the specified axis.
    /// If any of the computed source indices are out of bounds, it returns 0.
    ///
    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION ValueType
    get_src_value(iType dst_idx[], std::index_sequence<Is...>) const {
      // Bounds check
      bool out_of_bounds = false;
      if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        const iType p = dst_idx[DstViewType::rank() - 1];
        const auto [start, extent] =
            bin_mapping(iType(m_merged_size), iType(m_nprocs), p);
        iType src_indices[DstViewType::rank() - 1] = {
            merge_indices(dst_idx[Is], start, extent, Is, m_axis)...};
        for (std::size_t i = 0; i < DstViewType::rank() - 1; ++i) {
          if (src_indices[m_map[i]] >= iType(m_src_extents[i]) ||
              src_indices[m_map[i]] == -1) {
            out_of_bounds = true;
          }
        }
        return out_of_bounds ? ValueType(0) : m_src(src_indices[m_map[Is]]...);
      } else {
        const iType p = dst_idx[0];
        const auto [start, extent] =
            bin_mapping(iType(m_merged_size), iType(m_nprocs), p);
        iType src_indices[DstViewType::rank() - 1] = {
            merge_indices(dst_idx[Is + 1], start, extent, Is, m_axis)...};
        for (std::size_t i = 0; i < DstViewType::rank() - 1; ++i) {
          if (src_indices[m_map[i]] >= iType(m_src_extents[i]) ||
              src_indices[m_map[i]] == -1) {
            out_of_bounds = true;
          }
        }
        return out_of_bounds ? ValueType(0) : m_src(src_indices[m_map[Is]]...);
      }
    }
  };
};

/// \brief Unpack data from N+1D source view to ND destination view by merging
/// along a specified axis. The source view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          typename iType>
struct Unpack {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(SrcViewType::rank(), std::size_t(6));
  using ShapeType = Kokkos::Array<std::size_t, DstViewType::rank()>;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] x The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const SrcViewType& x) const {
    if constexpr (SrcViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, x.extent(0));
    } else {
      using LayoutType = typename SrcViewType::array_layout;
      static const Kokkos::Iterate outer_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::outer_iteration_pattern;
      static const Kokkos::Iterate inner_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::inner_iteration_pattern;
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, outer_iteration_pattern,
                       inner_iteration_pattern>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins = {};
      Kokkos::Array<std::size_t, m_rank_truncated> ends   = {};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = x.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructor for the Unpack functor.
  ///
  /// \param[in] src The input Kokkos view to be unpacked
  /// \param[out] dst The output Kokkos view to be unpacked
  /// \param[in] axis The axis to be merged
  /// \param[in] merged_size The extent of the dimension to be merged
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Unpack(const SrcViewType& src, const DstViewType& dst, const ShapeType& map,
         const std::size_t axis, const std::size_t merged_size,
         const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::Distributed::Unpack",
                         get_policy(exec_space, src),
                         UnpackInternal(src, dst, map, axis, merged_size));
  }

  struct UnpackInternal {
    using LayoutType = typename DstViewType::array_layout;
    using ValueType  = typename SrcViewType::non_const_value_type;
    SrcViewType m_src;
    DstViewType m_dst;
    std::size_t m_axis, m_merged_size, m_nprocs = 1;
    ShapeType m_map;
    ShapeType m_dst_extents;
    ShapeType m_src_extents;

    UnpackInternal(const SrcViewType& src, const DstViewType& dst,
                   const ShapeType& map, const std::size_t axis,
                   const std::size_t merged_size)
        : m_src(src),
          m_dst(dst),
          m_axis(axis),
          m_merged_size(merged_size),
          m_map(map) {
      for (std::size_t i = 0; i < DstViewType::rank(); ++i) {
        m_src_extents[i] = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                               ? src.extent(i + 1)
                               : src.extent(i);
      }
      m_nprocs = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                     ? src.extent(0)
                     : src.extent(SrcViewType::rank() - 1);
      for (std::size_t i = 0; i < DstViewType::rank(); ++i) {
        m_dst_extents[i] = dst.extent(i);
      }
    }

    template <typename... IndicesType>
    KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
      if constexpr (SrcViewType::rank() <= 6) {
        iType src_indices[SrcViewType::rank()] = {
            static_cast<iType>(indices)...};
        auto src_value = m_src(indices...);
        set_dst_value(src_indices, src_value,
                      std::make_index_sequence<SrcViewType::rank() - 1>{});
      } else if constexpr (SrcViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          iType src_indices[SrcViewType::rank()] = {
              static_cast<iType>(indices)..., i6};
          auto src_value = m_src(indices..., i6);
          set_dst_value(src_indices, src_value,
                        std::make_index_sequence<SrcViewType::rank() - 1>{});
        }
      } else if constexpr (SrcViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_dst.extent(7)); i7++) {
            iType src_indices[SrcViewType::rank()] = {
                static_cast<iType>(indices)..., i6, i7};
            auto src_value = m_src(indices..., i6, i7);
            set_dst_value(src_indices, src_value,
                          std::make_index_sequence<SrcViewType::rank() - 1>{});
          }
        }
      }
    }

    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION void set_dst_value(
        iType src_idx[], ValueType src_value,
        std::index_sequence<Is...>) const {
      // Bounds check
      bool in_bounds = true;
      if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        const iType p = src_idx[SrcViewType::rank() - 1];
        const auto [start, extent] =
            bin_mapping(iType(m_merged_size), iType(m_nprocs), p);
        iType dst_indices[SrcViewType::rank() - 1] = {
            merge_indices(src_idx[Is], start, extent, Is, m_axis)...};
        for (std::size_t i = 0; i < SrcViewType::rank() - 1; ++i) {
          if (dst_indices[m_map[i]] >= iType(m_dst_extents[i]) ||
              dst_indices[m_map[i]] == -1)
            in_bounds = false;
        }
        if (in_bounds) {
          m_dst(dst_indices[m_map[Is]]...) = src_value;
        }
      } else {
        const iType p = src_idx[0];
        const auto [start, extent] =
            bin_mapping(iType(m_merged_size), iType(m_nprocs), p);
        iType dst_indices[SrcViewType::rank() - 1] = {
            merge_indices(src_idx[Is + 1], start, extent, Is, m_axis)...};
        for (std::size_t i = 0; i < SrcViewType::rank() - 1; ++i) {
          if (dst_indices[m_map[i]] >= iType(m_dst_extents[i]) ||
              dst_indices[m_map[i]] == -1)
            in_bounds = false;
        }
        if (in_bounds) {
          m_dst(dst_indices[m_map[Is]]...) = src_value;
        }
      }
    }
  };
};

/// \brief Pack data from ND source view to (N+1)D destination view by splitting
/// along a specified axis. The destination view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam SrcViewType Kokkos View type of the source data
/// \tparam DstViewType Kokkos View type of the destination data
/// \tparam DIM Dimension of the mapping array
///
/// \param[in] exec_space The Kokkos execution space to be used
/// \param[in] src The input Kokkos view to be packed
/// \param[out] dst The output Kokkos view to be packed
/// \param[in] src_map The mapping of source view dimensions to destination view
/// dimensions
/// \param[in] axis The axis to be split
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          std::size_t DIM>
void pack(const ExecutionSpace& exec_space, const SrcViewType& src,
          const DstViewType& dst, std::array<std::size_t, DIM> src_map,
          std::size_t axis) {
  static_assert(SrcViewType::rank() >= 2);
  static_assert(DstViewType::rank() == SrcViewType::rank() + 1);
  Kokkos::Array<std::size_t, DIM> src_array =
      KokkosFFT::Impl::to_array(src_map);
  std::size_t merged_size =
      src.extent(KokkosFFT::Impl::get_index(src_map, axis));
  if (src.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      dst.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Pack<ExecutionSpace, SrcViewType, DstViewType, int64_t>(
        src, dst, src_array, axis, merged_size, exec_space);
  } else {
    Pack<ExecutionSpace, SrcViewType, DstViewType, int>(
        src, dst, src_array, axis, merged_size, exec_space);
  }
}

/// \brief Unpack data from (N+1)D source view to ND destination view by merging
/// along a specified axis. The source view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam SrcViewType Kokkos View type of the source data
/// \tparam DstViewType Kokkos View type of the destination data
/// \tparam DIM Dimension of the mapping array
/// \param[in] exec_space The Kokkos execution space to be used
/// \param[in] src The input Kokkos view to be unpacked
/// \param[out] dst The output Kokkos view to be unpacked
/// \param[in] dst_map The mapping of destination view dimensions to source view
/// dimensions
/// \param[in] axis The axis to be merged
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          std::size_t DIM>
void unpack(const ExecutionSpace& exec_space, const SrcViewType& src,
            const DstViewType& dst, std::array<std::size_t, DIM> dst_map,
            std::size_t axis) {
  static_assert(DstViewType::rank() >= 2);
  static_assert(SrcViewType::rank() == DstViewType::rank() + 1);
  Kokkos::Array<std::size_t, DIM> dst_array =
      KokkosFFT::Impl::to_array(dst_map);
  std::size_t merged_size =
      dst.extent(KokkosFFT::Impl::get_index(dst_map, axis));
  if (dst.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      src.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Unpack<ExecutionSpace, SrcViewType, DstViewType, int64_t>(
        src, dst, dst_array, axis, merged_size, exec_space);
  } else {
    Unpack<ExecutionSpace, SrcViewType, DstViewType, int>(
        src, dst, dst_array, axis, merged_size, exec_space);
  }
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
