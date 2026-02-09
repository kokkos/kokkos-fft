// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP
#define KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP

#include <algorithm>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_ContainerAnalyses.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Compute padded extents from the extents in Fourier space
///
/// Example, if the first FFT dimension is the 2nd dimension
/// in extents (real): (8, 7, 8)
/// out extents (complex): (8, 7, 5)
/// axes: (0, 1, 2)
/// FFT is operated from 2nd axis, so we have
/// padded extents (real): (8, 7, 10)
///
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] in_extents Extents of the global input View.
/// \param[in] out_extents Extents of the global output View.
/// \param[in] axes Axes of the transform
/// \return A extents of the permuted view
template <std::size_t DIM>
auto compute_padded_extents(const std::array<std::size_t, DIM> &extents,
                            const std::array<std::size_t, DIM> &axes) {
  std::array<std::size_t, DIM> padded_extents = extents;
  auto last_axis                              = axes.back();
  padded_extents.at(last_axis) *= 2;

  return padded_extents;
}

/// \brief Calculate the buffer extents based on the global extents,
/// the in-topology, and the out-topology.
///
/// Example
/// Global View extents (n0, n1, n2, n3)
/// in-topology = {1, p0, p1, 1} // X-pencil
/// out-topology = {p0, 1, p1, 1} // Y-pencil
/// Buffer View (p0, n0/p0, n1/p0, n2/p1, n3)
///
/// \tparam LayoutType The layout type of the buffer view (e.g.,
/// Kokkos::LayoutRight).
/// \tparam iType The integer type used for extents and topology.
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, typename iType, std::size_t DIM>
auto compute_buffer_extents(const std::array<iType, DIM> &extents,
                            const std::array<iType, DIM> &in_topology,
                            const std::array<iType, DIM> &out_topology) {
  static_assert(std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                    std::is_same_v<LayoutType, Kokkos::LayoutRight>,
                "compute_buffer_extents: We only accept LayoutLeft or "
                "LayoutRight for the buffer View.");
  std::array<iType, DIM + 1> buffer_extents{};
  auto merged_topology = merge_topology(in_topology, out_topology);
  auto p0 = diff_topology(merged_topology, in_topology);  // return 1 or p0
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    buffer_extents.at(0) = p0;
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i + 1) =
          (extents.at(i) - 1) / merged_topology.at(i) + 1;
    }
  } else {
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i) = (extents.at(i) - 1) / merged_topology.at(i) + 1;
    }
    buffer_extents.back() = p0;
  }
  return buffer_extents;
}

/// \brief Calculate the buffer extents based on the global extents,
/// the in-topology, and the out-topology.
///
/// Example
/// Global View extents (n0, n1, n2, n3)
/// in-topology = {1, p0, p1, 1} // X-pencil
/// out-topology = {p0, 1, p1, 1} // Y-pencil
/// Buffer View (p0, n0/p0, n1/p0, n2/p1, n3)
///
/// \tparam LayoutType The layout type of the buffer view (e.g.,
/// Kokkos::LayoutRight).
/// \tparam iType The integer type used for extents and topology.
/// \tparam DIM The number of dimensions of the extents.
/// \tparam InLayoutType The layout type of the in-topology (e.g.,
/// Kokkos::LayoutRight).
/// \tparam OutLayoutType The layout type of the out-topology (e.g.,
/// Kokkos::LayoutRight).
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, typename iType, std::size_t DIM = 1,
          typename InLayoutType  = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
auto compute_buffer_extents(
    const std::array<iType, DIM> &extents,
    const Topology<iType, DIM, InLayoutType> &in_topology,
    const Topology<iType, DIM, OutLayoutType> &out_topology) {
  return compute_buffer_extents<LayoutType>(extents, in_topology.array(),
                                            out_topology.array());
}

/// \brief Calculate the permuted extents based on the map
///
/// Example
/// View extents: (n0, n1, n2, n3)
/// map: (0, 2, 3, 1)
/// Next extents: (n0, n2, n3, n1)
///
/// \tparam ContainerType The container type
/// \tparam iType The integer type used for extents
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the View.
/// \param[in] map A map representing how the data is permuted
/// \return A extents of the permuted view
/// \throws std::runtime_error if the size of map is not equal to DIM
template <typename ContainerType, typename iType, std::size_t DIM>
auto compute_mapped_extents(const std::array<iType, DIM> &extents,
                            const ContainerType &map) {
  using value_type = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
  static_assert(std::is_integral_v<value_type>,
                "compute_mapped_extents: Map container value type must be an "
                "integral type");
  KOKKOSFFT_THROW_IF(map.size() != DIM,
                     "extents size must be equal to map size.");
  std::array<iType, DIM> mapped_extents{};
  std::transform(
      map.begin(), map.end(), mapped_extents.begin(),
      [&](std::size_t mapped_idx) { return extents.at(mapped_idx); });

  return mapped_extents;
}

/// \brief Compute the larger extents. Larger one corresponds to
/// the extents to FFT library. This is a helper for vendor library
/// which supports 2D or 3D non-batched FFTs.
///
/// Example
/// in extents: (8, 7, 8)
/// out extents: (8, 7, 5)
/// axes: (0, 1, 2)
/// FFT is operated from 2nd axis, so we have
/// fft extents: (8, 7, 8)
///
/// \tparam iType The integer type used for extents
/// \tparam DIM The number of dimensions of the extents.
/// \tparam FFT_DIM The number of dimensions of the FFT.
///
/// \param[in] in_extents Extents of the global input View.
/// \param[in] out_extents Extents of the global output View.
/// \param[in] axes Axes of the transform
/// \return A extents of the permuted view
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
auto compute_fft_extents(const std::array<iType, DIM> &in_extents,
                         const std::array<iType, DIM> &out_extents,
                         const std::array<iType, FFT_DIM> &axes) {
  static_assert(std::is_integral_v<iType>,
                "compute_fft_extents: iType must be an integral type");
  static_assert(
      FFT_DIM >= 1 && FFT_DIM <= KokkosFFT::MAX_FFT_DIM,
      "compute_fft_extents: the Rank of FFT axes must be between 1 and 3");
  static_assert(
      DIM >= FFT_DIM,
      "compute_fft_extents: View rank must be larger than or equal to "
      "the Rank of FFT axes");

  std::array<iType, FFT_DIM> fft_extents{};
  std::transform(axes.begin(), axes.end(), fft_extents.begin(),
                 [&](iType axis) {
                   return std::max(in_extents.at(axis), out_extents.at(axis));
                 });

  return fft_extents;
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
