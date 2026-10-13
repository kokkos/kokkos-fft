// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP
#define KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_ContainerAnalyses.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Get the topology type from the given topology container
/// Empty topology: at least one element is 0
/// Shared topology: no elements differ from 1
/// Slab topology: exactly 1 element differs from 1
/// Pencil topology: exactly 2 elements differ from 1
/// Brick topology: exactly 3 elements differ from 1
/// Invalid topology: more than 3 elements differ from 1
///
/// \tparam ContainerType Topology container type (std::array or Topology)
/// \param[in] topology Topology container
/// \return TopologyType enum value representing the topology type
template <typename ContainerType>
inline auto to_topology_type(const ContainerType& topology) {
  static_assert(
      (is_allowed_topology_v<ContainerType>),
      "to_topology_type: topologies must be either in std::array or Topology");
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*topology.begin())>>;
  static_assert(
      std::is_integral_v<value_type>,
      "to_topology_type: Container value type must be an integral type");

  for (const auto& value : topology) {
    if (value == 0) return TopologyType::Empty;
  }

  switch (count_non_ones(topology)) {
    case 0: return TopologyType::Shared;
    case 1: return TopologyType::Slab;
    case 2: return TopologyType::Pencil;
    case 3: return TopologyType::Brick;
    default: return TopologyType::Invalid;
  }
}

/// \brief Check if all given topologies are of specified type
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topology_type a topology type of interest
/// \param[in] topologies Topology containers
/// \return true if all topologies are of the specified type, false otherwise
template <class... Topologies>
inline bool are_specified_topologies(const TopologyType topology_type,
                                     const Topologies&... topologies) {
  static_assert(
      sizeof...(Topologies) > 0,
      "are_specified_topologies: at least one topology must be provided");
  static_assert((are_allowed_topologies_v<Topologies...>),
                "are_specified_topologies: topologies must be either in "
                "std::array or Topology");
  auto is_specified_topology = [topology_type](const auto& topology) {
    return to_topology_type(topology) == topology_type;
  };
  return (is_specified_topology(topologies) && ...);
}

/// \brief Get the topology type from the given topology containers
///
/// \tparam FirstTopology The type of the first topology container
/// \tparam RestTopologies Variadic template parameter for the rest of the
/// topology container types
/// \param[in] first_topology The first topology container
/// \param[in] rest_topologies The rest of the topology containers
/// \return TopologyType::Empty if any topology is empty; otherwise the common
/// topology type if all topologies have the same non-empty type; otherwise
/// TopologyType::Invalid
template <class FirstTopology, class... RestTopologies>
inline auto get_common_topology_type(const FirstTopology& first_topology,
                                     const RestTopologies&... rest_topologies) {
  static_assert((are_allowed_topologies_v<FirstTopology, RestTopologies...>),
                "get_common_topology_type: topologies must be either in "
                "std::array or Topology");

  const auto common_topology_type = to_topology_type(first_topology);
  if (common_topology_type == TopologyType::Empty) {
    return TopologyType::Empty;
  }

  if constexpr (sizeof...(RestTopologies) > 0) {
    const bool has_empty =
        ((to_topology_type(rest_topologies) == TopologyType::Empty) || ...);
    if (has_empty) {
      return TopologyType::Empty;
    }
  }

  bool has_mismatch        = false;
  auto check_topology_type = [&](const auto& topology) {
    const auto topology_type = to_topology_type(topology);
    if (topology_type != common_topology_type) {
      has_mismatch = true;
    }
  };
  if constexpr (sizeof...(RestTopologies) > 0) {
    (check_topology_type(rest_topologies), ...);
  }

  return has_mismatch ? TopologyType::Invalid : common_topology_type;
}

/// \brief Get the axes of the input and output slab topologies that are
/// different
/// Example
/// (1, P) -> (P, 1): y-slab to x-slab
/// (P, 1) -> (1, p): x-slab to y-slab
/// (1, 1, P) -> (1, P, 1): z-slab to y-slab
/// (P, 1, 1) -> (1, P, 1): x-slab to y-slab
///
/// \tparam iType The type of the index in the topology.
/// \tparam DIM The number of dimensions of the topology.
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \return A tuple of two size_t representing the axes that are different
/// \throws std::runtime_error if the input and output topologies do not have
/// the same size or if they are identical
/// \throws std::runtime_error if the input and output topologies are not slab
/// topologies
template <typename iType, std::size_t DIM>
auto slab_in_out_axes(const std::array<iType, DIM>& in_topology,
                      const std::array<iType, DIM>& out_topology) {
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  KOKKOSFFT_THROW_IF(in_topology == out_topology,
                     "Input and output topologies must be different.");

  bool is_slab =
      are_specified_topologies(TopologyType::Slab, in_topology, out_topology);
  KOKKOSFFT_THROW_IF(!is_slab,
                     "Input and output topologies must be slab topologies.");

  std::size_t in_axis = 0, out_axis = 0;
  for (std::size_t i = 0; i < DIM; ++i) {
    if (in_topology.at(i) > 1 && out_topology.at(i) == 1) {
      out_axis = i;
    }
    if (in_topology.at(i) == 1 && out_topology.at(i) > 1) {
      in_axis = i;
    }
  }

  return std::make_tuple(in_axis, out_axis);
}

/// \brief Get the axes of the input and output topologies that are different
///
/// Example
/// (1, Px, Py, 1) -> (Px, 1, Py, 1): 0-pencil to 1-pencil
/// (1, 1, P) -> (1, P, 1): 1-pencil to 2-pencil
/// (P, 1, 1) -> (1, P, 1): 1-pencil to 0-pencil
///
/// \tparam iType The type of the index in the topology.
/// \tparam DIM The number of dimensions of the topology.
///
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \return A tuple of two size_t representing the axes that are different
/// \throws std::runtime_error if the input and output topologies do not have
/// at least one non-trivial dimension
/// \throws std::runtime_error if the input and output topologies do not have
/// the same size
template <typename iType, std::size_t DIM>
auto pencil_in_out_axes(const std::array<iType, DIM>& in_topology,
                        const std::array<iType, DIM>& out_topology) {
  // Extract topology that is common between in_topology and out_topology
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size == 1 || out_size == 1,
                     "Input and output topologies must have at least one "
                     "non-trivial dimension.");

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  std::size_t in_axis = 0, out_axis = 0;
  for (std::size_t i = 0; i < DIM; ++i) {
    if (in_topology.at(i) != out_topology.at(i)) {
      if (in_topology.at(i) == 1) in_axis = i;
      if (out_topology.at(i) == 1) out_axis = i;
    }
  }

  return std::make_tuple(in_axis, out_axis);
}

/// \brief Decompose the FFT axes into vectors
///        The first vector includes the axes for FFT without transpose
///        The second vector includes the axes for FFT after transpose
///        The third vector includes the axes for remaining FFT
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param[in] topologies The vector of topologies.
/// \param[in] axes The axes along which the FFT is performed.
/// \return A vector of vectors of axes.
/// \throws std::runtime_error if the total size of decomposed axes does not
/// match the original axes size
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
std::vector<std::vector<iType>> decompose_axes(
    const std::vector<std::array<std::size_t, DIM>>& topologies,
    const std::array<iType, FFT_DIM>& axes) {
  auto non_negative_axes = KokkosFFT::Impl::convert_base_int_type<std::size_t>(
      KokkosFFT::Impl::convert_negative_axes(axes, DIM));

  // Reverse the axes e.g. {0, 2, 1} -> {1, 2, 0}
  std::vector<std::size_t> axes_reversed =
      KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(non_negative_axes));

  std::vector<std::vector<iType>> all_axes{};
  for (auto topology : topologies) {
    std::vector<iType> ready_axes;
    for (auto axis : axes_reversed) {
      if (topology.at(axis) > 1) break;
      ready_axes.push_back(axis);
    }
    // We need to reverse the axes again
    // i.e. {1, 2} -> {2, 1}
    all_axes.push_back(KokkosFFT::Impl::reversed(ready_axes));

    // Remove already registered axes
    for (auto axis : ready_axes) {
      auto it = std::find(axes_reversed.begin(), axes_reversed.end(), axis);
      if (it != axes_reversed.end()) {
        axes_reversed.erase(it);
      }
    }
  }

  auto error_msg = [&axes, &all_axes,
                    &topologies](std::string_view details) -> std::string {
    std::string msg(details);
    msg += KokkosFFT::Impl::container_to_string(" Input axes: ", axes);
    msg += "\n";
    msg += "Decomposed axes: \n";
    for (std::size_t i = 0; i < all_axes.size(); ++i) {
      auto topology = topologies.at(i);
      msg += "at ";
      msg += KokkosFFT::Impl::container_to_string("topology: ", topology);
      msg += ": Ready axes: ";
      if (all_axes.at(i).empty()) {
        msg += "None";
      } else {
        msg += KokkosFFT::Impl::container_to_string("", all_axes.at(i));
      }
      msg += "\n";
    }
    return msg;
  };

  std::size_t total_axes = 0;
  for (auto ready_axes : all_axes) {
    total_axes += ready_axes.size();
  }

  KOKKOSFFT_THROW_IF(total_axes != axes.size(),
                     error_msg("Axes are not decomposed correctly:"));

  return all_axes;
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
