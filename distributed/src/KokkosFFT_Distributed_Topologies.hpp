// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP
#define KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP

#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_ContainerAnalyses.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Get the topology type from the given topology container
/// Empty topology: 0 is included in the topology
/// Shared topology: non-one element is not included in the topology
/// Slab topology: 1 non-one element is included in the topology
/// Pencil topology: 2 non-one elements are included in the topology
/// Brick topology: 3 non-one elements are included in the topology
/// Invalid topology: more than 3 non-one elements are included in the topology
///
/// \tparam ContainerType Topology container type (std::array or Topology)
/// \param[in] topology Topology container
/// \return TopologyType enum value representing the topology type
template <typename ContainerType>
inline auto to_topology_type(const ContainerType& topology) {
  static_assert(
      (is_allowed_topology_v<ContainerType>),
      "to_topology_type: topologies must be either in std::array or Topology");

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
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topologies Topology containers to compare
/// \return TopologyType::Empty if any topology is empty; otherwise the common
/// topology type if all topologies have the same non-empty type; otherwise
/// TopologyType::Invalid
template <class... Topologies>
inline auto get_common_topology_type(const Topologies&... topologies) {
  static_assert(
      sizeof...(Topologies) > 0,
      "get_common_topology_type: at least one topology must be provided");
  static_assert((are_allowed_topologies_v<Topologies...>),
                "get_common_topology_type: topologies must be either in "
                "std::array or Topology");

  // Quick return if empty topology is found
  auto is_empty = [](const auto& topology) {
    return to_topology_type(topology) == TopologyType::Empty;
  };
  if ((is_empty(topologies) || ...)) {
    return TopologyType::Empty;
  }

  const std::array<TopologyType, 4> all_topology_types = {
      TopologyType::Shared, TopologyType::Slab, TopologyType::Pencil,
      TopologyType::Brick};
  for (TopologyType t : all_topology_types) {
    if (are_specified_topologies(t, topologies...)) {
      return t;
    }
  }

  return TopologyType::Invalid;
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
