// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP
#define KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP

#include <type_traits>
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

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
