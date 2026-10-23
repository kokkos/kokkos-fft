// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP
#define KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP

#include <algorithm>
#include <array>
#include <queue>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

/// \brief Divide the extents by the topology to get the local extents for each
/// process This is not exactly same as the local extents, but is used to check
/// whether the local extents include 0 or not.
/// \tparam ExtentsType The extents type (std::array)
/// \tparam TopologyContainerType The topology container type (std::array or
/// TopologyType)
/// \param[in] extents The extents
/// \param[in] topology The topology
/// \return The local extents for each process
template <typename ExtentsType, typename TopologyContainerType>
auto divide_by_topology(const ExtentsType& extents,
                        const TopologyContainerType& topology) {
  ExtentsType result;
  for (std::size_t i = 0; i < result.size(); ++i) {
    result.at(i) = extents.at(i) / topology.at(i);
  }
  return result;
}

/// \brief Progress of the FFT axes along a path of topologies.
/// The axes are processed from right to left: axes.back() must become
/// undistributed (1) first, then the one before it, and so on. The progress is
/// described by a single number, remaining: axes[0, remaining) are still
/// distributed and axes[remaining, axes.size()) have been consumed. The axes
/// consumed on reaching a state are derived from the remaining of consecutive
/// states, so they are never stored separately.
///
/// \tparam iType The integer type used for the axes
template <typename iType>
class FFTAxesProgress {
  const std::vector<iType>& m_axes;

 public:
  /// \param[in] axes The FFT axes. They must outlive this object.
  explicit FFTAxesProgress(const std::vector<iType>& axes) : m_axes(axes) {}

  /// \brief The number of axes still distributed after reaching a topology
  /// \param[in] topology The topology reached
  /// \param[in] remaining The number of axes still distributed before
  /// \return The number of axes still distributed after
  template <std::size_t DIM>
  std::size_t advance(const std::array<std::size_t, DIM>& topology,
                      std::size_t remaining) const {
    while (remaining > 0 && topology[m_axes[remaining - 1]] == std::size_t{1}) {
      --remaining;
    }
    return remaining;
  }

  /// \brief The number of axes still distributed on the input topology
  /// \param[in] topology The input topology
  template <std::size_t DIM>
  std::size_t initial(const std::array<std::size_t, DIM>& topology) const {
    return advance(topology, m_axes.size());
  }

  /// \brief Whether no axis has been consumed, i.e. no FFT has been performed
  bool none_consumed(std::size_t remaining) const {
    return remaining == m_axes.size();
  }

  /// \brief The FFT axes that become ready on each state of a path, in the
  /// original order. The first state consumes the axes already local on the
  /// input topology.
  /// \tparam StateType The type of the states, with a member m_remaining
  /// \param[in] path The states from the input to the output topology
  /// \return One vector of axes per state
  template <typename StateType>
  std::vector<std::vector<iType>> consumed_along(
      const std::vector<StateType>& path) const {
    std::vector<std::vector<iType>> all_axes;
    std::size_t prev_remaining = m_axes.size();
    for (const auto& state : path) {
      all_axes.emplace_back(m_axes.begin() + state.m_remaining,
                            m_axes.begin() + prev_remaining);
      prev_remaining = state.m_remaining;
    }
    return all_axes;
  }
};

struct SlabState {
  std::size_t m_pos;
  std::size_t m_p;

  // The number of FFT axes still distributed (see FFTAxesProgress)
  std::size_t m_remaining;

  SlabState(std::size_t pos, std::size_t p, std::size_t remaining)
      : m_pos(pos), m_p(p), m_remaining(remaining) {}

  bool operator==(const SlabState& other) const {
    return m_pos == other.m_pos && m_remaining == other.m_remaining;
  }

  template <std::size_t DIM>
  auto topology() const {
    std::array<std::size_t, DIM> temp;
    temp.fill(std::size_t{1});
    temp[m_pos] = m_p;
    return temp;
  }
};

struct SlabStateHash {
  std::size_t operator()(const SlabState& s) const {
    std::size_t h = std::hash<std::size_t>{}(s.m_pos);
    h ^= std::hash<std::size_t>{}(s.m_remaining) +
         static_cast<std::size_t>(0x9e3779b9) + (h << 6) + (h >> 2);
    return h;
  }
};

/// \brief Find the minimal sequence of slab topologies from the input to the
/// output topology such that every FFT axis becomes undistributed (1) at some
/// point along the way.
///
/// A slab topology has exactly one non-one entry P. Each step of the path
/// moves P to another position, which corresponds to a single transpose. The
/// axes are processed from right to left: axes.back() must be undistributed
/// first, then the one before it, and so on. The output topology can only be
/// reached after all axes have been satisfied. A breadth-first search is used,
/// so the returned path has the minimum number of transposes. Among the next
/// states, those consuming more axes are explored first, then those with a
/// smaller position of P. An intermediate topology is rejected if the extent
/// along P is smaller than P. The extents are gin_extents until the first axis
/// is consumed, and gout_extents afterwards.
///
/// Example: 3D case
/// In topology: (1, 1, P)
/// Out topology: (1, 1, P)
/// axes: {0, 1, 2}
/// Output: {(1, 1, P), (P, 1, 1), (1, 1, P)}
///   (1, 1, P) -> (P, 1, 1): axes 2 and 1 are undistributed
///   (P, 1, 1) -> (1, 1, P): axis 0 is undistributed
///
/// \tparam iType The integer type used for the axes.
/// \tparam DIM The dimensionality of the topologies.
/// \param[in] gin_extents The global input extents.
/// \param[in] gout_extents The global output extents.
/// \param[in] in_topology The input slab topology.
/// \param[in] out_topology The output slab topology.
/// \param[in] axes The FFT axes, each of which must be undistributed at some
/// point along the path (processed from right to left).
/// \return A vector of states representing the minimal path from the input to
/// the output topology, including both ends. If the input already equals the
/// output and all axes are satisfied, a single state is returned.
/// \throws std::runtime_error if no valid path satisfies the conditions
template <typename iType, std::size_t DIM>
std::vector<SlabState> find_minimal_slab_path(
    const std::array<std::size_t, DIM>& gin_extents,
    const std::array<std::size_t, DIM>& gout_extents,
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::vector<iType>& axes) {
  const FFTAxesProgress<iType> progress(axes);

  auto non_one_pos = [](const std::array<std::size_t, DIM>& topology) {
    return static_cast<std::size_t>(
        std::find_if(topology.begin(), topology.end(),
                     [](std::size_t value) { return value != 1; }) -
        topology.begin());
  };

  const std::size_t p = KokkosFFT::Impl::total_size(in_topology);

  // Reconstruct the array represented by the position of p.
  auto make_array = [&](std::size_t pos) {
    std::array<std::size_t, DIM> temp;
    temp.fill(std::size_t{1});
    temp[pos] = p;
    return temp;
  };

  // axes already satisfied by the input are removed first.
  const SlabState in{non_one_pos(in_topology), p,
                     progress.initial(in_topology)};
  const SlabState out{non_one_pos(out_topology), p, 0};

  if (in == out) return {in};

  std::queue<SlabState> queue;
  std::unordered_set<SlabState, SlabStateHash> visited;
  std::unordered_map<SlabState, SlabState, SlabStateHash> parent;

  queue.push(in);
  visited.insert(in);

  bool found = false;

  while (!queue.empty()) {
    const SlabState current = queue.front();
    queue.pop();
    if (current == out) {
      found = true;
      break;
    }

    // No FFT has been performed until the first axis is consumed.
    const auto& extents = progress.none_consumed(current.m_remaining)
                              ? gin_extents
                              : gout_extents;

    std::vector<SlabState> candidates;
    for (std::size_t pos = 0; pos < DIM; ++pos) {
      if (pos == current.m_pos) continue;

      const SlabState next{
          pos, p, progress.advance(make_array(pos), current.m_remaining)};

      // The output topology is already validated by the caller.
      if (!(next == out) && extents[pos] / p == 0) continue;

      candidates.push_back(next);
    }

    // Prefer the states consuming more axes. The order of positions is kept
    // for the states consuming the same number of axes.
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const SlabState& a, const SlabState& b) {
                       return a.m_remaining < b.m_remaining;
                     });

    for (const auto& next : candidates) {
      if (visited.insert(next).second) {
        parent.emplace(next, current);
        queue.push(next);
      }
    }
  }

  KOKKOSFFT_THROW_IF(!found, "No valid path satisfies the conditions");

  // Reconstruct the shortest path.
  std::vector<SlabState> states;

  for (SlabState current = out;; current = parent.at(current)) {
    states.push_back(current);
    if (current == in) break;
  }

  std::reverse(states.begin(), states.end());
  return states;
}

/// \brief Get all slab topologies for a given input and output topology
///
/// Example: 3D case
/// In topology: (1, 1, P)
/// Out topology: (1, 1, P)
/// axes: {0, 1, 2}
/// Output: {(1, 1, P), (P, 1, 1), (1, 1, P)}
/// Operation:
/// Transpose -> FFT2 ax = {1, 2} -> Transpose -> FFT1 ax = {0}
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param[in] gin_extents The global input extents of the data.
/// \param[in] gout_extents The global output extents of the data.
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \return A tuple of (all_topologies, all_axes). all_topologies is the
/// sequence of slab topologies from the input to the output topology. all_axes
/// are the FFT axes that become ready on each topology, identical to
/// decompose_axes(all_topologies, axes).
/// \throws std::runtime_error
/// 1. if the input and output topologies are not slab topologies
/// 2. if the input and output topologies do not have the same size
/// 3. if the input and output extents include 0
/// 4. if no valid path of slab topologies is found for the given input
/// topology and FFT axes
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
auto get_all_slab_topologies(const std::array<std::size_t, DIM>& gin_extents,
                             const std::array<std::size_t, DIM>& gout_extents,
                             const std::array<std::size_t, DIM>& in_topology,
                             const std::array<std::size_t, DIM>& out_topology,
                             const std::array<iType, FFT_DIM>& axes) {
  static_assert(FFT_DIM >= 1 && FFT_DIM <= 3, "FFT_DIM must be in [1, 3]");
  static_assert(DIM >= 2 && DIM >= FFT_DIM, "DIM >= 2 and DIM >= FFT_DIM");
  static_assert(std::is_unsigned_v<iType>,
                "get_all_slab_topologies: axes must be unsigned");

  // Firstly, check if input topologies are slabs with the same size
  bool is_slab =
      are_specified_topologies(TopologyType::Slab, in_topology, out_topology);

  KOKKOSFFT_THROW_IF(!is_slab,
                     "Input and output topologies must be slab topologies.");

  auto in_topology_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_topology_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_topology_size != out_topology_size,
                     "Input and output topologies must have the same size.");

  // Check if local input and output sizes are not zero
  auto in_extents  = divide_by_topology(gin_extents, in_topology);
  auto out_extents = divide_by_topology(gout_extents, out_topology);

  auto in_size  = KokkosFFT::Impl::total_size(in_extents);
  auto out_size = KokkosFFT::Impl::total_size(out_extents);

  KOKKOSFFT_THROW_IF(in_size == 0 || out_size == 0,
                     "Input and output extents must not include 0.");

  const auto axes_vec = KokkosFFT::Impl::to_vector(axes);

  auto all_states = find_minimal_slab_path(gin_extents, gout_extents,
                                           in_topology, out_topology, axes_vec);

  std::vector<std::array<std::size_t, DIM>> all_topologies;
  for (const auto& state : all_states) {
    all_topologies.push_back(state.template topology<DIM>());
  }
  auto all_axes = FFTAxesProgress<iType>(axes_vec).consumed_along(all_states);

  return std::make_tuple(all_topologies, all_axes);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
