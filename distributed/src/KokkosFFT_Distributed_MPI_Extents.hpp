// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_MPI_EXTENTS_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_EXTENTS_HPP

#include <type_traits>
#include <string_view>
#include <string>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Types.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Compute the global extents of the distributed View
/// Examples:
///  - with LayoutRight and Topology (1, 2, 4)
///      rank0: (0, 0)
///      rank1: (0, 1)
///      rank2: (0, 2)
///      rank3: (0, 3)
///      rank4: (1, 0)
///      rank5: (1, 1)
///      rank6: (1, 2)
///      rank7: (1, 3)
///
/// - with LayoutLeftout and Topology (1, 2, 4)
///      rank0: (0, 0)
///      rank1: (1, 0)
///      rank2: (0, 1)
///      rank3: (1, 1)
///      rank4: (0, 2)
///      rank5: (1, 2)
///      rank6: (0, 3)
///      rank7: (1, 3)
///
/// \tparam ViewType The Kokkos View type
/// \tparam LayoutType The layout type of the Topology (default is
/// Kokkos::LayoutRight)
/// \param[in] v The Kokkos View
/// \param[in] topology The topology representing the distribution of the data
/// \param[in] comm The MPI communicator
/// \return The global extents of the distributed View
template <typename ViewType, typename LayoutType = Kokkos::LayoutRight>
std::array<std::size_t, ViewType::rank()> compute_global_extents(
    const ViewType &v,
    const Topology<std::size_t, ViewType::rank(), LayoutType> &topology,
    MPI_Comm comm) {
  auto extents    = KokkosFFT::Impl::extract_extents(v);
  auto total_size = KokkosFFT::Impl::total_size(topology);

  std::vector<std::size_t> gathered_extents(ViewType::rank() * total_size);
  std::array<std::size_t, ViewType::rank()> global_extents{};
  MPI_Datatype mpi_data_type = mpi_datatype_v<std::size_t>;

  // Data are stored as
  // rank0: extents0
  // rank1: extents1
  // ...
  // rankn: extentsn
  MPI_Allgather(extents.data(), extents.size(), mpi_data_type,
                gathered_extents.data(), extents.size(), mpi_data_type, comm);

  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    std::size_t stride = total_size;
    for (std::size_t i = 0; i < topology.size(); i++) {
      if (topology.at(i) == 1) {
        global_extents.at(i) = extents.at(i);
      } else {
        // Maybe better to check that the shape is something like
        // n, n, n, n_remain
        std::size_t sum = 0;
        stride /= topology.at(i);
        for (std::size_t j = 0; j < topology.at(i); j++) {
          sum += gathered_extents.at(i + extents.size() * stride * j);
        }
        global_extents.at(i) = sum;
      }
    }
  } else {
    std::size_t stride = 1;
    for (std::size_t i = 0; i < topology.size(); i++) {
      if (topology.at(i) == 1) {
        global_extents.at(i) = extents.at(i);
      } else {
        std::size_t sum = 0;
        for (std::size_t j = 0; j < topology.at(i); j++) {
          sum += gathered_extents.at(i + extents.size() * stride * j);
        }
        stride *= topology.at(i);
        global_extents.at(i) = sum;
      }
    }
  }
  return global_extents;
}

/// \brief Compute the global extents of the distributed View
/// Overload using a std::array for the topology
/// \tparam ViewType The Kokkos View type
/// \param[in] v The Kokkos View
/// \param[in] topology The topology representing the distribution of the data
/// \param[in] comm The MPI communicator
/// \return The global extents of the distributed View
template <typename ViewType>
auto compute_global_extents(
    const ViewType &v,
    const std::array<std::size_t, ViewType::rank()> &topology, MPI_Comm comm) {
  return compute_global_extents(
      v, Topology<std::size_t, ViewType::rank()>(topology), comm);
}

/// \brief Compute the local extents for the next block given the current rank
/// and layout (compile time version)
/// Examples:
/// Global extents: (X, Y, Z)
/// Next Topology: (n, 1, 1)
/// Map: (0, 1, 2)
/// Next Extents: (X/n, Y, Z) or (X/n+1, Y, Z) if X is not divisible by n
/// The first X % n GPUs each own (X/n+1)*Y*Z elements and the remaining GPUs
/// each own (X/n)*Y*Z elements.
///
/// \tparam DIM Number of dimensions
/// \tparam LayoutType Layout type for the Input Topology (default is
/// Kokkos::LayoutRight)
/// \param[in] extents Global extents
/// \param[in] topology Topology of the next block
/// \param[in] map Map of the next block
/// \param[in] rank MPI rank
/// \return The local extents for the next block
template <std::size_t DIM, typename LayoutType = Kokkos::LayoutRight>
auto compute_next_extents(
    const std::array<std::size_t, DIM> &extents,
    const Topology<std::size_t, DIM, LayoutType> &topology,
    const std::array<std::size_t, DIM> &map, std::size_t rank) {
  std::array<std::size_t, DIM> local_extents{}, next_extents{};
  std::copy(extents.begin(), extents.end(), local_extents.begin());

  auto coords = rank_to_coord(topology, rank);
  for (std::size_t i = 0; i < extents.size(); i++) {
    if (topology.at(i) != 1) {
      std::size_t n = extents.at(i);
      std::size_t t = topology.at(i);

      std::size_t quotient  = n / t;
      std::size_t remainder = n % t;
      // Distribute the remainder across the first few elements
      local_extents.at(i) =
          (coords.at(i) < remainder) ? quotient + 1 : quotient;
    }
  }

  for (std::size_t i = 0; i < extents.size(); i++) {
    std::size_t mapped_idx = map.at(i);
    next_extents.at(i)     = local_extents.at(mapped_idx);
  }

  return next_extents;
}

/// \brief Compute the local extents for the next block given the current rank
/// and layout (run time version)
/// \tparam DIM Number of dimensions
/// \param[in] extents Global extents
/// \param[in] topology Topology of the next block
/// \param[in] map Map of the next block
/// \param[in] rank MPI rank
/// \param[in] is_layout_right Layout type for the Input Topology (default is
/// true)
/// \return The local extents for the next block
/// \throws std::runtime_error if the total size of next extents is 0
template <std::size_t DIM>
auto compute_next_extents(const std::array<std::size_t, DIM> &extents,
                          const std::array<std::size_t, DIM> &topology,
                          const std::array<std::size_t, DIM> &map,
                          std::size_t rank, bool is_layout_right = true) {
  if (is_layout_right) {
    return compute_next_extents(
        extents, Topology<std::size_t, DIM, Kokkos::LayoutRight>(topology), map,
        rank);
  } else {
    return compute_next_extents(
        extents, Topology<std::size_t, DIM, Kokkos::LayoutLeft>(topology), map,
        rank);
  }
}

/// \brief Compute the maximum value in the given container across all ranks
/// \tparam ContainerType Type of the container
/// \param[in] values Container holding the values to be compared
/// \param[in] comm MPI communicator
/// \return The maximum value across all ranks
template <typename ContainerType>
auto compute_global_max(const ContainerType &values, MPI_Comm comm) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  MPI_Datatype mpi_data_type = mpi_datatype_v<value_type>;
  value_type max_value       = 0;
  value_type lmax_value      = *std::max_element(values.begin(), values.end());

  MPI_Allreduce(&lmax_value, &max_value, 1, mpi_data_type, MPI_MAX, comm);
  return max_value;
}

/// \brief Compute the minimum value in the given container across all ranks
/// \tparam ContainerType Type of the container
/// \param[in] values Container holding the values to be compared
/// \param[in] comm MPI communicator
/// \return The minimum value across all ranks
template <typename ContainerType>
auto compute_global_min(const ContainerType &values, MPI_Comm comm) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  MPI_Datatype mpi_data_type = mpi_datatype_v<value_type>;
  value_type min_value       = 0;
  value_type lmin_value      = *std::min_element(values.begin(), values.end());

  MPI_Allreduce(&lmin_value, &min_value, 1, mpi_data_type, MPI_MIN, comm);
  return min_value;
}

/// \brief Check if input and output views have valid extents for the given
/// axes, FFT type, and topologies
///
/// Compute the global extents of the local input and output views and check if
/// they satisfy the requirements for each transform type:
///
/// For C2C transform:
/// - The 'in_extent' and 'out_extent' must have the same extent in all
/// dimensions
///
/// For R2C transform:
/// - The 'out_extent' of the transform must be equal to 'in_extent'/2 + 1
///
/// For C2R transform:
/// - The 'in_extent' of the transform must be equal to 'out_extent' / 2 + 1
///
/// \tparam InViewType Type of the input view
/// \tparam OutViewType Type of the output view
/// \tparam SizeType Type of the size (e.g., std::size_t)
/// \tparam DIM Number of dimensions for the FFT axes
/// \tparam InLayoutType Layout type for the input Topology (default is
/// Kokkos::LayoutRight)
/// \tparam OutLayoutType Layout type for the output Topology (default is
/// Kokkos::LayoutRight)
/// \param[in] in The input view
/// \param[in] out The output view
/// \param[in] axes The axes over which the FFT is performed
/// \param[in] in_topology The FFT topology for the input view
/// \param[in] out_topology The FFT topology for the output view
/// \param[in] comm The MPI communicator (default is MPI_COMM_WORLD)
/// \return true if the input and output views have valid extents, false
/// otherwise
/// \throws std::runtime_error if the axes are not valid for the input view
/// or if the extents do not satisfy the requirements for each transform type
template <typename InViewType, typename OutViewType, typename SizeType,
          std::size_t DIM, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
bool are_valid_extents(
    const InViewType &in, const OutViewType &out,
    KokkosFFT::axis_type<DIM> axes,
    const Topology<SizeType, InViewType::rank(), InLayoutType> &in_topology,
    const Topology<SizeType, OutViewType::rank(), OutLayoutType> &out_topology,
    const MPI_Comm &comm = MPI_COMM_WORLD) {
  using in_value_type     = typename InViewType::non_const_value_type;
  using out_value_type    = typename OutViewType::non_const_value_type;
  using array_layout_type = typename InViewType::array_layout;

  static_assert(!(KokkosFFT::Impl::is_real_v<in_value_type> &&
                  KokkosFFT::Impl::is_real_v<out_value_type>),
                "are_valid_extents: real to real transform is not supported");

  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "input axes are not valid for the view");

  constexpr std::size_t rank = InViewType::rank;
  [[maybe_unused]] std::size_t inner_most_axis =
      std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft>
          ? 0
          : (rank - 1);

  // index map after transpose over axis
  auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(in, axes);

  // Get global shape to define buffer and next shape
  auto gin_extents  = compute_global_extents(in, in_topology, comm);
  auto gout_extents = compute_global_extents(out, out_topology, comm);

  auto in_extents  = compute_mapped_extents(gin_extents, map);
  auto out_extents = compute_mapped_extents(gout_extents, map);

  auto mismatched_extents = [&in, &out, &in_extents, &out_extents](
                                std::string_view msg) -> std::string {
    std::string message(msg);
    message += in.label();
    message += "(";
    message += std::to_string(in_extents.at(0));
    for (std::size_t r = 1; r < in_extents.size(); r++) {
      message += ",";
      message += std::to_string(in_extents.at(r));
    }
    message += "), ";
    message += out.label();
    message += "(";
    message += std::to_string(out_extents.at(0));
    for (std::size_t r = 1; r < out_extents.size(); r++) {
      message += ",";
      message += std::to_string(out_extents.at(r));
    }
    message += ")";
    return message;
  };

  for (std::size_t i = 0; i < rank; i++) {
    // The requirement for inner_most_axis is different for transform type
    if (i == inner_most_axis) continue;
    KOKKOSFFT_THROW_IF(
        in_extents.at(i) != out_extents.at(i),
        mismatched_extents(
            "input and output extents must be the same except for "
            "the transform axis: "));
  }

  if constexpr (KokkosFFT::Impl::is_complex_v<in_value_type> &&
                KokkosFFT::Impl::is_complex_v<out_value_type>) {
    // Then C2C
    KOKKOSFFT_THROW_IF(
        in_extents.at(inner_most_axis) != out_extents.at(inner_most_axis),
        mismatched_extents(
            "input and output extents must be the same for C2C transform: "));
  }

  if constexpr (KokkosFFT::Impl::is_real_v<in_value_type>) {
    // Then R2C
    KOKKOSFFT_THROW_IF(
        out_extents.at(inner_most_axis) !=
            in_extents.at(inner_most_axis) / 2 + 1,
        mismatched_extents("For R2C, the 'output extent' of transform must be "
                           "equal to 'input extent'/2 + 1: "));
  }

  if constexpr (KokkosFFT::Impl::is_real_v<out_value_type>) {
    // Then C2R
    KOKKOSFFT_THROW_IF(
        in_extents.at(inner_most_axis) !=
            out_extents.at(inner_most_axis) / 2 + 1,
        mismatched_extents("For C2R, the 'input extent' of transform must be "
                           "equal to 'output extent' / 2 + 1: "));
  }
  return true;
}

/// \brief Check if input and output views have valid extents for the given
/// axes, FFT type, and topologies
/// Overload using a std::array for the topologies
///
/// Compute the global extents of the local input and output views and check if
/// they satisfy the requirements for each transform type:
///
/// For C2C transform:
/// - The 'in_extent' and 'out_extent' must have the same extent in all
/// dimensions
///
/// For R2C transform:
/// - The 'out_extent' of the transform must be equal to 'in_extent'/2 + 1
///
/// For C2R transform:
/// - The 'in_extent' of the transform must be equal to 'out_extent' / 2 + 1
///
/// \tparam InViewType Type of the input view
/// \tparam OutViewType Type of the output view
/// \tparam SizeType Type of the size (e.g., std::size_t)
/// \tparam DIM Number of dimensions for the FFT axes
/// \param[in] in The input view
/// \param[in] out The output view
/// \param[in] axes The axes over which the FFT is performed
/// \param[in] in_topology The FFT topology for the input view
/// \param[in] out_topology The FFT topology for the output view
/// \param[in] comm The MPI communicator (default is MPI_COMM_WORLD)
/// \return true if the input and output views have valid extents, false
/// otherwise
/// \throws std::runtime_error if the axes are not valid for the input view
/// or if the extents do not satisfy the requirements for each transform type
template <typename InViewType, typename OutViewType, typename SizeType,
          std::size_t DIM>
bool are_valid_extents(
    const InViewType &in, const OutViewType &out,
    KokkosFFT::axis_type<DIM> axes,
    const std::array<SizeType, InViewType::rank()> &in_topology,
    const std::array<SizeType, OutViewType::rank()> &out_topology,
    const MPI_Comm &comm = MPI_COMM_WORLD) {
  return are_valid_extents(
      in, out, axes, Topology<SizeType, InViewType::rank()>(in_topology),
      Topology<SizeType, OutViewType::rank()>(out_topology), comm);
}
}  // namespace Impl

/// \brief Convert rank to coordinate based on the given topology
/// Examples:
///  - with LayoutRight and Topology (2, 2)
///      rank0: (0, 0)
///      rank1: (0, 1)
///      rank2: (1, 0)
///      rank3: (1, 1)
///
/// - with LayoutLeftout and Topology (2, 2)
///      rank0: (0, 0)
///      rank1: (1, 0)
///      rank2: (0, 1)
///      rank3: (1, 1)
///
/// \tparam DIM Number of dimensions
/// \tparam LayoutType Layout type for the Topology (default is
/// Kokkos::LayoutRight)
/// \param[in] topology Topology of the distributed data
/// \param[in] rank MPI rank
/// \return The coordinate corresponding to the given rank
/// \throws std::runtime_error if the rank is out of range
template <std::size_t DIM, typename LayoutType = Kokkos::LayoutRight>
auto rank_to_coord(const Topology<std::size_t, DIM, LayoutType> &topology,
                   const std::size_t rank) {
  std::array<std::size_t, DIM> coord{};
  std::size_t rank_tmp = rank;
  auto topology_size   = KokkosFFT::Impl::total_size(topology);
  KOKKOSFFT_THROW_IF(rank >= topology_size,
                     "rank must be less than topology size.");

  int64_t topology_rank = topology.size();
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    for (int64_t i = topology_rank - 1; i >= 0; i--) {
      coord.at(i) = rank_tmp % topology.at(i);
      rank_tmp /= topology.at(i);
    }
  } else {
    for (int64_t i = 0; i < topology_rank; i++) {
      coord.at(i) = rank_tmp % topology.at(i);
      rank_tmp /= topology.at(i);
    }
  }

  return coord;
}

/// \brief Convert rank to coordinate based on the given topology
/// Overload using a std::array for the topology
/// \tparam DIM Number of dimensions
/// \param[in] topology Topology of the distributed data
/// \param[in] rank MPI rank
/// \return The coordinate corresponding to the given rank
template <std::size_t DIM>
auto rank_to_coord(const std::array<std::size_t, DIM> &topology,
                   const std::size_t rank) {
  return rank_to_coord(Topology<std::size_t, DIM>(topology), rank);
}

/// \brief Compute the local extents and starts of the distributed View
/// Examples:
/// Global extents: (10, Y, Z)
/// Topology: (3, 1, 1)
/// Local Extents:
/// (4, Y, Z), (3, Y, Z), (3, Y, Z)
/// Local Starts:
/// (0, 0, 0), (4, 0, 0), (7, 0, 0)
/// \tparam DIM Number of dimensions
/// \tparam LayoutType Layout type for the Topology (default is
/// Kokkos::LayoutRight)
/// \param[in] extents Extents of the global View
/// \param[in] topology Topology of the data distribution
/// \param[in] comm MPI communicator
/// \return A tuple of local extents and starts of the distributed View
template <std::size_t DIM, typename LayoutType = Kokkos::LayoutRight>
auto compute_local_extents(
    const std::array<std::size_t, DIM> &extents,
    const Topology<std::size_t, DIM, LayoutType> &topology, MPI_Comm comm) {
  // Check that topology includes two or less non-one elements
  std::array<std::size_t, DIM> local_extents{};
  std::array<std::size_t, DIM> local_starts{};
  std::copy(extents.begin(), extents.end(), local_extents.begin());
  auto total_size = KokkosFFT::Impl::total_size(topology);

  int rank, nprocs;
  ::MPI_Comm_rank(comm, &rank);
  ::MPI_Comm_size(comm, &nprocs);

  KOKKOSFFT_THROW_IF(total_size != static_cast<std::size_t>(nprocs),
                     "topology size must be identical to mpi size.");

  std::array<std::size_t, DIM> coords =
      rank_to_coord(topology, static_cast<std::size_t>(rank));

  for (std::size_t i = 0; i < extents.size(); i++) {
    if (topology.at(i) != 1) {
      std::size_t n = extents.at(i);
      std::size_t t = topology.at(i);

      std::size_t quotient  = n / t;
      std::size_t remainder = n % t;

      // Distribute the remainder acrocss the first few elements
      local_extents.at(i) =
          (coords.at(i) < remainder) ? quotient + 1 : quotient;
    }
  }

  std::vector<std::size_t> gathered_extents(DIM * total_size);
  MPI_Datatype mpi_data_type = Impl::mpi_datatype_v<std::size_t>;

  // Data are stored as
  // rank0: extents0
  // rank1: extents1
  // ...
  // rankn: extentsn
  MPI_Allgather(local_extents.data(), local_extents.size(), mpi_data_type,
                gathered_extents.data(), local_extents.size(), mpi_data_type,
                comm);

  std::size_t stride = total_size;
  for (std::size_t i = 0; i < topology.size(); i++) {
    if (topology.at(i) != 1) {
      // Maybe better to check that the shape is something like
      // n, n, n, n_remain
      std::size_t sum = 0;
      stride /= topology.at(i);
      for (std::size_t j = 0; j < coords.at(i); j++) {
        sum += gathered_extents.at(i + extents.size() * stride * j);
      }
      local_starts.at(i) = sum;
    }
  }

  return std::make_tuple(local_extents, local_starts);
}

/// \brief Compute the local extents and starts of the distributed View
/// Overload using a std::array for the topologies
/// \tparam DIM Number of dimensions
/// \param[in] extents Extents of the global View
/// \param[in] topology Topology of the data distribution
/// \param[in] comm MPI communicator
/// \return A tuple of local extents and starts of the distributed View
template <std::size_t DIM>
auto compute_local_extents(const std::array<std::size_t, DIM> &extents,
                           const std::array<std::size_t, DIM> &topology,
                           MPI_Comm comm) {
  return compute_local_extents(extents, Topology<std::size_t, DIM>(topology),
                               comm);
}

}  // namespace Distributed
}  // namespace KokkosFFT

#endif
