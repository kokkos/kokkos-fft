// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief MPI all-to-all communication for distributed data redistribution
/// This class implements MPI_Alltoall communication pattern on Kokkos Views,
/// used in distributed FFT operations for data redistribution between different
/// domain decompositions. The class handles the computation of send counts
/// based on the layout type and performs the actual MPI communication.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// The outermost dimension corresponds to the number of processes involved in
/// the communication.
/// For LayoutLeft, we expect the input and output views to have the shape
/// (n0, n1, ..., nprocs).
/// For LayoutRight, we expect the input and output views to have the shape
/// (nprocs, n0, ..., n_N).
/// The send_count and recv_count are the product of the other dimensions of the
/// input and output views. It will raise an error if the nprocs obtained from
/// the input and output views is not the same as the MPI size.
template <typename ExecutionSpace, typename ViewType>
struct All2All {
  static_assert(ViewType::rank() >= 2,
                "All2All: View rank must be larger than or equal to 2");
  using value_type = typename ViewType::non_const_value_type;

  ExecutionSpace m_exec_space;
  MPI_Comm m_comm;
  MPI_Datatype m_mpi_data_type;

  /// \brief Constructor for All2All communication
  /// \param[in] send Input view to be sent
  /// \param[in] recv Output view to be received
  /// \param[in] comm MPI communicator (default to MPI_COMM_WORLD)
  /// \param[in] exec_space Execution space (default to ExecutionSpace()
  /// instance)
  /// \throws std::runtime_error if the extent of the dimension to be transposed
  /// does not match MPI size
  All2All(const ViewType& send, const ViewType& recv,
          const MPI_Comm& comm            = MPI_COMM_WORLD,
          const ExecutionSpace exec_space = ExecutionSpace())
      : m_exec_space(exec_space),
        m_comm(comm),
        m_mpi_data_type(MPIDataType<value_type>::type()) {
    using LayoutType = typename ViewType::array_layout;
    int size_send    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                           ? send.extent_int(ViewType::rank() - 1)
                           : send.extent_int(0);
    int size_recv    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                           ? recv.extent_int(ViewType::rank() - 1)
                           : recv.extent_int(0);

    int size = 0;
    ::MPI_Comm_size(m_comm, &size);
    KOKKOSFFT_THROW_IF(
        (size_send != size) || (size_recv != size),
        "Extent of dimension to be transposed of send (" +
            std::to_string(size_send) + ") or recv (" +
            std::to_string(size_recv) +
            ") buffer does not match MPI size: " + std::to_string(size));

    // Compute the outermost dimension size
    int send_count = static_cast<int>(send.size()) / size_send;
    ::MPI_Alltoall(send.data(), send_count, m_mpi_data_type, recv.data(),
                   send_count, m_mpi_data_type, m_comm);
  }
};

/// \brief MPI all-to-all communication for distributed data redistribution
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// \param[in] exec_space Execution space
/// \param[in] send Input view to be sent
/// \param[in] recv Output view to be received
/// \param[in] comm MPI communicator
template <typename ExecutionSpace, typename ViewType>
void all2all(const ExecutionSpace& exec_space, const ViewType& send,
             const ViewType& recv, const MPI_Comm& comm) {
  static_assert(ViewType::rank() >= 2,
                "all2all: View rank must be larger than or equal to 2");
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::Distributed::all2all");
  All2All(send, recv, comm, exec_space);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
