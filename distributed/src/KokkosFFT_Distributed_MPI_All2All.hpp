// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_MPI_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_ALL2ALL_HPP

#include <type_traits>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Comm.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief MPI all-to-all communication for distributed data redistribution
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// \param[in] send Input view to be sent
/// \param[out] recv Output view to be received
/// \param[in] comm MPI communicator
template <typename ViewType>
void all2all(const ViewType& send, const ViewType& recv, const MPI_Comm& comm) {
  static_assert(ViewType::rank() >= 2,
                "all2all: View rank must be larger than or equal to 2");
  using value_type = typename ViewType::non_const_value_type;
  using LayoutType = typename ViewType::array_layout;
  std::string msg  = KokkosFFT::Impl::is_real_v<value_type>
                         ? "KokkosFFT::Distributed::all2all[TPL_MPI,real]"
                         : "KokkosFFT::Distributed::all2all[TPL_MPI,complex]";
  Kokkos::Profiling::ScopedRegion region(msg);
  int size_send = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                      ? send.extent_int(ViewType::rank() - 1)
                      : send.extent_int(0);
  int size_recv = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                      ? recv.extent_int(ViewType::rank() - 1)
                      : recv.extent_int(0);
  int size      = 0;
  ::MPI_Comm_size(comm, &size);

  auto size_mismatch_msg = [size, size_send, size_recv]() -> std::string {
    std::string message;
    message = "Extent of dimension to be transposed of send (" +
              std::to_string(size_send) + ") or recv (" +
              std::to_string(size_recv) +
              ") buffer does not match MPI size: " + std::to_string(size);
    return message;
  };

  KOKKOSFFT_THROW_IF((size_send != size) || (size_recv != size),
                     size_mismatch_msg());

  // Compute the outermost dimension size
  auto mpi_data_type = mpi_datatype_v<value_type>;
  int send_count     = static_cast<int>(send.size()) / size_send;
  ::MPI_Alltoall(send.data(), send_count, mpi_data_type, recv.data(),
                 send_count, mpi_data_type, comm);
}

/// \brief MPI all-to-all communication for distributed data redistribution
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// \param[in] send Input view to be sent
/// \param[out] recv Output view to be received
/// \param[in] comm MPI communicator wrapper
template <typename ExecutionSpace, typename ViewType>
void all2all(const ViewType& send, const ViewType& recv,
             const ScopedMPIComm<ExecutionSpace>& comm) {
  all2all(send, recv, comm.comm());
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
