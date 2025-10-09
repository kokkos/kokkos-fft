// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename ViewType>
struct All2All {
  static_assert(ViewType::rank() >= 2,
                "All2All: View rank must be larger than or equal to 2");
  using value_type = typename ViewType::non_const_value_type;
  using LayoutType = typename ViewType::array_layout;

  ExecutionSpace m_exec_space;
  MPI_Comm m_comm;
  MPI_Datatype m_mpi_data_type;

  All2All(const ViewType& send, const ViewType& recv,
          const MPI_Comm& comm            = MPI_COMM_WORLD,
          const ExecutionSpace exec_space = ExecutionSpace())
      : m_exec_space(exec_space),
        m_comm(comm),
        m_mpi_data_type(MPIDataType<value_type>::type()) {
    // Compute the outermost dimension size
    int send_count = 0;
    if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      send_count = send.size() / send.extent(ViewType::rank() - 1);
    } else {
      send_count = send.size() / send.extent(0);
    }

    ::MPI_Alltoall(send.data(), send_count, m_mpi_data_type, recv.data(),
                   send_count, m_mpi_data_type, m_comm);
  }
};

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
