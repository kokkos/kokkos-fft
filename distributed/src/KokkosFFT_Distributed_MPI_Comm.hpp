// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_MPI_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_COMM_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Scoped wrapper for MPI_Comm
/// This class provides a scoped wrapper around an MPI_Comm object
/// just handles rank, size, comm and execution space.
///
/// \tparam ExecutionSpace Kokkos execution space type
template <typename ExecutionSpace>
struct ScopedMPIComm {
 private:
  using execution_space = ExecutionSpace;
  MPI_Comm m_comm       = MPI_COMM_NULL;
  execution_space m_exec_space;
  int m_rank = 0;
  int m_size = 1;

 public:
  /// \brief Construct a ScopedMPIComm
  /// \param[in] comm MPI communicator
  /// \param[in] exec_space Kokkos execution space
  explicit ScopedMPIComm(MPI_Comm comm, const ExecutionSpace &exec_space)
      : m_comm(comm), m_exec_space(exec_space) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);
  }

  /// \brief Construct a ScopedMPIComm
  /// \param[in] comm MPI communicator
  explicit ScopedMPIComm(MPI_Comm comm)
      : ScopedMPIComm(comm, ExecutionSpace{}) {}

  ScopedMPIComm() = delete;

  // Delete copy semantics
  ScopedMPIComm(const ScopedMPIComm &)            = delete;
  ScopedMPIComm &operator=(const ScopedMPIComm &) = delete;

  /// \brief Move constructor
  /// \param[in] other Another ScopedMPIComm to move from
  ScopedMPIComm(ScopedMPIComm &&other) noexcept
      : m_comm(other.m_comm),
        m_exec_space(other.m_exec_space),
        m_rank(other.m_rank),
        m_size(other.m_size) {
    other.m_comm = MPI_COMM_NULL;
  }

  /// \brief Move assignment operator
  /// \param[in] other Another ScopedMPIComm to move from
  ScopedMPIComm &operator=(ScopedMPIComm &&other) noexcept {
    if (this != &other) {
      m_comm       = other.m_comm;
      m_exec_space = other.m_exec_space;
      m_rank       = other.m_rank;
      m_size       = other.m_size;
      other.m_comm = MPI_COMM_NULL;
    }
    return *this;
  }

  ~ScopedMPIComm() = default;

  /// \brief Get the MPI communicator
  /// \return MPI_Comm object
  MPI_Comm comm() const { return m_comm; }

  /// \brief Get the execution space
  /// \return Execution space object
  execution_space exec_space() const { return m_exec_space; }

  /// \brief Get the rank of the process
  /// \return Rank of the process
  int rank() const { return m_rank; }

  /// \brief Get the size of the communicator
  /// \return Size of the communicator
  int size() const { return m_size; }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
