// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_MPI_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_TYPES_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ValueType>
struct MPIDataType {};

template <>
struct MPIDataType<int> {
  static inline MPI_Datatype type() noexcept { return MPI_INT32_T; }
};

template <>
struct MPIDataType<std::size_t> {
  static inline MPI_Datatype type() noexcept { return MPI_UINT64_T; }
};

template <>
struct MPIDataType<float> {
  static inline MPI_Datatype type() noexcept { return MPI_FLOAT; }
};

template <>
struct MPIDataType<double> {
  static inline MPI_Datatype type() noexcept { return MPI_DOUBLE; }
};

template <>
struct MPIDataType<Kokkos::complex<float>> {
  static inline MPI_Datatype type() noexcept { return MPI_CXX_FLOAT_COMPLEX; }
};

template <>
struct MPIDataType<Kokkos::complex<double>> {
  static inline MPI_Datatype type() noexcept { return MPI_CXX_DOUBLE_COMPLEX; }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
